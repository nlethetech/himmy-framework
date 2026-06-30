"""Multi-agent orchestration: handoff (peer transfer) + supervisor (delegation).

A team is a set of :class:`TeamMember`s, each a persona with its own tool set, model,
and two kinds of edges to other members:

* **handoff** (``handoffs``) — the member may *transfer control* to a peer. The
  orchestrator binds a synthetic ``transfer_to_<peer>`` tool; when the model calls it,
  control moves to that peer and the conversation continues on the SAME thread
  (swarm-style routing).
* **delegate** (``delegates``) — the member may *call a worker as a tool* and get its
  result back. The orchestrator binds a synthetic ``ask_<worker>`` tool whose handler
  runs the worker to completion in its own sub-thread and returns its answer; control
  then stays with the manager (supervisor / manager-worker).

Handoff is detected by the orchestrator from ``RunResult.tool_calls`` between turns;
delegation resolves inside the normal tool pipeline. The design mirrors
:class:`~himmy.orchestrators.workflow.WorkflowOrchestrator` (shared ``_emit`` plumbing)
and reuses the runtime's own continuation step via ``runtime.continue_turn``.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import ChatThread
from himmy.agents.personas.persona import Persona
from himmy.core import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.runtime.single_agent import _SYNTHESIS_NUDGE, RunResult, SingleAgentRuntime
from himmy.runtime.termination import (
    FINAL_ANSWER_TOOL,
    final_answer_text,
    is_no_progress,
    register_final_answer_tool,
)
from himmy.services.inference.models import LLMConfig, ResponseFormat
from himmy.services.tools.registry import ToolRegistry, register_local_tool

OnEvent = Callable[[RunEvent], Awaitable[None]]

HANDOFF_PREFIX = "transfer_to_"
DELEGATE_PREFIX = "ask_"


#: Marker key a delegate handler returns when its WORKER called an approval-gated tool.
#: The delegate sub-run's own raise is swallowed by the tool service into a generic error
#: return, so the manager loop instead detects this structured marker on the ``ask_*``
#: tool return and fails the whole run closed (naming the tool).
_HITL_UNSUPPORTED_MARKER = "__hitl_unsupported__"


class HitlUnsupportedError(HimmyError):
    """An approval-gated tool fired in an orchestration kind without nested HITL.

    Robust nested HITL ships for the ``graph``/workflow kind only; ``multi_agent`` and
    ``group_chat`` (and the multi_agent delegate sub-run) FAIL CLOSED — the gated tool
    is denied by the single tool gate and never runs un-approved, and rather than
    silently continuing past that denial the run is failed loudly, naming the tool.
    """


def _gated_tool_denied(result: RunResult) -> str | None:
    """The name of an approval-gated tool that was DENIED in this turn, if any.

    Mirrors the single-agent runtime's pending-approvals detection EXACTLY (a ``denied``
    outcome with ``error_code == 'POLICY_BLOCKED'``) so an ordinary policy denial is not
    misclassified — only a ``requires_approval`` gate is. A CAPABILITY/RBAC denial now
    carries the distinct ``CAPABILITY_DENIED`` code (red-team r6), so it is naturally
    excluded here and surfaces as an ordinary tool failure, not a fail-closed HITL gate.
    Returns the first such tool name (for the fail-closed error), else ``None``.
    """
    denied_ids = {
        r.tool_call_id
        for r in result.tool_returns
        if r.outcome == "denied"
        and (r.metadata or {}).get("error_code") == "POLICY_BLOCKED"
    }
    for call in result.tool_calls:
        if call.tool_call_id in denied_ids:
            return call.tool_name
    return None


def _delegate_marker(result: RunResult) -> dict[str, Any] | None:
    """The HITL-unsupported marker a delegate (``ask_*``) sub-run surfaced, if any.

    A delegate whose WORKER called an approval-gated tool returns the structured
    :data:`_HITL_UNSUPPORTED_MARKER` (its own raise is swallowed by the tool service);
    that marker rides back on the ``ask_*`` tool return's content. Returns the marker
    dict so the manager loop can fail the whole run closed naming the worker's tool.
    """
    for ret in result.tool_returns:
        content = ret.content
        if isinstance(content, dict) and content.get(_HITL_UNSUPPORTED_MARKER):
            return content
    return None


def _fail_closed_on_gated_tool(result: RunResult, *, kind: str) -> None:
    """Raise :class:`HitlUnsupportedError` if ``result`` denied an approval-gated tool.

    Called after each member turn. Two cases fail closed: the member itself called a
    ``requires_approval`` tool (denied with ``POLICY_BLOCKED``), OR a DELEGATE sub-run
    one level down did (surfaced via the structured marker on the ``ask_*`` return). The
    gated tool has ALREADY been denied by the tool gate (it never ran), so this only
    converts that silent denial into a clean FAILED run naming the tool — never a
    privileged-tool execution and never a silent continue.
    """
    tool = _gated_tool_denied(result)
    detail = ""
    if tool is None:
        marker = _delegate_marker(result)
        if marker is not None:
            tool = str(marker.get("tool") or "")
            kind = str(marker.get("kind") or kind)
            worker = marker.get("worker")
            detail = f" (delegated to worker {worker!r})" if worker else ""
    if tool:
        raise HitlUnsupportedError(
            f"approval-gated tool {tool!r} was called inside a {kind} run{detail}, but "
            f"human-in-the-loop approval is not supported for the {kind} kind "
            "(use a workflow or a graph team for HITL); the tool was NOT executed."
        )

#: Serializes synthetic-tool registration on a SHARED :class:`ToolRegistry`. Two
#: orchestrators constructed concurrently (e.g. per-request wiring on different
#: threads) must not interleave the check-then-register sequence below, or a genuine
#: name collision could slip past the identity check and silently clobber a tool.
_REGISTRATION_LOCK = threading.Lock()

#: WS4.6 — the caller-supplied ``task_context`` of the *current* team run. Published by
#: :meth:`MultiAgentOrchestrator.run` for the life of the run so per-turn contexts and
#: the delegate handlers (constructed at wiring time but executed mid-turn) can thread
#: ``context_subject_id`` into every member turn and worker sub-run. A
#: :class:`contextvars.ContextVar` keeps this correct under concurrent runs.
_RUN_TASK_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("himmy_team_run_ctx", default=None)
)


def _subject_ctx() -> dict[str, Any]:
    """The active run's propagated ``context_subject_id`` (``{}`` outside a run).

    Merged into every member-turn / delegate-sub-run task context so the runtime's
    own per-entry-point subject scopes resolve the SAME subject as the team run —
    otherwise each nested scope would re-publish ``None`` and the turn's spine
    records would lose their subject (and be dropped by a fail-closed
    ``ConsentAwareRegistry``) after a handoff.
    """
    base = _RUN_TASK_CONTEXT.get() or {}
    subject = base.get("context_subject_id")
    return {"context_subject_id": subject} if subject else {}


class TeamMember(BaseModel):
    """One agent in a team: a persona plus its tools, model, and collaboration edges."""

    name: str
    persona: Persona
    tools: list[str] = []
    model_key: str = "default"
    handoffs: list[str] = []
    delegates: list[str] = []


class AgentTeam(BaseModel):
    """A named set of members with a designated entry member."""

    members: list[TeamMember]
    entry: str

    def get(self, name: str) -> TeamMember | None:
        """Return the member named ``name`` (or ``None``)."""
        return next((m for m in self.members if m.name == name), None)

    def require(self, name: str) -> TeamMember:
        """Return the member named ``name`` or raise a clear error."""
        member = self.get(name)
        if member is None:
            raise HimmyError(f"team has no member {name!r}")
        return member


@dataclass
class MultiAgentResult:
    """The outcome of a team run: the thread, every turn, and the routing trail."""

    thread: ChatThread
    turns: list[tuple[str, RunResult]] = field(default_factory=list)
    final_agent: str = ""
    output_text: str | None = None
    handoff_chain: list[str] = field(default_factory=list)
    # final | final_answer | no_progress | max_turns | max_turns_synthesized | error
    stopped_reason: str = "final"
    total_cost: float = 0.0

    @property
    def turn_count(self) -> int:
        """The number of agent turns executed."""
        return len(self.turns)


class MultiAgentOrchestrator:
    """Routes a task across a team via handoff + delegation over a shared thread."""

    def __init__(
        self,
        runtime: SingleAgentRuntime,
        team: AgentTeam,
        tool_registry: ToolRegistry,
        *,
        on_event: OnEvent | list[OnEvent] | None = None,
        max_turns: int = 12,
        default_temperature: float = 0.1,
        delegate_max_turns: int = 6,
        synthesize_on_max_turns: bool = True,
    ) -> None:
        """Wire the team and register the synthetic handoff/delegate tools.

        ``synthesize_on_max_turns`` mirrors the runtime's ``synthesize_empty``: when
        the team hits ``max_turns`` mid-act (tool calls, no ``final_answer``), run one
        forced tool-less turn so the caller gets a plain-text answer instead of raw
        tool output. Pass ``False`` to keep the bare ``max_turns`` stop.
        """
        self._runtime = runtime
        self._team = team
        self._registry = tool_registry
        self._max_turns = max_turns
        self._temperature = default_temperature
        self._delegate_max_turns = delegate_max_turns
        self._synthesize_on_max_turns = synthesize_on_max_turns
        if on_event is None:
            self._on_event: list[OnEvent] = []
        elif callable(on_event):
            self._on_event = [on_event]
        else:
            self._on_event = list(on_event)
        self._register_synthetic_tools()

    # --------------------------------------------------------------- synthetic tools

    def _register_synthetic_tools(self) -> None:
        """Register ``transfer_to_<peer>``, ``ask_<worker>``, and ``final_answer``.

        Registration on a possibly SHARED registry is serialized by a module-level
        lock and identity-checked: re-registering an equivalent synthetic tool (same
        kind + target — e.g. a second orchestrator wired onto the same registry) is
        idempotent and rebinds the handler to this orchestrator, while a genuine name
        collision with a foreign tool (a user tool that happens to be called
        ``ask_<member>``) raises instead of silently clobbering it.
        """
        handoff_targets = {t for m in self._team.members for t in m.handoffs}
        delegate_targets = {t for m in self._team.members for t in m.delegates}

        with _REGISTRATION_LOCK:
            self._check_collision(FINAL_ANSWER_TOOL, "termination", None)
            register_final_answer_tool(self._registry)
            for target in handoff_targets:
                self._team.require(target)
                self._check_collision(f"{HANDOFF_PREFIX}{target}", "handoff", target)
                register_local_tool(
                    self._registry,
                    name=f"{HANDOFF_PREFIX}{target}",
                    handler=self._make_handoff_handler(target),
                    description=f"Transfer control of the conversation to {target}.",
                    args_json_schema={
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    metadata={"synthetic": "handoff", "target": target},
                )
            for target in delegate_targets:
                self._team.require(target)
                self._check_collision(f"{DELEGATE_PREFIX}{target}", "delegate", target)
                register_local_tool(
                    self._registry,
                    name=f"{DELEGATE_PREFIX}{target}",
                    handler=self._make_delegate_handler(target),
                    description=(
                        f"Ask {target} to handle a subtask and return its answer."
                    ),
                    args_json_schema={
                        "type": "object",
                        "properties": {"task": {"type": "string"}},
                        "required": ["task"],
                        "additionalProperties": False,
                    },
                    metadata={"synthetic": "delegate", "target": target},
                )

    def _check_collision(self, name: str, kind: str, target: str | None) -> None:
        """Raise when ``name`` is already registered as anything but OUR synthetic tool.

        An existing definition with the same synthetic identity (``kind`` + ``target``)
        is an equivalent tool from another (or a prior) orchestrator — re-registration
        is idempotent, so it passes. Anything else under that name belongs to someone
        else and must not be silently replaced.
        """
        existing = self._registry.get(name)
        if existing is None:
            return
        meta = existing.metadata or {}
        if meta.get("synthetic") == kind and meta.get("target") == target:
            return
        raise HimmyError(
            f"synthetic tool name collision: {name!r} is already registered on this "
            f"ToolRegistry and is not a synthetic {kind} tool for target {target!r} — "
            "rename the team member or give this team its own ToolRegistry."
        )

    def _make_handoff_handler(
        self, target: str
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """A handoff tool just signals intent; the orchestrator does the switch."""

        def _handler(args: dict[str, Any]) -> dict[str, Any]:
            return {"status": "transfer_requested", "to": target}

        return _handler

    def _make_delegate_handler(
        self, worker_name: str
    ) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        """Run the worker to completion in its own sub-thread; return its answer."""

        async def _handler(args: dict[str, Any]) -> dict[str, Any]:
            worker = self._team.require(worker_name)
            subtask = str(args.get("task", ""))
            sub_thread = ChatThread()
            # WS4.6: thread the run's subject into the worker's sub-run; without it the
            # worker's run_agent_loop would publish a None subject and its sub-thread
            # records would be dropped by a fail-closed ConsentAwareRegistry.
            task = Task(
                title=f"delegate-{worker_name}",
                prompt=subtask,
                context={
                    "tool_names": worker.tools,
                    "model_key": worker.model_key,
                    **_subject_ctx(),
                },
            )
            loop = await self._runtime.run_agent_loop(
                worker.persona,
                task,
                sub_thread,
                max_turns=self._delegate_max_turns,
                llm_config=self._cfg(worker),
            )
            # Fail closed if the WORKER (one level down) called an approval-gated tool:
            # the delegate sub-run is a nested loop the flat checkpoint cannot represent,
            # so the gated tool was denied and must not silently continue. The handler's
            # raise is swallowed by the tool service into a generic error return, so
            # instead we surface a STRUCTURED marker the manager loop detects (see
            # ``_fail_closed_on_gated_tool``) — guaranteeing the run fails loudly naming
            # the tool rather than the manager continuing past the denied delegate.
            for _turn in loop.turns:
                gated = _gated_tool_denied(_turn)
                if gated is not None:
                    return {
                        _HITL_UNSUPPORTED_MARKER: True,
                        "kind": "multi_agent delegate",
                        "tool": gated,
                        "worker": worker_name,
                    }
            answer = loop.final.output_text or ""
            await self._emit(
                RunEvent(
                    event_type=EventType.AGENT_DELEGATED,
                    thread_id=sub_thread.thread_id,
                    agent_id=worker.persona.agent_id,
                    payload={"worker": worker_name, "task": subtask, "answer": answer},
                )
            )
            return {"agent": worker_name, "answer": answer}

        return _handler

    # ------------------------------------------------------------------------- run

    async def run(
        self,
        prompt: str,
        *,
        initial_thread: ChatThread | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> MultiAgentResult:
        """Route ``prompt`` through the team until a final answer or ``max_turns``.

        ``task_context`` carries run-level context; the recognized key today is the
        WS4.6 ``context_subject_id`` of the participating human subject. It is
        published around the WHOLE run (mirroring the single-agent runtime's
        ``_subject_scope`` on every public entry point) and threaded into every
        member turn and delegate sub-run, so the orchestrator's own
        ``AGENT_HANDOFF``/``AGENT_DELEGATED`` events — emitted BETWEEN runtime turns,
        outside any per-turn scope — and every turn's spine records keep their
        subject under a governed ``ConsentAwareRegistry``. A no-op without a
        governed ``consent_decider`` on the runtime (the zero-config path is
        unchanged).
        """
        base_ctx = dict(task_context or {})
        token = _RUN_TASK_CONTEXT.set(base_ctx)
        try:
            with self._runtime._subject_scope(base_ctx):
                return await self._run_team(prompt, initial_thread)
        finally:
            _RUN_TASK_CONTEXT.reset(token)

    async def _run_team(
        self, prompt: str, initial_thread: ChatThread | None
    ) -> MultiAgentResult:
        """The routing loop proper (subject scope published by :meth:`run`)."""
        active = self._team.require(self._team.entry)
        thread = initial_thread or ChatThread()
        turns: list[tuple[str, RunResult]] = []
        chain = [active.name]

        result = await self._runtime.run_task_detailed(
            active.persona,
            Task(
                title=f"{active.name}-entry", prompt=prompt, context=self._ctx(active)
            ),
            thread=thread,
            llm_config=self._cfg(active),
        )
        _fail_closed_on_gated_tool(result, kind="multi_agent")
        turns.append((active.name, result))

        while True:
            target = self._detect_handoff(result, active)
            if target is not None:
                source = active.name
                active = self._team.require(target)
                chain.append(active.name)
                # The shared thread still carries the SOURCE persona's SYSTEM message;
                # re-render + re-inject the TARGET persona's so the next turn actually
                # runs under the new persona's instructions (not the old one).
                await self._runtime.reinject_system_prompt(
                    active.persona, thread, task_context=self._ctx(active)
                )
                await self._emit(
                    RunEvent(
                        event_type=EventType.AGENT_HANDOFF,
                        thread_id=thread.thread_id,
                        agent_id=active.persona.agent_id,
                        payload={"from": source, "to": active.name},
                    )
                )
            else:
                answer = final_answer_text(result)
                if answer is not None:
                    return self._finish(
                        thread, turns, active.name, chain, "final_answer", answer
                    )
                if not result.tool_calls:
                    return self._finish(thread, turns, active.name, chain, "final")
                if is_no_progress([r for _, r in turns]):
                    return self._finish(
                        thread, turns, active.name, chain, "no_progress"
                    )

            if len(turns) >= self._max_turns:
                if (
                    self._synthesize_on_max_turns
                    and result.tool_calls
                    and final_answer_text(result) is None
                ):
                    synth = await self._synthesize(thread, active)
                    turns.append((active.name, synth))
                    return self._finish(
                        thread,
                        turns,
                        active.name,
                        chain,
                        "max_turns_synthesized",
                        final_answer_text(synth) or synth.output_text,
                    )
                return self._finish(thread, turns, active.name, chain, "max_turns")

            result = await self._runtime.continue_turn(
                active.persona,
                thread,
                task_context=self._ctx(active),
                llm_config=self._cfg(active),
            )
            _fail_closed_on_gated_tool(result, kind="multi_agent")
            turns.append((active.name, result))

    # --------------------------------------------------------------------- helpers

    async def _synthesize(self, thread: ChatThread, member: TeamMember) -> RunResult:
        """One forced final turn when the team hit ``max_turns`` mid-act.

        Mirrors the runtime's ``_maybe_synthesize``: the loop stopped while the active
        agent was still calling tools (no ``final_answer``), so the caller would get
        raw tool output. Run one more turn with tools unbound and an explicit
        instruction to answer from the results already gathered.
        """
        nudge = Task(
            title="synthesis",
            prompt=_SYNTHESIS_NUDGE,
            # Unbind tools to force a text answer; keep the run's subject (WS4.6).
            context={
                "model_key": member.model_key,
                "tool_names": [],
                **_subject_ctx(),
            },
        )
        return await self._runtime.run_task_detailed(
            member.persona,
            nudge,
            thread=thread,
            llm_config=LLMConfig(
                model_key=member.model_key, temperature=self._temperature
            ),
        )

    def _detect_handoff(self, result: RunResult, active: TeamMember) -> str | None:
        """Return the declared handoff target the agent transferred to, if any."""
        allowed = {f"{HANDOFF_PREFIX}{h}" for h in active.handoffs}
        for call in result.tool_calls:
            if call.tool_name in allowed:
                return call.tool_name[len(HANDOFF_PREFIX) :]
        return None

    def _ctx(self, member: TeamMember) -> dict[str, Any]:
        """The per-turn run context: this member's bound tools + model.

        ``tool_names`` is ALWAYS set (an empty list binds NO tools) — leaving it unset
        would make the runtime bind every registered tool, which would leak other
        members' handoff/delegate tools into an agent that should not have them.
        """
        names = list(member.tools)
        names += [f"{HANDOFF_PREFIX}{h}" for h in member.handoffs]
        names += [f"{DELEGATE_PREFIX}{d}" for d in member.delegates]
        # A tool-using member also gets final_answer so it can end cleanly instead of
        # spinning; a tool-less member already terminates via a plain text answer.
        if names:
            names.append(FINAL_ANSWER_TOOL)
        # WS4.6: every member turn carries the run's subject so the runtime's nested
        # per-entry-point subject scopes resolve the SAME subject across handoffs.
        return {"model_key": member.model_key, "tool_names": names, **_subject_ctx()}

    def _cfg(self, member: TeamMember) -> LLMConfig:
        """LLMConfig for a member: AUTO_TOOLS when it has any callable edges/tools."""
        has_tools = bool(member.tools or member.handoffs or member.delegates)
        return LLMConfig(
            model_key=member.model_key,
            temperature=self._temperature,
            response_format=ResponseFormat.AUTO_TOOLS if has_tools else None,
        )

    def _finish(
        self,
        thread: ChatThread,
        turns: list[tuple[str, RunResult]],
        final_agent: str,
        chain: list[str],
        reason: str,
        answer: str | None = None,
    ) -> MultiAgentResult:
        """Assemble the final :class:`MultiAgentResult`."""
        last = turns[-1][1] if turns else None
        output = answer if answer is not None else (last.output_text if last else None)
        return MultiAgentResult(
            thread=thread,
            turns=turns,
            final_agent=final_agent,
            output_text=output,
            handoff_chain=chain,
            stopped_reason=reason,
            total_cost=sum(r.cost for _, r in turns),
        )

    async def _emit(self, event: RunEvent) -> None:
        """Best-effort emit to the runtime's store/registry + caller callbacks."""
        memory_store = getattr(self._runtime, "memory_store", None)
        appender = getattr(memory_store, "append_event", None) if memory_store else None
        if appender is not None:
            try:
                await appender(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                pass
        registry = getattr(self._runtime, "entity_registry", None)
        if registry is not None:
            try:
                # WS4.6: stamp the published run subject (None ⇒ unchanged record) so a
                # governed ConsentAwareRegistry can resolve — rather than fail closed
                # and drop — the orchestrator's handoff/delegation events.
                stamp = getattr(self._runtime, "_subject_metadata", None)
                metadata = stamp() if callable(stamp) else None
                registry.register(event.to_record(metadata=metadata))
            except Exception:  # pragma: no cover - defensive
                pass
        for callback in self._on_event:
            try:
                await callback(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - never let a listener break a run
                pass


__all__ = [
    "TeamMember",
    "AgentTeam",
    "MultiAgentOrchestrator",
    "MultiAgentResult",
    "OnEvent",
    "HitlUnsupportedError",
]
