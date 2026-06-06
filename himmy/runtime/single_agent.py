"""Runtime kernel: SingleAgentRuntime — the per-task conductor of one agent run.

``SingleAgentRuntime.run_task`` is the single public entry point that turns a
persona + task into an answered :class:`~himmy.agents.base_agent.thread.ChatThread`
plus a complete audit trail. It resolves/builds a context snapshot, renders the
system + task prompts, calls inference, replays tool exchanges onto the thread,
appends the assistant message, registers entities, and emits the full RunEvent
sequence. Every dependency except ``inference_service`` is optional and the
runtime degrades cleanly when one is absent.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.runtime.checkpoint import (
    APPROVED,
    AWAITING_APPROVAL,
    REJECTED,
    AgentCheckpoint,
    CheckpointStore,
    PendingToolCall,
)
from himmy.runtime.termination import final_answer_text, is_no_progress
from himmy.services.inference.models import (
    BoundTool,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    LLMConfig,
    ResponseFormat,
    ToolCallRecord,
    ToolReturnRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.entities.registry import EntityRegistry
    from himmy.services.context.service import ContextService
    from himmy.services.guardrails.base import GuardrailPipeline
    from himmy.services.inference.service import InferenceService, StreamDelta
    from himmy.services.prompts.manager import PromptManager
    from himmy.services.prompts.mapper import ContextPromptMapper
    from himmy.services.storage.service import MemoryStore


# An optional caller-facing event callback (RO-6). Invoked best-effort inside
# ``_emit`` alongside the storage/registry/observability sinks so a UI driving a
# long run can receive incremental progress without polling storage.
OnEvent = Callable[[RunEvent], Awaitable[None]]


@runtime_checkable
class ToolServiceProtocol(Protocol):
    """The minimal tool-service surface the runtime depends on (RO-12).

    Replaces the previous ``Any`` typing of ``tool_service`` with a structural
    contract: anything exposing ``bound_tools(names) -> list[BoundTool]`` (the
    offline binding the runtime feeds to ``InferenceRequest.bound_tools``)
    satisfies the runtime. ``ToolService`` conforms to this without inheritance.
    """

    def bound_tools(
        self, names: list[str] | None = None
    ) -> list[BoundTool]:  # pragma: no cover - structural typing
        ...

    async def execute(self, invocation: Any) -> Any:  # pragma: no cover - structural
        """Execute one tool invocation (used by HITL resume to run an approved tool)."""
        ...


@dataclass
class RunResult:
    """A typed view of one ``run_task`` outcome for the application layer (RO-5).

    ``run_task`` still returns the :class:`ChatThread` for back-compat;
    :meth:`SingleAgentRuntime.run_task_detailed` returns this richer object so a
    caller can read status/cost/structured output and the real typed tool
    exchanges (RO-4) without scraping thread rows or catching exceptions.
    """

    thread: ChatThread
    status: str
    output_text: str | None = None
    output_structured: Any = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tool_returns: list[ToolReturnRecord] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    model_path: str = ""
    provider_name: str = ""
    request_id: str | None = None
    trace_id: str | None = None
    workflow: Any = None
    workflow_complete: bool | None = None
    # True when the provider ran the FULL tool round-trip internally (e.g. pydantic-ai),
    # so this turn already holds the final answer and the loop must not continue.
    round_trip_complete: bool = False

    @property
    def succeeded(self) -> bool:
        """True when the terminal inference status is SUCCESS."""
        return self.status == InferenceStatus.SUCCESS.value


@dataclass
class AgentLoopResult:
    """The outcome of a runtime-owned multi-turn agent loop.

    ``turns`` is every :class:`RunResult` in order; ``final`` is the last.
    ``stopped_reason`` is one of ``final`` (model answered with no tool calls),
    ``max_turns``, ``budget``, or ``error``. Token/cost totals are summed across
    turns so a caller can see the whole run's spend.
    """

    thread: ChatThread
    turns: list[RunResult] = field(default_factory=list)
    stopped_reason: str = "final"
    checkpoint_id: str | None = None

    @property
    def final(self) -> RunResult:
        """The last turn's result."""
        return self.turns[-1]

    @property
    def turn_count(self) -> int:
        """How many model turns the loop ran."""
        return len(self.turns)

    @property
    def total_cost(self) -> float:
        """Summed provider cost across all turns."""
        return sum(t.cost for t in self.turns)

    @property
    def total_input_tokens(self) -> int:
        """Summed input tokens across all turns."""
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        """Summed output tokens across all turns."""
        return sum(t.output_tokens for t in self.turns)


#: The forced final-turn instruction when a tool-using loop ends with no answer.
_SYNTHESIS_NUDGE = (
    "You have already gathered the information you need from the tools above. "
    "Now answer the user's original question directly and completely, using only "
    "those results. Do not call any tools."
)

# Raw-I/O capture (debug inspector): per-field and per-message size caps so an
# opt-in capture can never bloat the event log unboundedly.
_IO_FIELD_CAP = 4000
_IO_MSG_CAP = 1500
_IO_MAX_MESSAGES = 16


def _truncate(text: str, cap: int) -> str:
    text = text or ""
    return text if len(text) <= cap else text[: cap - 1] + "…"


def build_io_capture(request: Any, response: Any) -> dict[str, Any]:
    """A bounded snapshot of one inference's raw I/O (for the trace inspector).

    Captures the messages sent to the model, the bound tool names, the raw response
    text, and the parsed tool calls — each size-capped. Opt-in (see the runtime's
    ``capture_io`` flag); never on by default, so there's zero cost or exposure unless
    a developer explicitly turns it on.
    """
    messages = []
    for m in (request.messages or [])[-_IO_MAX_MESSAGES:]:
        messages.append(
            {
                "role": getattr(m, "role", "?"),
                "content": _truncate(getattr(m, "content", "") or "", _IO_MSG_CAP),
            }
        )
    tool_calls = [
        {"tool": c.tool_name, "args": c.args} for c in (response.tool_calls or [])
    ]
    return {
        "model": getattr(response, "model_path", None),
        "messages": messages,
        "tools": [t.name for t in (request.bound_tools or [])],
        "response_text": _truncate(
            getattr(response, "output_text", "") or "", _IO_FIELD_CAP
        ),
        "tool_calls": tool_calls,
    }


class SingleAgentRuntime:
    """Conducts one agent run end-to-end: persona + task in, answered thread out.

    The runtime is stateless and per-task; multi-agent orchestrations compose
    multiple ``run_task`` calls. Pass only ``inference_service`` for a minimal,
    offline run; wire memory/context/tools/registry to gain persistence,
    evidenced context, tool calling, and lineage respectively.
    """

    def __init__(
        self,
        *,
        inference_service: InferenceService,
        memory_store: MemoryStore | None = None,
        tool_service: ToolServiceProtocol | None = None,
        context_service: ContextService | None = None,
        prompt_manager: PromptManager | None = None,
        context_prompt_mapper: ContextPromptMapper | None = None,
        entity_registry: EntityRegistry | None = None,
        default_model_key: str = "default",
        save_threads: bool = True,
        default_deadline_seconds: float | None = None,
        strict_snapshot: bool = False,
        on_event: OnEvent | list[OnEvent] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        input_guardrail: GuardrailPipeline | None = None,
        output_guardrail: GuardrailPipeline | None = None,
        capture_io: bool | None = None,
    ) -> None:
        """Wire the runtime; auto-create prompt manager/mapper when omitted.

        ``default_deadline_seconds`` (RO-1) is an optional wall-clock budget for
        the whole run (snapshot build + render + inference + persistence),
        overridable per call via ``run_task(..., deadline_seconds=...)``. A
        cancelled or timed-out run still emits a terminal ``AGENT_RUN_FINISHED``
        (``error='cancelled'``) and saves the partial thread before re-raising.
        ``strict_snapshot`` (RO-11) makes an explicitly-requested-but-failed
        snapshot raise instead of degrading silently. ``on_event`` (RO-6) is an
        optional caller-facing callback (or list) for streaming/progress.
        """
        self.inference_service = inference_service
        self.memory_store = memory_store
        self.tool_service = tool_service
        self.context_service = context_service
        self.entity_registry = entity_registry
        self.default_model_key = default_model_key
        self.save_threads = save_threads
        self.default_deadline_seconds = default_deadline_seconds
        self.strict_snapshot = strict_snapshot
        self._checkpoint_store = checkpoint_store
        self._input_guardrail = input_guardrail
        self._output_guardrail = output_guardrail
        # Opt-in raw-I/O capture for the trace inspector (off unless asked, or the
        # HIMMY_CAPTURE_IO env is truthy).
        self._capture_io = (
            capture_io
            if capture_io is not None
            else os.environ.get("HIMMY_CAPTURE_IO", "").lower() in ("1", "true", "yes")
        )
        self._on_event: list[OnEvent] = self._coerce_callbacks(on_event)

        # Auto-create the prompt primitives when available; they have no required
        # dependencies and the framework expects rendered prompts by default.
        if prompt_manager is None:
            try:
                from himmy.services.prompts.manager import PromptManager

                prompt_manager = PromptManager()
            except Exception:  # pragma: no cover - defensive
                prompt_manager = None
        if context_prompt_mapper is None:
            try:
                from himmy.services.prompts.mapper import ContextPromptMapper

                context_prompt_mapper = ContextPromptMapper()
            except Exception:  # pragma: no cover - defensive
                context_prompt_mapper = None
        self.prompt_manager = prompt_manager
        self.context_prompt_mapper = context_prompt_mapper

    @staticmethod
    def _coerce_callbacks(
        on_event: OnEvent | list[OnEvent] | None,
    ) -> list[OnEvent]:
        """Normalize the ``on_event`` argument to a list of callables."""
        if on_event is None:
            return []
        if isinstance(on_event, list):
            return [cb for cb in on_event if cb is not None]
        return [on_event]

    def add_event_listener(self, callback: OnEvent) -> None:
        """Register an additional caller-facing event callback (RO-6)."""
        self._on_event.append(callback)

    # ------------------------------------------------------------------ public
    async def run_task(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        llm_config: LLMConfig | None = None,
        snapshot_id: str | None = None,
        deadline_seconds: float | None = None,
    ) -> ChatThread:
        """Run one task for a persona and return the (appended) chat thread.

        Mirrors the documented sequence: snapshot resolve/build, prompt render +
        context projection, message appends (SYSTEM on first turn, USER, TOOL
        rows, ASSISTANT), entity registration/links, and the full event series.
        ``llm_config`` takes precedence over ``task.context`` for model knobs.

        Back-compat surface (returns the thread). The terminal assistant message
        metadata carries the full status/cost/token/error/structured contract for
        callers that cannot accept a new return type; :meth:`run_task_detailed`
        exposes the same as a typed :class:`RunResult`.
        """
        result = await self.run_task_detailed(
            persona,
            task,
            thread,
            llm_config=llm_config,
            snapshot_id=snapshot_id,
            deadline_seconds=deadline_seconds,
        )
        return result.thread

    async def run_task_detailed(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        llm_config: LLMConfig | None = None,
        snapshot_id: str | None = None,
        deadline_seconds: float | None = None,
    ) -> RunResult:
        """Run one task and return a typed :class:`RunResult` (RO-5).

        Identical pipeline to :meth:`run_task` but returns status/cost/structured
        output and the real typed ``tool_calls``/``tool_returns`` records, so the
        application layer can detect FAILED runs (invariant #4) without scraping
        thread rows or catching exceptions. An optional ``deadline_seconds``
        (RO-1) bounds the whole run; on timeout/cancellation a terminal
        ``AGENT_RUN_FINISHED(error='cancelled')`` is emitted and the partial
        thread saved before the :class:`asyncio.CancelledError` re-raises.
        """
        from himmy.agents.base_agent.thread import ChatThread

        ctx = dict(task.context or {})
        is_new_thread = thread is None
        if thread is None:
            thread = ChatThread(agent_id=persona.agent_id)
        trace_id = f"{thread.thread_id}:{task.task_id}"

        deadline = (
            deadline_seconds
            if deadline_seconds is not None
            else self.default_deadline_seconds
        )

        try:
            if deadline is not None and deadline > 0:
                async with _timeout(deadline):
                    return await self._run_task_body(
                        persona,
                        task,
                        thread,
                        ctx,
                        trace_id,
                        is_new_thread=is_new_thread,
                        llm_config=llm_config,
                        snapshot_id=snapshot_id,
                    )
            return await self._run_task_body(
                persona,
                task,
                thread,
                ctx,
                trace_id,
                is_new_thread=is_new_thread,
                llm_config=llm_config,
                snapshot_id=snapshot_id,
            )
        except (TimeoutError, asyncio.CancelledError):
            # RO-1: a cancelled run (external cancellation -> CancelledError) or a
            # deadline expiry (asyncio.timeout surfaces TimeoutError on exit) still
            # records a terminal event and persists the partial thread. We always
            # re-raise CancelledError so the run unwinds as a cancellation.
            await self._emit(
                RunEvent(
                    event_type=EventType.AGENT_RUN_FINISHED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    error="cancelled",
                    payload={"status": "CANCELLED"},
                )
            )
            await self._maybe_save_thread(thread)
            raise asyncio.CancelledError() from None

    async def run_agent_loop(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        max_turns: int = 6,
        cost_budget: float | None = None,
        llm_config: LLMConfig | None = None,
        hitl: bool = False,
        stop_on_no_progress: bool = False,
        synthesize_empty: bool = True,
        route_tools: bool = False,
        route_max_tools: int = 4,
    ) -> AgentLoopResult:
        """Run a bounded, runtime-owned agentic loop: act -> observe -> re-invoke.

        The first turn is a normal run; while a turn calls tools (so the model
        likely wants to act on their results), the runtime feeds the updated thread
        back for another model turn — until the model answers WITHOUT tool calls
        (``final``), or ``max_turns`` / ``cost_budget`` is reached, or a turn FAILS.
        Unlike delegating to a provider's opaque loop, the runtime *bounds* the
        turns, accrues spend, and emits an ``AGENT_TURN_COMPLETED`` event per turn.

        With ``hitl=True`` (requires a ``checkpoint_store``) the loop PAUSES when a
        turn calls a tool that requires approval: it persists an
        :class:`~himmy.runtime.checkpoint.AgentCheckpoint`, emits
        ``APPROVAL_REQUIRED``, and returns with ``stopped_reason='awaiting_approval'``
        and a ``checkpoint_id`` for :meth:`resume_agent_loop`.
        """
        if max_turns < 1:
            raise HimmyError("run_agent_loop requires max_turns >= 1.")
        if hitl and self._checkpoint_store is None:
            raise HimmyError("hitl=True requires a checkpoint_store on the runtime.")

        if route_tools:
            task = await self._route_tools(task, route_max_tools)

        first = await self.run_task_detailed(
            persona, task, thread, llm_config=llm_config
        )
        thread = first.thread
        ctx = dict(task.context or {})
        trace_id = f"{thread.thread_id}:{task.task_id}"
        await self._emit_turn_completed(trace_id, thread, persona, 1, first)

        result = await self._drive_loop(
            persona,
            task,
            thread,
            ctx,
            trace_id,
            turns=[first],
            max_turns=max_turns,
            cost_budget=cost_budget,
            llm_config=llm_config,
            hitl=hitl,
            stop_on_no_progress=stop_on_no_progress,
            turns_offset=0,
            cost_offset=0.0,
        )
        if synthesize_empty:
            result = await self._maybe_synthesize(result, persona, trace_id, llm_config)
        return result

    async def _route_tools(self, task: Task, max_tools: int) -> Task:
        """Narrow the bound tools to the relevant few for this task (Tier 1.3).

        A no-op unless a tool service is wired, the task hasn't already pinned
        ``tool_names``, and there are more candidate tools than ``max_tools``. Returns
        a copy of the task with ``context['tool_names']`` set to the routed subset.
        """
        if self.tool_service is None or max_tools < 1:
            return task
        ctx = task.context or {}
        if ctx.get("tool_names") is not None:
            return task  # caller already chose the tools — respect that
        registry = getattr(self.tool_service, "registry", None)
        if registry is None:
            return task
        candidates = [(d.name, d.description) for d in registry.list()]
        if len(candidates) <= max_tools:
            return task

        from himmy.runtime.tool_router import select_tools

        # Skills contribute "use this when …" hints so the router knows which tools a
        # capability implies for this request, beyond the bare prompt.
        query = task.prompt
        hints = ctx.get("skill_routing_hints") or []
        if hints:
            query = f"{query}\n\nRelevant capabilities:\n" + "\n".join(
                f"- {h}" for h in hints
            )

        selected = await select_tools(
            self.inference_service,
            query,
            candidates,
            max_tools=max_tools,
            model_key=str(ctx.get("model_key") or self.default_model_key),
        )
        return task.model_copy(update={"context": {**ctx, "tool_names": selected}})

    async def _maybe_synthesize(
        self,
        result: AgentLoopResult,
        persona: Persona,
        trace_id: str,
        llm_config: LLMConfig | None,
    ) -> AgentLoopResult:
        """One forced final turn when a tool-using loop ended with no answer (Tier 1.1).

        Small models often call a tool, get the result, then fail to write the final
        answer (an empty reply). When the loop stops with an empty answer but tools
        WERE used, run one more turn with tools unbound and an explicit instruction to
        answer from the results already gathered — converting an empty into an answer.
        """
        # Only rescue the genuine "model fell silent" stops. ``no_progress`` is an
        # opt-in deliberate halt whose stop reason callers rely on, so leave it.
        if result.stopped_reason not in ("final", "max_turns"):
            return result
        if (result.final.output_text or "").strip():
            return result  # already answered — nothing to nudge
        if not any(t.tool_calls for t in result.turns):
            return result  # no tools were used — synthesis has nothing to work from

        from himmy.agents.base_agent.task import Task

        nudge = Task(
            title="synthesis",
            prompt=_SYNTHESIS_NUDGE,
            context={"tool_names": []},  # unbind tools: force a text answer
        )
        synth = await self.run_task_detailed(
            persona, nudge, thread=result.thread, llm_config=llm_config
        )
        index = result.turn_count + 1
        await self._emit_turn_completed(trace_id, synth.thread, persona, index, synth)
        return AgentLoopResult(
            thread=synth.thread,
            turns=[*result.turns, synth],
            stopped_reason="synthesized",
        )

    async def continue_turn(
        self,
        persona: Persona,
        thread: ChatThread,
        *,
        task_context: dict[str, Any] | None = None,
        llm_config: LLMConfig | None = None,
    ) -> RunResult:
        """Run ONE more inference turn on an existing thread (no new user prompt).

        The model sees the thread as-is (including any prior tool results) and either
        calls more tools or produces a final answer. ``task_context`` carries the
        recognized run knobs (``tool_names``, ``model_key``, ``output_schema``), so a
        multi-agent orchestrator can switch the bound tool set / model per turn. This
        is the public seam over the runtime's own continuation step (used by
        :meth:`run_agent_loop`).
        """
        ctx = dict(task_context or {})
        trace_id = f"{thread.thread_id}:continue"
        return await self._continue_turn(
            persona, thread, ctx, trace_id, llm_config=llm_config
        )

    async def stream_task(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        llm_config: LLMConfig | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream one task's assistant reply as :class:`StreamDelta` chunks.

        Mirrors :meth:`run_task_detailed`'s pre-inference setup (snapshot, prompt
        render, system/user message appends) but delegates to
        :meth:`InferenceService.run_stream`, yielding incremental text. The final
        ``done`` delta carries the materialized response; the assistant message is
        appended to the thread before that delta is yielded. Single-turn (no tool
        loop) — intended for streaming a chat reply to a UI/stdout.
        """
        from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole

        if thread is None:
            thread = ChatThread(agent_id=persona.agent_id)
        ctx = dict(task.context or {})
        snapshot, _snapshot_id, _err = await self._resolve_snapshot(
            persona, task, ctx, None
        )
        system_prompt, task_prompt, _missing = self._render_prompts(
            persona, task, ctx, snapshot
        )
        if not any(m.role == MessageRole.SYSTEM for m in thread.messages):
            sys_msg = Message(role=MessageRole.SYSTEM, content=system_prompt)
            thread.append_message(sys_msg)
            self._register_message(sys_msg)
        user_msg = Message(
            role=MessageRole.USER, content=self._guard_input(task_prompt)
        )
        thread.append_message(user_msg)
        self._register_message(user_msg)

        request, _tool_names = self._build_request(thread, ctx, llm_config)
        async for delta in self.inference_service.run_stream(request):
            if delta.done and delta.response is not None:
                assistant = Message(
                    role=MessageRole.ASSISTANT,
                    content=delta.response.output_text or "",
                    metadata={"request_id": request.request_id, "streamed": True},
                )
                thread.append_message(assistant)
                self._register_message(assistant)
                self._register_thread_version(thread)
            yield delta

    async def resume_agent_loop(
        self,
        checkpoint_id: str,
        *,
        approved: bool,
        llm_config: LLMConfig | None = None,
        hitl: bool = True,
    ) -> AgentLoopResult:
        """Resume a paused agent run after a human approves or rejects the action.

        Rehydrates the checkpoint, applies the decision to each pending tool call
        — executing it (``approved=True``) and recording the real result, or
        recording a rejection — then drives one more model turn (so the model sees
        the outcome) and continues the loop. Idempotency is enforced: a checkpoint
        already ``approved``/``rejected`` cannot be resumed twice.
        """
        if self._checkpoint_store is None:
            raise HimmyError("resume_agent_loop requires a checkpoint_store.")
        checkpoint = self._checkpoint_store.load(checkpoint_id)
        if checkpoint is None:
            raise HimmyError(f"unknown checkpoint {checkpoint_id!r}.")
        if checkpoint.status != AWAITING_APPROVAL:
            raise HimmyError(
                f"checkpoint {checkpoint_id!r} already resolved ({checkpoint.status})."
            )
        if approved and self.tool_service is None:
            raise HimmyError(
                "cannot resume approved: no tool_service to execute the action."
            )

        from himmy.agents.base_agent.task import Task as _Task
        from himmy.agents.base_agent.thread import ChatThread as _ChatThread
        from himmy.agents.personas.persona import Persona as _Persona
        from himmy.services.tools.models import ToolInvocation

        persona = _Persona.model_validate(checkpoint.persona)
        task = _Task.model_validate(checkpoint.task)
        thread = _ChatThread.model_validate(checkpoint.thread)
        ctx = dict(checkpoint.ctx)
        trace_id = f"{thread.thread_id}:{task.task_id}"
        resume_llm = llm_config
        if resume_llm is None and checkpoint.llm_config is not None:
            resume_llm = LLMConfig.model_validate(checkpoint.llm_config)

        # Apply the human decision to each pending tool call, recording the outcome
        # on the thread as a TOOL message (so the next model turn sees it).
        for call in checkpoint.pending_tool_calls:
            if approved:
                assert self.tool_service is not None
                execution = await self.tool_service.execute(
                    ToolInvocation(
                        tool_name=call.tool_name,
                        args=dict(call.args),
                        metadata={"approved": True},
                    )
                )
                tool_returns = [
                    ToolReturnRecord(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        content=execution.result,
                        outcome=execution.outcome,
                        metadata={
                            "approved_by": "human",
                            "error_code": execution.error_code.value
                            if execution.error_code
                            else None,
                        },
                    )
                ]
                event_type = EventType.APPROVAL_GRANTED
            else:
                tool_returns = [
                    ToolReturnRecord(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        content={"rejected": True, "reason": "rejected by human"},
                        outcome="rejected",
                        metadata={"approved_by": "human"},
                    )
                ]
                event_type = EventType.APPROVAL_REJECTED
            synthetic = InferenceResponse(
                request_id=f"resume:{checkpoint_id}",
                status=InferenceStatus.SUCCESS,
                tool_calls=[
                    ToolCallRecord(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        args=dict(call.args),
                    )
                ],
                tool_returns=tool_returns,
            )
            await self._append_tool_messages(
                thread,
                synthetic,
                request_id=synthetic.request_id,
                trace_id=trace_id,
                agent_id=persona.agent_id,
            )
            await self._emit(
                RunEvent(
                    event_type=event_type,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={
                        "checkpoint_id": checkpoint_id,
                        "tool_name": call.tool_name,
                    },
                )
            )

        # Resolve the checkpoint exactly once (idempotency guard above).
        self._checkpoint_store.save(
            checkpoint.model_copy(update={"status": APPROVED if approved else REJECTED})
        )
        thread.version += 1
        self._register_thread_version(thread)

        # One continuation turn so the model reacts to the decision, then drive on.
        index = checkpoint.turns_completed + 1
        await self._emit(
            RunEvent(
                event_type=EventType.AGENT_TURN_STARTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload={"turn": index},
            )
        )
        result = await self._continue_turn(
            persona, thread, ctx, trace_id, llm_config=resume_llm
        )
        await self._emit_turn_completed(trace_id, thread, persona, index, result)
        return await self._drive_loop(
            persona,
            task,
            thread,
            ctx,
            trace_id,
            turns=[result],
            max_turns=checkpoint.max_turns,
            cost_budget=checkpoint.cost_budget,
            llm_config=resume_llm,
            hitl=hitl,
            stop_on_no_progress=False,
            turns_offset=checkpoint.turns_completed,
            cost_offset=checkpoint.cost_completed,
        )

    async def _drive_loop(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        *,
        turns: list[RunResult],
        max_turns: int,
        cost_budget: float | None,
        llm_config: LLMConfig | None,
        hitl: bool,
        stop_on_no_progress: bool,
        turns_offset: int,
        cost_offset: float,
    ) -> AgentLoopResult:
        """Drive continuation turns until a stop condition (shared by run/resume)."""
        while True:
            last = turns[-1]
            if not last.succeeded:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="error"
                )
            if hitl:
                pending = self._pending_approvals(last)
                if pending:
                    checkpoint = self._save_checkpoint(
                        persona,
                        task,
                        thread,
                        ctx,
                        llm_config,
                        max_turns,
                        cost_budget,
                        turns_offset + len(turns),
                        cost_offset + sum(t.cost for t in turns),
                        pending,
                    )
                    await self._emit(
                        RunEvent(
                            event_type=EventType.APPROVAL_REQUIRED,
                            trace_id=trace_id,
                            thread_id=thread.thread_id,
                            agent_id=persona.agent_id,
                            payload={
                                "checkpoint_id": checkpoint.checkpoint_id,
                                "tools": [p.tool_name for p in pending],
                            },
                        )
                    )
                    return AgentLoopResult(
                        thread=thread,
                        turns=turns,
                        stopped_reason="awaiting_approval",
                        checkpoint_id=checkpoint.checkpoint_id,
                    )
            if not last.tool_calls:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final"
                )
            if last.round_trip_complete:
                # The provider already ran the tool round-trip and produced the final
                # answer this turn (e.g. pydantic-ai/OpenAI); continuing would re-send a
                # history the strict API rejects. Stop here.
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final"
                )
            if final_answer_text(last) is not None:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final_answer"
                )
            if stop_on_no_progress and is_no_progress(turns):
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="no_progress"
                )
            if turns_offset + len(turns) >= max_turns:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="max_turns"
                )
            if (
                cost_budget is not None
                and cost_offset + sum(t.cost for t in turns) >= cost_budget
            ):
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="budget"
                )
            index = turns_offset + len(turns) + 1
            await self._emit(
                RunEvent(
                    event_type=EventType.AGENT_TURN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={"turn": index},
                )
            )
            result = await self._continue_turn(
                persona, thread, ctx, trace_id, llm_config=llm_config
            )
            turns.append(result)
            await self._emit_turn_completed(trace_id, thread, persona, index, result)

    @staticmethod
    def _pending_approvals(result: RunResult) -> list[PendingToolCall]:
        """The tool calls in a turn that were denied for lack of approval."""
        denied = {
            r.tool_call_id
            for r in result.tool_returns
            if r.outcome == "denied"
            and (r.metadata or {}).get("error_code") == "POLICY_BLOCKED"
        }
        return [
            PendingToolCall(
                tool_call_id=c.tool_call_id, tool_name=c.tool_name, args=dict(c.args)
            )
            for c in result.tool_calls
            if c.tool_call_id in denied
        ]

    def _save_checkpoint(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        llm_config: LLMConfig | None,
        max_turns: int,
        cost_budget: float | None,
        turns_completed: int,
        cost_completed: float,
        pending: list[PendingToolCall],
    ) -> AgentCheckpoint:
        """Persist a paused run as a durable checkpoint and return it."""
        assert self._checkpoint_store is not None
        checkpoint = AgentCheckpoint(
            persona=persona.model_dump(mode="json"),
            task=task.model_dump(mode="json"),
            thread=thread.model_dump(mode="json"),
            ctx=ctx,
            llm_config=llm_config.model_dump(mode="json")
            if llm_config is not None
            else None,
            max_turns=max_turns,
            cost_budget=cost_budget,
            turns_completed=turns_completed,
            cost_completed=cost_completed,
            pending_tool_calls=pending,
        )
        self._checkpoint_store.save(checkpoint)
        return checkpoint

    async def _emit_turn_completed(
        self,
        trace_id: str,
        thread: ChatThread,
        persona: Persona,
        index: int,
        result: RunResult,
    ) -> None:
        await self._emit(
            RunEvent(
                event_type=EventType.AGENT_TURN_COMPLETED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                cost=result.cost,
                payload={
                    "turn": index,
                    "tool_calls": len(result.tool_calls),
                    "status": result.status,
                },
            )
        )

    async def _maybe_compact(
        self,
        persona: Persona,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        llm_config: LLMConfig | None,
    ) -> None:
        """Summarize old turns in-place when the thread outgrows its token budget.

        Opt-in via ``ctx['compaction_spec']``. Keeps the system head + recent tail,
        replaces the middle with one model-written summary message, and emits a
        ``CONTEXT_COMPACTED`` event (the audit trail of what was condensed). A no-op
        when not configured or under budget.
        """
        spec = ctx.get("compaction_spec")
        if not spec:
            return
        from himmy.agents.base_agent.thread import Message, MessageRole
        from himmy.runtime.compaction import (
            SUMMARY_INSTRUCTION,
            ContextCompactor,
            estimate_tokens,
        )

        compactor = ContextCompactor(
            max_tokens=int(spec.get("max_tokens", 3000)),
            keep_recent=int(spec.get("keep_recent", 6)),
        )
        plan = compactor.plan(thread.messages)
        if not plan.should_compact:
            return

        span_text = compactor.render_span(plan.summarize)
        model_key = str(ctx.get("model_key") or self.default_model_key)
        summary_req = InferenceRequest(
            model_key=model_key,
            response_format=ResponseFormat.TEXT,
            messages=[
                InferenceMessage(role="system", content=SUMMARY_INSTRUCTION),
                InferenceMessage(role="user", content=span_text),
            ],
        )
        summary_resp = await self.inference_service.run(summary_req)
        summary_text = (summary_resp.output_text or "").strip()
        if not summary_text:
            return  # summarization failed/empty — leave history intact (safe)

        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[Summary of earlier conversation]\n{summary_text}",
            metadata={"compacted": True},
        )
        # Only apply if the summary is actually smaller than what it replaces — a verbose
        # summary of a tiny span would otherwise grow the context, not shrink it.
        if estimate_tokens(summary_msg.content) >= compactor.estimate(plan.summarize):
            return
        head = list(thread.messages[: plan.head_count])
        tail = list(thread.messages[plan.tail_start :])
        compacted_count = len(plan.summarize)
        thread.messages[:] = [*head, summary_msg, *tail]
        thread.version += 1

        await self._emit(
            RunEvent(
                event_type=EventType.CONTEXT_COMPACTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload={
                    "summarized_messages": compacted_count,
                    "before_tokens": plan.before_tokens,
                    "after_tokens": compactor.estimate(thread.messages),
                    "kept_recent": len(tail),
                },
            )
        )

    async def _continue_turn(
        self,
        persona: Persona,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        *,
        llm_config: LLMConfig | None,
    ) -> RunResult:
        """One more inference turn on an existing thread (no new user prompt).

        Builds the request from the thread as-is (so the model sees prior tool
        results), runs inference, replays tool exchanges, and appends the assistant
        turn. Persona/prompt lineage was linked by the first turn, so this only
        registers the new message + the bumped thread version.
        """
        from himmy.agents.base_agent.thread import Message, MessageRole

        await self._maybe_compact(persona, thread, ctx, trace_id, llm_config)
        request, tool_names = self._build_request(thread, ctx, llm_config)
        await self._emit(
            RunEvent(
                event_type=EventType.INFERENCE_REQUESTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                request_id=request.request_id,
                payload={"model_key": request.model_key, "tool_names": tool_names},
            )
        )
        response = await self.inference_service.run(request)
        await self._emit(
            RunEvent(
                event_type=(
                    EventType.INFERENCE_SUCCEEDED
                    if response.status == InferenceStatus.SUCCESS
                    else EventType.INFERENCE_FAILED
                ),
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                request_id=request.request_id,
                latency_ms=response.latency_ms,
                cost=response.cost,
                error=(response.error.message if response.error else None),
                payload={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    **(
                        {"io": build_io_capture(request, response)}
                        if self._capture_io
                        else {}
                    ),
                },
            )
        )
        await self._append_tool_messages(
            thread,
            response,
            request_id=request.request_id,
            trace_id=trace_id,
            agent_id=persona.agent_id,
        )

        assistant_text = response.output_text
        if assistant_text is None and response.output_structured is not None:
            assistant_text = json.dumps(response.output_structured, default=str)
        assistant_text = self._guard_output(assistant_text)
        error_message = response.error.message if response.error else None
        error_code = response.error.code.value if response.error else None
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=assistant_text or "",
            metadata={
                "request_id": request.request_id,
                "trace_id": trace_id,
                "status": response.status.value,
                "error": error_message,
                "error_code": error_code,
                "cost": response.cost,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "output_structured": response.output_structured,
            },
        )
        thread.append_message(assistant_message)
        thread.version += 1
        self._register_message(assistant_message)
        self._register_thread_version(thread)

        return RunResult(
            thread=thread,
            status=response.status.value,
            output_text=assistant_text,
            output_structured=response.output_structured,
            tool_calls=list(response.tool_calls),
            tool_returns=list(response.tool_returns),
            error=error_message,
            error_code=error_code,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
            latency_ms=response.latency_ms,
            model_path=response.model_path,
            provider_name=response.provider_name,
            request_id=request.request_id,
            trace_id=trace_id,
            round_trip_complete=bool(response.metadata.get("round_trip_complete")),
        )

    async def _run_task_body(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        *,
        is_new_thread: bool,
        llm_config: LLMConfig | None,
        snapshot_id: str | None,
    ) -> RunResult:
        """The actual run pipeline (wrapped by ``run_task_detailed`` for deadline)."""
        from himmy.agents.base_agent.thread import Message, MessageRole

        # --- 1. snapshot resolve/build -------------------------------------
        snapshot, snapshot_id, snapshot_error = await self._resolve_snapshot(
            persona, task, ctx, snapshot_id
        )

        # --- 2. render prompts (+ project snapshot keys) -------------------
        system_prompt, task_prompt, _missing = self._render_prompts(
            persona, task, ctx, snapshot
        )

        # --- 3. append SYSTEM (first turn) + USER --------------------------
        first_turn = not any(m.role == MessageRole.SYSTEM for m in thread.messages)
        if first_turn and system_prompt:
            system_message = Message(role=MessageRole.SYSTEM, content=system_prompt)
            thread.append_message(system_message)
            self._register_message(system_message)
        user_message = Message(
            role=MessageRole.USER, content=self._guard_input(task_prompt)
        )
        thread.append_message(user_message)
        self._register_message(user_message)

        # --- 4. register persona/agent/prompt entities --------------------
        persona_record = self._register_entity(persona)
        prompt_record = self._register_entity(task)

        # --- 5. AGENT_RUN_STARTED -----------------------------------------
        started_payload: dict[str, Any] = {
            "model_key": self._effective_model_key(ctx, llm_config),
            "snapshot_id": snapshot_id,
            "persona_name": persona.name,
        }
        if snapshot_error is not None:
            started_payload["snapshot_error"] = snapshot_error
        await self._emit(
            RunEvent(
                event_type=EventType.AGENT_RUN_STARTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload=started_payload,
            )
        )

        # --- 6. build the inference request -------------------------------
        request, tool_names = self._build_request(thread, ctx, llm_config)

        await self._emit(
            RunEvent(
                event_type=EventType.INFERENCE_REQUESTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                request_id=request.request_id,
                payload={
                    "model_key": request.model_key,
                    "route_override": request.route_override,
                    "response_format": request.response_format.value
                    if request.response_format
                    else None,
                    "rendered_prompt": task_prompt,
                    "retrieval_ctx": list((snapshot.fields or {}).keys())
                    if snapshot is not None
                    else [],
                    "tool_names": tool_names,
                },
            )
        )

        # --- 7. call inference --------------------------------------------
        # InferenceService.run never raises for provider/manager errors
        # (invariant #3); CancelledError still propagates and is handled by the
        # deadline wrapper above.
        response = await self.inference_service.run(request)

        if response.status == InferenceStatus.SUCCESS:
            await self._emit(
                RunEvent(
                    event_type=EventType.INFERENCE_SUCCEEDED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    request_id=request.request_id,
                    latency_ms=response.latency_ms,
                    cost=response.cost,
                    payload={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "model_path": response.model_path,
                        "provider_name": response.provider_name,
                        **(
                            {"io": build_io_capture(request, response)}
                            if self._capture_io
                            else {}
                        ),
                    },
                )
            )
        else:
            await self._emit(
                RunEvent(
                    event_type=EventType.INFERENCE_FAILED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    request_id=request.request_id,
                    latency_ms=response.latency_ms,
                    error=response.error.message
                    if response.error
                    else "inference failed",
                    payload={
                        "error_code": response.error.code.value
                        if response.error
                        else None
                    },
                )
            )

        # --- 8. replay TOOL exchanges onto the thread + emit tool events ---
        await self._append_tool_messages(
            thread,
            response,
            request_id=request.request_id,
            trace_id=trace_id,
            agent_id=persona.agent_id,
        )

        # --- 9. append ASSISTANT message ----------------------------------
        assistant_text = response.output_text
        if assistant_text is None and response.output_structured is not None:
            assistant_text = json.dumps(response.output_structured, default=str)
        assistant_text = self._guard_output(assistant_text)
        error_message = response.error.message if response.error else None
        error_code = response.error.code.value if response.error else None
        assistant_metadata: dict[str, Any] = {
            "request_id": request.request_id,
            "trace_id": trace_id,
            "latency_ms": response.latency_ms,
            "model_path": response.model_path,
            "provider_name": response.provider_name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost": response.cost,
            "status": response.status.value,
            # Invariant #4: stamp error + structured output so the application
            # layer can detect FAILED runs without exceptions.
            "error": error_message,
            "error_code": error_code,
            "output_structured": response.output_structured,
        }
        if response.workflow is not None:
            assistant_metadata["workflow_complete"] = response.workflow.is_complete
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=assistant_text or "",
            metadata=assistant_metadata,
        )
        thread.append_message(assistant_message)

        # --- 10. register message + chat_thread version + links -----------
        # RO-8: bump the thread version on the 2nd+ turn regardless of whether a
        # registry is wired, so persisted thread versions are correct even
        # without lineage.
        if not is_new_thread:
            thread.version += 1
        self._register_message(assistant_message)
        thread_record = self._register_thread_version(thread)
        self._link_lineage(
            persona_record=persona_record,
            prompt_record=prompt_record,
            thread_record=thread_record,
            snapshot=snapshot,
            persona=persona,
            thread=thread,
        )

        # --- 11. AGENT_RUN_FINISHED + save thread -------------------------
        await self._emit(
            RunEvent(
                event_type=EventType.AGENT_RUN_FINISHED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                latency_ms=response.latency_ms,
                cost=response.cost,
                error=error_message
                if response.status != InferenceStatus.SUCCESS
                else None,
                payload={"status": response.status.value},
            )
        )
        await self._maybe_save_thread(thread)

        return RunResult(
            thread=thread,
            status=response.status.value,
            output_text=assistant_text,
            output_structured=response.output_structured,
            tool_calls=list(response.tool_calls),
            tool_returns=list(response.tool_returns),
            error=error_message,
            error_code=error_code,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
            latency_ms=response.latency_ms,
            model_path=response.model_path,
            provider_name=response.provider_name,
            request_id=request.request_id,
            trace_id=trace_id,
            workflow=response.workflow,
            workflow_complete=(
                response.workflow.is_complete if response.workflow is not None else None
            ),
            round_trip_complete=bool(response.metadata.get("round_trip_complete")),
        )

    # ------------------------------------------------------------- snapshot
    def _guard_input(self, text: str) -> str:
        """Apply the input guardrail to a user prompt (redact); ``None`` → passthrough."""
        if self._input_guardrail is None:
            return text
        return self._input_guardrail.inspect(text, context={"stage": "input"}).text

    def _guard_output(self, text: str | None) -> str | None:
        """Apply the output guardrail to an assistant reply (redact); passthrough None."""
        if self._output_guardrail is None or text is None:
            return text
        return self._output_guardrail.inspect(text, context={"stage": "output"}).text

    async def _resolve_snapshot(
        self,
        persona: Persona,
        task: Task,
        ctx: dict[str, Any],
        snapshot_id: str | None,
    ) -> tuple[Any, str | None, str | None]:
        """Resolve a snapshot from arg/context, or build one.

        Returns ``(snapshot, resolved_id, snapshot_error)``. A snapshot was
        explicitly *requested* when a ``snapshot_id`` was supplied or a
        ``context_build_spec`` is present; in that case a load/build failure is
        diagnosed (RO-11): the error is captured for the AGENT_RUN_STARTED /
        CONTEXT_SNAPSHOT_BUILT payload, and — when ``strict_snapshot`` is on —
        re-raised as an :class:`HimmyError` so the caller knows the requested
        evidence was unavailable instead of silently running without it.
        """
        snapshot: Any = None
        snapshot_error: str | None = None
        resolved_id = snapshot_id or ctx.get("snapshot_id")
        requested = bool(
            snapshot_id
            or ctx.get("snapshot_id")
            or ctx.get("context_build_spec") is not None
        )

        if self.context_service is None:
            if requested and self.strict_snapshot:
                raise HimmyError("snapshot requested but no context_service is wired")
            return (
                None,
                resolved_id,
                ("no context_service wired" if requested else None),
            )

        # Load an existing snapshot when an id was supplied and storage is present.
        if resolved_id and self.memory_store is not None:
            loader = getattr(self.memory_store, "load_snapshot", None)
            if loader is not None:
                try:
                    snapshot = await loader(resolved_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - diagnose, don't crash
                    snapshot_error = f"snapshot load failed: {exc}"
                if snapshot is None and snapshot_error is None:
                    snapshot_error = f"snapshot {resolved_id!r} not found"

        # Otherwise build one from a declared build spec.
        if snapshot is None and ctx.get("context_build_spec") is not None:
            subject_id = ctx.get("context_subject_id") or persona.agent_id
            try:
                snapshot = await self.context_service.build_snapshot(
                    subject_id=subject_id,
                    task_id=task.task_id,
                    build_spec=ctx["context_build_spec"],
                    metadata=ctx.get("context_metadata"),
                )
                resolved_id = snapshot.snapshot_id
                snapshot_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - diagnose, don't crash
                snapshot = None
                snapshot_error = f"snapshot build failed: {exc}"

        if snapshot is not None:
            await self._emit(
                RunEvent(
                    event_type=EventType.CONTEXT_SNAPSHOT_BUILT,
                    thread_id=None,
                    agent_id=persona.agent_id,
                    payload={
                        "snapshot_id": getattr(snapshot, "snapshot_id", None),
                        "subject_id": getattr(snapshot, "subject_id", None),
                        "missing_required_keys": list(
                            getattr(snapshot, "missing_required_keys", []) or []
                        ),
                    },
                )
            )
            resolved_id = getattr(snapshot, "snapshot_id", resolved_id)
        elif snapshot_error is not None:
            # RO-11: surface the failure on the audit trail so 'requested but
            # unavailable' is distinguishable from 'no snapshot requested'.
            await self._emit(
                RunEvent(
                    event_type=EventType.CONTEXT_SNAPSHOT_BUILT,
                    thread_id=None,
                    agent_id=persona.agent_id,
                    error=snapshot_error,
                    payload={
                        "snapshot_id": resolved_id,
                        "snapshot_error": snapshot_error,
                    },
                )
            )
            if requested and self.strict_snapshot:
                raise HimmyError(snapshot_error)
        return snapshot, resolved_id, snapshot_error

    # --------------------------------------------------------------- prompts
    def _render_prompts(
        self,
        persona: Persona,
        task: Task,
        ctx: dict[str, Any],
        snapshot: Any,
    ) -> tuple[str, str, list[str]]:
        """Render system + task prompts and append projected snapshot blocks."""
        system_prompt = ""
        task_prompt = task.prompt
        missing: list[str] = []

        if self.prompt_manager is not None:
            from himmy.services.prompts.manager import (
                SystemPromptVariables,
                TaskPromptVariables,
            )

            # The persona's instructions are its directives — render them as
            # objectives so they reach the model EVEN WHEN a description is set.
            # (Previously the background used `description or instructions`, which
            # silently dropped every instruction whenever a description existed.)
            objectives = list(persona.instructions or [])
            objectives += list(getattr(persona, "objectives", []) or [])
            objectives += list(ctx.get("objectives", []) or [])
            # Skills: ctx override wins, else persona.metadata.skills/required_skills.
            skills = ctx.get("skills")
            if skills is None:
                skills = persona.metadata.get("skills") or list(
                    getattr(persona, "required_skills", []) or []
                )
            skills = list(skills or [])

            system_vars = SystemPromptVariables(
                role=ctx.get("role") or persona.role,
                persona=persona.description,
                objectives=objectives,
                skills=skills,
                datetime=ctx.get("datetime", ""),
            )
            system_prompt = self.prompt_manager.get_system_prompt(system_vars)

            task_vars = TaskPromptVariables(
                task=task.prompt,
                output_format=ctx.get("output_format", ""),
                output_schema=ctx.get("output_schema"),
            )
            task_prompt = self.prompt_manager.get_task_prompt(task_vars) or task.prompt

        # Prepend any system_prefix.
        prefix = ctx.get("system_prefix")
        if prefix:
            system_prompt = f"{prefix}\n\n{system_prompt}".strip()

        # Project snapshot keys into system/task blocks.
        map_spec = ctx.get("context_prompt_map_spec")
        if (
            self.context_prompt_mapper is not None
            and map_spec is not None
            and snapshot is not None
        ):
            try:
                sys_block, task_block, missing = self.context_prompt_mapper.project(
                    snapshot, map_spec
                )
                if sys_block:
                    system_prompt = f"{system_prompt}\n\n{sys_block}".strip()
                if task_block:
                    task_prompt = f"{task_prompt}\n\n{task_block}".strip()
            except Exception:  # pragma: no cover - defensive
                missing = []
        return system_prompt, task_prompt, missing

    # ----------------------------------------------------------- inference
    def _effective_model_key(
        self, ctx: dict[str, Any], llm_config: LLMConfig | None
    ) -> str:
        """Resolve the effective model key (llm_config > task.context > default)."""
        if llm_config is not None and llm_config.model_key:
            return llm_config.model_key
        return ctx.get("model_key") or self.default_model_key

    def _build_request(
        self,
        thread: Any,
        ctx: dict[str, Any],
        llm_config: LLMConfig | None,
    ) -> tuple[InferenceRequest, list[str] | None]:
        """Build the typed InferenceRequest with llm_config-over-context precedence."""
        from himmy.agents.base_agent.thread import MessageRole

        messages = [
            InferenceMessage(
                role=m.role.value if isinstance(m.role, MessageRole) else str(m.role),
                content=m.content,
                metadata=dict(m.metadata),
                tool_call_id=m.metadata.get("tool_call_id"),
                name=m.metadata.get("tool_name"),
            )
            for m in thread.messages
        ]

        model_key = self._effective_model_key(ctx, llm_config)
        generation_params: dict[str, Any] = {}
        response_format: ResponseFormat | None = None
        output_json_schema: dict[str, Any] | None = None
        workflow = None
        route_override = None
        timeout_seconds: float | None = None
        tool_names = ctx.get("tool_names")

        if llm_config is not None:
            response_format = llm_config.response_format
            output_json_schema = llm_config.output_json_schema
            workflow = llm_config.workflow
            route_override = llm_config.route_override
            timeout_seconds = llm_config.timeout_seconds
            if llm_config.temperature is not None:
                generation_params["temperature"] = llm_config.temperature
            if llm_config.max_tokens is not None:
                generation_params["max_tokens"] = llm_config.max_tokens
            if llm_config.top_p is not None:
                generation_params["top_p"] = llm_config.top_p
            if llm_config.use_cache is not None:
                # Forward the cache lever so InferenceService's response cache
                # (honored via generation_params['use_cache']) actually engages.
                generation_params["use_cache"] = llm_config.use_cache
            generation_params.update(llm_config.extra_params or {})
        else:
            # Fall back to task.context for the schema / format hints.
            output_json_schema = ctx.get("output_schema")
            fmt = ctx.get("response_format")
            if isinstance(fmt, ResponseFormat):
                response_format = fmt
            elif isinstance(fmt, str):
                try:
                    response_format = ResponseFormat(fmt)
                except ValueError:
                    response_format = None

        # Bind tools when a tool service is present.
        # RO-9: compute the WORKFLOW single-tool override OUTSIDE the tool_service
        # guard so the event payload always reflects the intended single tool,
        # and fail fast with a clear message when WORKFLOW can't actually bind it.
        bound_tools: list[BoundTool] = []
        tool_names_override: list[str] | None = None
        is_forced_workflow = (
            response_format == ResponseFormat.WORKFLOW
            and workflow is not None
            and workflow.current_tool_name is not None
        )
        if is_forced_workflow:
            tool_names_override = [workflow.current_tool_name]  # type: ignore[union-attr,list-item]

        if self.tool_service is not None:
            if is_forced_workflow:
                bound_tools = self.tool_service.bound_tools(tool_names_override)
                bound_names = {bt.name for bt in bound_tools}
                step_tool = tool_names_override[0]  # type: ignore[index]
                if step_tool not in bound_names:
                    raise HimmyError(
                        f"WORKFLOW response_format requires the step tool "
                        f"{step_tool!r} to be bound, but it is not registered "
                        f"with the tool_service"
                    )
            else:
                bound_tools = self.tool_service.bound_tools(tool_names)
        elif is_forced_workflow:
            # A WORKFLOW run with no tool_service can never bind the step tool;
            # surface the real cause instead of a generic INFERENCE_FAILED later.
            raise HimmyError(
                "WORKFLOW response_format requires a tool_service with the "
                f"named step tool {tool_names_override[0]!r} bound; none is wired"  # type: ignore[index]
            )

        request = InferenceRequest(
            model_key=model_key,
            messages=messages,
            response_format=response_format,
            output_json_schema=output_json_schema,
            workflow=workflow,
            generation_params=generation_params,
            route_override=route_override,
            bound_tools=bound_tools,
            tool_names_override=tool_names_override,
        )
        if timeout_seconds is not None:
            request.timeout_seconds = timeout_seconds
        return request, tool_names_override or tool_names

    # ------------------------------------------------------- tool messages
    async def _append_tool_messages(
        self,
        thread: Any,
        response: InferenceResponse,
        *,
        request_id: str,
        trace_id: str,
        agent_id: str | None,
    ) -> None:
        """Replay each tool call/return pair as a TOOL Message (full metadata).

        RO-2: per tool exchange this also emits a ``TOOL_CALLED`` event for the
        call and a ``TOOL_COMPLETED`` / ``TOOL_FAILED`` event for the paired
        return (keyed on ``ret.outcome``), threading tool_call_id / tool_name /
        tool_args / request_id / trace_id so the events link to the run like the
        other emissions and power the ai_call_log / lineage view.
        """
        from himmy.agents.base_agent.thread import Message, MessageRole

        returns_by_id: dict[str, ToolReturnRecord] = {
            r.tool_call_id: r for r in response.tool_returns
        }
        for call in response.tool_calls:
            ret = returns_by_id.get(call.tool_call_id)

            # TOOL_CALLED: emitted before the return is recorded.
            await self._emit(
                RunEvent(
                    event_type=EventType.TOOL_CALLED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=agent_id,
                    request_id=request_id,
                    tool_call_id=call.tool_call_id,
                    payload={
                        "tool_name": call.tool_name,
                        "tool_args": dict(call.args),
                    },
                )
            )

            content = ret.content if ret is not None else None
            try:
                content_text = (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, default=str)
                )
            except TypeError:  # pragma: no cover - defensive
                content_text = str(content)
            # Surface tool failures as a clear ERROR line so the model can adapt
            # instead of seeing a bare ``null`` content.
            if ret is not None and ret.outcome in ("failed", "denied"):
                meta = ret.metadata or {}
                code = meta.get("error_code", ret.outcome.upper())
                detail = meta.get("error_message") or content_text or ""
                content_text = f"ERROR: {code}: {detail}".strip().rstrip(":")
            message = Message(
                role=MessageRole.TOOL,
                content=content_text,
                metadata={
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "tool_outcome": ret.outcome if ret is not None else "unknown",
                    "tool_args": dict(call.args),
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "timestamp": message_timestamp(),
                    "tool_return_metadata": dict(ret.metadata)
                    if ret is not None
                    else {},
                },
            )
            thread.append_message(message)
            self._register_message(message)

            # TOOL_COMPLETED / TOOL_FAILED keyed on the return's outcome.
            outcome = ret.outcome if ret is not None else "unknown"
            completed = outcome == "success"
            await self._emit(
                RunEvent(
                    event_type=(
                        EventType.TOOL_COMPLETED if completed else EventType.TOOL_FAILED
                    ),
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=agent_id,
                    request_id=request_id,
                    tool_call_id=call.tool_call_id,
                    error=None if completed else f"tool outcome: {outcome}",
                    payload={
                        "tool_name": call.tool_name,
                        "tool_outcome": outcome,
                    },
                )
            )

    # --------------------------------------------------------------- entities
    def _register_entity(self, obj: Any) -> Any:
        """Register a domain object's record when a registry is wired; else None."""
        if self.entity_registry is None:
            return None
        to_record = getattr(obj, "to_record", None)
        if to_record is None:
            return None
        try:
            record = to_record()
            return self.entity_registry.register(record)
        except Exception:  # pragma: no cover - defensive
            return None

    def _register_message(self, message: Any) -> Any:
        """Register a Message entity (kind="message") when a registry is present."""
        return self._register_entity(message)

    def _register_thread_version(self, thread: Any) -> Any:
        """Project the current chat_thread version into a record (when a registry).

        RO-8: the version bump now happens in the run body (regardless of
        registry), so this helper only projects the record at the already-bumped
        version. Returns ``None`` when no registry is wired.
        """
        if self.entity_registry is None:
            return None
        try:
            record = thread.to_record()
            return self.entity_registry.register(record)
        except Exception:  # pragma: no cover - defensive
            return None

    def _link_lineage(
        self,
        *,
        persona_record: Any,
        prompt_record: Any,
        thread_record: Any,
        snapshot: Any,
        persona: Persona,
        thread: Any,
    ) -> None:
        """Wire the documented lineage relations between run artefacts."""
        if self.entity_registry is None or thread_record is None:
            return
        link = self.entity_registry.link
        try:
            if persona_record is not None:
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=persona_record.record_id,
                    relation="uses_persona",
                )
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=persona_record.record_id,
                    relation="thread_for_agent",
                )
            if prompt_record is not None:
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=prompt_record.record_id,
                    relation="in_thread",
                )
            if snapshot is not None:
                snapshot_record = getattr(snapshot, "to_record", None)
                if snapshot_record is not None:
                    sr = self.entity_registry.register(snapshot.to_record())
                    link(
                        from_record_id=thread_record.record_id,
                        to_record_id=sr.record_id,
                        relation="built_from",
                    )
                    link(
                        from_record_id=sr.record_id,
                        to_record_id=thread_record.record_id,
                        relation="observed_in_run",
                    )
        except Exception:  # pragma: no cover - defensive
            pass

    # --------------------------------------------------------------- helpers
    async def _emit(self, event: RunEvent) -> None:
        """Best-effort fan-out of one event to all configured sinks.

        Order: storage (the durable spine) -> entity registry -> observability
        span (invariant #6) -> caller-facing ``on_event`` callbacks (RO-6). Every
        sink is isolated so one failing sink can never break the run or starve
        the others, and ``CancelledError`` is honored so a cancelled run unwinds.
        """
        if self.memory_store is not None:
            appender = getattr(self.memory_store, "append_event", None)
            if appender is not None:
                try:
                    await appender(event)
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover - defensive
                    pass
        if self.entity_registry is not None:
            try:
                self.entity_registry.register(event.to_record())
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            from himmy.services.observability import emit_event_span

            emit_event_span(event)
        except Exception:  # pragma: no cover - defensive
            pass
        # RO-6: stream incremental progress to caller-supplied callbacks.
        for callback in self._on_event:
            try:
                await callback(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - never let a listener break the run
                pass

    async def _maybe_save_thread(self, thread: Any) -> None:
        """Persist the thread when ``save_threads`` and a memory store are present."""
        if not self.save_threads or self.memory_store is None:
            return
        saver = getattr(self.memory_store, "save_thread", None)
        if saver is None:
            return
        try:
            await saver(thread)
        except Exception:  # pragma: no cover - defensive
            pass


def message_timestamp() -> str:
    """Return an ISO timestamp for a tool message (delegates to the core helper)."""
    from himmy.core.ids import utc_now_iso

    return utc_now_iso()


def _timeout(seconds: float):
    """Return an ``asyncio.timeout(seconds)`` context manager (RO-1).

    ``asyncio.timeout`` exists on Python 3.11+ (the project targets 3.12). It
    raises :class:`asyncio.CancelledError` inside the block on expiry, which the
    runtime catches to emit a terminal cancelled event before re-raising.
    """
    # ``asyncio.timeout`` is always present on 3.11+; keep the lookup defensive
    # so the module still imports on any interpreter.
    timeout_cm = getattr(asyncio, "timeout", None)
    if timeout_cm is None:  # pragma: no cover - 3.10 fallback only
        raise HimmyError("deadline_seconds requires Python 3.11+ (asyncio.timeout)")
    return timeout_cm(seconds)


__all__ = [
    "SingleAgentRuntime",
    "RunResult",
    "AgentLoopResult",
    "ToolServiceProtocol",
    "OnEvent",
]
