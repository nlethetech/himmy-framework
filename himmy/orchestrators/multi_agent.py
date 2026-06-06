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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import ChatThread
from himmy.agents.personas.persona import Persona
from himmy.core import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.runtime.single_agent import RunResult, SingleAgentRuntime
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
    stopped_reason: str = "final"  # final | max_turns | error
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
    ) -> None:
        """Wire the team and register the synthetic handoff/delegate tools."""
        self._runtime = runtime
        self._team = team
        self._registry = tool_registry
        self._max_turns = max_turns
        self._temperature = default_temperature
        self._delegate_max_turns = delegate_max_turns
        if on_event is None:
            self._on_event: list[OnEvent] = []
        elif callable(on_event):
            self._on_event = [on_event]
        else:
            self._on_event = list(on_event)
        self._register_synthetic_tools()

    # --------------------------------------------------------------- synthetic tools

    def _register_synthetic_tools(self) -> None:
        """Register ``transfer_to_<peer>``, ``ask_<worker>``, and ``final_answer``."""
        register_final_answer_tool(self._registry)
        handoff_targets = {t for m in self._team.members for t in m.handoffs}
        delegate_targets = {t for m in self._team.members for t in m.delegates}

        for target in handoff_targets:
            self._team.require(target)
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
            register_local_tool(
                self._registry,
                name=f"{DELEGATE_PREFIX}{target}",
                handler=self._make_delegate_handler(target),
                description=f"Ask {target} to handle a subtask and return its answer.",
                args_json_schema={
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                    "additionalProperties": False,
                },
                metadata={"synthetic": "delegate", "target": target},
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
            task = Task(
                title=f"delegate-{worker_name}",
                prompt=subtask,
                context={"tool_names": worker.tools, "model_key": worker.model_key},
            )
            loop = await self._runtime.run_agent_loop(
                worker.persona,
                task,
                sub_thread,
                max_turns=self._delegate_max_turns,
                llm_config=self._cfg(worker),
            )
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
        self, prompt: str, *, initial_thread: ChatThread | None = None
    ) -> MultiAgentResult:
        """Route ``prompt`` through the team until a final answer or ``max_turns``."""
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
        turns.append((active.name, result))

        while True:
            target = self._detect_handoff(result, active)
            if target is not None:
                source = active.name
                active = self._team.require(target)
                chain.append(active.name)
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
                return self._finish(thread, turns, active.name, chain, "max_turns")

            result = await self._runtime.continue_turn(
                active.persona,
                thread,
                task_context=self._ctx(active),
                llm_config=self._cfg(active),
            )
            turns.append((active.name, result))

    # --------------------------------------------------------------------- helpers

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
        return {"model_key": member.model_key, "tool_names": names}

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
                registry.register(event.to_record())
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
]
