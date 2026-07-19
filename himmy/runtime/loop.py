"""Multi-turn drive / routing / synthesis for :class:`SingleAgentRuntime`.

Extracted verbatim from ``single_agent.py`` (P3 decomposition, lane ``runtime``
step ``loop``). :class:`LoopDriver` owns the continuation-turn machinery:

* the ``_drive_loop`` stop-condition ladder shared by run/resume — the exact
  ``error``/``awaiting_approval``/``final``/``final_answer``/``no_progress``/
  ``max_turns``/``budget`` order and their ``stopped_reason`` strings;
* between-turns steering (``_drain_steer_queue``);
* the forced final-turn rescue (``_maybe_synthesize``); and
* adaptive tool routing (``_should_route`` / ``_route_tools``).

The runtime constructs one of these in ``__init__`` and its former methods become
thin delegating shims. ``_pending_approvals`` (a staticmethod tests call on the
class) and ``_save_checkpoint`` / ``_continue_turn`` / ``_emit_turn_completed`` /
``run_task_detailed`` stay on the runtime; this driver back-references them off
``self._rt`` so runtime wiring is read LIVE at call time. Behaviour — event ORDER,
stop-reason strings, turn bounds — is byte-for-byte identical to the
pre-extraction inline code.
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any

from himmy.core.events import EventType, RunEvent
from himmy.runtime.termination import final_answer_text, is_no_progress

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.runtime.single_agent import (
        AgentLoopResult,
        RunResult,
        SingleAgentRuntime,
    )
    from himmy.services.inference.models import LLMConfig


class LoopDriver:
    """Drives continuation turns / routing / synthesis for one runtime.

    Holds a back-reference to the owning :class:`SingleAgentRuntime` and reads its
    live wiring (``tool_service``, ``inference_service``, ``default_model_key``,
    ``_emit``, ``_continue_turn``, ``_save_checkpoint``, ``_pending_approvals``,
    ``run_task_detailed``, ``_emit_turn_completed``) at call time, so runtime
    reconfiguration between runs is honored exactly as when the logic lived inline
    on the runtime.
    """

    def __init__(self, runtime: SingleAgentRuntime) -> None:
        self._rt = runtime

    def should_route(self, route_tools: bool | None) -> bool:
        """Resolve the tri-state routing flag (explicit wins; None = adaptive)."""
        rt = self._rt
        if route_tools is not None:
            return route_tools
        registry = getattr(rt.tool_service, "registry", None)
        if registry is None:
            return False
        try:
            return len(registry.list()) > rt.AUTO_ROUTE_OVER_TOOLS
        except Exception:  # noqa: BLE001 - routing is an optimization, never a crash
            return False

    async def route_tools(self, task: Task, max_tools: int) -> Task:
        """Narrow the bound tools to the relevant few for this task (Tier 1.3).

        A no-op unless a tool service is wired, the task hasn't already pinned
        ``tool_names``, and there are more candidate tools than ``max_tools``. Returns
        a copy of the task with ``context['tool_names']`` set to the routed subset.
        """
        rt = self._rt
        if rt.tool_service is None or max_tools < 1:
            return task
        ctx = task.context or {}
        if ctx.get("tool_names") is not None:
            return task  # caller already chose the tools — respect that
        registry = getattr(rt.tool_service, "registry", None)
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
            rt.inference_service,
            query,
            candidates,
            max_tools=max_tools,
            model_key=str(ctx.get("model_key") or rt.default_model_key),
        )
        return task.model_copy(update={"context": {**ctx, "tool_names": selected}})

    async def maybe_synthesize(
        self,
        result: AgentLoopResult,
        persona: Persona,
        trace_id: str,
        llm_config: LLMConfig | None,
        ctx: dict[str, Any] | None = None,
    ) -> AgentLoopResult:
        """One forced final turn when a tool-using loop ended with no answer (Tier 1.1).

        Small models often call a tool, get the result, then fail to write the final
        answer (an empty reply). When the loop stops with an empty answer but tools
        WERE used, run one more turn with tools unbound and an explicit instruction to
        answer from the results already gathered — converting an empty into an answer.
        """
        import himmy.runtime.single_agent as _single_agent

        rt = self._rt
        # Only rescue the genuine "model fell silent" stops. ``no_progress`` is an
        # opt-in deliberate halt whose stop reason callers rely on, so leave it.
        if result.stopped_reason not in ("final", "max_turns"):
            return result
        if (result.final.output_text or "").strip():
            return result  # already answered — nothing to nudge
        if not any(t.tool_calls for t in result.turns):
            return result  # no tools were used — synthesis has nothing to work from

        from himmy.agents.base_agent.task import Task

        # Thread the ORIGINAL run context onto the rescue turn so it keeps the run's
        # model_key (a custom model must still write the final answer) AND its
        # ``context_subject_id`` (a governed run's synthesis message/thread/event
        # records must resolve to the subject, else they fail-closed DROP from the
        # spine). Only ``tool_names`` is overridden (unbind tools: force a text
        # answer); ``output_schema`` is dropped so a schema-constrained original run
        # doesn't force the free-text rescue answer through structured validation.
        nudge_ctx = {**(ctx or {}), "tool_names": []}
        nudge_ctx.pop("output_schema", None)
        nudge = Task(
            title="synthesis",
            prompt=_single_agent._SYNTHESIS_NUDGE,
            context=nudge_ctx,
        )
        synth = await rt.run_task_detailed(
            persona, nudge, thread=result.thread, llm_config=llm_config
        )
        index = result.turn_count + 1
        await rt._emit_turn_completed(trace_id, synth.thread, persona, index, synth)
        return _single_agent.AgentLoopResult(
            thread=synth.thread,
            turns=[*result.turns, synth],
            stopped_reason="synthesized",
        )

    async def drive_loop(
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
        steer_queue: queue.Queue[str] | None = None,
    ) -> AgentLoopResult:
        """Drive continuation turns until a stop condition (shared by run/resume)."""
        import himmy.runtime.single_agent as _single_agent

        rt = self._rt
        AgentLoopResult = _single_agent.AgentLoopResult
        while True:
            last = turns[-1]
            if not last.succeeded:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="error"
                )
            if hitl:
                pending = rt._pending_approvals(last)
                if pending:
                    checkpoint = rt._save_checkpoint(
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
                    await rt._emit(
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
            # Between-turns steering (opt-in): drain queued user guidance at the
            # top of this continuation turn so the next request includes it.
            if steer_queue is not None:
                rt._drain_steer_queue(steer_queue, thread)
            await rt._emit(
                RunEvent(
                    event_type=EventType.AGENT_TURN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={"turn": index},
                )
            )
            result = await rt._continue_turn(
                persona, thread, ctx, trace_id, llm_config=llm_config
            )
            turns.append(result)
            await rt._emit_turn_completed(trace_id, thread, persona, index, result)

    def drain_steer_queue(
        self, steer_queue: queue.Queue[str], thread: ChatThread
    ) -> None:
        """Append every queued steering text as a USER message on the thread.

        Each non-empty text becomes one USER message (``metadata={'steer': True}``)
        in arrival order, so the very next ``_continue_turn`` request — built from
        the thread as-is — carries the guidance. Thread-safe by construction:
        ``queue.Queue`` may be fed from any thread (an HTTP handler steering a
        background mission) while the loop drains it here on the event loop.
        """
        from himmy.agents.base_agent.thread import Message, MessageRole

        rt = self._rt
        injected = False
        while True:
            try:
                text = steer_queue.get_nowait()
            except queue.Empty:
                break
            content = str(text).strip()
            if not content:
                continue
            message = Message(
                role=MessageRole.USER, content=content, metadata={"steer": True}
            )
            thread.append_message(message)
            rt._register_message(message)
            injected = True
        if injected:
            thread.version += 1
            rt._register_thread_version(thread)
