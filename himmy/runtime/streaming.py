"""Streaming surfaces for :class:`SingleAgentRuntime`.

Extracted verbatim from ``single_agent.py`` (P3 decomposition, lane ``runtime``
step ``stream``). :class:`StreamDriver` owns the incremental-delta machinery:

* ``stream_task`` — stream one single-turn assistant reply as ``StreamDelta``
  chunks (snapshot → prompt render → message appends → ``run_stream``), with the
  output guardrail enforced in BUFFER vs GUARD-AFTER regimes and a deterministic
  early-close of the provider stream;
* ``stream_agent_loop`` — stream tokens THROUGH the whole multi-turn tool loop,
  mirroring ``run_agent_loop``'s bounding / no-progress / cost-budget / HITL
  logic and emitting ``tool_call`` / ``tool_result`` / ``turn_end`` deltas plus a
  terminal ``done`` carrying the ``AgentLoopResult``;
* ``_stream_drive_loop`` — the streamed continuation-turn stop-condition ladder;
* ``_result_from_response`` / ``_text_deltas`` / ``_tool_deltas`` — the helpers
  that reconstruct a ``RunResult`` from the streamed first turn and re-chunk /
  surface continuation-turn text + tool exchanges.

The PUBLIC ``stream_task`` / ``stream_agent_loop`` async generators stay ON the
runtime and delegate here via ``async for d in inner: yield d`` wrapped in a
``try/finally: await inner.aclose()``, so GeneratorExit (client dropped the
stream) / CancelledError propagates into this driver's generators — and their
own ``finally``s that close the provider stream / inner turn generators — exactly
as when the logic lived inline on the runtime. This driver holds a back-reference
to the owning :class:`SingleAgentRuntime` and reads its live wiring
(``inference_service``, ``_checkpoint_store``, ``_output_guardrail``,
``_subject_scope``, ``_resolve_snapshot``, ``_render_guarded_prompts``,
``_guard_input``, ``_guard_output``, ``_emit``, ``_build_request``,
``_effective_model_key``, ``_register_message``, ``_register_thread_version``,
``_should_route``, ``_route_tools``, ``stream_task``, ``_append_tool_messages``,
``_emit_turn_completed``, ``_maybe_synthesize``, ``_pending_approvals``,
``_save_checkpoint``, ``_continue_turn``) at call time. Behaviour — event ORDER,
delta shapes/order, stop-reason strings, prompt bytes, exception types — is
byte-for-byte identical to the pre-extraction inline code.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Iterator
from typing import TYPE_CHECKING, Any, cast

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.runtime.termination import final_answer_text, is_no_progress
from himmy.services.inference.models import InferenceStatus

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.runtime.single_agent import (
        AgentLoopResult,
        RunResult,
        SingleAgentRuntime,
    )
    from himmy.services.inference.models import InferenceResponse, LLMConfig
    from himmy.services.inference.service import StreamDelta


class StreamDriver:
    """Runs the streaming surfaces for one runtime.

    Holds a back-reference to the owning :class:`SingleAgentRuntime` and reads its
    live wiring at call time, so runtime reconfiguration between runs is honored
    exactly as when the logic lived inline on the runtime. The public
    ``stream_task`` / ``stream_agent_loop`` async generators stay on the runtime
    and delegate into these bodies; early-close cleanup (provider stream / inner
    turn generators) is preserved byte-for-byte.
    """

    def __init__(self, runtime: SingleAgentRuntime) -> None:
        self._rt = runtime

    async def stream_task(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        llm_config: LLMConfig | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Body of :meth:`SingleAgentRuntime.stream_task` (see that docstring)."""
        import himmy.runtime.single_agent as _single_agent
        from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
        from himmy.services.inference.service import StreamDelta

        rt = self._rt
        if thread is None:
            thread = ChatThread(agent_id=persona.agent_id)
        ctx = _single_agent._validated_ctx(task.context, "stream_task")
        # WS4.6: publish the subject so the streamed turn's system/user/assistant
        # messages and the bumped thread version resolve to it (governed only; a no-op
        # when no consent_decider is wired).
        with rt._subject_scope(ctx):
            snapshot, _snapshot_id, _err = await rt._resolve_snapshot(
                persona, task, ctx, None
            )
            # Injected context (recalled memory / retrieved KB docs) is guarded here
            # so a poisoned memory/KB chunk is redacted/blocked before it reaches the
            # model (indirect prompt-injection seam) — parity with run_task_detailed.
            system_prompt, task_prompt, _missing = await rt._render_guarded_prompts(
                persona, task, ctx, snapshot, thread_id=thread.thread_id
            )
            if not any(m.role == MessageRole.SYSTEM for m in thread.messages):
                sys_msg = Message(role=MessageRole.SYSTEM, content=system_prompt)
                thread.append_message(sys_msg)
                rt._register_message(sys_msg)
            user_msg = Message(
                role=MessageRole.USER,
                content=await rt._guard_input(
                    task_prompt,
                    agent_id=persona.agent_id,
                    thread_id=thread.thread_id,
                ),
            )
            thread.append_message(user_msg)
            rt._register_message(user_msg)

            trace_id = f"{thread.thread_id}:{task.task_id}"
            # Audit parity with run_task_detailed: a streamed run is a run — it
            # opens with AGENT_RUN_STARTED and ALWAYS closes with a terminal
            # AGENT_RUN_FINISHED (success, failure, or 'cancelled' when the client
            # drops the stream / the consuming task is cancelled mid-run).
            await rt._emit(
                RunEvent(
                    event_type=EventType.AGENT_RUN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={
                        "model_key": rt._effective_model_key(ctx, llm_config),
                        "persona_name": persona.name,
                        "streamed": True,
                    },
                )
            )

            request, _tool_names = rt._build_request(
                thread, ctx, llm_config, trace_id=trace_id
            )
            # The output guardrail must be enforced on a streamed answer too (audit
            # parity with the non-streaming paths). Two regimes:
            #  * BUFFER: the output guard can WITHHOLD blocked content (DLP ``…:block``
            #    / blocklist / blocking injection). An already-streamed secret can't be
            #    recalled, so we must NOT stream — we drain the provider stream silently,
            #    guard the full answer, then emit it as one ``done`` delta.
            #  * GUARD-AFTER: a redact-only / grounding-only output guard. Stream the
            #    tokens, then guard the final text; if it changed, the PERSISTED message
            #    and the ``done`` payload carry the guarded text (the already-streamed
            #    deltas can't be recalled, but the durable copy must be clean) and a
            #    correction delta is emitted so a consumer that materializes from
            #    ``delta`` rather than ``response`` also lands on the guarded text.
            buffer_output = (
                rt._output_guardrail is not None
                and rt._output_guardrail.suppresses_output_content()
            )
            # ``run_stream`` is an async generator; own the reference so an early
            # close/cancellation of THIS generator can close it in the finally.
            stream = rt.inference_service.run_stream(request)
            run_finished = False
            try:
                async for delta in stream:
                    if delta.done and delta.response is not None:
                        raw_text = delta.response.output_text or ""
                        guarded_text = (
                            await rt._guard_output(
                                raw_text,
                                agent_id=persona.agent_id,
                                trace_id=trace_id,
                                thread_id=thread.thread_id,
                            )
                            or ""
                        )
                        corrected = guarded_text != raw_text
                        assistant = Message(
                            role=MessageRole.ASSISTANT,
                            content=guarded_text,
                            metadata={
                                "request_id": request.request_id,
                                "streamed": True,
                                **({"guarded": True} if corrected else {}),
                            },
                        )
                        thread.append_message(assistant)
                        rt._register_message(assistant)
                        rt._register_thread_version(thread)
                        await rt._emit(
                            RunEvent(
                                event_type=EventType.AGENT_RUN_FINISHED,
                                trace_id=trace_id,
                                thread_id=thread.thread_id,
                                agent_id=persona.agent_id,
                                latency_ms=delta.response.latency_ms,
                                cost=delta.response.cost,
                                error=(
                                    delta.response.error.message
                                    if delta.response.error is not None
                                    and delta.response.status != InferenceStatus.SUCCESS
                                    else None
                                ),
                                payload={
                                    "status": delta.response.status.value,
                                    "streamed": True,
                                },
                            )
                        )
                        run_finished = True
                        # The ``done`` delta must never carry the pre-guard text. Rewrite
                        # its response (and the textual ``delta``) to the guarded answer.
                        guarded_response = delta.response.model_copy(
                            update={"output_text": guarded_text}
                        )
                        if buffer_output:
                            # Nothing was streamed; deliver the whole guarded answer now.
                            yield StreamDelta(
                                request_id=request.request_id,
                                delta=guarded_text,
                                index=delta.index,
                                done=True,
                                response=guarded_response,
                            )
                        else:
                            if corrected:
                                # The already-streamed tokens were the raw answer; emit a
                                # correction so a delta-materializing consumer ends clean.
                                yield StreamDelta(
                                    request_id=request.request_id,
                                    delta="",
                                    index=delta.index,
                                    event_type="guarded_output",
                                    event_payload={"output_text": guarded_text},
                                )
                            yield delta.model_copy(update={"response": guarded_response})
                        continue
                    if buffer_output and not delta.done:
                        # Suppress intermediate text in BUFFER mode — the answer is only
                        # safe to surface after the guard runs on the ``done`` delta.
                        continue
                    yield delta
            except (GeneratorExit, asyncio.CancelledError):
                # Early termination at a yield point: record the terminal event
                # (mirrors run_task_detailed's cancellation leg) before unwinding.
                if not run_finished:
                    await rt._emit(
                        RunEvent(
                            event_type=EventType.AGENT_RUN_FINISHED,
                            trace_id=trace_id,
                            thread_id=thread.thread_id,
                            agent_id=persona.agent_id,
                            error="cancelled",
                            payload={"status": "CANCELLED", "streamed": True},
                        )
                    )
                raise
            finally:
                # GeneratorExit (client closed the stream) / CancelledError land
                # here at a yield point; close the provider stream NOW so its own
                # cleanup runs before we unwind, then the exception re-raises.
                await stream.aclose()

    async def stream_agent_loop(
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
        route_tools: bool | None = None,
        route_max_tools: int = 4,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Body of :meth:`SingleAgentRuntime.stream_agent_loop` (see that docstring)."""
        import himmy.runtime.single_agent as _single_agent
        from himmy.services.inference.service import StreamDelta

        rt = self._rt
        _single_agent._validate_max_turns(max_turns, "stream_agent_loop")
        # Reject a malformed context BEFORE the tool router / first turn runs.
        _single_agent._validated_ctx(task.context, "stream_agent_loop")
        if hitl and rt._checkpoint_store is None:
            raise HimmyError("hitl=True requires a checkpoint_store on the runtime.")

        if rt._should_route(route_tools):
            task = await rt._route_tools(task, route_max_tools)

        from himmy.agents.base_agent.thread import ChatThread as _ChatThread

        # Own the thread reference up front so we keep operating on the SAME thread
        # stream_task mutates (it creates one internally when None — we pre-create it
        # here instead so the continuation turns and tool replay land on it).
        if thread is None:
            thread = _ChatThread(agent_id=persona.agent_id)

        ctx = dict(task.context or {})
        # WS4.6: publish the subject for the WHOLE streamed loop (governed only; a
        # no-op when no consent_decider is wired) so the per-turn continuation
        # messages, turn-completed events, and any synthesis turn — all emitted
        # OUTSIDE the first turn's own scope — resolve to the subject. The inner
        # stream_task / run_task_detailed scopes nest cleanly inside this one.
        with rt._subject_scope(ctx):
            # Own the inner turn generators so EARLY termination — the client
            # closing this generator (GeneratorExit) or the consuming task being
            # cancelled (CancelledError) at any yield point — closes the in-flight
            # turn in the ``finally`` below instead of leaving the suspended inner
            # generators (and the provider stream beneath them) dangling until the
            # event loop's lazy async-generator finalizer runs.
            first_stream = rt.stream_task(persona, task, thread, llm_config=llm_config)
            drive: AsyncGenerator[StreamDelta | AgentLoopResult, None] | None = None
            try:
                # --- first turn: stream tokens via stream_task, materialize it ---
                first_response: InferenceResponse | None = None
                async for delta in first_stream:
                    if delta.done:
                        # Swallow the single-turn ``done`` delta: this loop owns the
                        # one terminal ``done`` (carrying the AgentLoopResult).
                        # Capture the materialized response so we can reconstruct
                        # the turn result.
                        first_response = delta.response
                        continue
                    yield delta

                # run_stream always yields a done delta
                assert first_response is not None
                trace_id = f"{thread.thread_id}:{task.task_id}"

                first = self._result_from_response(first_response, trace_id=trace_id)
                first.thread = thread  # the real thread stream_task ran on
                # stream_task does NOT replay TOOL exchanges; replay them now (so a
                # continuation turn sees the tool results), surface them as events.
                await rt._append_tool_messages(
                    thread,
                    first_response,
                    request_id=first.request_id or first_response.request_id,
                    trace_id=trace_id,
                    agent_id=persona.agent_id,
                )
                # stream_task already registered a chat_thread version snapshot that
                # only held the ASSISTANT message (taken BEFORE these tools existed).
                # Bump + re-project so the registered version includes the replayed
                # TOOL messages, matching the resume / non-streaming paths. (The
                # in-memory order stays [ASSISTANT, TOOL...] here — a full reorder to
                # the non-streaming [TOOL..., ASSISTANT] is DEFERRED: stream_task owns
                # the ASSISTANT append and never replays tools, so moving it risks the
                # single-turn contract / delta ordering for LOW benefit.)
                if first.tool_calls:
                    thread.version += 1
                    rt._register_thread_version(thread)
                for tool_delta in self._tool_deltas(first):
                    yield tool_delta
                await rt._emit_turn_completed(trace_id, thread, persona, 1, first)
                yield StreamDelta(
                    request_id=first.request_id or first_response.request_id,
                    event_type="turn_end",
                    event_payload={"turn": 1, "tool_calls": len(first.tool_calls)},
                )

                # --- continuation turns: mirror _drive_loop, streaming each turn ---
                result: AgentLoopResult | None = None
                drive = self._stream_drive_loop(
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
                )
                async for item in drive:
                    if isinstance(item, _single_agent.AgentLoopResult):
                        result = item
                        break
                    yield item

                assert result is not None
                if synthesize_empty:
                    # Reuse the exact non-streaming synthesis rescue so an empty
                    # tool-using answer is converted to a text answer identically.
                    result = await rt._maybe_synthesize(
                        result, persona, trace_id, llm_config, ctx
                    )
                yield StreamDelta(
                    request_id=first.request_id or first_response.request_id,
                    done=True,
                    event_type="done",
                    event_payload={"result": result},
                )
            finally:
                # Runs on normal completion (both acloses are no-ops on exhausted
                # generators) AND on GeneratorExit/CancelledError, which then
                # re-raise naturally after the inner generators are closed.
                await first_stream.aclose()
                if drive is not None:
                    await drive.aclose()

    async def _stream_drive_loop(
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
    ) -> AsyncGenerator[StreamDelta | AgentLoopResult, None]:
        """Drive streamed continuation turns until a stop condition.

        Mirrors :meth:`_drive_loop`'s EXACT stop-condition / no-progress /
        cost-budget / HITL logic (run from a continuation perspective, so the
        ``turns_offset`` / ``cost_offset`` are zero), but per continuation turn it
        runs the turn and yields its text + ``tool_call`` / ``tool_result`` /
        ``turn_end`` :class:`StreamDelta`s. The final yielded item is the terminal
        :class:`AgentLoopResult` (so the caller stops iterating and owns the single
        ``done`` delta). The checkpoint / pending-approval helpers are shared with
        the non-streaming loop — nothing is duplicated.
        """
        import himmy.runtime.single_agent as _single_agent
        from himmy.services.inference.service import StreamDelta

        rt = self._rt
        AgentLoopResult = _single_agent.AgentLoopResult
        while True:
            last = turns[-1]
            if not last.succeeded:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="error"
                )
                return
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
                        len(turns),
                        sum(t.cost for t in turns),
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
                    yield AgentLoopResult(
                        thread=thread,
                        turns=turns,
                        stopped_reason="awaiting_approval",
                        checkpoint_id=checkpoint.checkpoint_id,
                    )
                    return
            if not last.tool_calls:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final"
                )
                return
            if last.round_trip_complete:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final"
                )
                return
            if final_answer_text(last) is not None:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final_answer"
                )
                return
            if stop_on_no_progress and is_no_progress(turns):
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="no_progress"
                )
                return
            if len(turns) >= max_turns:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="max_turns"
                )
                return
            if cost_budget is not None and sum(t.cost for t in turns) >= cost_budget:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="budget"
                )
                return
            index = len(turns) + 1
            await rt._emit(
                RunEvent(
                    event_type=EventType.AGENT_TURN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={"turn": index},
                )
            )
            # Continuation turns buffer then re-chunk for deterministic offline
            # replay (the stub streams in 24-char chunks; matching that keeps the
            # reassembled text identical across turns).
            result = await rt._continue_turn(
                persona, thread, ctx, trace_id, llm_config=llm_config
            )
            turns.append(result)
            for text_delta in self._text_deltas(result):
                yield text_delta
            for tool_delta in self._tool_deltas(result):
                yield tool_delta
            await rt._emit_turn_completed(trace_id, thread, persona, index, result)
            yield StreamDelta(
                request_id=result.request_id or "",
                event_type="turn_end",
                event_payload={"turn": index, "tool_calls": len(result.tool_calls)},
            )

    @staticmethod
    def _result_from_response(
        response: InferenceResponse, *, trace_id: str
    ) -> RunResult:
        """Reconstruct a :class:`RunResult` from a streamed first turn's response.

        :meth:`stream_task` yields its terminal ``done`` delta carrying the
        materialized :class:`InferenceResponse` but no typed result; this rebuilds
        the same :class:`RunResult` shape :meth:`_continue_turn` produces so the
        streamed first turn drives the loop identically to a non-streamed one.
        """
        import himmy.runtime.single_agent as _single_agent

        assistant_text = response.output_text
        if assistant_text is None and response.output_structured is not None:
            assistant_text = json.dumps(response.output_structured, default=str)
        error_message = response.error.message if response.error else None
        error_code = response.error.code.value if response.error else None
        return _single_agent.RunResult(
            thread=cast("ChatThread", None),  # not used by the loop's stop logic
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
            request_id=response.request_id,
            trace_id=trace_id,
            workflow=response.workflow,
            workflow_complete=(
                response.workflow.is_complete if response.workflow is not None else None
            ),
            round_trip_complete=bool(response.metadata.get("round_trip_complete")),
        )

    @staticmethod
    def _text_deltas(result: RunResult) -> Iterator[StreamDelta]:
        """Re-chunk a continuation turn's text at 24 chars (stub-faithful)."""
        from himmy.services.inference.service import StreamDelta

        text = result.output_text or ""
        request_id = result.request_id or ""
        index = 0
        for start in range(0, len(text), 24):
            yield StreamDelta(
                request_id=request_id, delta=text[start : start + 24], index=index
            )
            index += 1

    @staticmethod
    def _tool_deltas(result: RunResult) -> Iterator[StreamDelta]:
        """Yield a ``tool_call`` + paired ``tool_result`` delta per tool exchange."""
        from himmy.services.inference.service import StreamDelta

        returns_by_id = {r.tool_call_id: r for r in result.tool_returns}
        request_id = result.request_id or ""
        for call in result.tool_calls:
            yield StreamDelta(
                request_id=request_id,
                event_type="tool_call",
                event_payload={
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "tool_args": dict(call.args),
                },
            )
            ret = returns_by_id.get(call.tool_call_id)
            yield StreamDelta(
                request_id=request_id,
                event_type="tool_result",
                event_payload={
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "outcome": ret.outcome if ret is not None else "unknown",
                    "content": ret.content if ret is not None else None,
                },
            )
