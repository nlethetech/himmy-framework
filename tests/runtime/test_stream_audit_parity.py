"""Audit parity: streamed runs emit the same run-level events as buffered runs.

``run_task_detailed`` has always opened with ``AGENT_RUN_STARTED`` and closed
with a terminal ``AGENT_RUN_FINISHED`` (including an ``error='cancelled'``
terminal on cancellation/timeout). The streamed surfaces (``stream_task``, and
``stream_agent_loop`` whose first turn runs through ``stream_task``) previously
emitted neither, so streamed activity was invisible at the run level of the
audit trail. These tests pin the parity:

1. a fully-consumed ``stream_task`` emits exactly one STARTED and one FINISHED
   (success), tagged ``streamed: True``;
2. a client closing the stream early still produces the terminal
   ``FINISHED(error='cancelled')`` — silence is not an option in an audit log;
3. ``stream_agent_loop`` produces the same run-level event shape as
   ``run_agent_loop`` on an identical scripted conversation.

Offline-first: scripted in-memory managers; nothing to skip.
"""

from __future__ import annotations

import asyncio
from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType, RunEvent
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


class _ScriptedManager:
    """Returns a fixed sequence of turns (last repeats); no ``generate_stream``."""

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        self._specs = specs
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return f"scripted:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:
        spec = self._specs[min(self.calls, len(self._specs) - 1)]
        self.calls += 1
        cid = f"c{self.calls}"
        has_tools = bool(spec.get("tools"))
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=spec.get("text", ""),
            tool_calls=(
                [ToolCallRecord(tool_call_id=cid, tool_name="probe", args={})]
                if has_tools
                else []
            ),
            tool_returns=(
                [
                    ToolReturnRecord(
                        tool_call_id=cid, tool_name="probe", content={"ok": True}
                    )
                ]
                if has_tools
                else []
            ),
            input_tokens=1,
            output_tokens=1,
        )


def _runtime(
    specs: list[dict[str, Any]],
) -> tuple[SingleAgentRuntime, list[RunEvent]]:
    events: list[RunEvent] = []
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(_ScriptedManager(specs)),
        memory_store=StorageService(),
        on_event=lambda e: events.append(e),
    )
    return runtime, events


def _run_events(events: list[RunEvent]) -> list[RunEvent]:
    return [
        e
        for e in events
        if e.event_type in (EventType.AGENT_RUN_STARTED, EventType.AGENT_RUN_FINISHED)
    ]


def _go() -> tuple[Persona, Task]:
    return Persona(name="analyst"), Task(title="t", prompt="investigate")


# ------------------------------------------------------------------ stream_task
def test_stream_task_emits_started_and_finished() -> None:
    """A fully-consumed streamed run opens and closes the run-level trail."""
    runtime, events = _runtime([{"tools": False, "text": "answer"}])
    persona, task = _go()

    async def consume() -> None:
        async for _delta in runtime.stream_task(persona, task):
            pass

    run_async(consume())
    run_level = _run_events(events)
    assert [e.event_type for e in run_level] == [
        EventType.AGENT_RUN_STARTED,
        EventType.AGENT_RUN_FINISHED,
    ]
    started, finished = run_level
    assert started.payload["streamed"] is True
    assert finished.payload == {"status": "SUCCESS", "streamed": True}
    assert finished.error is None
    # Both events belong to the same run.
    assert started.trace_id == finished.trace_id


def test_stream_task_early_close_emits_cancelled_terminal() -> None:
    """A client dropping the stream still records FINISHED(error='cancelled')."""
    runtime, events = _runtime([{"tools": False, "text": "a" * 200}])
    persona, task = _go()

    async def drop_after_first() -> None:
        gen = runtime.stream_task(persona, task)
        async for _delta in gen:
            break  # client disconnects after the first chunk
        await gen.aclose()

    run_async(drop_after_first())
    run_level = _run_events(events)
    assert [e.event_type for e in run_level] == [
        EventType.AGENT_RUN_STARTED,
        EventType.AGENT_RUN_FINISHED,
    ]
    finished = run_level[-1]
    assert finished.error == "cancelled"
    assert finished.payload == {"status": "CANCELLED", "streamed": True}


def test_stream_task_cancellation_emits_cancelled_terminal() -> None:
    """Task cancellation mid-stream records the same terminal event."""
    runtime, events = _runtime([{"tools": False, "text": "a" * 200}])
    persona, task = _go()

    async def cancel_mid_stream() -> None:
        started = asyncio.Event()

        async def consume() -> None:
            async for _delta in runtime.stream_task(persona, task):
                started.set()
                await asyncio.sleep(3600)  # park so cancellation lands mid-run

        consumer = asyncio.create_task(consume())
        await started.wait()
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

    run_async(cancel_mid_stream())
    finished = _run_events(events)[-1]
    assert finished.event_type == EventType.AGENT_RUN_FINISHED
    assert finished.error == "cancelled"


def test_no_double_terminal_when_closed_after_done() -> None:
    """Closing the generator after the done delta does not re-emit FINISHED."""
    runtime, events = _runtime([{"tools": False, "text": "short"}])
    persona, task = _go()

    async def consume_then_close() -> None:
        gen = runtime.stream_task(persona, task)
        async for _delta in gen:
            pass
        await gen.aclose()

    run_async(consume_then_close())
    finished = [e for e in events if e.event_type == EventType.AGENT_RUN_FINISHED]
    assert len(finished) == 1
    assert finished[0].error is None


# ------------------------------------------------------------ loop-level parity
def test_stream_agent_loop_matches_run_agent_loop_event_shape() -> None:
    """Identical scripted runs leave identical run-level audit trails."""
    specs = [{"tools": True, "text": ""}, {"tools": False, "text": "done"}]
    persona, task = _go()

    buffered_rt, buffered_events = _runtime(specs)
    run_async(buffered_rt.run_agent_loop(persona, task, max_turns=3))

    streamed_rt, streamed_events = _runtime(specs)

    async def consume() -> None:
        async for _delta in streamed_rt.stream_agent_loop(persona, task, max_turns=3):
            pass

    run_async(consume())

    buffered_shape = [(e.event_type, e.error) for e in _run_events(buffered_events)]
    streamed_shape = [(e.event_type, e.error) for e in _run_events(streamed_events)]
    assert streamed_shape == buffered_shape
    assert buffered_shape == [
        (EventType.AGENT_RUN_STARTED, None),
        (EventType.AGENT_RUN_FINISHED, None),
    ]
