"""Chaos: tool handlers that genuinely sleep past their configured timeout.

The tools unit suite already asserts the TIMEOUT *mapping* for one slow handler;
these tests inject real latency and lock the failure-handling contract around
it: the timed-out handler coroutine is actually CANCELLED (not left running as
an orphan task), ``retry_hints`` govern how many real attempts run before the
TIMEOUT classification is final, and a tool timing out inside a full runtime
run surfaces as a typed failed TOOL row + ``TOOL_FAILED`` event instead of
hanging the run. Every awaited scenario is guarded by ``asyncio.wait_for`` so a
regression to a hang fails in seconds.
"""

from __future__ import annotations

import asyncio
from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType, RunEvent
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.models import ToolErrorCode, ToolInvocation
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async

#: Hard guard on every awaited scenario: a hang fails fast, not forever.
_GUARD_SECONDS = 5.0


class _SinkRecorder:
    """A minimal EventSink capturing every emitted event in order."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        self.events.append(event)


def _orphan_tasks() -> list[asyncio.Task[Any]]:
    """Every still-pending task on the loop other than the current one."""
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]


def test_sleeping_handler_times_out_is_classified_and_cancelled() -> None:
    """A handler sleeping far past its timeout yields TIMEOUT, is cancelled, no orphans."""
    sink = _SinkRecorder()
    registry = ToolRegistry()
    cancelled: list[bool] = []

    async def sleepy(args: dict[str, Any]) -> dict[str, Any]:
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return {"ok": True}  # pragma: no cover - must never be reached

    register_local_tool(
        registry,
        name="sleepy",
        handler=sleepy,
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
    )
    service = ToolService(registry, event_sink=sink)

    async def _scenario() -> tuple[Any, list[asyncio.Task[Any]]]:
        result = await asyncio.wait_for(
            service.execute(ToolInvocation(tool_name="sleepy", args={})),
            timeout=_GUARD_SECONDS,
        )
        return result, _orphan_tasks()

    result, orphans = run_async(_scenario())

    # Typed classification: TIMEOUT, not a generic execution error.
    assert result.outcome == "failed"
    assert result.error_code is ToolErrorCode.TIMEOUT
    assert "timeout" in (result.error_message or "")
    # The handler's coroutine really was cancelled — no orphan task keeps sleeping.
    assert cancelled == [True]
    assert orphans == []
    # The failure is observable: TOOL_CALLED then TOOL_FAILED with the typed code.
    types = [e.event_type for e in sink.events]
    assert types == [EventType.TOOL_CALLED, EventType.TOOL_FAILED]
    assert sink.events[1].payload["error_code"] == ToolErrorCode.TIMEOUT.value


def test_timeout_retries_run_real_attempts_then_classify() -> None:
    """retry_hints drive real re-attempts of a timing-out handler before TIMEOUT is final."""
    registry = ToolRegistry()
    attempts: list[int] = []

    async def sleepy(args: dict[str, Any]) -> dict[str, Any]:
        attempts.append(len(attempts) + 1)
        await asyncio.sleep(30.0)
        return {"ok": True}  # pragma: no cover - must never be reached

    register_local_tool(
        registry,
        name="sleepy",
        handler=sleepy,
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
        retry_hints={"max_attempts": 2},
    )
    service = ToolService(registry)

    result = run_async(
        asyncio.wait_for(
            service.execute(ToolInvocation(tool_name="sleepy", args={})),
            timeout=_GUARD_SECONDS,
        )
    )
    # TIMEOUT is retryable, so the handler genuinely ran twice — then failed typed.
    assert attempts == [1, 2]
    assert result.outcome == "failed"
    assert result.error_code is ToolErrorCode.TIMEOUT


def test_runtime_run_with_timing_out_tool_completes_typed_not_hung() -> None:
    """A full runtime run whose tool times out finishes promptly with a typed failure.

    The run itself must NOT hang or crash: the timed-out tool surfaces as an
    ``ERROR: TIMEOUT`` TOOL row (so the model can adapt) plus a runtime
    ``TOOL_FAILED`` event linked to the run's trace, and the turn still
    completes with a typed result.
    """
    storage = StorageService()
    registry = ToolRegistry()

    async def sleepy(args: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(30.0)
        return {"ok": True}  # pragma: no cover - must never be reached

    register_local_tool(
        registry,
        name="sleepy",
        handler=sleepy,
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
    )
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        tool_service=ToolService(registry, event_sink=storage),
    )
    persona = Persona(name="A")
    task = Task(
        title="t",
        prompt="go",
        context={"tool_names": ["sleepy"], "response_format": "AUTO_TOOLS"},
    )

    async def _scenario() -> tuple[Any, list[asyncio.Task[Any]]]:
        result = await asyncio.wait_for(
            rt.run_task_detailed(persona, task), timeout=_GUARD_SECONDS
        )
        return result, _orphan_tasks()

    result, orphans = run_async(_scenario())

    # The run terminated normally (typed result), with no orphan sleeper left.
    assert result.succeeded
    assert orphans == []
    # The tool exchange carries the typed TIMEOUT failure end-to-end.
    tool_rows = [m for m in result.thread.messages if m.role == MessageRole.TOOL]
    assert len(tool_rows) == 1
    assert tool_rows[0].metadata["tool_outcome"] == "failed"
    assert tool_rows[0].metadata["tool_return_metadata"]["error_code"] == "TIMEOUT"
    assert tool_rows[0].content.startswith("ERROR: TIMEOUT")
    # The runtime emitted a trace-linked TOOL_FAILED event for the exchange.
    events = run_async(storage.list_events())
    failed = [
        e
        for e in events
        if e.event_type == EventType.TOOL_FAILED and e.trace_id is not None
    ]
    assert len(failed) == 1
    assert failed[0].payload["tool_name"] == "sleepy"
    assert failed[0].payload["tool_outcome"] == "failed"
