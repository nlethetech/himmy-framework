"""Turn-level transient tool retry (timeout / rate-limited) in the runtime.

A transiently-failed tool call (the tools kernel's ``RETRYABLE_TOOL_CODES``
taxonomy) used to flow straight into the turn as a failed tool_return the agent
loop just carried on with — only provider-level errors were retried. The
runtime now retries the affected TOOL execution (never the whole inference) a
bounded number of times, emitting a ``TOOL_CALLED`` event tagged
``transient_retry`` per retry so the trace shows it. Non-transient failures and
exhausted retries keep the prior behavior exactly.

Offline-first: everything runs on the StubClientManager with no provider/DB.
"""

from __future__ import annotations

import asyncio
from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType, RunEvent
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


def _runtime_with_tool(
    handler: Any,
) -> tuple[SingleAgentRuntime, list[RunEvent]]:
    """A tool-wired runtime whose emitted events are collected into a list."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="probe",
        handler=handler,
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,  # a sleeping handler trips a genuine TIMEOUT
    )
    events: list[RunEvent] = []

    async def collect(event: RunEvent) -> None:
        events.append(event)

    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=StorageService(),
        tool_service=ToolService(registry),
        on_event=collect,
    )
    return runtime, events


def _flaky_handler(fail_times: int) -> tuple[Any, dict[str, int]]:
    """An async handler that times out for the first ``fail_times`` calls."""
    calls = {"count": 0}

    async def probe(args: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] <= fail_times:
            await asyncio.sleep(0.5)  # > the 0.05s tool timeout -> TIMEOUT
        return {"ok": True}

    return probe, calls


def _task(**ctx: Any) -> Task:
    return Task(
        title="t",
        prompt="probe it",
        context={
            "tool_names": ["probe"],
            "response_format": "AUTO_TOOLS",
            "tool_retry_backoff_seconds": 0.0,  # keep the tests fast
            **ctx,
        },
    )


def _retry_events(events: list[RunEvent]) -> list[RunEvent]:
    return [
        e
        for e in events
        if e.event_type == EventType.TOOL_CALLED and "transient_retry" in e.payload
    ]


def test_transient_failure_retried_then_turn_succeeds() -> None:
    """A tool failing once with TIMEOUT is retried; the turn carries the success."""
    handler, calls = _flaky_handler(fail_times=1)
    rt, events = _runtime_with_tool(handler)
    result = run_async(rt.run_task_detailed(Persona(name="a"), _task()))
    assert result.succeeded
    assert calls["count"] == 2  # initial attempt + one retry
    assert [r.outcome for r in result.tool_returns] == ["success"]
    retries = _retry_events(events)
    assert len(retries) == 1
    assert retries[0].payload["tool_name"] == "probe"
    assert retries[0].payload["transient_retry"] == 1
    assert retries[0].payload["error_code"] == "TIMEOUT"
    assert retries[0].trace_id == result.trace_id  # linked to the run


def test_persistent_transient_failure_is_bounded() -> None:
    """A persistently timing-out tool stops after the default bounded attempts."""
    handler, calls = _flaky_handler(fail_times=100)
    rt, events = _runtime_with_tool(handler)
    result = run_async(rt.run_task_detailed(Persona(name="a"), _task()))
    assert calls["count"] == 3  # initial + DEFAULT_TOOL_RETRY_ATTEMPTS (2)
    assert len(_retry_events(events)) == 2
    # Exhausted retries keep the prior behavior: the turn itself is still a
    # SUCCESS and the failed tool_return flows into the transcript.
    assert result.succeeded
    assert result.tool_returns[0].outcome == "failed"
    assert result.tool_returns[0].metadata.get("error_code") == "TIMEOUT"


def test_retry_count_configurable_via_ctx() -> None:
    """``tool_retry_attempts`` in task.context overrides the default bound."""
    handler, calls = _flaky_handler(fail_times=100)
    rt, events = _runtime_with_tool(handler)
    run_async(rt.run_task_detailed(Persona(name="a"), _task(tool_retry_attempts=1)))
    assert calls["count"] == 2  # initial + the single configured retry
    assert len(_retry_events(events)) == 1


def test_retry_disabled_with_zero_attempts() -> None:
    """``tool_retry_attempts: 0`` restores the exact pre-retry behavior."""
    handler, calls = _flaky_handler(fail_times=100)
    rt, events = _runtime_with_tool(handler)
    result = run_async(
        rt.run_task_detailed(Persona(name="a"), _task(tool_retry_attempts=0))
    )
    assert calls["count"] == 1
    assert _retry_events(events) == []
    assert result.tool_returns[0].outcome == "failed"


def test_non_transient_failure_not_retried() -> None:
    """An EXECUTION_ERROR (handler raised) keeps current behavior: no retry."""
    calls = {"count": 0}

    def broken(args: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        raise ValueError("boom")

    rt, events = _runtime_with_tool(broken)
    result = run_async(rt.run_task_detailed(Persona(name="a"), _task()))
    assert calls["count"] == 1
    assert _retry_events(events) == []
    assert result.tool_returns[0].outcome == "failed"
    assert result.tool_returns[0].metadata.get("error_code") == "EXECUTION_ERROR"


def test_agent_loop_turns_retry_transient_failures() -> None:
    """The retry seam applies inside run_agent_loop turns (the loop path)."""
    handler, calls = _flaky_handler(fail_times=1)
    rt, events = _runtime_with_tool(handler)
    result = run_async(rt.run_agent_loop(Persona(name="a"), _task(), max_turns=2))
    # Turn 1: failed attempt + successful retry (the stub's continuation
    # turn answers without re-calling the tool).
    assert calls["count"] == 2
    assert len(_retry_events(events)) == 1
    assert result.turns[0].tool_returns[0].outcome == "success"
    assert all(t.succeeded for t in result.turns)
