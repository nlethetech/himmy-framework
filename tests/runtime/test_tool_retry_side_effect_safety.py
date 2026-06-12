"""Transient tool retry must not re-fire SIDE-EFFECTING tools on TIMEOUT.

A ``TIMEOUT`` does not mean the tool didn't run — an HTTP POST / write may have
committed server-side after the client gave up. The runtime's turn-level transient
retry used to re-execute the call regardless of intent, duplicating writes/sends/
charges. The fix: a TIMEOUT is only retried for a provably read-only tool; a write
(or an ambiguously-named) tool is not re-run on timeout. The OTHER transient codes
(``RATE_LIMITED`` / ``PROVIDER_UNAVAILABLE``) mean the call never reached the tool, so
they stay retryable for every tool.

Offline-first: a sleeping handler trips a real TIMEOUT; a raising handler simulates a
RATE_LIMITED upstream — no provider/DB.
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
from himmy.services.tools.models import ToolErrorCode
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


def _runtime(registry: ToolRegistry) -> tuple[SingleAgentRuntime, list[RunEvent]]:
    events: list[RunEvent] = []

    async def collect(event: RunEvent) -> None:
        events.append(event)

    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=StorageService(),
        tool_service=ToolService(registry),
        on_event=collect,
    )
    return rt, events


def _timeout_handler() -> tuple[Any, dict[str, int]]:
    """A handler that always exceeds the per-tool timeout (a genuine TIMEOUT)."""
    calls = {"count": 0}

    async def _h(args: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        await asyncio.sleep(0.5)  # > the 0.05s tool timeout
        return {"ok": True}

    return _h, calls


def _rate_limited_handler() -> tuple[Any, dict[str, int]]:
    """A handler that raises a RATE_LIMITED dispatch error (the call never lands)."""
    from himmy.services.tools.service import _ToolDispatchError

    calls = {"count": 0}

    def _h(args: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        raise _ToolDispatchError(ToolErrorCode.RATE_LIMITED, "slow down")

    return _h, calls


def _task(tool: str) -> Task:
    return Task(
        title="t",
        prompt="do it",
        context={
            "tool_names": [tool],
            "response_format": "AUTO_TOOLS",
            "tool_retry_backoff_seconds": 0.0,
        },
    )


def _retries(events: list[RunEvent]) -> list[RunEvent]:
    return [
        e
        for e in events
        if e.event_type == EventType.TOOL_CALLED and "transient_retry" in e.payload
    ]


def test_write_tool_not_retried_on_timeout() -> None:
    """A write (non-read-only) tool that TIMES OUT is NOT re-fired (no duplicate side effect)."""
    registry = ToolRegistry()
    handler, calls = _timeout_handler()
    register_local_tool(
        registry,
        name="send_payment",  # a write verb -> classified side-effecting
        handler=handler,
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
    )
    rt, events = _runtime(registry)
    result = run_async(rt.run_task_detailed(Persona(name="a"), _task("send_payment")))

    assert calls["count"] == 1  # ran exactly once — never re-fired on the timeout
    assert _retries(events) == []
    assert result.tool_returns[0].outcome == "failed"
    assert result.tool_returns[0].metadata.get("error_code") == "TIMEOUT"


def test_explicit_non_read_only_tool_not_retried_on_timeout() -> None:
    """An explicit ``read_only=False`` overrides an otherwise read-looking name."""
    registry = ToolRegistry()
    handler, calls = _timeout_handler()
    register_local_tool(
        registry,
        name="get_then_charge",  # name looks read-y; the flag says it writes
        handler=handler,
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
        read_only=False,
    )
    rt, events = _runtime(registry)
    run_async(rt.run_task_detailed(Persona(name="a"), _task("get_then_charge")))

    assert calls["count"] == 1
    assert _retries(events) == []


def test_ambiguous_tool_not_retried_on_timeout() -> None:
    """An ambiguously-named tool (intent unknown) is treated as side-effecting."""
    registry = ToolRegistry()
    handler, calls = _timeout_handler()
    register_local_tool(
        registry,
        name="probe",  # classify_read_only -> None (ambiguous)
        handler=handler,
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
    )
    rt, events = _runtime(registry)
    run_async(rt.run_task_detailed(Persona(name="a"), _task("probe")))

    assert calls["count"] == 1
    assert _retries(events) == []


def test_read_only_tool_still_retried_on_timeout() -> None:
    """A read-only tool has no side effect, so a TIMEOUT is still safely retried."""
    registry = ToolRegistry()
    handler, calls = _timeout_handler()
    register_local_tool(
        registry,
        name="get_quote",  # a read verb -> read-only
        handler=handler,
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
    )
    rt, events = _runtime(registry)
    run_async(rt.run_task_detailed(Persona(name="a"), _task("get_quote")))

    assert calls["count"] == 3  # initial + DEFAULT_TOOL_RETRY_ATTEMPTS (2)
    assert len(_retries(events)) == 2


def test_write_tool_still_retried_on_rate_limited() -> None:
    """A RATE_LIMITED write IS retried — the call never reached the tool, so it's safe."""
    registry = ToolRegistry()
    handler, calls = _rate_limited_handler()
    register_local_tool(
        registry,
        name="send_payment",
        handler=handler,
        args_json_schema={"type": "object", "properties": {}},
    )
    rt, events = _runtime(registry)
    result = run_async(rt.run_task_detailed(Persona(name="a"), _task("send_payment")))

    # The runtime retried the rate-limited write (initial + 2 retries).
    assert calls["count"] == 3
    assert len(_retries(events)) == 2
    assert result.tool_returns[0].metadata.get("error_code") == "RATE_LIMITED"
