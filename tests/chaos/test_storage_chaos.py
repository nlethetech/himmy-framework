"""Chaos: storage backends that fail like an unreachable database mid-run.

The runtime's persistence contract is best-effort isolation: a dead or dying
storage seam (event append, thread save, snapshot load) must never hang or
crash a run — the run still completes and returns a *typed* ``RunResult``.
These tests inject real connection-style failures (including a genuine
``SqliteEventStore`` whose connection is closed mid-run) and lock that
contract, plus the same isolation on the inference and tool event sinks.
Every awaited scenario is guarded by ``asyncio.wait_for``.
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
from himmy.services.inference.models import (
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
)
from himmy.services.inference.service import InferenceService
from himmy.services.observability.trace import SqliteEventStore
from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async

#: Hard guard on every awaited scenario: a hang fails fast, not forever.
_GUARD_SECONDS = 5.0


class _DeadStore:
    """A storage seam whose every method fails like an unreachable database."""

    def __init__(self) -> None:
        self.calls = 0

    async def append_event(self, event: RunEvent) -> None:
        self.calls += 1
        raise ConnectionError("database unavailable")

    async def save_thread(self, thread: Any) -> Any:
        self.calls += 1
        raise ConnectionError("database unavailable")

    async def load_thread(self, thread_id: str) -> Any:
        self.calls += 1
        raise ConnectionError("database unavailable")

    async def load_snapshot(self, snapshot_id: str) -> Any:
        self.calls += 1
        raise ConnectionError("database unavailable")


class _MidRunDeathStore:
    """A store that accepts the first event then dies for the rest of the run."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []
        self.failures = 0

    async def append_event(self, event: RunEvent) -> None:
        if self.events:
            self.failures += 1
            raise ConnectionError("database connection lost mid-run")
        self.events.append(event)

    async def save_thread(self, thread: Any) -> Any:
        self.failures += 1
        raise ConnectionError("database connection lost mid-run")


def test_run_with_dead_storage_returns_typed_result_not_hang() -> None:
    """A store down from the first call cannot hang or fail the run itself."""
    store = _DeadStore()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=store,
    )
    persona = Persona(name="A")
    result = run_async(
        asyncio.wait_for(
            rt.run_task_detailed(persona, Task(title="t", prompt="hi")),
            timeout=_GUARD_SECONDS,
        )
    )
    # The failures were really injected (the seam was exercised, not skipped) …
    assert store.calls > 0
    # … yet the run completed with a typed, successful result and a full thread.
    assert result.succeeded
    assert result.thread.last_message is not None
    assert result.thread.last_message.role == MessageRole.ASSISTANT


def test_storage_dying_mid_run_is_isolated_from_the_run() -> None:
    """A store that dies AFTER the run started keeps the run alive and typed."""
    store = _MidRunDeathStore()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=store,
    )
    persona = Persona(name="A")
    result = run_async(
        asyncio.wait_for(
            rt.run_task_detailed(persona, Task(title="t", prompt="hi")),
            timeout=_GUARD_SECONDS,
        )
    )
    # The first event landed before the outage; later appends genuinely failed.
    assert [e.event_type for e in store.events] == [EventType.AGENT_RUN_STARTED]
    assert store.failures > 0
    assert result.succeeded


def test_sqlite_event_store_connection_closed_mid_run() -> None:
    """The real SQLite event log losing its connection mid-run is survivable.

    The store's connection is closed right after the first event lands (via the
    caller-facing ``on_event`` hook, which runs after the storage append), so
    every later ``append_event`` raises ``sqlite3.ProgrammingError`` — the
    closed-database analogue of a DB outage. The run must still complete.
    """
    store = SqliteEventStore()
    seen: list[str] = []

    async def _kill_db(event: RunEvent) -> None:
        if not seen:
            store.close()
        seen.append(event.event_type.value)

    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=store,
        on_event=_kill_db,
    )
    persona = Persona(name="A")
    result = run_async(
        asyncio.wait_for(
            rt.run_task_detailed(persona, Task(title="t", prompt="hi")),
            timeout=_GUARD_SECONDS,
        )
    )
    assert result.succeeded
    # The run kept emitting events past the outage (they reached on_event even
    # though the durable store was down) — proof later appends really fired.
    assert seen[0] == EventType.AGENT_RUN_STARTED.value
    assert EventType.AGENT_RUN_FINISHED.value in seen


def test_dead_event_sink_never_breaks_inference() -> None:
    """An InferenceService with a dead event sink still returns typed responses."""
    sink = _DeadStore()
    service = InferenceService(StubClientManager(), event_sink=sink)
    response = run_async(
        asyncio.wait_for(
            service.run(
                InferenceRequest(messages=[InferenceMessage(role="user", content="hi")])
            ),
            timeout=_GUARD_SECONDS,
        )
    )
    assert sink.calls > 0  # the sink was really hit (and really failed)
    assert response.status == InferenceStatus.SUCCESS


def test_dead_event_sink_never_breaks_tool_execution() -> None:
    """A ToolService with a dead event sink still executes tools successfully."""
    sink = _DeadStore()
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="ping",
        handler=lambda args: {"pong": True},
        args_json_schema={"type": "object", "properties": {}},
    )
    service = ToolService(registry, event_sink=sink)
    result = run_async(
        asyncio.wait_for(
            service.execute(ToolInvocation(tool_name="ping", args={})),
            timeout=_GUARD_SECONDS,
        )
    )
    assert sink.calls > 0
    assert result.outcome == "success"
    assert result.result == {"pong": True}
