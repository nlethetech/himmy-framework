"""Tests for the ToolService idempotency seam (exactly-once keyed execution).

The seam exists so resume-style paths (HITL approve/resume, future replay paths)
can re-drive a transcript without executing a state-mutating tool twice: an
invocation carrying ``metadata["idempotency_key"]`` plus a ``ToolIdempotencyStore``
records its result once and replays it on every later call with the same key.
"""

from __future__ import annotations

from himmy.core.events import EventType
from himmy.services.storage.service import StorageService
from himmy.services.tools import (
    ToolErrorCode,
    ToolExecutionResult,
    ToolInvocation,
    ToolRegistry,
    ToolService,
    register_local_tool,
)
from tests.conftest import run_async


class _DictIdempotencyStore:
    """A minimal in-memory ``ToolIdempotencyStore`` for tests."""

    def __init__(self) -> None:
        self.records: dict[str, ToolExecutionResult] = {}

    def get(self, key: str) -> ToolExecutionResult | None:
        found = self.records.get(key)
        return found.model_copy(deep=True) if found is not None else None

    def put(self, key: str, result: ToolExecutionResult) -> None:
        self.records[key] = result.model_copy(deep=True)


def _counting_service(
    calls: list[dict],
    *,
    requires_approval: bool = False,
    sink: StorageService | None = None,
) -> ToolService:
    registry = ToolRegistry()

    def transfer(args: dict) -> dict:
        calls.append(dict(args))
        return {"transferred": True, "count": len(calls)}

    register_local_tool(
        registry,
        name="transfer_money",
        handler=transfer,
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=requires_approval,
    )
    return ToolService(registry, event_sink=sink)


def _invocation(key: str | None = None, **metadata: object) -> ToolInvocation:
    meta: dict[str, object] = dict(metadata)
    if key is not None:
        meta["idempotency_key"] = key
    return ToolInvocation(tool_name="transfer_money", args={}, metadata=meta)


def test_keyed_execution_runs_once_and_replays() -> None:
    """The same key + store executes the handler exactly once; replays after."""
    calls: list[dict] = []
    svc = _counting_service(calls)
    store = _DictIdempotencyStore()

    first = run_async(svc.execute(_invocation("call-1"), idempotency_store=store))
    second = run_async(svc.execute(_invocation("call-1"), idempotency_store=store))

    assert len(calls) == 1  # the state-mutating handler ran exactly once
    assert first.outcome == "success"
    assert second.outcome == "success"
    assert second.result == first.result  # the recorded result is reused verbatim
    assert second.metadata.get("idempotent_replay") is True
    assert "idempotent_replay" not in first.metadata


def test_without_store_or_key_behaves_as_before() -> None:
    """No store, or no key, means no dedup — fully backward compatible."""
    calls: list[dict] = []
    svc = _counting_service(calls)
    store = _DictIdempotencyStore()

    run_async(svc.execute(_invocation("call-1")))  # key but no store
    run_async(svc.execute(_invocation("call-1")))
    assert len(calls) == 2

    run_async(svc.execute(_invocation(), idempotency_store=store))  # store, no key
    run_async(svc.execute(_invocation(), idempotency_store=store))
    assert len(calls) == 4
    assert store.records == {}


def test_denied_is_not_recorded_so_an_approved_retry_still_runs() -> None:
    """An approval denial is never recorded; the approved retry executes."""
    calls: list[dict] = []
    svc = _counting_service(calls, requires_approval=True)
    store = _DictIdempotencyStore()

    denied = run_async(svc.execute(_invocation("call-1"), idempotency_store=store))
    assert denied.outcome == "denied"
    assert denied.error_code is ToolErrorCode.POLICY_BLOCKED
    assert store.records == {}  # the denial must not poison the key

    approved = run_async(
        svc.execute(_invocation("call-1", approved=True), idempotency_store=store)
    )
    assert approved.outcome == "success"
    assert len(calls) == 1
    assert "call-1" in store.records


def test_failures_are_recorded_and_replayed() -> None:
    """A post-dispatch failure is pinned to its key — retries replay, not re-run."""
    registry = ToolRegistry()
    attempts: list[int] = []

    def explode(args: dict) -> dict:
        attempts.append(1)
        raise RuntimeError("wire transfer state unknown")

    register_local_tool(
        registry,
        name="transfer_money",
        handler=explode,
        args_json_schema={"type": "object", "properties": {}},
    )
    svc = ToolService(registry)
    store = _DictIdempotencyStore()

    first = run_async(svc.execute(_invocation("call-1"), idempotency_store=store))
    second = run_async(svc.execute(_invocation("call-1"), idempotency_store=store))

    assert len(attempts) == 1  # the maybe-mutating handler is never re-attempted
    assert first.outcome == "failed"
    assert second.outcome == "failed"
    assert second.error_code is ToolErrorCode.EXECUTION_ERROR
    assert second.metadata.get("idempotent_replay") is True


def test_replay_emits_no_duplicate_tool_events() -> None:
    """A replay is not an execution: no second TOOL_CALLED / TOOL_COMPLETED."""
    sink = StorageService()
    calls: list[dict] = []
    svc = _counting_service(calls, sink=sink)
    store = _DictIdempotencyStore()

    run_async(svc.execute(_invocation("call-1"), idempotency_store=store))
    run_async(svc.execute(_invocation("call-1"), idempotency_store=store))

    events = run_async(sink.list_events())
    tool_events = [
        e
        for e in events
        if e.event_type in (EventType.TOOL_CALLED, EventType.TOOL_COMPLETED)
    ]
    assert len(tool_events) == 2  # exactly one CALLED + one COMPLETED, not four
