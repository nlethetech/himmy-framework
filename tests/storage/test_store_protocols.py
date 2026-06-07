"""Conformance tests for the focused storage protocols (SE decomposition).

These verify — entirely offline, no Postgres, no asyncpg — that:

- each in-memory store satisfies its single-responsibility protocol,
- the ``StorageService`` facade satisfies every focused protocol (it composes and
  re-exports all of them), and
- ``PostgresStorageService`` (constructible without a live pool) satisfies the same
  protocols, so the two backends stay surface-compatible.

This is the structural safety net for keeping both backends in lockstep when the
Postgres internals are later split into per-concern stores.
"""

from __future__ import annotations

from himmy.services.storage import (
    ContextStore,
    EvaluationStore,
    EventLog,
    InMemoryContextStore,
    InMemoryEvaluationStore,
    InMemoryEventLog,
    InMemoryOrchestrationStore,
    InMemoryRecommendationStore,
    InMemoryRunStore,
    InMemoryThreadStore,
    OrchestrationStore,
    PostgresStorageService,
    RecommendationStore,
    RunStore,
    StorageService,
    ThreadEventStore,
    ThreadStore,
)

# Every focused protocol the facade and the Postgres backend must satisfy.
_ALL_PROTOCOLS = (
    ThreadStore,
    EventLog,
    ThreadEventStore,
    ContextStore,
    RunStore,
    RecommendationStore,
    EvaluationStore,
    OrchestrationStore,
)


def test_inmemory_stores_satisfy_their_protocols() -> None:
    """Each focused in-memory store is a structural instance of its protocol."""
    assert isinstance(InMemoryThreadStore(), ThreadStore)
    assert isinstance(InMemoryEventLog(), EventLog)
    assert isinstance(InMemoryContextStore(), ContextStore)
    assert isinstance(InMemoryRunStore(), RunStore)
    assert isinstance(InMemoryRecommendationStore(), RecommendationStore)
    assert isinstance(InMemoryEvaluationStore(), EvaluationStore)
    assert isinstance(InMemoryOrchestrationStore(), OrchestrationStore)


def test_facade_satisfies_every_focused_protocol() -> None:
    """The StorageService facade exposes the full surface of every focused store."""
    storage = StorageService()
    for protocol in _ALL_PROTOCOLS:
        assert isinstance(storage, protocol), f"facade missing {protocol.__name__}"


def test_postgres_backend_satisfies_every_focused_protocol() -> None:
    """PostgresStorageService stays surface-compatible with the facade (no DB)."""
    pg = PostgresStorageService()  # constructible without a live pool
    for protocol in _ALL_PROTOCOLS:
        assert isinstance(pg, protocol), f"postgres backend missing {protocol.__name__}"
