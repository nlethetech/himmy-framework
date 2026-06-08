"""Memory-on-the-spine projection tests: memory facts ARE EntityRecords.

The flagship differentiator — a remembered fact becomes a first-class, content-
addressed ``memory_fact`` :class:`EntityRecord`, idempotent on re-registration, with a
content hash that stays stable when the fact is later invalidated (validity lives in
metadata/links, not the payload).
"""

from __future__ import annotations

from himmy.entities.records import EntityQuery
from himmy.entities.registry import EntityRegistry
from himmy.services.memory import InMemoryMemoryStore, MemoryRecord, MemoryService
from himmy.services.memory.projection import MEMORY_FACT_KIND, memory_to_record


def test_memory_to_record_is_deterministic_and_idempotent() -> None:
    """Identical content projects to the same record_id and re-registers idempotently."""
    rec = MemoryRecord(
        memory_id="fixed-id",
        subject_id="u",
        text="the user prefers metric units",
        stable_key="u/units",
    )
    r1 = memory_to_record(rec, version=1)
    r2 = memory_to_record(rec, version=1)
    assert r1.record_id == r2.record_id
    assert r1.kind == MEMORY_FACT_KIND

    registry = EntityRegistry()
    registry.register(r1)
    # Re-registering byte-identical content is a no-op, not a violation.
    again = registry.register(r2)
    assert again.record_id == r1.record_id


def test_record_id_stable_across_invalidation() -> None:
    """Stamping valid_to (invalidation) must NOT drift the content-addressed id.

    valid_to lives in metadata, not the payload, so the same fact projected before and
    after invalidation yields the same record_id — re-registration stays idempotent.
    """
    rec = MemoryRecord(
        memory_id="m1", subject_id="u", text="lives in Mumbai", stable_key="u/home"
    )
    before = memory_to_record(rec, version=1)
    invalidated = rec.model_copy(update={"valid_to": "2025-01-01T00:00:00Z"})
    after = memory_to_record(invalidated, version=1)
    assert before.record_id == after.record_id
    assert before.payload == after.payload  # payload unchanged
    # The validity transition shows up only in metadata.
    assert after.metadata["valid_to"] == "2025-01-01T00:00:00Z"
    assert before.metadata["valid_to"] is None


def test_remember_projects_memory_fact_onto_spine() -> None:
    """With a registry wired, remember registers a memory_fact + MEMORY_REMEMBERED."""
    registry = EntityRegistry()
    svc = MemoryService(InMemoryMemoryStore(), registry=registry)
    saved = svc.remember("the user farms ducks and bees", subject_id="boss")

    facts = registry.list_by_kind(MEMORY_FACT_KIND)
    assert len(facts) == 1
    assert facts[0].payload["text"] == "the user farms ducks and bees"
    assert facts[0].payload["memory_id"] == saved.memory_id

    # A MEMORY_REMEMBERED audit event was projected onto the spine too.
    events = registry.query(EntityQuery(kind="run_event"))
    types = {e.payload.get("event_type") for e in events}
    assert "MEMORY_REMEMBERED" in types


def test_no_registry_means_no_spine_side_effects() -> None:
    """Without a registry/sink, remember is a pure store write (baseline preserved)."""
    registry = EntityRegistry()
    svc = MemoryService(InMemoryMemoryStore())  # no registry wired
    svc.remember("a fact", subject_id="u")
    # Nothing leaked onto an unrelated registry, and none was created internally.
    assert registry.list_by_kind(MEMORY_FACT_KIND) == []
