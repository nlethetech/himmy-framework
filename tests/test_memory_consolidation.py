"""Audited consolidation tests: ADD/UPDATE/DELETE/NOOP, bi-temporal apply, replay.

Covers the Mem0-style reconcile-don't-append loop, the Graphiti invalidate-not-delete
apply, the spine supersedes/invalidated_by links, the MEMORY_CONSOLIDATED audit event
(to a fake sink AND the registry), and the opt-in LLM decision path that still runs
fully offline through the StubClientManager.
"""

from __future__ import annotations

import time

from himmy.core.events import EventType, RunEvent
from himmy.entities.records import EntityQuery
from himmy.entities.registry import EntityRegistry
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.memory import InMemoryMemoryStore, MemoryService
from himmy.services.memory.consolidation import MemoryConsolidator
from himmy.services.memory.projection import MEMORY_FACT_KIND
from tests.conftest import run_async


class _CapturingSink:
    """A minimal in-memory EventSink that records every appended event."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        self.events.append(event)


def test_first_fact_is_added() -> None:
    """A candidate with no related memory is ADDed."""
    svc = MemoryService(InMemoryMemoryStore())
    cons = MemoryConsolidator(svc, update_threshold=0.2)
    result = run_async(cons.consolidate("the user lives in Mumbai", subject_id="u"))
    assert result.action == "ADD"
    assert result.record is not None
    active = run_async(svc.recall("user lives", subject_id="u", active_only=True))
    assert len(active) == 1


def test_changed_fact_updates_and_invalidates_old() -> None:
    """A closely-related but changed fact supersedes (invalidates) the prior one."""
    svc = MemoryService(InMemoryMemoryStore())
    cons = MemoryConsolidator(svc, update_threshold=0.2)
    first = run_async(cons.consolidate("the user lives in Mumbai", subject_id="u"))
    time.sleep(0.01)
    second = run_async(cons.consolidate("the user lives in Mumbai now", subject_id="u"))
    assert second.action == "UPDATE"
    assert second.target_id == first.record.memory_id  # type: ignore[union-attr]

    # Only the new value is active; the old fact is invalidated, not deleted.
    active = run_async(svc.recall("user lives", subject_id="u", active_only=True))
    assert [h.record.text for h in active] == ["the user lives in Mumbai now"]
    old = svc.get(first.record.memory_id)  # type: ignore[union-attr]
    assert old is not None and old.valid_to is not None
    assert old.superseded_by == second.record.memory_id  # type: ignore[union-attr]


def test_duplicate_fact_is_noop() -> None:
    """An identical candidate is a NOOP — no second record is created."""
    svc = MemoryService(InMemoryMemoryStore())
    cons = MemoryConsolidator(svc, dup_threshold=0.95, update_threshold=0.2)
    run_async(cons.consolidate("the user prefers metric units", subject_id="u"))
    result = run_async(
        cons.consolidate("the user prefers metric units", subject_id="u")
    )
    assert result.action == "NOOP"
    # Exactly one stored fact remains.
    assert len(svc.store.list("u")) == 1


def test_distinct_fact_is_added_not_update() -> None:
    """An unrelated candidate is ADDed (does not wrongly supersede an existing fact)."""
    svc = MemoryService(InMemoryMemoryStore())
    cons = MemoryConsolidator(svc, dup_threshold=0.95, update_threshold=0.8)
    run_async(cons.consolidate("the orchard has apple and pear trees", subject_id="u"))
    result = run_async(
        cons.consolidate("the pond holds fish and ducks", subject_id="u")
    )
    assert result.action == "ADD"
    assert len(svc.store.list("u")) == 2


def test_consolidation_audited_to_sink_and_registry_and_replayable() -> None:
    """A consolidate emits MEMORY_CONSOLIDATED to the sink AND projects to the spine.

    The supersedes/invalidated_by links make the fact's evolution traceable via the
    existing registry.trace — the replayable audit loop that is the differentiator.
    """
    registry = EntityRegistry()
    sink = _CapturingSink()
    svc = MemoryService(InMemoryMemoryStore(), registry=registry)
    cons = MemoryConsolidator(
        svc, registry=registry, event_sink=sink, update_threshold=0.2
    )

    first = run_async(cons.consolidate("the user lives in Mumbai", subject_id="u"))
    second = run_async(cons.consolidate("the user lives in Mumbai now", subject_id="u"))
    assert second.action == "UPDATE"

    # The fake sink saw the consolidation decisions.
    consolidated = [
        e for e in sink.events if e.event_type == EventType.MEMORY_CONSOLIDATED
    ]
    assert any(e.payload["action"] == "UPDATE" for e in consolidated)

    # A run_event record AND two memory_fact records exist on the spine.
    run_events = registry.query(EntityQuery(kind="run_event"))
    assert any(e.payload.get("event_type") == "MEMORY_CONSOLIDATED" for e in run_events)
    facts = registry.list_by_kind(MEMORY_FACT_KIND)
    assert len(facts) == 2

    # trace(new, supersedes) reaches the old fact (the evolution is graph-traceable).
    new_rid = svc.get(second.record.memory_id).to_record(version=2).record_id  # type: ignore[union-attr]
    old_rid = svc.get(first.record.memory_id).to_record(version=1).record_id  # type: ignore[union-attr]
    graph = registry.trace(new_rid, relations={"supersedes"})
    assert old_rid in graph.record_ids()
    assert "supersedes" in graph.relations()


def test_llm_path_uses_structured_output_and_runs_offline() -> None:
    """The opt-in LLM path drives a STRUCTURED_OUTPUT decision via the offline stub.

    No network is touched (StubClientManager), and whatever the stub decides maps to a
    valid action — proving the schema-constrained extraction loop works with no keys.
    """
    captured: dict[str, object] = {}

    class _SchemaAssertingManager(StubClientManager):
        async def generate(self, request):  # type: ignore[override]
            # The consolidation request MUST carry the decision schema.
            captured["schema"] = request.output_json_schema
            captured["format"] = request.response_format
            return await super().generate(request)

    svc = MemoryService(InMemoryMemoryStore())
    cons = MemoryConsolidator(
        svc, client_manager=_SchemaAssertingManager(), use_llm=True
    )
    result = run_async(cons.consolidate("the user prefers dark mode", subject_id="u"))

    assert result.action in ("ADD", "UPDATE", "DELETE", "NOOP")
    schema = captured["schema"]
    assert isinstance(schema, dict)
    assert "action" in schema["properties"]
    assert set(schema["properties"]["action"]["enum"]) == {
        "ADD",
        "UPDATE",
        "DELETE",
        "NOOP",
    }
