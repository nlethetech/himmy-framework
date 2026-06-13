"""P0-C: the memory audit spine is ON — writes project + audit; recall is scrubbed.

The projection plumbing was fully built but dormant (no production caller passed
registry/event_sink). These pin that it now fires: a remember lands a memory_fact
record + a MEMORY_REMEMBERED audit event, recall audits WITHOUT the raw query, and
the Studio path writes onto the durable entity spine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.core.events import EventType
from himmy.entities.registry import EntityRegistry
from himmy.services.memory import InMemoryMemoryStore, MemoryService
from tests.conftest import run_async


def _run_events_of(reg: EntityRegistry, event_type: EventType) -> list:
    return [
        r
        for r in reg.list_by_kind("run_event")
        if r.payload.get("event_type") == event_type.value
    ]


def test_remember_projects_memory_fact_and_audits() -> None:
    reg = EntityRegistry()
    mem = MemoryService(InMemoryMemoryStore(), registry=reg)
    mem.remember("the farm raises ducks and bees", subject_id="boss")

    facts = reg.list_by_kind("memory_fact")
    assert len(facts) == 1
    assert "ducks" in str(facts[0].payload)

    assert _run_events_of(reg, EventType.MEMORY_REMEMBERED)


def test_recall_event_scrubs_the_raw_query() -> None:
    reg = EntityRegistry()
    mem = MemoryService(InMemoryMemoryStore(), registry=reg)
    mem.remember("ducks", subject_id="boss")

    query = "where exactly is my secret farm located"
    run_async(mem.recall(query, subject_id="boss"))

    recalled = _run_events_of(reg, EventType.MEMORY_RECALLED)
    assert recalled
    inner = recalled[0].payload["payload"]
    assert "query" not in inner  # the raw user prompt is never persisted
    assert inner["query_chars"] == len(query)


def test_studio_memory_projects_to_durable_spine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from himmy.api.studio_entities import get_entity_registry, reset_entity_registry
    from himmy.api.studio_memory import add_memory, reset_memory_service

    reset_memory_service()
    reset_entity_registry()

    add_memory("the orchard has 40 mango trees", subject_id="boss")

    facts = get_entity_registry().list_by_kind("memory_fact")
    assert facts and any("mango" in str(f.payload) for f in facts)
