"""Tests for typed relational links between memories (graph-memory write side).

Covers link persistence + retrieval + dedup across BOTH stores (in-memory and the
durable sqlite file via ``tmp_path``), plus the canonical relation vocabulary.
"""

from __future__ import annotations

from pathlib import Path

from himmy.services.memory import (
    InMemoryMemoryStore,
    MemoryLink,
    MemoryService,
    SqliteMemoryStore,
)
from himmy.services.memory.store import (
    ABOUT_ENTITY,
    CAUSED_BY,
    CONTRADICTS,
    DEPENDS_ON,
    MEMORY_RELATIONS,
    PART_OF,
    RELATES_TO,
    SUPERSEDES,
)


def test_canonical_relations_are_the_documented_set() -> None:
    """The canonical vocabulary matches the documented constants (relation stays str)."""
    assert MEMORY_RELATIONS == (
        RELATES_TO,
        CAUSED_BY,
        CONTRADICTS,
        ABOUT_ENTITY,
        SUPERSEDES,
        PART_OF,
        DEPENDS_ON,
    )
    # ``relation`` is an open string: a custom relation is accepted unchanged.
    link = MemoryLink(from_memory_id="a", to_memory_id="b", relation="my_custom_rel")
    assert link.relation == "my_custom_rel"
    # Default relation is the generic association.
    assert MemoryLink(from_memory_id="a", to_memory_id="b").relation == RELATES_TO


def test_inmemory_link_persistence_and_retrieval() -> None:
    """save_link + links_from/links_to round-trip on the volatile store."""
    store = InMemoryMemoryStore()
    link = store.save_link(
        MemoryLink(from_memory_id="m1", to_memory_id="m2", relation=CAUSED_BY)
    )
    out = store.links_from("m1")
    assert [link_.link_id for link_ in out] == [link.link_id]
    assert out[0].relation == CAUSED_BY
    # The reverse index resolves the same edge.
    back = store.links_to("m2")
    assert [link_.link_id for link_ in back] == [link.link_id]
    # An unrelated node has no edges either way.
    assert store.links_from("m2") == []
    assert store.links_to("m1") == []


def test_sqlite_link_persistence_round_trips_via_tmp_path(tmp_path: Path) -> None:
    """A link survives a reopen of the durable sqlite file (metadata preserved)."""
    db = str(tmp_path / "memory.db")
    store = SqliteMemoryStore(db)
    saved = store.save_link(
        MemoryLink(
            from_memory_id="m1",
            to_memory_id="m2",
            relation=ABOUT_ENTITY,
            metadata={"weight": 3, "note": "primary"},
        )
    )
    store.close()

    reopened = SqliteMemoryStore(db)
    out = reopened.links_from("m1")
    assert len(out) == 1
    assert out[0].link_id == saved.link_id
    assert out[0].relation == ABOUT_ENTITY
    assert out[0].metadata == {"weight": 3, "note": "primary"}
    assert reopened.links_to("m2")[0].link_id == saved.link_id
    reopened.close()


def test_sqlite_save_link_is_dedup_by_link_id(tmp_path: Path) -> None:
    """Re-saving the same link_id replaces, not duplicates (INSERT OR REPLACE)."""
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    link = MemoryLink(from_memory_id="m1", to_memory_id="m2", relation=RELATES_TO)
    store.save_link(link)
    store.save_link(link.model_copy(update={"relation": CONTRADICTS}))
    out = store.links_from("m1")
    assert len(out) == 1  # not duplicated
    assert out[0].relation == CONTRADICTS  # latest write wins
    store.close()


def test_inmemory_save_link_is_dedup_by_link_id() -> None:
    """Re-saving the same link_id replaces, not duplicates, in the volatile store."""
    store = InMemoryMemoryStore()
    link = MemoryLink(from_memory_id="m1", to_memory_id="m2")
    store.save_link(link)
    store.save_link(link.model_copy(update={"relation": DEPENDS_ON}))
    out = store.links_from("m1")
    assert len(out) == 1
    assert out[0].relation == DEPENDS_ON


def test_service_link_helper_persists_to_store() -> None:
    """``MemoryService.link`` draws a typed edge and persists it to the store."""
    svc = MemoryService(InMemoryMemoryStore())
    a = svc.remember("a fact", subject_id="u")
    b = svc.remember("a related fact", subject_id="u")
    link = svc.link(a.memory_id, b.memory_id, relation=PART_OF, metadata={"k": "v"})
    assert link.relation == PART_OF
    assert link.metadata == {"k": "v"}
    stored = svc.store.links_from(a.memory_id)
    assert [link_.link_id for link_ in stored] == [link.link_id]
