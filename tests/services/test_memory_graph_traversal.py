"""Tests for bounded multi-hop memory graph traversal (graph-memory read side).

Covers a chain ``A -> B -> C`` walked at varying ``max_depth`` (with the ``truncated``
flag), bi-temporal ``as_of`` endpoint filtering, and tier + subject scoping. Uses the
in-memory store; the durable path shares the exact same traversal core.
"""

from __future__ import annotations

from himmy.services.memory import (
    InMemoryMemoryStore,
    MemoryGraph,
    MemoryRecord,
    MemoryService,
)
from himmy.services.memory.store import CAUSED_BY, RELATES_TO
from tests.conftest import run_async


def _chain(svc: MemoryService, subject: str = "u") -> tuple[str, str, str]:
    """Remember three facts A, B, C and link A -> B -> C; return their ids."""
    a = svc.remember("fact A", subject_id=subject)
    b = svc.remember("fact B", subject_id=subject)
    c = svc.remember("fact C", subject_id=subject)
    svc.link(a.memory_id, b.memory_id, relation=CAUSED_BY)
    svc.link(b.memory_id, c.memory_id, relation=CAUSED_BY)
    return a.memory_id, b.memory_id, c.memory_id


def test_chain_traversal_reaches_all_at_depth_two() -> None:
    """A -> B -> C: depth=2 from A reaches all three, two edges, not truncated."""
    svc = MemoryService(InMemoryMemoryStore())
    a, b, c = _chain(svc)
    graph = run_async(svc.traverse_graph(a, max_depth=2))
    assert isinstance(graph, MemoryGraph)
    assert graph.root_memory_id == a
    assert graph.memory_ids() == {a, b, c}
    assert graph.edge_count == 2
    assert graph.relations() == {CAUSED_BY}
    assert graph.truncated is False


def test_chain_traversal_truncates_at_depth_one() -> None:
    """depth=1 from A reaches only A and B and flags truncated (C unexpanded)."""
    svc = MemoryService(InMemoryMemoryStore())
    a, b, c = _chain(svc)
    graph = run_async(svc.traverse_graph(a, max_depth=1))
    assert graph.memory_ids() == {a, b}
    assert graph.edge_count == 1
    assert graph.truncated is True


def test_traversal_both_directions_and_dedups_edges() -> None:
    """Walking from the middle reaches both ends; the shared edge is not duplicated."""
    svc = MemoryService(InMemoryMemoryStore())
    a, b, c = _chain(svc)
    graph = run_async(svc.traverse_graph(b, max_depth=1))
    # From B the IN edge (A->B) and OUT edge (B->C) are both followed.
    assert graph.memory_ids() == {a, b, c}
    assert graph.edge_count == 2
    assert graph.truncated is False


def test_depth_zero_returns_only_the_seed() -> None:
    """max_depth=0 returns just the seed node, no edges, truncated when neighbours exist."""
    svc = MemoryService(InMemoryMemoryStore())
    a, b, c = _chain(svc)
    graph = run_async(svc.traverse_graph(a, max_depth=0))
    assert graph.memory_ids() == {a}
    assert graph.edge_count == 0
    assert graph.truncated is True


def test_as_of_filtering_prunes_invalid_endpoints() -> None:
    """An endpoint invalid at ``as_of`` is pruned from the walk (and its edge).

    Timestamps are pinned explicitly (rather than the wall-clock ``created_at``) so the
    ``as_of`` instants sit unambiguously inside/outside each fact's validity window:
    all three become valid at ``2026-01-01``, but B is invalidated at ``2026-02-01``.
    """
    store = InMemoryMemoryStore()
    svc = MemoryService(store)
    base = "2026-01-01T00:00:00Z"
    a = store.save(MemoryRecord(subject_id="u", text="fact A", valid_from=base))
    b = store.save(
        MemoryRecord(
            subject_id="u",
            text="fact B",
            valid_from=base,
            valid_to="2026-02-01T00:00:00Z",
        )
    )
    c = store.save(MemoryRecord(subject_id="u", text="fact C", valid_from=base))
    svc.link(a.memory_id, b.memory_id)
    svc.link(b.memory_id, c.memory_id)
    # As of an instant AFTER B's valid_to: B is pruned, severing the only path to C.
    graph = run_async(
        svc.traverse_graph(a.memory_id, max_depth=2, as_of="2026-03-01T00:00:00Z")
    )
    assert graph.memory_ids() == {a.memory_id}
    assert graph.edge_count == 0
    # As of an instant WITHIN B's window, the full chain is reachable.
    early = run_async(
        svc.traverse_graph(a.memory_id, max_depth=2, as_of="2026-01-15T00:00:00Z")
    )
    assert early.memory_ids() == {a.memory_id, b.memory_id, c.memory_id}


def test_active_only_skips_invalidated_endpoints() -> None:
    """``active_only`` drops invalidated endpoints from the traversal."""
    svc = MemoryService(InMemoryMemoryStore())
    a = svc.remember("fact A", subject_id="u")
    b = svc.remember("fact B", subject_id="u")
    svc.link(a.memory_id, b.memory_id)
    svc.invalidate(b.memory_id, valid_to="2026-02-01T00:00:00Z")
    graph = run_async(svc.traverse_graph(a.memory_id, max_depth=2, active_only=True))
    assert graph.memory_ids() == {a.memory_id}
    # Without active_only the (invalidated-but-present) neighbour is still reachable.
    full = run_async(svc.traverse_graph(a.memory_id, max_depth=2))
    assert full.memory_ids() == {a.memory_id, b.memory_id}


def test_tier_scoping_prunes_other_tiers() -> None:
    """A ``tier`` filter prunes endpoints in a different Letta tier."""
    svc = MemoryService(InMemoryMemoryStore())
    a = svc.remember("core fact A", subject_id="u", tier="core")
    b = svc.remember("recall fact B", subject_id="u", tier="recall")
    c = svc.remember("core fact C", subject_id="u", tier="core")
    svc.link(a.memory_id, b.memory_id)
    svc.link(b.memory_id, c.memory_id)
    graph = run_async(svc.traverse_graph(a.memory_id, max_depth=2, tier="core"))
    # B (recall) is pruned, which also severs the only path to C.
    assert graph.memory_ids() == {a.memory_id}


def test_subject_scoping_prunes_other_subjects() -> None:
    """A ``subject_id`` filter prunes endpoints belonging to another subject."""
    svc = MemoryService(InMemoryMemoryStore())
    a = svc.remember("alice fact", subject_id="alice")
    b = svc.remember("bob fact", subject_id="bob")
    svc.link(a.memory_id, b.memory_id)
    graph = run_async(svc.traverse_graph(a.memory_id, max_depth=2, subject_id="alice"))
    assert graph.memory_ids() == {a.memory_id}
    # Without scoping the cross-subject neighbour is reachable.
    full = run_async(svc.traverse_graph(a.memory_id, max_depth=2))
    assert full.memory_ids() == {a.memory_id, b.memory_id}


def test_unknown_seed_returns_empty_graph() -> None:
    """An unknown seed yields an empty, non-truncated graph (no crash)."""
    svc = MemoryService(InMemoryMemoryStore())
    graph = run_async(svc.traverse_graph("nope", max_depth=2))
    assert graph.memory_ids() == set()
    assert graph.edge_count == 0
    assert graph.truncated is False


def test_relations_filter_in_pure_walker() -> None:
    """The pure walker can restrict to a relation set, severing other-relation paths."""
    from himmy.services.memory.graph import traverse_memory_graph

    svc = MemoryService(InMemoryMemoryStore())
    a = svc.remember("A", subject_id="u")
    b = svc.remember("B", subject_id="u")
    c = svc.remember("C", subject_id="u")
    svc.link(a.memory_id, b.memory_id, relation=CAUSED_BY)
    svc.link(b.memory_id, c.memory_id, relation=RELATES_TO)
    store = svc.store

    def neighbors(mid: str) -> list:
        return [*store.links_from(mid), *store.links_to(mid)]

    graph = traverse_memory_graph(
        a.memory_id,
        neighbors=neighbors,
        get=store.get,
        accept=lambda _r: True,
        max_depth=2,
        relations={CAUSED_BY},
    )
    # Only the caused_by edge A->B is followed; the relates_to B->C path is severed.
    assert graph.memory_ids() == {a.memory_id, b.memory_id}
    assert graph.relations() == {CAUSED_BY}
