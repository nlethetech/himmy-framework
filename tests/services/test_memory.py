"""Tests for the long-term memory module (store + service + adapter)."""

from __future__ import annotations

from pathlib import Path

from himmy.services.memory import (
    InMemoryMemoryStore,
    MemoryContextAdapter,
    MemoryService,
    SqliteMemoryStore,
)
from himmy.services.memory.service import ALWAYS_INCLUDE
from tests.conftest import run_async


def test_remember_then_recall_ranks_by_overlap() -> None:
    """Recall returns the most token-similar memory first (offline embedder)."""
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("the orchard has apple and pear trees", subject_id="u")
    svc.remember("the pond holds fish and ducks", subject_id="u")
    # Both share the leading "the ... has/holds" tokens, so ALWAYS_INCLUDE surfaces
    # both and lets us assert the ordering without the noise floor intervening.
    hits = run_async(
        svc.recall(
            "apple and pear trees orchard",
            subject_id="u",
            top_k=2,
            similarity_threshold=ALWAYS_INCLUDE,
        )
    )
    assert len(hits) == 2
    assert hits[0].record.text.startswith("the orchard")
    assert hits[0].similarity > hits[1].similarity


def test_recall_is_subject_scoped() -> None:
    """A subject only recalls its own memories."""
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("alice likes metric units", subject_id="alice")
    svc.remember("bob likes imperial units", subject_id="bob")
    hits = run_async(svc.recall("units", subject_id="alice"))
    assert len(hits) == 1
    assert "alice" in hits[0].record.text


def test_empty_recall() -> None:
    """Recall on an empty subject returns nothing."""
    svc = MemoryService(InMemoryMemoryStore())
    assert run_async(svc.recall("anything", subject_id="nobody")) == []


def test_sqlite_durability_across_reopen(tmp_path: Path) -> None:
    """Memories persist across a closed + re-opened SQLite store."""
    db = str(tmp_path / "mem.db")
    store_a = SqliteMemoryStore(db)
    MemoryService(store_a).remember("bees pollinate the orchard", subject_id="farm")
    store_a.close()

    store_b = SqliteMemoryStore(db)
    hits = run_async(
        MemoryService(store_b).recall("bees pollinate orchard", subject_id="farm")
    )
    store_b.close()
    assert len(hits) == 1
    assert "bees" in hits[0].record.text


def test_recall_default_applies_noise_floor() -> None:
    """The default now drops a zero-overlap hit (the safe ``> 0.0`` noise floor).

    Previously recall always surfaced the single best (possibly similarity-0.0) hit;
    aligning with the sibling knowledge service, the default now recalls NOTHING for an
    orthogonal query instead of a best-of-the-irrelevant one.
    """
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("the pond holds fish and ducks", subject_id="u")
    # The query shares no tokens with the only memory -> similarity 0.0, so the
    # default noise floor drops it.
    hits = run_async(svc.recall("quantum chromodynamics", subject_id="u"))
    assert hits == []


def test_recall_always_include_returns_top_hit_even_at_zero() -> None:
    """The explicit ALWAYS_INCLUDE path still returns the top hit at similarity 0.0.

    This is the always-in-context working set used by the ``core`` tier: it must keep
    returning its set even when nothing clears the noise floor.
    """
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("the pond holds fish and ducks", subject_id="u")
    hits = run_async(
        svc.recall(
            "quantum chromodynamics",
            subject_id="u",
            similarity_threshold=ALWAYS_INCLUDE,
        )
    )
    assert len(hits) == 1
    assert hits[0].similarity == 0.0


def test_explicit_zero_threshold_keeps_zero_similarity_hits() -> None:
    """An explicit threshold of 0.0 keeps similarity>=0.0 hits (caller opts in)."""
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("the pond holds fish and ducks", subject_id="u")
    hits = run_async(
        svc.recall("quantum chromodynamics", subject_id="u", similarity_threshold=0.0)
    )
    assert len(hits) == 1
    assert hits[0].similarity == 0.0


def test_recall_threshold_filters_irrelevant_hits() -> None:
    """An explicit threshold lets recall return ZERO results for an off-topic query."""
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("the orchard has apple and pear trees", subject_id="u")
    svc.remember("the pond holds fish and ducks", subject_id="u")

    # On-topic: the orchard memory clears the bar, the (orthogonal) pond memory does not.
    relevant = run_async(
        svc.recall(
            "apple and pear trees orchard", subject_id="u", similarity_threshold=0.1
        )
    )
    assert len(relevant) == 1
    assert relevant[0].record.text.startswith("the orchard")
    assert relevant[0].similarity >= 0.1

    # Off-topic: nothing clears the bar, so recall correctly returns nothing.
    irrelevant = run_async(
        svc.recall("quantum chromodynamics", subject_id="u", similarity_threshold=0.1)
    )
    assert irrelevant == []


def test_forget_removes_memory() -> None:
    svc = MemoryService(InMemoryMemoryStore())
    rec = svc.remember("temporary note", subject_id="u")
    assert svc.forget(rec.memory_id) is True
    assert run_async(svc.recall("temporary", subject_id="u")) == []


def test_context_adapter_injects_recalled_memories() -> None:
    """The adapter renders recalled memories into a ContextField."""
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("the user farms ducks and bees", subject_id="boss")
    adapter = MemoryContextAdapter(svc)
    field = run_async(
        adapter.fetch("memory", {"subject_id": "boss", "query": "ducks and bees"})
    )
    assert field is not None
    assert field.source == "memory"
    assert "ducks" in field.value["rendered_text"]


def test_context_adapter_returns_none_without_query() -> None:
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("x", subject_id="boss")
    adapter = MemoryContextAdapter(svc)
    assert run_async(adapter.fetch("memory", {"subject_id": "boss"})) is None
