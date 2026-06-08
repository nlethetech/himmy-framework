"""Semantic recall threshold tests: the recall floor and its baseline-preserving default.

Proves (a) a positive floor drops a fact the historical code WOULD have returned, and
(b) the zero/None default reproduces the exact current ordering — so the 1216-test
baseline (and the always-return-the-top-hit contract) is untouched.
"""

from __future__ import annotations

from himmy.services.memory import InMemoryMemoryStore, MemoryService
from tests.conftest import run_async


def test_service_min_similarity_floor_drops_irrelevant_fact() -> None:
    """A ctor-level min_similarity drops a fact the default would have surfaced."""
    svc = MemoryService(InMemoryMemoryStore(), min_similarity=0.5)
    svc.remember("the pond holds fish and ducks", subject_id="u")

    # The default (no floor) WOULD return this orthogonal fact at similarity 0.0.
    baseline = MemoryService(InMemoryMemoryStore())
    baseline.remember("the pond holds fish and ducks", subject_id="u")
    baseline_hits = run_async(baseline.recall("quantum chromodynamics", subject_id="u"))
    assert len(baseline_hits) == 1 and baseline_hits[0].similarity == 0.0

    # With the floor, the orthogonal query correctly recalls nothing.
    hits = run_async(svc.recall("quantum chromodynamics", subject_id="u"))
    assert hits == []


def test_per_call_threshold_overrides_service_default() -> None:
    """A per-call similarity_threshold takes precedence over the ctor floor."""
    svc = MemoryService(InMemoryMemoryStore(), min_similarity=0.9)
    svc.remember("the orchard has apple and pear trees", subject_id="u")

    # Service floor 0.9 would drop everything; an explicit 0.0 restores the top hit.
    hits = run_async(
        svc.recall("apple pear orchard", subject_id="u", similarity_threshold=0.0)
    )
    assert len(hits) == 1


def test_default_none_reproduces_top_hit_ordering() -> None:
    """min_similarity=None (the default) preserves always-return-the-top-hit."""
    svc = MemoryService(InMemoryMemoryStore())  # no floor
    svc.remember("the orchard has apple and pear trees", subject_id="u")
    svc.remember("the pond holds fish and ducks", subject_id="u")
    hits = run_async(svc.recall("apple and pear trees orchard", subject_id="u"))
    # Both returned, most-similar first — identical to the historical contract.
    assert len(hits) == 2
    assert hits[0].record.text.startswith("the orchard")
    assert hits[0].similarity >= hits[1].similarity


def test_floor_keeps_relevant_drops_irrelevant() -> None:
    """A floor keeps an on-topic fact and drops the orthogonal one in one call."""
    svc = MemoryService(InMemoryMemoryStore(), min_similarity=0.1)
    svc.remember("the orchard has apple and pear trees", subject_id="u")
    svc.remember("the pond holds fish and ducks", subject_id="u")
    hits = run_async(svc.recall("apple and pear trees orchard", subject_id="u"))
    assert len(hits) == 1
    assert hits[0].record.text.startswith("the orchard")
