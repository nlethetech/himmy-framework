"""Semantic recall threshold tests: the recall noise floor and its override knobs.

Proves (a) a positive floor drops a fact the zero/explicit-0.0 path WOULD return, (b)
a per-call threshold overrides the service default, and (c) the ``None`` default applies
the safe ``> 0.0`` noise floor (an orthogonal query recalls nothing) — aligning memory
recall with the sibling knowledge service rather than always surfacing the top hit.
"""

from __future__ import annotations

from himmy.services.memory import InMemoryMemoryStore, MemoryService
from himmy.services.memory.service import ALWAYS_INCLUDE
from tests.conftest import run_async


def test_service_min_similarity_floor_drops_irrelevant_fact() -> None:
    """A ctor-level min_similarity drops a fact the explicit-0.0 path would surface."""
    svc = MemoryService(InMemoryMemoryStore(), min_similarity=0.5)
    svc.remember("the pond holds fish and ducks", subject_id="u")

    # An explicit 0.0 threshold WOULD return this orthogonal fact at similarity 0.0.
    baseline = MemoryService(InMemoryMemoryStore())
    baseline.remember("the pond holds fish and ducks", subject_id="u")
    baseline_hits = run_async(
        baseline.recall(
            "quantum chromodynamics", subject_id="u", similarity_threshold=0.0
        )
    )
    assert len(baseline_hits) == 1 and baseline_hits[0].similarity == 0.0

    # With the service floor, the orthogonal query correctly recalls nothing.
    hits = run_async(svc.recall("quantum chromodynamics", subject_id="u"))
    assert hits == []


def test_default_none_applies_noise_floor() -> None:
    """The ``None`` default now applies the safe ``> 0.0`` noise floor."""
    svc = MemoryService(InMemoryMemoryStore())  # no explicit floor
    svc.remember("the pond holds fish and ducks", subject_id="u")
    # Zero-overlap query: the default floor drops the only (similarity-0.0) memory.
    assert run_async(svc.recall("quantum chromodynamics", subject_id="u")) == []
    # ALWAYS_INCLUDE bypasses the floor and still returns it.
    forced = run_async(
        svc.recall(
            "quantum chromodynamics",
            subject_id="u",
            similarity_threshold=ALWAYS_INCLUDE,
        )
    )
    assert len(forced) == 1 and forced[0].similarity == 0.0


def test_per_call_threshold_overrides_service_default() -> None:
    """A per-call similarity_threshold takes precedence over the ctor floor."""
    svc = MemoryService(InMemoryMemoryStore(), min_similarity=0.9)
    svc.remember("the orchard has apple and pear trees", subject_id="u")

    # Service floor 0.9 would drop everything; an explicit 0.0 restores the top hit.
    hits = run_async(
        svc.recall("apple pear orchard", subject_id="u", similarity_threshold=0.0)
    )
    assert len(hits) == 1


def test_default_floor_drops_zero_overlap_while_always_include_keeps_order() -> None:
    """The default floors the zero-overlap memory; ALWAYS_INCLUDE keeps both ordered."""
    svc = MemoryService(InMemoryMemoryStore())  # no explicit floor
    svc.remember("the orchard has apple and pear trees", subject_id="u")
    svc.remember("the pond holds fish and ducks", subject_id="u")

    # Default: the orchard memory clears the floor, the orthogonal pond memory (0.0)
    # is dropped.
    hits = run_async(svc.recall("apple and pear trees orchard", subject_id="u"))
    assert len(hits) == 1
    assert hits[0].record.text.startswith("the orchard")

    # ALWAYS_INCLUDE bypasses the floor and returns both, most-similar first.
    forced = run_async(
        svc.recall(
            "apple and pear trees orchard",
            subject_id="u",
            similarity_threshold=ALWAYS_INCLUDE,
        )
    )
    assert len(forced) == 2
    assert forced[0].record.text.startswith("the orchard")
    assert forced[0].similarity >= forced[1].similarity


def test_floor_keeps_relevant_drops_irrelevant() -> None:
    """A floor keeps an on-topic fact and drops the orthogonal one in one call."""
    svc = MemoryService(InMemoryMemoryStore(), min_similarity=0.1)
    svc.remember("the orchard has apple and pear trees", subject_id="u")
    svc.remember("the pond holds fish and ducks", subject_id="u")
    hits = run_async(svc.recall("apple and pear trees orchard", subject_id="u"))
    assert len(hits) == 1
    assert hits[0].record.text.startswith("the orchard")
