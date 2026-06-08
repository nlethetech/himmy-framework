"""Behavioral tests for Reciprocal Rank Fusion (deterministic, no models)."""

from __future__ import annotations

import math

from himmy.services.knowledge.retrieval.fusion import (
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
)


def test_rrf_exact_scores_single_list() -> None:
    """RRF score of rank r in one list is exactly 1/(k+r)."""
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=10)
    scores = dict(fused)
    assert math.isclose(scores["a"], 1 / 11)
    assert math.isclose(scores["b"], 1 / 12)
    assert math.isclose(scores["c"], 1 / 13)
    # Order is by descending score.
    assert [cid for cid, _ in fused] == ["a", "b", "c"]


def test_rrf_sums_contributions_across_lists() -> None:
    """An item's fused score is the sum of 1/(k+rank) over every list it is in."""
    dense = ["x", "y", "z"]
    lexical = ["y", "x", "w"]
    fused = dict(reciprocal_rank_fusion([dense, lexical], k=60))
    # x: rank 1 (dense) + rank 2 (lexical)
    assert math.isclose(fused["x"], 1 / 61 + 1 / 62)
    # y: rank 2 (dense) + rank 1 (lexical)
    assert math.isclose(fused["y"], 1 / 62 + 1 / 61)
    # z only in dense (rank 3), w only in lexical (rank 3)
    assert math.isclose(fused["z"], 1 / 63)
    assert math.isclose(fused["w"], 1 / 63)


def test_rrf_consensus_beats_single_list_top() -> None:
    """An item ranked highly in BOTH lists beats one ranked top in only one.

    This is the whole point of fusion: consensus across retrievers wins.
    """
    dense = ["consensus", "dense_only"]
    lexical = ["consensus", "lexical_only"]
    ranked = reciprocal_rank_fusion([dense, lexical], k=60)
    order = [cid for cid, _ in ranked]
    assert order[0] == "consensus"
    # The two single-list items are tied; both rank below consensus.
    assert set(order[1:]) == {"dense_only", "lexical_only"}


def test_rrf_k_controls_top_rank_weighting() -> None:
    """A smaller k sharpens the advantage of rank-1 over rank-2; a larger k flattens it."""
    small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    large_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    small_gap = small_k["a"] - small_k["b"]
    large_gap = large_k["a"] - large_k["b"]
    assert small_gap > large_gap


def test_rrf_deterministic_tie_break_by_id() -> None:
    """Equal fused scores break ties by id ascending, so output is stable."""
    fused = reciprocal_rank_fusion([["b"], ["a"]], k=60)
    # Both at rank 1 in their own list -> equal score; 'a' sorts first.
    assert [cid for cid, _ in fused] == ["a", "b"]


def test_rrf_default_k_is_sixty() -> None:
    """The documented default k is 60 (Cormack et al.)."""
    assert DEFAULT_RRF_K == 60
    fused = dict(reciprocal_rank_fusion([["only"]]))
    assert math.isclose(fused["only"], 1 / 61)


def test_rrf_empty_and_invalid_k() -> None:
    """No lists -> empty result; a non-positive k is a clear error."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    try:
        reciprocal_rank_fusion([["a"]], k=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError for k=0")
