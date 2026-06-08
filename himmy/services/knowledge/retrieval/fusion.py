"""Reciprocal Rank Fusion (RRF) — combine ranked lists by rank, not score.

BM25 (lexical) and cosine (dense) produce scores on incompatible scales, so you
cannot simply add or average them. RRF sidesteps the scale mismatch entirely by
fusing on *rank position*: each item earns ``1 / (k + rank)`` from every list it
appears in, and the per-list contributions are summed. An item ranked highly in
both lists beats one ranked highly in only one — the production-proven way to make
hybrid retrieval consistently beat either retriever alone on nDCG.

``k`` (default 60, the value from the original Cormack et al. RRF paper) damps the
influence of top ranks: a larger ``k`` flattens the contribution curve so deeper
ranks matter more relative to the very top.

This is a pure function — no models, no I/O — so it is deterministic and fully
offline.
"""

from __future__ import annotations

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = DEFAULT_RRF_K
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one, ordered by descending RRF score.

    Each input is a list of ids ordered best-first (rank 1 is index 0). The RRF
    score of an id is ``sum(1 / (k + rank))`` over every list it appears in, where
    ``rank`` is 1-based. Ties (equal fused score) are broken deterministically by
    id so the output order is stable across runs.

    Args:
        rankings: One ranked id list per retriever (e.g. dense, lexical). Empty
            lists are ignored; an id absent from a list contributes nothing from it.
        k: The RRF damping constant (must be positive). Defaults to 60.

    Returns:
        ``(id, fused_score)`` pairs sorted by fused score descending, then id
        ascending. The fused score is strictly positive for any returned id.
    """
    if k <= 0:
        raise ValueError("RRF k must be a positive integer")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for index, item_id in enumerate(ranking):
            rank = index + 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


__all__ = ["reciprocal_rank_fusion", "DEFAULT_RRF_K"]
