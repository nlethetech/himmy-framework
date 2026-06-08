"""HybridRetriever — orchestrates dense + lexical -> RRF -> rerank -> top_k.

This is the engine behind ``RetrievalConfig(mode='hybrid')``. It is decoupled from
*where* the legs run: the KnowledgeBase injects three async callables — a dense leg,
a lexical leg, and a re-hydrator that turns a winning ``chunk_id`` into a
:class:`RetrievedChunk` — so the same orchestration serves both the in-memory KB and
a lexical-capable backend without duplicating fusion logic.

The pipeline, per query:

1. **candidate generation** — run the dense and lexical legs (each ``candidate_pool``
   deep), optionally once per rewritten query variant;
2. **fusion** — Reciprocal Rank Fusion over the per-leg ``chunk_id`` rank lists,
   producing one fused ordering that needs no score-scale reconciliation;
3. **rerank** (optional) — cross-encode the fused top candidates and re-order;
4. **cut + hydrate** — keep ``top_k`` and re-hydrate each into a RetrievedChunk whose
   ``similarity`` is the fused/reranked score and whose metadata records *how* it was
   ranked (dense_rank, dense_sim, lexical_rank, lexical_score, rrf_score,
   rerank_score) — so the audit trail explains the ranking.

A post-fusion ``similarity_threshold`` is applied on the final score with the same
semantics as the dense path (``None`` => keep ``> 0``; a float => keep ``>=``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from himmy.services.knowledge.models import RetrievedChunk
from himmy.services.knowledge.retrieval.config import RetrievalConfig
from himmy.services.knowledge.retrieval.fusion import reciprocal_rank_fusion

#: A dense leg: (query, pool_size) -> ranked [(chunk_id, cosine_similarity)].
DenseLeg = Callable[[str, int], Awaitable[list[tuple[str, float]]]]
#: A lexical leg: (query, pool_size) -> ranked [(chunk_id, bm25_score)].
LexicalLeg = Callable[[str, int], Awaitable[list[tuple[str, float]]]]
#: Re-hydrate a winning chunk into a RetrievedChunk (None if it vanished).
Hydrator = Callable[[str], Awaitable[RetrievedChunk | None]]
#: Fetch a chunk's text for the reranker (None if it has none, e.g. an image).
TextLookup = Callable[[str], Awaitable[str | None]]


class HybridRetriever:
    """Fuse a dense and a lexical retriever (and optional rerank/rewrite)."""

    def __init__(
        self,
        *,
        dense_leg: DenseLeg,
        lexical_leg: LexicalLeg,
        hydrate: Hydrator,
        text_lookup: TextLookup,
        config: RetrievalConfig,
    ) -> None:
        """Wire the orchestrator to its legs and the active config."""
        self._dense = dense_leg
        self._lexical = lexical_leg
        self._hydrate = hydrate
        self._text_lookup = text_lookup
        self._config = config

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        similarity_threshold: float | None,
    ) -> list[RetrievedChunk]:
        """Run the full hybrid pipeline for ``query`` and return ``top_k`` chunks."""
        queries = await self._expand_queries(query)
        pool = max(self._config.candidate_pool, top_k)

        # Collect per-query, per-leg rankings + remember each leg's raw scores/ranks
        # so the breadcrumbs reflect the BEST rank a chunk earned in either list.
        rankings: list[list[str]] = []
        dense_rank: dict[str, int] = {}
        dense_sim: dict[str, float] = {}
        lexical_rank: dict[str, int] = {}
        lexical_score: dict[str, float] = {}

        for q in queries:
            dense_hits = await self._dense(q, pool)
            lexical_hits = await self._lexical(q, pool)
            rankings.append([cid for (cid, _) in dense_hits])
            rankings.append([cid for (cid, _) in lexical_hits])
            for rank, (cid, sim) in enumerate(dense_hits, start=1):
                if cid not in dense_rank or rank < dense_rank[cid]:
                    dense_rank[cid] = rank
                    dense_sim[cid] = sim
            for rank, (cid, score) in enumerate(lexical_hits, start=1):
                if cid not in lexical_rank or rank < lexical_rank[cid]:
                    lexical_rank[cid] = rank
                    lexical_score[cid] = score

        fused = reciprocal_rank_fusion(rankings, k=self._config.rrf_k)
        if not fused:
            return []
        rrf_score = {cid: score for (cid, score) in fused}

        ordered_ids = [cid for (cid, _) in fused]
        rerank_score: dict[str, float] = {}
        if self._config.rerank and self._config.reranker is not None:
            ordered_ids, rerank_score = await self._rerank(query, ordered_ids[:pool])

        results: list[RetrievedChunk] = []
        for cid in ordered_ids:
            final_score = rerank_score.get(cid, rrf_score.get(cid, 0.0))
            if not self._keep(final_score, similarity_threshold):
                continue
            chunk = await self._hydrate(cid)
            if chunk is None:
                continue
            chunk.similarity = final_score
            chunk.metadata = {
                **chunk.metadata,
                "dense_rank": dense_rank.get(cid),
                "dense_sim": dense_sim.get(cid),
                "lexical_rank": lexical_rank.get(cid),
                "lexical_score": lexical_score.get(cid),
                "rrf_score": rrf_score.get(cid),
                "rerank_score": rerank_score.get(cid),
                "retrieval_mode": "hybrid",
            }
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    async def _expand_queries(self, query: str) -> list[str]:
        """Apply the (optional) query rewriter, always retaining the original."""
        if self._config.query_rewrite and self._config.rewriter is not None:
            variants = await self._config.rewriter.rewrite(query)
            return variants or [query]
        return [query]

    async def _rerank(
        self, query: str, candidate_ids: list[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Cross-encode the fused candidates; return reordered ids + their scores."""
        assert self._config.reranker is not None  # noqa: S101 - guarded by caller
        pairs: list[tuple[str, str]] = []
        for cid in candidate_ids:
            text = await self._text_lookup(cid)
            if text:
                pairs.append((cid, text))
        if not pairs:
            return candidate_ids, {}
        reranked = await self._config.reranker.rerank(query, pairs)
        scores = {cid: score for (cid, score) in reranked}
        # Reranked ids first (in their new order), then any that had no text to
        # score, so the cut never silently drops candidates the reranker skipped.
        ordered = [cid for (cid, _) in reranked]
        ordered.extend(cid for cid in candidate_ids if cid not in scores)
        return ordered, scores

    @staticmethod
    def _keep(score: float, threshold: float | None) -> bool:
        """Apply the dense path's threshold semantics to the final fused score."""
        if threshold is None:
            return score > 0.0
        return score >= threshold


__all__ = ["HybridRetriever", "DenseLeg", "LexicalLeg", "Hydrator", "TextLookup"]
