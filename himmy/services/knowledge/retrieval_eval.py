"""Knowledge kernel: retrieval-quality evaluation (recall@k / MRR / nDCG).

The framework can score agent *outputs* but had no way to score *retrieval* — so
every RAG knob (chunker, embedder, similarity threshold, a future reranker) was a
blind change. This module closes that loop: define a golden set of
:class:`RetrievalEvalCase` (a query + the documents that should come back), run it
through :func:`evaluate_retrieval`, and get per-case + aggregate
recall@k / precision@k / MRR / nDCG / hit-rate. Relevance is judged at the
**document** level (the granularity ``KnowledgeBase.search`` returns), so a golden
set only needs to name the relevant ``document_id``s.

It runs fully offline against the :class:`DeterministicEmbedder`, so it can guard
retrieval regressions in CI just like the rest of the suite.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.knowledge.models import RetrievedChunk
    from himmy.services.knowledge.service import KnowledgeBase


class RetrievalEvalCase(BaseModel):
    """One golden query and the document ids that *should* be retrieved for it."""

    query: str
    relevant_document_ids: list[str] = Field(default_factory=list)
    top_k: int = 5
    label: str | None = None


class RetrievalScore(BaseModel):
    """Per-case retrieval metrics (all in ``[0, 1]`` except the counts)."""

    label: str | None = None
    query: str
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_similarity: float
    hit: bool
    num_relevant: int
    num_retrieved: int


class RetrievalEvalReport(BaseModel):
    """A scorecard over a golden set: per-case scores plus their means."""

    cases: list[RetrievalScore] = Field(default_factory=list)
    num_cases: int = 0
    mean_recall_at_k: float = 0.0
    mean_precision_at_k: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg_at_k: float = 0.0
    mean_similarity: float = 0.0
    hit_rate: float = 0.0


def _ranked_unique_docs(results: list[RetrievedChunk]) -> list[str]:
    """Document ids in retrieval order, de-duplicated to first appearance."""
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in results:
        if chunk.document_id not in seen:
            seen.add(chunk.document_id)
            ordered.append(chunk.document_id)
    return ordered


def score_retrieval(
    case: RetrievalEvalCase, results: list[RetrievedChunk]
) -> RetrievalScore:
    """Compute recall/precision/MRR/nDCG/similarity for one query's results.

    Document-level binary relevance: a retrieved document counts as a hit when its
    id is in the case's ``relevant_document_ids``. nDCG uses the standard binary
    gain with an ideal ranking of ``min(|relevant|, k)`` documents.
    """
    k = max(1, case.top_k)
    relevant = set(case.relevant_document_ids)
    ranked = _ranked_unique_docs(results)
    top = ranked[:k]
    hits = [doc for doc in top if doc in relevant]

    recall = len(set(top) & relevant) / len(relevant) if relevant else 0.0
    precision = len(hits) / len(top) if top else 0.0

    mrr = 0.0
    for rank, doc in enumerate(top, start=1):
        if doc in relevant:
            mrr = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(top, start=1)
        if doc in relevant
    )
    ideal_hits = min(len(relevant), len(top))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    mean_sim = sum(c.similarity for c in results) / len(results) if results else 0.0

    return RetrievalScore(
        label=case.label,
        query=case.query,
        k=k,
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        ndcg_at_k=ndcg,
        mean_similarity=mean_sim,
        hit=bool(hits),
        num_relevant=len(relevant),
        num_retrieved=len(ranked),
    )


def aggregate_scores(scores: list[RetrievalScore]) -> RetrievalEvalReport:
    """Roll per-case scores up into a report with per-metric means."""
    n = len(scores)
    if n == 0:
        return RetrievalEvalReport()

    def _mean(values: list[float]) -> float:
        return sum(values) / n

    return RetrievalEvalReport(
        cases=scores,
        num_cases=n,
        mean_recall_at_k=_mean([s.recall_at_k for s in scores]),
        mean_precision_at_k=_mean([s.precision_at_k for s in scores]),
        mean_mrr=_mean([s.mrr for s in scores]),
        mean_ndcg_at_k=_mean([s.ndcg_at_k for s in scores]),
        mean_similarity=_mean([s.mean_similarity for s in scores]),
        hit_rate=_mean([1.0 if s.hit else 0.0 for s in scores]),
    )


async def evaluate_retrieval(
    kb: KnowledgeBase,
    kb_id: str,
    cases: list[RetrievalEvalCase],
    *,
    workspace_id: str | None = None,
    client_id: str | None = None,
    metadata_filters: dict[str, Any] | None = None,
) -> RetrievalEvalReport:
    """Run a golden set through ``KnowledgeBase.search`` and score the retrieval.

    Each case is searched with its own ``top_k`` (tenancy/metadata filters applied
    uniformly), scored, and aggregated into a :class:`RetrievalEvalReport`.
    """
    scores: list[RetrievalScore] = []
    for case in cases:
        results = await kb.search(
            kb_id,
            case.query,
            top_k=case.top_k,
            metadata_filters=metadata_filters,
            workspace_id=workspace_id,
            client_id=client_id,
        )
        scores.append(score_retrieval(case, results))
    return aggregate_scores(scores)


__all__ = [
    "RetrievalEvalCase",
    "RetrievalScore",
    "RetrievalEvalReport",
    "score_retrieval",
    "aggregate_scores",
    "evaluate_retrieval",
]
