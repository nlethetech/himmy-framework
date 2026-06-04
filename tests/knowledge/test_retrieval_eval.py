"""Tests for the retrieval-quality eval harness (recall@k / MRR / nDCG)."""

from __future__ import annotations

import math

from opensims.services.knowledge import (
    DeterministicEmbedder,
    KnowledgeBase,
    RetrievalEvalCase,
    RetrievedChunk,
    aggregate_scores,
    evaluate_retrieval,
    score_retrieval,
)
from opensims.services.storage.service import StorageService
from tests.conftest import run_async


def _chunk(doc_id: str, sim: float) -> RetrievedChunk:
    return RetrievedChunk(document_id=doc_id, similarity=sim, text=doc_id)


# ----------------------------------------------------------------- metric math
def test_score_retrieval_metric_values() -> None:
    """Exact recall/precision/MRR/nDCG for a known ranking with one relevant doc."""
    case = RetrievalEvalCase(query="q", relevant_document_ids=["d1"], top_k=3)
    results = [_chunk("d2", 0.9), _chunk("d1", 0.8), _chunk("d3", 0.1)]
    score = score_retrieval(case, results)

    assert score.recall_at_k == 1.0
    assert math.isclose(score.precision_at_k, 1 / 3, rel_tol=1e-9)
    assert score.mrr == 0.5  # first relevant at rank 2
    assert math.isclose(score.ndcg_at_k, 1.0 / math.log2(3), rel_tol=1e-9)
    assert math.isclose(score.mean_similarity, 0.6, rel_tol=1e-9)
    assert score.hit is True
    assert score.num_relevant == 1
    assert score.num_retrieved == 3


def test_score_retrieval_perfect_and_miss() -> None:
    """A perfect top-rank scores 1.0; a total miss scores 0.0 across the board."""
    perfect = score_retrieval(
        RetrievalEvalCase(query="q", relevant_document_ids=["d1", "d2"], top_k=3),
        [_chunk("d1", 0.9), _chunk("d2", 0.8), _chunk("d3", 0.1)],
    )
    assert perfect.recall_at_k == 1.0
    assert perfect.mrr == 1.0
    assert math.isclose(perfect.ndcg_at_k, 1.0, rel_tol=1e-9)

    miss = score_retrieval(
        RetrievalEvalCase(query="q", relevant_document_ids=["d9"], top_k=2),
        [_chunk("d1", 0.5), _chunk("d2", 0.4)],
    )
    assert miss.recall_at_k == 0.0
    assert miss.mrr == 0.0
    assert miss.ndcg_at_k == 0.0
    assert miss.hit is False


def test_aggregate_scores_means() -> None:
    """Aggregation averages each metric and computes the hit rate."""
    a = score_retrieval(
        RetrievalEvalCase(query="a", relevant_document_ids=["d1"], top_k=2),
        [_chunk("d1", 1.0)],
    )
    b = score_retrieval(
        RetrievalEvalCase(query="b", relevant_document_ids=["d9"], top_k=2),
        [_chunk("d1", 0.2)],
    )
    report = aggregate_scores([a, b])
    assert report.num_cases == 2
    assert report.mean_recall_at_k == 0.5  # 1.0 + 0.0 over 2
    assert report.hit_rate == 0.5
    assert aggregate_scores([]).num_cases == 0


# --------------------------------------------------- integration over a real KB
def test_evaluate_retrieval_over_real_kb() -> None:
    """A golden set scores high on relevant queries and flags a true miss."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    kb_id = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb")).kb_id
    acme = run_async(
        kb.ingest_text(
            kb_id, "Quarterly revenue of ACME grew on strong widget sales.", title="r"
        )
    )
    garden = run_async(
        kb.ingest_text(
            kb_id, "Notes on gardening tomatoes and basil in spring.", title="g"
        )
    )

    cases = [
        RetrievalEvalCase(
            query="ACME revenue widget sales",
            relevant_document_ids=[acme.document_id],
            top_k=2,
            label="acme",
        ),
        RetrievalEvalCase(
            query="gardening tomatoes basil",
            relevant_document_ids=[garden.document_id],
            top_k=2,
            label="garden",
        ),
    ]
    report = run_async(evaluate_retrieval(kb, kb_id, cases))
    assert report.num_cases == 2
    assert report.hit_rate == 1.0
    assert report.mean_recall_at_k == 1.0
    assert report.mean_mrr == 1.0

    # A query with no lexical overlap is correctly scored as a miss.
    miss = run_async(
        evaluate_retrieval(
            kb,
            kb_id,
            [
                RetrievalEvalCase(
                    query="rocket spaceship orbital telemetry",
                    relevant_document_ids=[acme.document_id],
                    top_k=2,
                )
            ],
        )
    )
    assert miss.hit_rate == 0.0
    assert miss.mean_recall_at_k == 0.0
