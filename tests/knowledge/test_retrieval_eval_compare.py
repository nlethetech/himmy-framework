"""The proof gate: compare_retrieval shows hybrid (and hybrid+rerank) >= dense.

This is the on-identity differentiator — every retrieval knob is provable via the
EXISTING retrieval_eval harness before it ships. The golden set is designed so the
pure-dense path (the real offline DeterministicEmbedder) genuinely under-retrieves
rare-token queries — the hashing-trick dilutes a single decisive token buried in a
long, common-word document — while the BM25 lexical leg recovers them. Everything
runs fully offline (no model downloads, a stub reranker for the rerank case).
"""

from __future__ import annotations

from himmy.services.knowledge import (
    DeterministicEmbedder,
    KnowledgeBase,
    RetrievalConfig,
    RetrievalEvalCase,
    compare_retrieval,
)
from himmy.services.knowledge.retrieval.lexical import tokenize
from himmy.services.storage.service import StorageService
from tests.conftest import run_async

_COMMON = (
    "report quarterly revenue growth analysis market segment "
    "customer region product line"
)


def _golden_kb() -> tuple[KnowledgeBase, str, list[RetrievalEvalCase]]:
    """A KB where rare-token recall separates dense from hybrid."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    cases: list[RetrievalEvalCase] = []
    for code in ("ZZQ42", "NX88K", "QR7VV", "KP3LM"):
        doc = run_async(
            kb.ingest_text(
                rec.kb_id,
                f"{_COMMON} {_COMMON} The internal reference {code} applies "
                f"here. {_COMMON} {_COMMON}",
                source_uri=code,
            )
        )
        cases.append(
            RetrievalEvalCase(
                query=code,
                relevant_document_ids=[doc.document_id],
                top_k=3,
                label=code,
            )
        )
    # Distractors: pure common text, no codes — these out-rank the diluted targets
    # under pure dense cosine.
    for i in range(12):
        run_async(
            kb.ingest_text(
                rec.kb_id,
                f"{_COMMON} extra filler number {i} {_COMMON}",
                source_uri=f"D{i}",
            )
        )
    return kb, rec.kb_id, cases


def test_hybrid_beats_dense_on_ndcg_and_recall() -> None:
    """mean nDCG@k and recall@k strictly improve from dense -> hybrid (the gate)."""
    kb, kb_id, cases = _golden_kb()
    reports = run_async(
        compare_retrieval(
            kb,
            kb_id,
            cases,
            {"dense": RetrievalConfig(), "hybrid": RetrievalConfig(mode="hybrid")},
        )
    )
    dense = reports["dense"]
    hybrid = reports["hybrid"]
    # The differentiator's promise: hybrid is never worse, and here is strictly better.
    assert hybrid.mean_ndcg_at_k >= dense.mean_ndcg_at_k
    assert hybrid.mean_recall_at_k >= dense.mean_recall_at_k
    assert hybrid.mean_ndcg_at_k > dense.mean_ndcg_at_k
    assert hybrid.hit_rate > dense.hit_rate


def test_rerank_does_not_regress_hybrid() -> None:
    """A stub reranker that promotes the relevant doc keeps hybrid's recall/nDCG.

    The reranker scores by exact rare-token overlap (a deterministic stand-in for a
    cross-encoder), so the relevant chunk stays on top — proving the rerank stage is
    wired in and gate-able without a model download.
    """

    class _RareTokenReranker:
        async def rerank(
            self, query: str, candidates: list[tuple[str, str]]
        ) -> list[tuple[str, float]]:
            wanted = set(tokenize(query))
            scored = [
                (cid, float(len(wanted & set(tokenize(text)))))
                for cid, text in candidates
            ]
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            return scored

    kb, kb_id, cases = _golden_kb()
    reports = run_async(
        compare_retrieval(
            kb,
            kb_id,
            cases,
            {
                "dense": RetrievalConfig(),
                "hybrid": RetrievalConfig(mode="hybrid"),
                "hybrid_rerank": RetrievalConfig(
                    mode="hybrid", rerank=True, reranker=_RareTokenReranker()
                ),
            },
        )
    )
    base = reports["hybrid"]
    reranked = reports["hybrid_rerank"]
    assert reranked.mean_recall_at_k >= base.mean_recall_at_k
    assert reranked.mean_ndcg_at_k >= base.mean_ndcg_at_k
    assert reranked.mean_recall_at_k > reports["dense"].mean_recall_at_k


def test_compare_restores_original_config() -> None:
    """compare_retrieval never leaves the KB reconfigured (config restored)."""
    kb, kb_id, cases = _golden_kb()
    assert kb._retrieval.mode == "dense"
    run_async(
        compare_retrieval(kb, kb_id, cases, {"hybrid": RetrievalConfig(mode="hybrid")})
    )
    assert kb._retrieval.mode == "dense"


def test_compare_restores_config_even_on_error() -> None:
    """An exception mid-comparison still restores the KB's original config."""

    class _BoomReranker:
        async def rerank(
            self, query: str, candidates: list[tuple[str, str]]
        ) -> list[tuple[str, float]]:
            raise RuntimeError("boom")

    kb, kb_id, cases = _golden_kb()
    try:
        run_async(
            compare_retrieval(
                kb,
                kb_id,
                cases,
                {
                    "broken": RetrievalConfig(
                        mode="hybrid", rerank=True, reranker=_BoomReranker()
                    )
                },
            )
        )
    except RuntimeError:
        pass
    assert kb._retrieval.mode == "dense"
