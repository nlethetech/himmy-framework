"""Real-data RAG quality evaluation against a live Ollama embedding model.

Stands the retrieval-eval harness up on a real corpus + the real ``qwen3-embedding``
model (4096-d) and measures dense vs hybrid (BM25+dense RRF) retrieval with
recall@k / MRR / nDCG@k / hit-rate. This is the regression gate for RAG quality: the
floors below are derived from measured live numbers, so a real drop in retrieval
quality (a worse embedder, a broken fusion path, a chunker regression) fails the test.

Marked ``integration`` and skipped unless Ollama is reachable with the embed model
pulled, so the offline unit lane is unaffected. Run it (and see the scorecard):

    pytest -m integration tests/integration/test_rag_eval.py -s -v
"""

from __future__ import annotations

import pytest

from himmy.services.knowledge import RetrievalConfig
from himmy.services.knowledge.local_embedders import (
    OllamaEmbedder,
    default_ollama_embed_model,
    ollama_embed_model_available,
    ollama_reachable,
)
from himmy.services.knowledge.retrieval_eval import (
    RetrievalEvalCase,
    compare_retrieval,
)
from himmy.services.knowledge.service import KnowledgeBase
from himmy.services.storage.service import StorageService
from tests.conftest import run_async
from tests.integration.rag_corpus import CASES, DOCS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (ollama_reachable() and ollama_embed_model_available()),
        reason=f"Ollama embed model {default_ollama_embed_model()!r} not pulled",
    ),
]

_WS, _CLIENT, _TOP_K = "rag-eval", "rag-eval", 5

# Floors derived from a measured live run on qwen3-embedding (29-doc corpus, 21 queries):
#   dense   recall@5=1.000  mrr=1.000  ndcg@5=1.000  hit=1.000
#   hybrid  recall@5=1.000  mrr=0.952  ndcg@5=0.965  hit=1.000
# Set below the observed numbers with margin so the gate catches a real regression (a
# worse embedder, a broken cosine/fusion path) without flaking on a model/version nudge.
_DENSE_RECALL_FLOOR = 0.92
_DENSE_NDCG_FLOOR = 0.92
_HYBRID_RECALL_FLOOR = 0.92
_HYBRID_NDCG_FLOOR = 0.90
_HITRATE_FLOOR = 0.95


def _build_kb() -> tuple[KnowledgeBase, str, list[RetrievalEvalCase]]:
    """Build a hybrid-capable KB on the real embed model, ingest the corpus, map cases."""
    embedder = OllamaEmbedder(model=default_ollama_embed_model())
    kb = KnowledgeBase(
        storage=StorageService(),
        embedder=embedder,
        # Construct with hybrid so the BM25 lexical index is built at ingest; the
        # comparison then swaps dense/hybrid at search time over the same index.
        retrieval=RetrievalConfig(mode="hybrid"),
    )
    rec = run_async(
        kb.create_kb(
            workspace_id=_WS, client_id=_CLIENT, name="corpus", vector_dim=embedder.dim
        )
    )
    key_to_id: dict[str, str] = {}
    for key, title, text in DOCS:
        doc = run_async(
            kb.ingest_text(rec.kb_id, text, title=title, metadata={"key": key})
        )
        key_to_id[key] = doc.document_id
    cases = [
        RetrievalEvalCase(
            query=q,
            relevant_document_ids=[key_to_id[k] for k in keys],
            top_k=_TOP_K,
            label=label,
        )
        for q, keys, label in CASES
    ]
    return kb, rec.kb_id, cases


def _print_report(name: str, report) -> None:  # noqa: ANN001 - test-local pretty printer
    print(
        f"  {name:14} recall@{_TOP_K}={report.mean_recall_at_k:.3f}  "
        f"mrr={report.mean_mrr:.3f}  ndcg@{_TOP_K}={report.mean_ndcg_at_k:.3f}  "
        f"hit_rate={report.hit_rate:.3f}  (n={report.num_cases})"
    )


def test_rag_quality_dense_vs_hybrid_on_real_embeddings() -> None:
    kb, kb_id, cases = _build_kb()

    reports = run_async(
        compare_retrieval(
            kb,
            kb_id,
            cases,
            {
                "dense": RetrievalConfig(mode="dense"),
                "hybrid": RetrievalConfig(mode="hybrid"),
            },
            workspace_id=_WS,
            client_id=_CLIENT,
        )
    )
    dense, hybrid = reports["dense"], reports["hybrid"]

    print(
        f"\nRAG scorecard — corpus={len(DOCS)} docs, {len(cases)} queries, qwen3-embedding:"
    )
    _print_report("dense", dense)
    _print_report("hybrid", hybrid)

    # Per-regime split (computed from the already-scored cases — no extra embed calls)
    # so we can see WHERE hybrid helps: rare-term/lexical queries vs paraphrase/semantic.
    def _regime(scores, regime):  # noqa: ANN001, ANN202 - test-local helper
        sub = [s for s in scores if s.label == regime]
        n = len(sub) or 1
        return (
            sum(s.recall_at_k for s in sub) / n,
            sum(s.ndcg_at_k for s in sub) / n,
            sum(1.0 for s in sub if s.hit) / n,
        )

    for regime in ("semantic", "lexical", "disambig"):
        dr, dn, dh = _regime(dense.cases, regime)
        hr, hn, hh = _regime(hybrid.cases, regime)
        nq = sum(1 for c in cases if c.label == regime)
        print(f"  -- regime={regime} ({nq} q):")
        print(f"       dense  recall={dr:.3f} ndcg={dn:.3f} hit={dh:.3f}")
        print(f"       hybrid recall={hr:.3f} ndcg={hn:.3f} hit={hh:.3f}")

    # FINDING (this corpus): dense qwen3-embedding is saturated — perfect on all 21
    # queries, including the confusable bank/star clusters where only the proper noun
    # disambiguates. The hybrid BM25 leg therefore adds no lift and slightly reorders
    # paraphrase queries (semantic nDCG 1.000 -> 0.926). Recommendation: keep dense the
    # default; hybrid earns its keep only with a weaker embedder or a larger, noisier
    # corpus. This test is the regression gate, not a "hybrid wins" claim.

    # Gate 1 — the production dense path must clear high absolute floors. A real embedder
    # regression (a worse model, a broken cosine path) drops below these and fails here.
    assert dense.mean_recall_at_k >= _DENSE_RECALL_FLOOR, dense.mean_recall_at_k
    assert dense.mean_ndcg_at_k >= _DENSE_NDCG_FLOOR, dense.mean_ndcg_at_k
    assert dense.hit_rate >= _HITRATE_FLOOR, dense.hit_rate

    # Gate 2 — the hybrid fusion path must still retrieve well, proving BM25+dense RRF is
    # wired and functional (a broken fusion tanks these). We do NOT assert hybrid > dense,
    # because the measured data shows it does not beat a saturated dense on this corpus.
    assert hybrid.mean_recall_at_k >= _HYBRID_RECALL_FLOOR, hybrid.mean_recall_at_k
    assert hybrid.mean_ndcg_at_k >= _HYBRID_NDCG_FLOOR, hybrid.mean_ndcg_at_k
    assert hybrid.hit_rate >= _HITRATE_FLOOR, hybrid.hit_rate
