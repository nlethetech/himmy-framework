"""Behavioral tests for hybrid (BM25 + dense RRF) retrieval on the KnowledgeBase.

The golden setup uses TWO complementary retrievers so the test is not hollow:

* the lexical (BM25) leg owns exact rare-token queries; and
* a small synonym-aware dense embedder owns paraphrase queries that share NO
  surface tokens with the target (so the lexical leg genuinely misses them).

Hybrid must retrieve BOTH where dense-only and lexical-only each miss one.
"""

from __future__ import annotations

import hashlib
import math

from himmy.services.knowledge import KnowledgeBase, RetrievalConfig
from himmy.services.knowledge.retrieval.lexical import tokenize
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


class SynonymEmbedder:
    """A tiny semantic embedder: synonyms collapse to one concept bucket.

    Two paraphrases that share no surface tokens (``reboot`` vs ``restart``) still
    embed near each other, so the dense leg can win a query the lexical leg misses —
    the complement of BM25's exact-token strength.
    """

    supports_images = False
    _CONCEPTS = {
        "reboot": "restart",
        "restart": "restart",
        "reset": "restart",
        "machine": "computer",
        "computer": "computer",
        "pc": "computer",
        "fix": "repair",
        "repair": "repair",
        "solve": "repair",
        "fast": "quick",
        "quick": "quick",
        "quickly": "quick",
        "problem": "issue",
        "issue": "issue",
    }

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            concept = self._CONCEPTS.get(tok, tok)
            digest = hashlib.sha256(concept.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return vec if norm == 0.0 else [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _build_kb() -> tuple[KnowledgeBase, str, dict[str, str]]:
    """Build a KB with a lexical-only target (A) and a paraphrase target (B)."""
    kb = KnowledgeBase(storage=StorageService(), embedder=SynonymEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    a = run_async(
        kb.ingest_text(
            rec.kb_id,
            f"{filler} {filler} The diagnostic code ZZQ42 means a checksum "
            f"failure. {filler} {filler}",
            source_uri="A",
        )
    )
    b = run_async(
        kb.ingest_text(
            rec.kb_id,
            "Restart the machine to fix the problem quickly.",
            source_uri="B",
        )
    )
    c = run_async(
        kb.ingest_text(
            rec.kb_id,
            "Cooking pasta requires boiling water and salt.",
            source_uri="C",
        )
    )
    return kb, rec.kb_id, {"A": a.document_id, "B": b.document_id, "C": c.document_id}


def test_hybrid_covers_both_legs() -> None:
    """Hybrid retrieves the rare-token doc AND the paraphrase doc."""
    kb, kb_id, ids = _build_kb()
    kb._retrieval = RetrievalConfig(mode="hybrid")

    rare = run_async(kb.search(kb_id, "ZZQ42", top_k=3))
    assert rare and rare[0].document_id == ids["A"]

    para = run_async(kb.search(kb_id, "reboot computer solve issue fast", top_k=3))
    assert para and para[0].document_id == ids["B"]


def test_lexical_only_query_is_a_dense_weakness() -> None:
    """The lexical leg, not dense, is what carries the rare-token query in hybrid."""
    kb, kb_id, ids = _build_kb()
    kb._retrieval = RetrievalConfig(mode="hybrid")
    hits = run_async(kb.search(kb_id, "ZZQ42", top_k=3))
    top = hits[0]
    assert top.document_id == ids["A"]
    # The breadcrumb proves the lexical leg ranked it #1.
    assert top.metadata["lexical_rank"] == 1


def test_paraphrase_query_is_a_lexical_blind_spot() -> None:
    """The paraphrase doc has NO lexical match — only the dense leg surfaces it."""
    kb, kb_id, ids = _build_kb()
    kb._retrieval = RetrievalConfig(mode="hybrid")
    hits = run_async(kb.search(kb_id, "reboot computer solve issue fast", top_k=3))
    winner = next(h for h in hits if h.document_id == ids["B"])
    # Lexical missed it entirely (no shared surface tokens); dense carried it.
    assert winner.metadata["lexical_rank"] is None
    assert winner.metadata["dense_rank"] is not None


def test_metadata_carries_rank_fusion_breadcrumbs() -> None:
    """Each hybrid result records HOW it was ranked (dense/lexical/rrf)."""
    kb, kb_id, _ = _build_kb()
    kb._retrieval = RetrievalConfig(mode="hybrid")
    hits = run_async(kb.search(kb_id, "ZZQ42", top_k=3))
    meta = hits[0].metadata
    for key in ("dense_rank", "lexical_rank", "rrf_score", "retrieval_mode"):
        assert key in meta
    assert meta["retrieval_mode"] == "hybrid"
    assert meta["rrf_score"] is not None
    # The result similarity is the fused RRF score (not a raw cosine).
    assert math.isclose(hits[0].similarity, meta["rrf_score"])


def test_threshold_applies_to_fused_score() -> None:
    """A similarity_threshold above any fused score yields no results under hybrid."""
    kb, kb_id, _ = _build_kb()
    kb._retrieval = RetrievalConfig(mode="hybrid")
    # Fused RRF scores are ~0.03; a 0.9 cutoff must drop everything.
    hits = run_async(kb.search(kb_id, "ZZQ42", top_k=3, similarity_threshold=0.9))
    assert hits == []


def test_metadata_filters_apply_to_both_legs() -> None:
    """metadata_filters restrict candidates in hybrid just like the dense path."""
    kb = KnowledgeBase(storage=StorageService(), embedder=SynonymEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    run_async(
        kb.ingest_text(
            rec.kb_id, "The code ZZQ42 fails.", source_uri="X", metadata={"team": "a"}
        )
    )
    run_async(
        kb.ingest_text(
            rec.kb_id,
            "The code ZZQ42 fails too.",
            source_uri="Y",
            metadata={"team": "b"},
        )
    )
    kb._retrieval = RetrievalConfig(mode="hybrid")
    hits = run_async(
        kb.search(rec.kb_id, "ZZQ42", top_k=5, metadata_filters={"team": "a"})
    )
    assert hits
    assert all(h.metadata.get("team") == "a" for h in hits)


def test_tenancy_guard_holds_under_hybrid() -> None:
    """A cross-tenant kb_id is rejected under hybrid mode (guard runs before search)."""
    from himmy.core.errors import HimmyError

    kb, kb_id, _ = _build_kb()
    kb._retrieval = RetrievalConfig(mode="hybrid")
    try:
        run_async(
            kb.search(kb_id, "ZZQ42", top_k=3, workspace_id="other", client_id="c")
        )
    except HimmyError as exc:
        assert "cross-tenant" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected a cross-tenant HimmyError")


def test_delete_prunes_lexical_in_lockstep() -> None:
    """Deleting a document removes it from hybrid results (lexical index rebuilt)."""
    kb, kb_id, ids = _build_kb()
    kb._retrieval = RetrievalConfig(mode="hybrid")
    before = run_async(kb.search(kb_id, "ZZQ42", top_k=3))
    assert any(h.document_id == ids["A"] for h in before)

    run_async(kb.delete_document(kb_id, ids["A"]))
    after = run_async(kb.search(kb_id, "ZZQ42", top_k=3))
    assert not any(h.document_id == ids["A"] for h in after)
