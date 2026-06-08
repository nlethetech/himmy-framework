"""Behavioral tests for the opt-in query rewriters (multi-query / HyDE)."""

from __future__ import annotations

from himmy.services.inference.models import (
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
)
from himmy.services.knowledge import KnowledgeBase, RetrievalConfig
from himmy.services.knowledge.retrieval.query_rewrite import (
    HyDERewriter,
    IdentityRewriter,
    MultiQueryRewriter,
    QueryRewriterProtocol,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async
from tests.knowledge.test_hybrid_retrieval import SynonymEmbedder


class _FakeClientManager:
    """A deterministic stand-in for a ClientManager: returns canned text."""

    def __init__(
        self, text: str, *, status: InferenceStatus = InferenceStatus.SUCCESS
    ) -> None:
        self._text = text
        self._status = status

    def resolve(self, model_key: str) -> str:
        return f"fake:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=self._status,
            output_text=self._text if self._status == InferenceStatus.SUCCESS else None,
        )


def test_identity_rewriter_is_noop() -> None:
    """The offline default returns the query unchanged (zero-config stays model-free)."""
    rewriter = IdentityRewriter()
    assert isinstance(rewriter, QueryRewriterProtocol)
    assert run_async(rewriter.rewrite("hello world")) == ["hello world"]


def test_multiquery_keeps_original_and_adds_expansions() -> None:
    """MultiQueryRewriter always keeps the original first, then model alternatives."""
    cm = _FakeClientManager("reboot the box\nrestart the machine\n- power cycle")
    rewriter = MultiQueryRewriter(cm, num_expansions=3)
    variants = run_async(rewriter.rewrite("how to fix a frozen computer"))
    assert variants[0] == "how to fix a frozen computer"
    assert "reboot the box" in variants
    assert "power cycle" in variants  # bullet marker stripped
    assert len(variants) <= 4


def test_multiquery_degrades_on_model_failure() -> None:
    """A failed model call falls back to the original query (no hard error)."""
    cm = _FakeClientManager("", status=InferenceStatus.FAILED)
    rewriter = MultiQueryRewriter(cm, num_expansions=3)
    assert run_async(rewriter.rewrite("query")) == ["query"]


def test_hyde_returns_hypothetical_document() -> None:
    """HyDE appends the generated passage; keep_original retains the raw query."""
    cm = _FakeClientManager("A frozen computer is fixed by restarting it.")
    rewriter = HyDERewriter(cm, keep_original=True)
    variants = run_async(rewriter.rewrite("how to fix a frozen computer"))
    assert variants[0] == "how to fix a frozen computer"
    assert any("restarting" in v for v in variants)


def test_expansions_fold_into_one_rrf_pool() -> None:
    """A rewritten variant that matches a doc the original misses is fused in.

    The original query shares no surface/concept tokens with the target, but an
    expansion does — proving every variant's candidates land in the same RRF pool.
    """
    kb = KnowledgeBase(storage=StorageService(), embedder=SynonymEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    target = run_async(
        kb.ingest_text(
            rec.kb_id,
            "Restart the machine to fix the problem quickly.",
            source_uri="T",
        )
    )
    run_async(
        kb.ingest_text(
            rec.kb_id,
            "Cooking pasta requires boiling water and salt.",
            source_uri="D",
        )
    )

    # Original query matches nothing; the expansion ("reboot computer ...") does.
    cm = _FakeClientManager("reboot computer solve issue fast")
    kb._retrieval = RetrievalConfig(
        mode="hybrid",
        query_rewrite=True,
        rewriter=MultiQueryRewriter(cm, num_expansions=1),
    )
    hits = run_async(kb.search(rec.kb_id, "xyzzy nonsense unrelated", top_k=3))
    assert any(h.document_id == target.document_id for h in hits)


def test_query_rewrite_off_by_default() -> None:
    """The default RetrievalConfig performs no rewrite (offline-first guarantee)."""
    cfg = RetrievalConfig()
    assert cfg.query_rewrite is False
    assert cfg.rewriter is None
