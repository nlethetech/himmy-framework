"""Guards for the additive promise: the dense default is unchanged + provenance holds.

These tests defend the two non-negotiables of the hybrid-retrieval change: a KB
built with no ``retrieval`` kwarg behaves byte-for-byte like the pre-change dense
path, and the in-run ``kb_search`` tool keeps emitting the same evidence shape +
RunEvents (now with rank-fusion breadcrumbs) under hybrid.
"""

from __future__ import annotations

from himmy.services.knowledge import (
    DeterministicEmbedder,
    KnowledgeBase,
    RetrievalConfig,
    register_kb_search_tool,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _seed_kb() -> tuple[KnowledgeBase, str]:
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "alpha beta gamma delta", source_uri="d1"))
    run_async(kb.ingest_text(rec.kb_id, "epsilon zeta eta theta", source_uri="d2"))
    run_async(kb.ingest_text(rec.kb_id, "alpha epsilon mixed terms", source_uri="d3"))
    return kb, rec.kb_id


def test_default_kb_matches_explicit_dense() -> None:
    """No retrieval kwarg == RetrievalConfig() (mode='dense') for the same corpus."""
    kb_default, kid = _seed_kb()
    assert kb_default._retrieval.mode == "dense"
    default_hits = run_async(kb_default.search(kid, "alpha", top_k=3))

    kb_explicit = KnowledgeBase(
        storage=StorageService(),
        embedder=DeterministicEmbedder(),
        retrieval=RetrievalConfig(),
    )
    rec = run_async(kb_explicit.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(
        kb_explicit.ingest_text(rec.kb_id, "alpha beta gamma delta", source_uri="d1")
    )
    run_async(
        kb_explicit.ingest_text(rec.kb_id, "epsilon zeta eta theta", source_uri="d2")
    )
    run_async(
        kb_explicit.ingest_text(rec.kb_id, "alpha epsilon mixed terms", source_uri="d3")
    )
    explicit_hits = run_async(kb_explicit.search(rec.kb_id, "alpha", top_k=3))

    assert [h.text for h in default_hits] == [h.text for h in explicit_hits]
    assert [h.similarity for h in default_hits] == [h.similarity for h in explicit_hits]


def test_dense_default_has_no_fusion_breadcrumbs() -> None:
    """The dense path never injects hybrid metadata (it is the original code path)."""
    kb, kid = _seed_kb()
    hits = run_async(kb.search(kid, "alpha", top_k=3))
    assert hits
    for h in hits:
        assert "rrf_score" not in h.metadata
        assert "retrieval_mode" not in h.metadata
        # The original keys are still present.
        assert "chunk_id" in h.metadata and "start_pos" in h.metadata


def test_dense_default_builds_no_lexical_index() -> None:
    """A pure-dense KB never pays for the BM25 index (no lexical work on dense)."""
    kb, kid = _seed_kb()
    run_async(kb.search(kid, "alpha", top_k=3))
    # The lexical cache stays empty for dense-mode searches.
    assert kb._lexical == {}


def test_kb_search_tool_hybrid_emits_events_and_breadcrumbs() -> None:
    """Under hybrid, kb_search keeps its evidence shape + emits TOOL_CALLED/COMPLETED."""
    from himmy.services.tools.models import ToolInvocation
    from himmy.services.tools.registry import ToolRegistry
    from himmy.services.tools.service import ToolService

    storage = StorageService()
    kb = KnowledgeBase(
        storage=storage,
        embedder=DeterministicEmbedder(),
        retrieval=RetrievalConfig(mode="hybrid"),
    )
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "ACME revenue rose on widget demand."))

    registry = ToolRegistry()
    register_kb_search_tool(
        registry, kb, default_workspace_id="w1", default_client_id="c1"
    )
    svc = ToolService(registry, event_sink=storage)
    result = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="kb_search",
                args={"query": "ACME widgets revenue", "kb_name": "kb"},
            )
        )
    )
    assert result.outcome == "success"
    out = result.result
    assert out["chunks"] and "rendered_text" in out
    # Evidence shape unchanged.
    ref = out["evidence_refs"][0]
    assert ref["source_type"] == "knowledge_base"
    assert ref["account_scope"]["workspace_id"] == "w1"
    # Rank-fusion breadcrumbs ride in the chunk metadata under hybrid.
    assert out["chunks"][0]["metadata"]["retrieval_mode"] == "hybrid"
    assert "rrf_score" in out["chunks"][0]["metadata"]
    # Both lifecycle events emitted.
    events = run_async(storage.list_events())
    types = {e.event_type.value for e in events}
    assert "TOOL_CALLED" in types
    assert "TOOL_COMPLETED" in types


def test_kb_search_mode_override_per_call() -> None:
    """A per-call mode='hybrid' arg turns a dense-default KB hybrid for that call only."""
    from himmy.services.tools.models import ToolInvocation
    from himmy.services.tools.registry import ToolRegistry
    from himmy.services.tools.service import ToolService

    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "The token QXZ99 is unique here."))

    registry = ToolRegistry()
    register_kb_search_tool(
        registry, kb, default_workspace_id="w1", default_client_id="c1"
    )
    svc = ToolService(registry)
    res = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="kb_search",
                args={"query": "QXZ99", "kb_name": "kb", "mode": "hybrid"},
            )
        )
    )
    assert res.outcome == "success"
    assert res.result["chunks"][0]["metadata"]["retrieval_mode"] == "hybrid"
    # The KB's configured default is restored after the call.
    assert kb._retrieval.mode == "dense"
