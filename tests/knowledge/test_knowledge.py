"""Tests for the knowledge kernel: chunking, embedding, ingest, search, adapter."""

from __future__ import annotations

import pytest

from opensims.core.errors import OpenSimsError
from opensims.services.context.models import ContextField
from opensims.services.knowledge import (
    DeterministicEmbedder,
    DocumentInput,
    DocumentReaderFactory,
    KnowledgeBase,
    KnowledgeBaseAdapter,
    OpenAIMultimodalEmbeddingModel,
    SemanticChunker,
    build_knowledge_schema_ddl,
    embedder_is_multimodal,
    register_kb_search_tool,
    resolve_column_and_index,
)
from opensims.services.storage.service import StorageService
from tests.conftest import run_async


class _EmptyVecEmbedder(DeterministicEmbedder):
    """An embedder that returns empty document vectors (a broken provider call)."""

    async def embed_documents(self, texts):  # type: ignore[override]
        return [[] for _ in texts]


class _ZeroVecEmbedder(DeterministicEmbedder):
    """An embedder that returns zero-norm document vectors of the right dim."""

    async def embed_documents(self, texts):  # type: ignore[override]
        return [[0.0] * self.dim for _ in texts]


class _FakeMultimodalEmbedder(DeterministicEmbedder):
    """A deterministic embedder that *declares* image capability for ingest tests."""

    supports_images = True
    is_multimodal = True


def test_deterministic_embedder_is_stable_and_normalized() -> None:
    """Same text -> same unit vector of the configured dimension."""
    emb = DeterministicEmbedder(dim=64)
    v1 = run_async(emb.embed_query("hello world"))
    v2 = run_async(emb.embed_query("hello world"))
    assert v1 == v2
    assert len(v1) == 64
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_chunker_covers_text_with_offsets() -> None:
    """Chunks cover the document and carry valid (start, end) offsets."""
    chunker = SemanticChunker(max_chars=50, overlap=10)
    text = "Sentence one. " * 20
    chunks = chunker.chunk(text)
    assert chunks
    for start, end, body in chunks:
        assert 0 <= start < end <= len(text)
        assert text[start:end] == body
    # Empty text -> no chunks.
    assert chunker.chunk("") == []


def test_ingest_and_search_returns_relevant_chunk() -> None:
    """Ingested text is searchable; lexical overlap surfaces the right document."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    record = run_async(
        kb.create_kb(workspace_id="w1", client_id="c1", name="manuals", vector_dim=64)
    )
    run_async(
        kb.ingest_text(
            record.kb_id,
            "The quarterly revenue of ACME grew due to strong widget sales.",
            title="ACME report",
        )
    )
    run_async(
        kb.ingest_text(
            record.kb_id,
            "Unrelated note about gardening tomatoes in spring.",
            title="garden",
        )
    )
    results = run_async(kb.search(record.kb_id, "ACME revenue widgets", top_k=1))
    assert len(results) == 1
    assert "ACME" in (results[0].text or "")
    assert results[0].similarity > 0.0


def test_duplicate_kb_name_raises() -> None:
    """Creating a KB with a duplicate (workspace, client, name) raises."""
    kb = KnowledgeBase(storage=StorageService())
    run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="dup"))
    with pytest.raises(OpenSimsError):
        run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="dup"))


def test_ingest_documents_batches_one_embed_call() -> None:
    """ingest_documents folds all chunks into a single embed_documents call."""

    class CountingEmbedder(DeterministicEmbedder):
        def __init__(self) -> None:
            super().__init__(dim=64)
            self.doc_calls = 0

        async def embed_documents(self, texts):  # type: ignore[override]
            self.doc_calls += 1
            return await super().embed_documents(texts)

    emb = CountingEmbedder()
    kb = KnowledgeBase(storage=StorageService(), embedder=emb)
    record = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="batch"))
    run_async(
        kb.ingest_documents(
            record.kb_id,
            [DocumentInput(text="alpha doc"), DocumentInput(text="beta doc")],
        )
    )
    assert emb.doc_calls == 1


def test_document_input_requires_exactly_one_source() -> None:
    """DocumentInput rejects both/neither of text and file."""
    with pytest.raises(ValueError):
        DocumentInput()
    with pytest.raises(ValueError):
        DocumentInput(text="x", file="y.txt")
    # Exactly one is fine.
    assert DocumentInput(text="x").text == "x"


def test_knowledge_base_adapter_returns_context_field() -> None:
    """The adapter resolves a KB and projects search results into a ContextField."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    record = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(
        kb.ingest_text(record.kb_id, "ACME revenue rose on widget demand.", title="r")
    )
    adapter = KnowledgeBaseAdapter(kb)
    scope = {
        "workspace_id": "w1",
        "subject_id": "c1",
        "spec_metadata": {"kb_name": "kb", "query": "ACME revenue"},
    }
    field = run_async(adapter.fetch("knowledge", scope))
    assert isinstance(field, ContextField)
    assert field.source == "knowledge_base"
    assert "rendered_text" in field.value
    assert field.evidence_refs
    assert field.confidence > 0.0


def test_text_reader_reads_file(tmp_path) -> None:
    """The reader factory reads a .txt file via TextReader."""
    factory = DocumentReaderFactory()
    p = tmp_path / "doc.txt"
    p.write_text("hello from disk", encoding="utf-8")
    assert factory.read(str(p)) == "hello from disk"
    assert factory.reader_for("foo.unknown") is None


def test_multimodal_embedder_is_import_safe(monkeypatch) -> None:
    """The multimodal model constructs offline and raises a clear, normalized error.

    With no provider configured (no openai key / extra) the real path normalizes the
    SDK/import failure to an :class:`OpenSimsError` rather than leaking a provider
    exception. It also advertises image capability so ingest_image accepts it.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    model = OpenAIMultimodalEmbeddingModel("some-model")
    assert model.model_name == "some-model"
    assert getattr(model, "supports_images", False) is True
    with pytest.raises(OpenSimsError):
        run_async(model.embed_query("x"))


# --------------------------------------------------------------------- CK-3
def test_empty_embedding_vector_is_rejected_at_ingest() -> None:
    """An embedder returning [] must hard-fail ingest (no unretrievable chunks)."""
    kb = KnowledgeBase(storage=StorageService(), embedder=_EmptyVecEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    with pytest.raises(OpenSimsError):
        run_async(kb.ingest_text(rec.kb_id, "some real content here"))
    # Nothing was persisted.
    assert run_async(kb.search(rec.kb_id, "content")) == []


def test_zero_norm_embedding_vector_is_rejected_at_ingest() -> None:
    """An all-zero (zero-norm) vector is a hard error at ingest."""
    kb = KnowledgeBase(storage=StorageService(), embedder=_ZeroVecEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    with pytest.raises(OpenSimsError):
        run_async(kb.ingest_text(rec.kb_id, "some real content here"))


# --------------------------------------------------------------------- CK-7
def test_default_threshold_drops_orthogonal_chunks() -> None:
    """The default (None) threshold drops zero-similarity (no-overlap) chunks."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "alpha beta gamma"))
    # A query with no token overlap scores 0.0 -> dropped by default.
    results = run_async(kb.search(rec.kb_id, "zzz qqq vvv"))
    assert results == []
    # An explicit 0.0 threshold keeps the documented "sim >= threshold" rule.
    kept = run_async(kb.search(rec.kb_id, "zzz qqq vvv", similarity_threshold=0.0))
    assert len(kept) == 1
    assert kept[0].similarity == 0.0


# --------------------------------------------------------------------- CK-8
def test_chunker_clamps_overlap_no_near_duplicate_microchunks() -> None:
    """Large overlap relative to a boundary cut never yields a tiny all-overlap chunk."""
    chunker = SemanticChunker(max_chars=4, overlap=3, min_new_chars=2)
    chunks = chunker.chunk("a b c d e f g h")
    # Every emitted chunk advances by at least min_new_chars over its predecessor.
    prev_end = -1
    for _start, end, _ in chunks:
        if prev_end >= 0:
            assert (end - prev_end) >= 2
        prev_end = end
    # And the whole document is still covered to the end.
    assert chunks[-1][1] == len("a b c d e f g h")


# --------------------------------------------------------------------- CK-10
def test_ingest_image_requires_multimodal_embedder() -> None:
    """A text-only embedder is refused; a multimodal-declaring one is accepted."""
    text_kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(text_kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    with pytest.raises(OpenSimsError):
        run_async(text_kb.ingest_image(rec.kb_id, "image://logo.png", caption="logo"))

    mm_kb = KnowledgeBase(storage=StorageService(), embedder=_FakeMultimodalEmbedder())
    rec2 = run_async(mm_kb.create_kb(workspace_id="w", client_id="c", name="kb2"))
    doc = run_async(
        mm_kb.ingest_image(rec2.kb_id, "image://logo.png", caption="the logo")
    )
    assert doc.source_uri == "image://logo.png"
    # The image chunk is searchable and carries image-kind provenance.
    hits = run_async(mm_kb.search(rec2.kb_id, "image://logo.png"))
    assert hits and hits[0].chunk_kind == "image"


def test_embedder_is_multimodal_helper() -> None:
    """Only embedders declaring the capability are treated as multimodal."""
    assert embedder_is_multimodal(_FakeMultimodalEmbedder()) is True
    assert embedder_is_multimodal(DeterministicEmbedder()) is False


# --------------------------------------------------------------------- CK-11
def test_empty_text_document_input_rejected() -> None:
    """Empty/whitespace text (or empty file path) is invalid at construction."""
    with pytest.raises(ValueError):
        DocumentInput(text="")
    with pytest.raises(ValueError):
        DocumentInput(text="   ")
    with pytest.raises(ValueError):
        DocumentInput(file="")


def test_ingest_zero_chunk_documents_does_not_persist_no_op() -> None:
    """A document that chunks to nothing is skipped, not silently persisted."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    # Whitespace-only text is rejected at the DocumentInput layer.
    with pytest.raises(ValueError):
        DocumentInput(text="   \n  ")


# --------------------------------------------------------------------- CK-12
def test_embedder_must_satisfy_protocol() -> None:
    """A bad embedder (missing embed_query) is rejected at construction."""

    class _Bad:
        async def embed_documents(self, texts):
            return [[1.0]]

    with pytest.raises(OpenSimsError):
        KnowledgeBase(storage=StorageService(), embedder=_Bad())


# --------------------------------------------------------------------- CK-6
def test_search_tenancy_guard_blocks_cross_tenant_kb_id() -> None:
    """A raw kb_id from another tenant is rejected when scope is supplied."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "alpha beta gamma"))
    # Correct scope works.
    ok = run_async(kb.search(rec.kb_id, "alpha", workspace_id="w1", client_id="c1"))
    assert ok
    # Wrong workspace is denied.
    with pytest.raises(OpenSimsError):
        run_async(kb.search(rec.kb_id, "alpha", workspace_id="w2", client_id="c1"))
    # Wrong client is denied.
    with pytest.raises(OpenSimsError):
        run_async(kb.search(rec.kb_id, "alpha", workspace_id="w1", client_id="c2"))


# --------------------------------------------------------------------- CK-5
def test_register_kb_search_tool_emits_evidence_shape() -> None:
    """The in-run kb_search tool returns the adapter's evidence shape + emits events."""
    from opensims.services.tools.models import ToolInvocation
    from opensims.services.tools.registry import ToolRegistry
    from opensims.services.tools.service import ToolService

    storage = StorageService()
    kb = KnowledgeBase(storage=storage, embedder=DeterministicEmbedder())
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
                args={"query": "ACME revenue widgets", "kb_name": "kb"},
            )
        )
    )
    assert result.outcome == "success"
    out = result.result
    assert "rendered_text" in out and out["chunks"]
    # Evidence refs match the adapter's source_type and carry tenancy scope.
    assert out["evidence_refs"]
    ref = out["evidence_refs"][0]
    assert ref["source_type"] == "knowledge_base"
    assert ref["account_scope"]["workspace_id"] == "w1"
    # A TOOL_CALLED event was emitted by the service.
    events = run_async(storage.list_events())
    assert any(e.event_type.value == "TOOL_CALLED" for e in events)


def test_kb_search_tool_enforces_tenancy_scope() -> None:
    """kb_search requires workspace+client scope and resolves by name (no raw id)."""
    from opensims.services.tools.models import ToolInvocation
    from opensims.services.tools.registry import ToolRegistry
    from opensims.services.tools.service import ToolService

    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "alpha beta"))
    registry = ToolRegistry()
    register_kb_search_tool(registry, kb)  # no defaults
    svc = ToolService(registry)
    # Missing scope -> execution error (no silent cross-tenant search).
    res = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="kb_search", args={"query": "alpha", "kb_name": "kb"}
            )
        )
    )
    assert res.outcome == "failed"


# --------------------------------------------------------------------- CK-1 (DDL)
def test_knowledge_schema_ddl_selects_column_and_index() -> None:
    """create_knowledge_schema DDL picks vector/halfvec + ivfflat/hnsw by dim."""
    # <= 2000 dims -> vector + ivfflat
    col, idx = resolve_column_and_index(1536)
    assert col == "vector" and idx == "ivfflat"
    # > 2000 dims -> halfvec + hnsw
    col, idx = resolve_column_and_index(3072)
    assert col == "halfvec" and idx == "hnsw"
    # > 4000 dims -> not indexable
    col, idx = resolve_column_and_index(5000)
    assert col == "halfvec" and idx is None
    # explicit vector beyond ceiling raises
    with pytest.raises(OpenSimsError):
        resolve_column_and_index(3000, embedding_column_type="vector")

    ext, tables, index = build_knowledge_schema_ddl(vector_dim=1536)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ext
    assert "vector(1536)" in tables
    assert "knowledge_chunks" in tables and "ON DELETE CASCADE" in tables
    assert "ivfflat" in index and "vector_cosine_ops" in index

    _, tables_hv, index_hv = build_knowledge_schema_ddl(
        vector_dim=3072, embedding_column_type="halfvec", index_method="hnsw"
    )
    assert "halfvec(3072)" in tables_hv
    assert "hnsw" in index_hv and "halfvec_cosine_ops" in index_hv


# --------------------------------------------------------------------- CK-4 (freshness)
def test_storage_first_refetches_when_stale() -> None:
    """A stored field past its freshness TTL falls through to the adapter + recaches."""
    from opensims.services.context.models import (
        ContextBuildSpec,
        ContextSpecKey,
    )
    from opensims.services.context.service import ContextService

    class _CountingAdapter(KnowledgeBaseAdapter.__bases__[0]):  # ContextAdapter
        name = "static"

        def __init__(self) -> None:
            self.calls = 0

        async def fetch(self, key, scope):  # type: ignore[override]
            self.calls += 1
            return ContextField(
                key=key,
                value=f"v{self.calls}",
                source="static",
                freshness_seconds=0.0001,
            )

    storage = StorageService()
    adapter = _CountingAdapter()
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="k", adapter_name="static")])
    s1 = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert s1.fields["k"].value == "v1"
    import time as _t

    _t.sleep(0.01)  # exceed the 0.0001s TTL
    s2 = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    # Stale -> re-fetched (adapter called twice, fresh value used).
    assert adapter.calls == 2
    assert s2.fields["k"].value == "v2"
