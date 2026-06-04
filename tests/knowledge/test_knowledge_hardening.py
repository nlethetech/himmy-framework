"""Knowledge kernel hardening: extra coverage for the CK production fixes.

This module complements ``tests/knowledge/test_knowledge.py`` by pinning the
finer-grained contract details the IMPROVEMENTS audit calls out — the edge cases
that turn a "headline" fix into a durable guarantee:

* CK-3  — empty / zero-norm / wrong-dim embeddings are rejected *before* any chunk
          is persisted (partial corruption is impossible; valid vectors still flow).
* CK-7  — explicit positive thresholds drop below-cutoff hits; the adapter forwards
          ``None`` so orthogonal chunks never leak into a snapshot.
* CK-8  — the clamped-overlap chunker never emits a chunk that is mostly overlap;
          construction rejects invalid overlap / min_new_chars.
* CK-5  — ``register_kb_search_tool`` emits TOOL_CALLED + TOOL_COMPLETED, returns the
          adapter's evidence shape, and enforces tenancy when an arg overrides the
          pinned scope toward another tenant.
* CK-10 — the multimodal capability gate routes image ingest and populates the
          image-kind chunk (uri + caption); a text-only embedder is refused.
* CK-11 — empty/whitespace ``DocumentInput`` is rejected; an all-empty batch raises,
          while a mixed batch ingests the real docs and skips the empties.

Gated real-embedder / pgvector paths live in ``test_knowledge_postgres.py`` and
skip offline; everything here runs with the deterministic embedder.
"""

from __future__ import annotations

import pytest

from opensims.core.errors import OpenSimsError
from opensims.services.context.models import ContextField
from opensims.services.knowledge import (
    DeterministicEmbedder,
    DocumentInput,
    KnowledgeBase,
    KnowledgeBaseAdapter,
    SemanticChunker,
    embedder_is_multimodal,
    register_kb_search_tool,
)
from opensims.services.storage.service import StorageService
from tests.conftest import run_async


# --------------------------------------------------------------- shared embedders
class _PartialEmptyEmbedder(DeterministicEmbedder):
    """Returns a valid vector for the first text and an empty one for the rest.

    Exercises CK-3's "reject before persisting" guarantee against a *partial*
    provider failure (the common case): even one bad vector must abort the whole
    ingest so no chunk is ever half-written.
    """

    async def embed_documents(self, texts):  # type: ignore[override]
        good = await super().embed_documents(texts[:1])
        return good + [[] for _ in texts[1:]]


class _WrongDimEmbedder(DeterministicEmbedder):
    """Returns vectors of the wrong dimension (a misconfigured provider/model)."""

    async def embed_documents(self, texts):  # type: ignore[override]
        return [[1.0] * (self.dim + 1) for _ in texts]


class _FakeMultimodalEmbedder(DeterministicEmbedder):
    """A deterministic embedder that *declares* image capability."""

    supports_images = True
    is_multimodal = True


def _kb(embedder=None) -> KnowledgeBase:
    return KnowledgeBase(
        storage=StorageService(), embedder=embedder or DeterministicEmbedder()
    )


# --------------------------------------------------------------------- CK-3
def test_partial_empty_vector_aborts_entire_ingest() -> None:
    """One empty vector in a batch fails the whole ingest — no partial persistence."""
    kb = _kb(_PartialEmptyEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    with pytest.raises(OpenSimsError):
        run_async(
            kb.ingest_documents(
                rec.kb_id,
                [
                    DocumentInput(text="first real doc"),
                    DocumentInput(text="second doc"),
                ],
            )
        )
    # Nothing was written for either document (no half-ingested chunks).
    assert run_async(kb.search(rec.kb_id, "first", similarity_threshold=0.0)) == []


def test_wrong_dimension_vector_is_rejected_at_ingest() -> None:
    """An embedding whose length != kb.vector_dim is a hard error (not stored)."""
    kb = _kb(_WrongDimEmbedder())
    rec = run_async(
        kb.create_kb(workspace_id="w", client_id="c", name="kb", vector_dim=64)
    )
    with pytest.raises(OpenSimsError):
        run_async(kb.ingest_text(rec.kb_id, "content that should not persist"))
    assert run_async(kb.search(rec.kb_id, "content", similarity_threshold=0.0)) == []


def test_valid_embeddings_still_ingest_after_validation() -> None:
    """The validator does not reject legitimate, correct-dim, non-zero vectors."""
    kb = _kb(DeterministicEmbedder(dim=64))
    rec = run_async(
        kb.create_kb(workspace_id="w", client_id="c", name="kb", vector_dim=64)
    )
    docs = run_async(kb.ingest_text(rec.kb_id, "ACME revenue grew on widget sales"))
    assert docs.document_id
    hits = run_async(kb.search(rec.kb_id, "ACME revenue"))
    assert hits and hits[0].similarity > 0.0


# --------------------------------------------------------------------- CK-7
def test_explicit_positive_threshold_drops_below_cutoff() -> None:
    """An explicit threshold keeps only chunks scoring >= it."""
    kb = _kb(DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "alpha beta gamma delta"))
    # A weak/partial-overlap query produces a small positive similarity.
    weak = run_async(
        kb.search(rec.kb_id, "alpha zzz qqq vvv", similarity_threshold=0.0)
    )
    assert weak, "sanity: there is a positive-but-small hit at threshold 0.0"
    sim = weak[0].similarity
    assert 0.0 < sim < 1.0
    # Raising the cutoff above that similarity drops the hit.
    above = run_async(
        kb.search(rec.kb_id, "alpha zzz qqq vvv", similarity_threshold=sim + 0.01)
    )
    assert above == []


def test_adapter_forwards_none_threshold_so_orthogonal_chunks_dont_leak() -> None:
    """With no explicit threshold the adapter returns None -> a no-overlap query empty."""
    kb = _kb(DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "alpha beta gamma"))
    adapter = KnowledgeBaseAdapter(kb)
    # An orthogonal query (no token overlap) -> all sims 0.0 -> dropped -> no field.
    scope = {
        "workspace_id": "w1",
        "subject_id": "c1",
        "spec_metadata": {"kb_name": "kb", "query": "zzz qqq vvv"},
    }
    assert run_async(adapter.fetch("knowledge", scope)) is None
    # A relevant query still produces a field.
    scope_ok = {
        "workspace_id": "w1",
        "subject_id": "c1",
        "spec_metadata": {"kb_name": "kb", "query": "alpha beta"},
    }
    field = run_async(adapter.fetch("knowledge", scope_ok))
    assert isinstance(field, ContextField) and field.confidence > 0.0


# --------------------------------------------------------------------- CK-8
def test_min_new_chars_bounds_overlap_so_no_near_duplicate_microchunks() -> None:
    """With a meaningful min_new_chars, every chunk advances by at least that much.

    This is the durable CK-8 guarantee: even with a pathologically large ``overlap``
    (9 of 10 chars), setting ``min_new_chars`` forbids the near-duplicate
    overlap-dominated micro-chunks the audit flagged — every emitted chunk after the
    first contributes at least ``min_new_chars`` brand-new characters.
    """
    chunker = SemanticChunker(max_chars=10, overlap=9, min_new_chars=3)
    text = "abcdefghij klmnopqrst uvwxyz0123 456789ABCD"
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    prev_end = -1
    for start, end, body in chunks:
        assert text[start:end] == body
        if prev_end >= 0:
            # Each chunk adds at least min_new_chars of brand-new content -> the
            # chunk is never dominated by carry-over overlap.
            assert (end - prev_end) >= 3, (start, end, prev_end)
        prev_end = end
    # Full coverage to the document end.
    assert chunks[-1][1] == len(text)


def test_overlap_is_clamped_to_half_the_cut_chunk_length() -> None:
    """Even with default min_new_chars, the realized overlap never exceeds half a cut chunk.

    The clamp ``effective_overlap = min(overlap, (end-start)//2)`` means a short
    boundary-cut chunk cannot hand almost its entire span to the next window; the
    new start is always at least halfway through the chunk that produced it.
    """
    chunker = SemanticChunker(max_chars=10, overlap=9, min_new_chars=1)
    text = "abcdefghij klmnopqrst uvwxyz0123 456789ABCD"
    chunks = chunker.chunk(text)
    # For consecutive emitted chunks, the next start advanced by >= half the prior
    # chunk's length (the overlap carried over is at most half of it).
    for (s0, e0, _), (s1, _e1, _) in zip(chunks, chunks[1:], strict=False):
        prior_len = e0 - s0
        carried_overlap = e0 - s1  # how much of the prior chunk the next reuses
        assert carried_overlap <= prior_len // 2, (s0, e0, s1)


def test_tail_is_folded_not_dropped_when_too_small_to_stand_alone() -> None:
    """A tail smaller than min_new_chars folds into the prior chunk (no lost text)."""
    chunker = SemanticChunker(max_chars=5, overlap=2, min_new_chars=3)
    text = "abcdefgh"  # final window would add < 3 new chars
    chunks = chunker.chunk(text)
    assert chunks[-1][1] == len(text), "document fully covered to the end"
    # No micro-chunk that adds fewer than min_new_chars over its predecessor.
    prev_end = -1
    for _, end, _ in chunks:
        if prev_end >= 0:
            assert (end - prev_end) >= 3
        prev_end = end


def test_chunker_rejects_invalid_overlap_and_min_new_chars() -> None:
    """Construction-time validation guards the chunker's invariants."""
    with pytest.raises(ValueError):
        SemanticChunker(max_chars=10, overlap=10)  # overlap must be < max_chars
    with pytest.raises(ValueError):
        SemanticChunker(max_chars=10, overlap=-1)  # overlap must be >= 0
    with pytest.raises(ValueError):
        SemanticChunker(max_chars=0)  # max_chars must be positive
    with pytest.raises(ValueError):
        SemanticChunker(max_chars=10, min_new_chars=0)  # must be >= 1


# --------------------------------------------------------------------- CK-5
def test_kb_search_tool_emits_called_and_completed_events() -> None:
    """The in-run tool emits both TOOL_CALLED and TOOL_COMPLETED around the handler."""
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
    types = {e.event_type.value for e in run_async(storage.list_events())}
    assert "TOOL_CALLED" in types
    assert "TOOL_COMPLETED" in types


def test_kb_search_tool_evidence_matches_adapter_shape() -> None:
    """The tool's evidence_refs/account_scope mirror the KnowledgeBaseAdapter exactly."""
    from opensims.services.tools.models import ToolInvocation
    from opensims.services.tools.registry import ToolRegistry
    from opensims.services.tools.service import ToolService

    storage = StorageService()
    kb = KnowledgeBase(storage=storage, embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "ACME revenue rose on widget demand."))

    # The declarative adapter result for the same query/scope.
    adapter = KnowledgeBaseAdapter(kb)
    adapter_field = run_async(
        adapter.fetch(
            "kb",
            {
                "workspace_id": "w1",
                "subject_id": "c1",
                "spec_metadata": {"kb_name": "kb", "query": "ACME revenue"},
            },
        )
    )
    assert adapter_field is not None
    adapter_ref = adapter_field.evidence_refs[0]

    # The in-run tool result for the same query/scope.
    registry = ToolRegistry()
    register_kb_search_tool(
        registry, kb, default_workspace_id="w1", default_client_id="c1"
    )
    svc = ToolService(registry, event_sink=storage)
    out = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="kb_search",
                args={"query": "ACME revenue", "kb_name": "kb"},
            )
        )
    ).result
    tool_ref = out["evidence_refs"][0]

    # Same source_type and account scope shape from both retrieval paths.
    assert tool_ref["source_type"] == adapter_ref.source_type == "knowledge_base"
    assert tool_ref["account_scope"] == adapter_ref.account_scope
    assert out["confidence"] == adapter_field.confidence


def test_kb_search_tool_blocks_arg_override_to_other_tenant() -> None:
    """A workspace_id arg pointing at another tenant cannot reach this KB's chunks."""
    from opensims.services.tools.models import ToolInvocation
    from opensims.services.tools.registry import ToolRegistry
    from opensims.services.tools.service import ToolService

    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "alpha beta gamma"))

    registry = ToolRegistry()
    register_kb_search_tool(
        registry, kb, default_workspace_id="w1", default_client_id="c1"
    )
    svc = ToolService(registry)
    # Override the pinned scope toward a different workspace -> resolve fails (no KB
    # by that name in 'w2'); the tool surfaces a failed execution, not a leak.
    res = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="kb_search",
                args={"query": "alpha", "kb_name": "kb", "workspace_id": "w2"},
            )
        )
    )
    assert res.outcome == "failed"


def test_kb_search_tool_uses_pinned_defaults_when_args_omit_scope() -> None:
    """Pinned default workspace/client are used when the agent omits them."""
    from opensims.services.tools.models import ToolInvocation
    from opensims.services.tools.registry import ToolRegistry
    from opensims.services.tools.service import ToolService

    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w1", client_id="c1", name="kb"))
    run_async(kb.ingest_text(rec.kb_id, "alpha beta gamma"))
    registry = ToolRegistry()
    register_kb_search_tool(
        registry, kb, default_workspace_id="w1", default_client_id="c1"
    )
    svc = ToolService(registry)
    res = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="kb_search", args={"query": "alpha", "kb_name": "kb"}
            )
        )
    )
    assert res.outcome == "success"
    assert res.result["chunks"]


# --------------------------------------------------------------------- CK-10
def test_multimodal_capability_gate_routes_image_ingest() -> None:
    """A multimodal-declaring embedder ingests an image-kind chunk with uri+caption."""
    assert embedder_is_multimodal(_FakeMultimodalEmbedder()) is True
    assert embedder_is_multimodal(DeterministicEmbedder()) is False

    kb = KnowledgeBase(storage=StorageService(), embedder=_FakeMultimodalEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    doc = run_async(
        kb.ingest_image(rec.kb_id, "image://logo.png", caption="the ACME logo")
    )
    assert doc.text is None and doc.source_uri == "image://logo.png"
    hits = run_async(kb.search(rec.kb_id, "image://logo.png"))
    assert hits, "the image chunk is retrievable"
    hit = hits[0]
    assert hit.chunk_kind == "image"


def test_text_only_embedder_refuses_image_ingest() -> None:
    """A text-only embedder cannot ingest images (no silent garbage embedding)."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    with pytest.raises(OpenSimsError):
        run_async(kb.ingest_image(rec.kb_id, "image://logo.png", caption="logo"))


# --------------------------------------------------------------------- CK-11
def test_document_input_rejects_empty_and_whitespace_sources() -> None:
    """Empty/whitespace text and empty file paths are invalid at construction."""
    for bad in ("", "   ", "\n\t "):
        with pytest.raises(ValueError):
            DocumentInput(text=bad)
    with pytest.raises(ValueError):
        DocumentInput(file="")
    with pytest.raises(ValueError):
        DocumentInput(file="   ")
    with pytest.raises(ValueError):
        DocumentInput()  # neither
    with pytest.raises(ValueError):
        DocumentInput(text="x", file="y.txt")  # both
    # Exactly one non-empty source is accepted.
    assert DocumentInput(text="real").text == "real"
    assert DocumentInput(file="doc.txt").file == "doc.txt"


def test_ingest_batch_skips_whitespace_file_and_keeps_real_docs() -> None:
    """A file whose contents are whitespace-only is skipped; real docs still ingest.

    The ``DocumentInput`` validator can't see *into* a file, so the skip happens at
    ingest time when the materialized text chunks to nothing.
    """
    import tempfile
    from pathlib import Path

    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))

    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "blank.txt"
        empty.write_text("   \n\t  ", encoding="utf-8")
        real = Path(d) / "real.txt"
        real.write_text("ACME revenue grew on widget sales", encoding="utf-8")
        created = run_async(
            kb.ingest_documents(
                rec.kb_id,
                [DocumentInput(file=str(empty)), DocumentInput(file=str(real))],
            )
        )
    # Only the real document was persisted (the whitespace file was skipped).
    assert len(created) == 1
    assert run_async(kb.search(rec.kb_id, "ACME revenue"))


def test_all_empty_batch_raises_rather_than_silent_no_op() -> None:
    """A batch of only zero-chunk files raises instead of silently ingesting nothing."""
    import tempfile
    from pathlib import Path

    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    rec = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb"))
    with tempfile.TemporaryDirectory() as d:
        blank = Path(d) / "blank.txt"
        blank.write_text("    \n   ", encoding="utf-8")
        with pytest.raises(OpenSimsError):
            run_async(kb.ingest_documents(rec.kb_id, [DocumentInput(file=str(blank))]))
