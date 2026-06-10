"""Embedder versioning on KB chunks: fingerprint stamping + mismatch detection.

Chunks used to store vectors with no record of WHICH embedder produced them, so
swapping embedding models made retrieval silently score garbage against
incompatible old vectors. These tests pin the fix:

* every chunk (text and image, in-memory and backend) is stamped at ingest with
  the active embedder's fingerprint (class + model + dim, hashed);
* searching with a *different* embedder logs a clear warning once per
  ``(kb, fingerprint)`` — and raises with ``strict_embedder_check=True``;
* searching with the *same* embedder stays silent;
* legacy fingerprint-less chunks are treated as unknown: warn-only, never an
  error, even in strict mode.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from himmy.core.errors import HimmyError
from himmy.services.knowledge import (
    EMBEDDER_FINGERPRINT_KEY,
    DeterministicEmbedder,
    KnowledgeBase,
    KnowledgeBaseRecord,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievedChunk,
    embedder_fingerprint,
)
from himmy.services.knowledge.retrieval.config import RetrievalConfig
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


# --------------------------------------------------------------------- helpers
class _SwappedEmbedder(DeterministicEmbedder):
    """Identical vectors to the default, but a different identity (class name).

    Same dimension and same deterministic algorithm, so searches still score and
    return chunks — isolating the *fingerprint* mismatch from a dim mismatch.
    """


class _FakeMultimodalEmbedder(DeterministicEmbedder):
    """A deterministic embedder that declares image capability."""

    supports_images = True
    is_multimodal = True


class _FakeBackend:
    """A minimal in-test KnowledgeBackendProtocol implementation.

    Stores documents/chunks in dicts and returns every chunk from ``search`` with
    its persisted metadata, mirroring how the pgvector backend round-trips the
    chunk metadata JSONB (where the fingerprint travels).
    """

    def __init__(self) -> None:
        self.kbs: dict[str, KnowledgeBaseRecord] = {}
        self.documents: dict[str, KnowledgeDocument] = {}
        self.chunks: dict[str, KnowledgeChunk] = {}

    async def create_kb(
        self, *, workspace_id: str, client_id: str, name: str, vector_dim: int
    ) -> KnowledgeBaseRecord:
        record = KnowledgeBaseRecord(
            workspace_id=workspace_id,
            client_id=client_id,
            name=name,
            vector_dim=vector_dim,
        )
        self.kbs[record.kb_id] = record
        return record

    async def get_kb(self, kb_id: str) -> KnowledgeBaseRecord | None:
        return self.kbs.get(kb_id)

    async def resolve_kb(
        self, *, workspace_id: str, client_id: str, name: str
    ) -> KnowledgeBaseRecord | None:
        for kb in self.kbs.values():
            if (kb.workspace_id, kb.client_id, kb.name) == (
                workspace_id,
                client_id,
                name,
            ):
                return kb
        return None

    async def delete_kb(self, kb_id: str) -> bool:
        return self.kbs.pop(kb_id, None) is not None

    async def persist_documents(
        self,
        kb_id: str,
        documents: list[KnowledgeDocument],
        chunks: list[KnowledgeChunk],
    ) -> None:
        for doc in documents:
            self.documents[doc.document_id] = doc
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        return self.documents.pop(document_id, None) is not None

    async def search(
        self,
        kb_id: str,
        query_vec: list[float],
        *,
        top_k: int,
        similarity_threshold: float | None,
        metadata_filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        results = [
            RetrievedChunk(
                text=chunk.text,
                similarity=0.5,
                document_id=chunk.document_id,
                metadata={**chunk.metadata, "chunk_id": chunk.chunk_id},
            )
            for chunk in self.chunks.values()
            if chunk.kb_id == kb_id
        ]
        return results[:top_k]


def _kb(embedder: Any = None, **kwargs: Any) -> KnowledgeBase:
    return KnowledgeBase(
        storage=StorageService(),
        embedder=embedder or DeterministicEmbedder(),
        **kwargs,
    )


def _ingested_kb_id(kb: KnowledgeBase) -> str:
    record = run_async(kb.create_kb(workspace_id="ws", client_id="cl", name="docs"))
    run_async(
        kb.ingest_text(record.kb_id, "the quick brown fox jumps over the lazy dog")
    )
    return record.kb_id


def _fingerprint_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "fingerprint" in r.getMessage()]


# ------------------------------------------------------ fingerprint computation
def test_fingerprint_varies_with_class_model_and_dim() -> None:
    """The fingerprint hashes the salient identity: class, model id, dimension."""
    base = embedder_fingerprint(DeterministicEmbedder(dim=64))
    assert base == embedder_fingerprint(DeterministicEmbedder(dim=64))
    assert base != embedder_fingerprint(DeterministicEmbedder(dim=128))
    assert base != embedder_fingerprint(_SwappedEmbedder(dim=64))

    class _Modeled:
        model = "bge-m3"
        dim = 64

    class _OtherModel:
        model = "nomic-embed-text"
        dim = 64

    assert embedder_fingerprint(_Modeled()) != embedder_fingerprint(_OtherModel())


def test_fingerprint_explicit_attribute_overrides_inference() -> None:
    """A non-empty string ``fingerprint`` attribute is used verbatim."""

    class _Pinned:
        fingerprint = "my-custom-identity"

    assert embedder_fingerprint(_Pinned()) == "my-custom-identity"


# ------------------------------------------------------------ stamping at ingest
def test_text_chunks_are_stamped_with_embedder_fingerprint() -> None:
    """Every ingested text chunk records which embedder produced its vector."""
    embedder = DeterministicEmbedder()
    kb = _kb(embedder)
    kb_id = _ingested_kb_id(kb)
    chunks = list(kb._chunks[kb_id].values())
    assert chunks
    expected = embedder_fingerprint(embedder)
    assert all(c.metadata[EMBEDDER_FINGERPRINT_KEY] == expected for c in chunks)


def test_image_chunks_are_stamped_with_embedder_fingerprint() -> None:
    """The image-ingest path stamps the fingerprint too."""
    embedder = _FakeMultimodalEmbedder()
    kb = _kb(embedder)
    record = run_async(kb.create_kb(workspace_id="ws", client_id="cl", name="imgs"))
    run_async(kb.ingest_image(record.kb_id, "image://photo.png", caption="a photo"))
    chunks = list(kb._chunks[record.kb_id].values())
    assert len(chunks) == 1
    assert chunks[0].metadata[EMBEDDER_FINGERPRINT_KEY] == embedder_fingerprint(
        embedder
    )


def test_backend_persisted_chunks_carry_fingerprint() -> None:
    """Chunks pushed through a backend carry the fingerprint in their metadata."""
    backend = _FakeBackend()
    embedder = DeterministicEmbedder()
    kb = _kb(embedder, backend=backend)
    record = run_async(kb.create_kb(workspace_id="ws", client_id="cl", name="docs"))
    run_async(kb.ingest_text(record.kb_id, "persisted through the backend"))
    assert backend.chunks
    expected = embedder_fingerprint(embedder)
    assert all(
        c.metadata[EMBEDDER_FINGERPRINT_KEY] == expected
        for c in backend.chunks.values()
    )


# -------------------------------------------------------- search-time detection
def test_same_embedder_search_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    """Searching with the embedder that ingested the chunks emits no warning."""
    kb = _kb()
    kb_id = _ingested_kb_id(kb)
    with caplog.at_level(logging.WARNING, logger="himmy.knowledge"):
        results = run_async(kb.search(kb_id, "quick fox"))
    assert results
    assert not _fingerprint_warnings(caplog)


def test_mismatched_embedder_warns_once_and_still_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ingest with embedder A, search with embedder B: one clear warning, no break."""
    kb = _kb(DeterministicEmbedder())
    kb_id = _ingested_kb_id(kb)
    kb._embedder = _SwappedEmbedder()  # same vectors, different identity
    with caplog.at_level(logging.WARNING, logger="himmy.knowledge"):
        first = run_async(kb.search(kb_id, "quick fox"))
        second = run_async(kb.search(kb_id, "lazy dog"))
    assert first and second  # default mode degrades gracefully, never breaks
    assert len(_fingerprint_warnings(caplog)) == 1  # deduped per (kb, fingerprint)


def test_strict_mode_raises_on_mismatch() -> None:
    """``strict_embedder_check=True`` turns the mismatch into a hard error."""
    kb = _kb(DeterministicEmbedder(), strict_embedder_check=True)
    kb_id = _ingested_kb_id(kb)
    kb._embedder = _SwappedEmbedder()
    with pytest.raises(HimmyError, match="different embedder"):
        run_async(kb.search(kb_id, "quick fox"))


def test_legacy_chunks_without_fingerprint_warn_only_even_in_strict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pre-fingerprint chunks are unknown: warn once, never raise, keep working."""
    kb = _kb(DeterministicEmbedder(), strict_embedder_check=True)
    kb_id = _ingested_kb_id(kb)
    for chunk in kb._chunks[kb_id].values():  # simulate a pre-change index
        chunk.metadata.pop(EMBEDDER_FINGERPRINT_KEY, None)
    with caplog.at_level(logging.WARNING, logger="himmy.knowledge"):
        first = run_async(kb.search(kb_id, "quick fox"))
        second = run_async(kb.search(kb_id, "lazy dog"))
    assert first and second
    warnings = _fingerprint_warnings(caplog)
    assert len(warnings) == 1
    assert "no embedder fingerprint" in warnings[0].getMessage()


def test_hybrid_mode_detects_mismatch_too(caplog: pytest.LogCaptureFixture) -> None:
    """The hybrid pipeline shares the same in-memory fingerprint check."""
    kb = _kb(DeterministicEmbedder(), retrieval=RetrievalConfig(mode="hybrid"))
    kb_id = _ingested_kb_id(kb)
    kb._embedder = _SwappedEmbedder()
    with caplog.at_level(logging.WARNING, logger="himmy.knowledge"):
        run_async(kb.search(kb_id, "quick fox"))
    assert len(_fingerprint_warnings(caplog)) == 1


def test_backend_search_detects_mismatch_from_result_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Backend-held chunks are checked from the fingerprints on returned results."""
    backend = _FakeBackend()
    kb = _kb(DeterministicEmbedder(), backend=backend)
    record = run_async(kb.create_kb(workspace_id="ws", client_id="cl", name="docs"))
    run_async(kb.ingest_text(record.kb_id, "persisted through the backend"))
    kb._embedder = _SwappedEmbedder()
    with caplog.at_level(logging.WARNING, logger="himmy.knowledge"):
        results = run_async(kb.search(record.kb_id, "backend"))
    assert results
    assert len(_fingerprint_warnings(caplog)) == 1

    strict = _kb(_SwappedEmbedder(), backend=backend, strict_embedder_check=True)
    with pytest.raises(HimmyError, match="different embedder"):
        run_async(strict.search(record.kb_id, "backend"))
