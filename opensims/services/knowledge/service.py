"""Knowledge kernel: the in-memory KnowledgeBase + the context-feeding adapter.

``KnowledgeBase`` is a per-client vector store: documents are chunked and embedded
at ingest time (one batched embed call per job) and retrieved by cosine top-k at
decision time, each result carrying its similarity and a context window pulled from
the parent document. ``KnowledgeBaseAdapter`` projects a search into a
:class:`ContextField` so a KB becomes just another context source.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

from opensims.core.errors import OpenSimsError
from opensims.services.context.adapters import ContextAdapter
from opensims.services.context.models import ContextField, EvidenceRef
from opensims.services.knowledge.chunker import SemanticChunker
from opensims.services.knowledge.embedder import (
    DeterministicEmbedder,
    EmbedderProtocol,
    embedder_is_multimodal,
)
from opensims.services.knowledge.models import (
    DocumentInput,
    KnowledgeBaseRecord,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievedChunk,
)
from opensims.services.knowledge.readers import DocumentReaderFactory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opensims.services.knowledge.backend import KnowledgeBackendProtocol
    from opensims.services.storage.service import StorageService


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (0.0 when degenerate)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class KnowledgeBase:
    """An in-memory, per-client vector knowledge base (default offline backend).

    The ``storage`` parameter is accepted for backend parity but the default
    implementation keeps documents and chunks in process-local dicts. Embedding is
    delegated to an :class:`EmbedderProtocol`; the offline default is the
    :class:`DeterministicEmbedder`.

    Pass a ``backend`` (a :class:`KnowledgeBackendProtocol`, e.g. the
    :class:`~opensims.services.knowledge.backend.PgVectorKnowledgeBackend`) to push
    KB/document/chunk persistence and cosine search down into Postgres + pgvector;
    when omitted, the in-memory dicts are the store. The two surfaces are
    behaviourally identical from the caller's perspective.
    """

    def __init__(
        self,
        *,
        storage: StorageService | Any,
        embedder: EmbedderProtocol | None = None,
        chunker: SemanticChunker | None = None,
        reader_factory: DocumentReaderFactory | None = None,
        max_embed_batch: int = 512,
        max_concurrent_embeds: int = 4,
        backend: KnowledgeBackendProtocol | None = None,
    ) -> None:
        """Wire the KB with a storage backend, embedder, chunker, and readers.

        The injected ``embedder`` is asserted to satisfy :class:`EmbedderProtocol`
        at construction time so a missing ``embed_query``/``embed_documents`` is a
        clear early error rather than a deferred failure at search time.
        """
        self._storage = storage
        self._embedder: EmbedderProtocol = embedder or DeterministicEmbedder()
        if not isinstance(self._embedder, EmbedderProtocol):  # runtime_checkable
            raise OpenSimsError(
                "embedder must implement EmbedderProtocol "
                "(async embed_query + embed_documents)."
            )
        self._chunker = chunker or SemanticChunker()
        self._readers = reader_factory or DocumentReaderFactory()
        self.max_embed_batch = max_embed_batch
        self.max_concurrent_embeds = max_concurrent_embeds
        self._backend: KnowledgeBackendProtocol | None = backend
        self._kbs: dict[str, KnowledgeBaseRecord] = {}
        # kb_id -> {document_id: KnowledgeDocument}
        self._documents: dict[str, dict[str, KnowledgeDocument]] = {}
        # kb_id -> {chunk_id: KnowledgeChunk}
        self._chunks: dict[str, dict[str, KnowledgeChunk]] = {}

    # ------------------------------------------------------------------ kb crud
    async def create_kb(
        self,
        *,
        workspace_id: str,
        client_id: str,
        name: str,
        vector_dim: int = 64,
    ) -> KnowledgeBaseRecord:
        """Create a KB unique on ``(workspace_id, client_id, name)``."""
        if self._backend is not None:
            return await self._backend.create_kb(
                workspace_id=workspace_id,
                client_id=client_id,
                name=name,
                vector_dim=vector_dim,
            )
        for kb in self._kbs.values():
            if (
                kb.workspace_id == workspace_id
                and kb.client_id == client_id
                and kb.name == name
            ):
                raise OpenSimsError(
                    f"KnowledgeBase {name!r} already exists for "
                    f"({workspace_id}, {client_id})."
                )
        record = KnowledgeBaseRecord(
            workspace_id=workspace_id,
            client_id=client_id,
            name=name,
            vector_dim=vector_dim,
        )
        self._kbs[record.kb_id] = record
        self._documents[record.kb_id] = {}
        self._chunks[record.kb_id] = {}
        return record

    async def get_kb(self, kb_id: str) -> KnowledgeBaseRecord | None:
        """Return a KB record by id, or None."""
        if self._backend is not None:
            return await self._backend.get_kb(kb_id)
        return self._kbs.get(kb_id)

    async def resolve_kb(
        self, *, workspace_id: str, client_id: str, name: str
    ) -> KnowledgeBaseRecord | None:
        """Resolve a KB by its scope keys ``(workspace_id, client_id, name)``."""
        if self._backend is not None:
            return await self._backend.resolve_kb(
                workspace_id=workspace_id, client_id=client_id, name=name
            )
        for kb in self._kbs.values():
            if (
                kb.workspace_id == workspace_id
                and kb.client_id == client_id
                and kb.name == name
            ):
                return kb
        return None

    async def delete_kb(
        self,
        kb_id: str,
        *,
        workspace_id: str | None = None,
        client_id: str | None = None,
    ) -> bool:
        """Cascade-delete a KB and all its documents and chunks; return success.

        When ``workspace_id``/``client_id`` are supplied they are verified against
        the resolved KB record so a raw ``kb_id`` from another tenant cannot reach
        this KB (see :meth:`_authorize_kb`).
        """
        await self._authorize_kb(kb_id, workspace_id, client_id, missing_ok=True)
        if self._backend is not None:
            return await self._backend.delete_kb(kb_id)
        if kb_id not in self._kbs:
            return False
        del self._kbs[kb_id]
        self._documents.pop(kb_id, None)
        self._chunks.pop(kb_id, None)
        return True

    # ------------------------------------------------------------------ ingest
    async def ingest_text(
        self,
        kb_id: str,
        text: str,
        *,
        title: str | None = None,
        source_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        """Ingest a single raw-text document (one embed call for its chunks)."""
        docs = await self.ingest_documents(
            kb_id,
            [
                DocumentInput(
                    text=text,
                    title=title,
                    source_uri=source_uri,
                    metadata=metadata or {},
                )
            ],
        )
        return docs[0]

    async def ingest_file(
        self,
        kb_id: str,
        path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        """Ingest a single file (read via the reader factory, then embedded)."""
        docs = await self.ingest_documents(
            kb_id, [DocumentInput(file=path, metadata=metadata or {})]
        )
        return docs[0]

    async def ingest_documents(
        self, kb_id: str, docs: list[DocumentInput]
    ) -> list[KnowledgeDocument]:
        """Ingest a batch of documents, folding all chunks into one embed call.

        Each input is read (file) or used directly (text), chunked, and the chunks
        across every input are embedded in a single ``embed_documents`` request so
        the provider sees one job, not one call per document. Inputs that produce
        zero chunks (empty/whitespace-only text) are skipped rather than persisted
        as silent no-op documents, and every produced embedding is validated to be a
        usable (non-empty, non-zero-norm, correct-dim) vector before any chunk is
        written.
        """
        kb = await self._resolve_kb_record(kb_id)
        created_docs: list[KnowledgeDocument] = []
        skipped: list[str] = []
        # Collect (document, [(start,end,text)]) so we can batch the embed call.
        pending: list[tuple[KnowledgeDocument, list[tuple[int, int, str]]]] = []
        all_chunk_texts: list[str] = []

        for item in docs:
            text = self._materialize_text(item)
            # Treat whitespace-only content (incl. from a file the DocumentInput
            # validator can't see into) as empty: it would chunk to a useless
            # whitespace span, so skip the document rather than persist a no-op.
            triples = self._chunker.chunk(text) if text.strip() else []
            if not triples:
                # An empty / whitespace-only document yields no chunks and would be
                # unretrievable — skip it loudly rather than persist a no-op doc.
                skipped.append(item.title or item.source_uri or item.file or "<text>")
                continue
            document = KnowledgeDocument(
                kb_id=kb_id,
                title=item.title,
                source_uri=item.source_uri or item.file,
                text=text,
                metadata=dict(item.metadata),
            )
            pending.append((document, triples))
            all_chunk_texts.extend(t for (_, _, t) in triples)
            created_docs.append(document)

        if not pending:
            if skipped:
                raise OpenSimsError(
                    "ingest_documents produced zero chunks for every input "
                    f"({len(skipped)} empty document(s)); nothing was ingested."
                )
            return []

        embeddings = await self._embed_batched(all_chunk_texts)
        if len(embeddings) != len(all_chunk_texts):
            raise OpenSimsError(
                "Embedder returned a mismatched number of vectors during ingest."
            )
        self._validate_embeddings(kb, embeddings)

        all_chunks: list[KnowledgeChunk] = []
        cursor = 0
        for document, triples in pending:
            for start, end, chunk_text in triples:
                embedding = embeddings[cursor]
                cursor += 1
                chunk = KnowledgeChunk(
                    document_id=document.document_id,
                    kb_id=kb_id,
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    embedding=embedding,
                    chunk_kind="text",
                    metadata=dict(document.metadata),
                )
                all_chunks.append(chunk)

        if self._backend is not None:
            await self._backend.persist_documents(kb_id, created_docs, all_chunks)
        else:
            for document in created_docs:
                self._documents[kb_id][document.document_id] = document
            for chunk in all_chunks:
                self._chunks[kb_id][chunk.chunk_id] = chunk
        return created_docs

    async def ingest_directory(
        self,
        kb_id: str,
        path: str,
        *,
        glob: str = "**/*",
        metadata: dict[str, Any] | None = None,
    ) -> list[KnowledgeDocument]:
        """Ingest every readable file under ``path`` in one batched embed call."""
        from pathlib import Path

        base = Path(path)
        inputs: list[DocumentInput] = []
        for fpath in sorted(base.glob(glob)):
            if not fpath.is_file():
                continue
            if self._readers.reader_for(str(fpath)) is None:
                continue
            inputs.append(DocumentInput(file=str(fpath), metadata=dict(metadata or {})))
        if not inputs:
            return []
        return await self.ingest_documents(kb_id, inputs)

    async def ingest_image(
        self,
        kb_id: str,
        image_uri: str,
        *,
        caption: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        """Ingest an image (requires a multimodal embedder; raises otherwise).

        The embedder must declare multimodal capability (``supports_images``/
        ``is_multimodal`` truthy) — a text-only embedder is rejected so callers never
        get silent garbage image embeddings. The image is embedded via the
        embedder's multimodal document path; the resulting chunk is ``chunk_kind=
        "image"`` carrying ``image_uri`` + ``caption`` (no text span).
        """
        if not embedder_is_multimodal(self._embedder):
            raise OpenSimsError(
                "ingest_image requires a multimodal embedder "
                "(e.g. OpenAIMultimodalEmbeddingModel via the [knowledge]/[providers] "
                "extras); the configured embedder does not support images."
            )
        kb = await self._resolve_kb_record(kb_id)
        meta = dict(metadata or {})
        document = KnowledgeDocument(
            kb_id=kb_id,
            title=meta.get("title") or caption,
            source_uri=image_uri,
            text=None,
            metadata=meta,
        )
        # The multimodal embedder encodes the image (by URI) on the document side.
        embeddings = await self._embedder.embed_documents([image_uri])
        if len(embeddings) != 1:
            raise OpenSimsError(
                "Multimodal embedder returned the wrong number of image vectors."
            )
        self._validate_embeddings(kb, embeddings)
        chunk = KnowledgeChunk(
            document_id=document.document_id,
            kb_id=kb_id,
            text=None,
            start_pos=0,
            end_pos=0,
            embedding=embeddings[0],
            chunk_kind="image",
            image_uri=image_uri,
            caption=caption,
            metadata=meta,
        )
        if self._backend is not None:
            await self._backend.persist_documents(kb_id, [document], [chunk])
        else:
            self._documents[kb_id][document.document_id] = document
            self._chunks[kb_id][chunk.chunk_id] = chunk
        return document

    # ------------------------------------------------------------------ search
    async def search(
        self,
        kb_id: str,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        metadata_filters: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        client_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks by cosine similarity, with a context window.

        ``metadata_filters`` restricts candidates to chunks whose top-level metadata
        matches every pair (exact equality only). The threshold semantics close the
        "orthogonal noise" footgun:

        - ``similarity_threshold=None`` (the default) drops every non-positive
          similarity, so a chunk with zero lexical/semantic overlap is never
          returned just because the default was ``0.0``.
        - An explicit float keeps the documented "drop ``sim < threshold``" rule, so
          callers who deliberately want ``0.0`` (or any cutoff) get exactly that.

        When ``workspace_id``/``client_id`` are supplied they are verified against
        the resolved KB record (tenancy guard) so a raw ``kb_id`` from another tenant
        cannot reach this KB's chunks.
        """
        await self._authorize_kb(kb_id, workspace_id, client_id)

        def _keep(sim: float) -> bool:
            if similarity_threshold is None:
                return sim > 0.0
            return sim >= similarity_threshold

        query_vec = await self._embedder.embed_query(query)

        if self._backend is not None:
            return await self._backend.search(
                kb_id,
                query_vec,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                metadata_filters=metadata_filters,
            )

        chunks = list(self._chunks.get(kb_id, {}).values())
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in chunks:
            if metadata_filters and any(
                chunk.metadata.get(k) != v for k, v in metadata_filters.items()
            ):
                continue
            sim = _cosine(query_vec, chunk.embedding)
            if not _keep(sim):
                continue
            scored.append((sim, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: list[RetrievedChunk] = []
        for sim, chunk in scored[:top_k]:
            document = self._documents.get(kb_id, {}).get(chunk.document_id)
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    similarity=sim,
                    context_window=self._context_window(document, chunk),
                    document_id=chunk.document_id,
                    source_uri=document.source_uri if document else None,
                    chunk_kind=chunk.chunk_kind,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                    },
                )
            )
        return results

    async def delete_document(
        self,
        kb_id: str,
        document_id: str,
        *,
        workspace_id: str | None = None,
        client_id: str | None = None,
    ) -> bool:
        """Delete a document and its chunks from a KB; return success.

        When ``workspace_id``/``client_id`` are supplied they are verified against
        the resolved KB record so a raw ``kb_id`` from another tenant cannot delete
        this KB's documents.
        """
        await self._authorize_kb(kb_id, workspace_id, client_id, missing_ok=True)
        if self._backend is not None:
            return await self._backend.delete_document(kb_id, document_id)
        docs = self._documents.get(kb_id, {})
        if document_id not in docs:
            return False
        del docs[document_id]
        chunks = self._chunks.get(kb_id, {})
        for cid in [c for c, ch in chunks.items() if ch.document_id == document_id]:
            del chunks[cid]
        return True

    # ------------------------------------------------------------------ helpers
    def _require_kb(self, kb_id: str) -> KnowledgeBaseRecord:
        """Return the in-memory KB record or raise a clear error."""
        kb = self._kbs.get(kb_id)
        if kb is None:
            raise OpenSimsError(f"KnowledgeBase {kb_id!r} not found.")
        return kb

    async def _resolve_kb_record(self, kb_id: str) -> KnowledgeBaseRecord:
        """Return the KB record (from the backend or in-memory) or raise."""
        if self._backend is not None:
            kb = await self._backend.get_kb(kb_id)
            if kb is None:
                raise OpenSimsError(f"KnowledgeBase {kb_id!r} not found.")
            return kb
        return self._require_kb(kb_id)

    async def _authorize_kb(
        self,
        kb_id: str,
        workspace_id: str | None,
        client_id: str | None,
        *,
        missing_ok: bool = False,
    ) -> KnowledgeBaseRecord | None:
        """Verify ``kb_id`` belongs to the (workspace, client) caller, if given.

        Tenancy is physical: when the caller declares a scope, a ``kb_id`` that
        resolves to a different workspace/client is rejected with a clear error so a
        raw id can never reach another tenant's data. When no scope is given the
        check is a no-op (the caller asserts trust). ``missing_ok`` lets delete-style
        callers treat an unknown id as "nothing to do" instead of an error.
        """
        if workspace_id is None and client_id is None:
            return None
        try:
            kb = await self._resolve_kb_record(kb_id)
        except OpenSimsError:
            if missing_ok:
                return None
            raise
        if workspace_id is not None and kb.workspace_id != workspace_id:
            raise OpenSimsError(
                f"KnowledgeBase {kb_id!r} is not reachable from workspace "
                f"{workspace_id!r} (cross-tenant access denied)."
            )
        if client_id is not None and kb.client_id != client_id:
            raise OpenSimsError(
                f"KnowledgeBase {kb_id!r} is not reachable from client "
                f"{client_id!r} (cross-tenant access denied)."
            )
        return kb

    def _materialize_text(self, item: DocumentInput) -> str:
        """Return the document text, reading from disk when ``file`` is set."""
        if item.text is not None:
            return item.text
        if item.file is not None:
            return self._readers.read(item.file)
        raise OpenSimsError("DocumentInput has neither text nor file.")

    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in provider-safe batches with bounded concurrency."""
        if not texts:
            return []
        batches = [
            texts[i : i + self.max_embed_batch]
            for i in range(0, len(texts), self.max_embed_batch)
        ]
        semaphore = asyncio.Semaphore(self.max_concurrent_embeds)

        async def _run(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._embedder.embed_documents(batch)

        results = await asyncio.gather(*(_run(b) for b in batches))
        flat: list[list[float]] = []
        for r in results:
            flat.extend(r)
        return flat

    def _validate_embeddings(
        self, kb: KnowledgeBaseRecord, embeddings: list[list[float]]
    ) -> None:
        """Reject empty, zero-norm, or wrong-dimension embeddings before persisting.

        A broken/partial provider call (or a non-multimodal stub) can return ``[]``
        or an all-zero vector; either would store an unretrievable chunk that
        forever scores 0.0, silently corrupting the index. Both are hard errors at
        ingest, alongside the original dimension check.
        """
        for vec in embeddings:
            if not vec:
                raise OpenSimsError(
                    "Embedder returned an empty embedding vector; refusing to ingest "
                    "an unretrievable chunk (broken provider call?)."
                )
            if len(vec) != kb.vector_dim:
                raise OpenSimsError(
                    f"Embedding dim {len(vec)} != KB vector_dim {kb.vector_dim}."
                )
            if not any(v != 0.0 for v in vec):
                raise OpenSimsError(
                    "Embedder returned a zero-norm embedding vector; refusing to "
                    "ingest an unretrievable chunk."
                )

    @staticmethod
    def _context_window(
        document: KnowledgeDocument | None, chunk: KnowledgeChunk
    ) -> str | None:
        """Return a padded window of the parent document around the chunk span."""
        if document is None or document.text is None:
            return chunk.text
        pad = 200
        start = max(0, chunk.start_pos - pad)
        end = min(len(document.text), chunk.end_pos + pad)
        return document.text[start:end]


class KnowledgeBaseAdapter(ContextAdapter):
    """A :class:`ContextAdapter` that routes context lookups through KB search.

    Resolves a KB from ``(workspace_id, client_id|subject_id, kb_name)`` declared in
    the spec metadata, runs :meth:`KnowledgeBase.search`, and returns a
    :class:`ContextField` whose value is ``{chunks, rendered_text}``, confidence is
    the max chunk similarity, and evidence refs point at the source chunks.
    """

    name = "knowledge_base"

    def __init__(self, kb_service: KnowledgeBase) -> None:
        """Wire the adapter to a :class:`KnowledgeBase` service."""
        self._kb = kb_service

    async def fetch(self, key: str, scope: dict) -> ContextField | None:
        """Resolve a KB-backed context field for ``key`` within ``scope``."""
        spec_metadata: dict[str, Any] = dict(scope.get("spec_metadata", {}))
        workspace_id = scope.get("workspace_id") or scope.get("workspace")
        subject_id = scope.get("subject_id")
        client_id = spec_metadata.get("client_id") or subject_id
        kb_name = spec_metadata.get("kb_name")
        if not workspace_id or not client_id or not kb_name:
            return None

        kb = await self._kb.resolve_kb(
            workspace_id=str(workspace_id),
            client_id=str(client_id),
            name=str(kb_name),
        )
        if kb is None:
            return None

        query = spec_metadata.get("query")
        if not query and spec_metadata.get("query_template"):
            query = str(spec_metadata["query_template"]).format(
                subject_id=subject_id, key=key, **spec_metadata
            )
        if not query:
            query = key

        top_k = int(spec_metadata.get("top_k", 5))
        # Only forward an explicit threshold; None preserves the "drop non-positive"
        # default so orthogonal/no-overlap chunks never leak into the snapshot.
        threshold = (
            float(spec_metadata["similarity_threshold"])
            if spec_metadata.get("similarity_threshold") is not None
            else None
        )
        metadata_filters = spec_metadata.get("metadata_filters")
        freshness_seconds = spec_metadata.get("freshness_seconds")

        chunks = await self._kb.search(
            kb.kb_id,
            str(query),
            top_k=top_k,
            similarity_threshold=threshold,
            metadata_filters=metadata_filters,
            workspace_id=str(workspace_id),
            client_id=str(client_id),
        )
        if not chunks:
            return None

        return build_kb_context_field(
            key=key,
            chunks=chunks,
            kb=kb,
            workspace_id=str(workspace_id),
            client_id=str(client_id),
            query=str(query),
            freshness_seconds=(
                float(freshness_seconds) if freshness_seconds is not None else None
            ),
        )


def build_kb_context_field(
    *,
    key: str,
    chunks: list[RetrievedChunk],
    kb: KnowledgeBaseRecord,
    workspace_id: str,
    client_id: str,
    query: str,
    freshness_seconds: float | None = None,
) -> ContextField:
    """Project KB search results into the canonical evidenced :class:`ContextField`.

    Shared by :class:`KnowledgeBaseAdapter` (declarative pre-run enrichment) and the
    in-run ``kb_search`` tool so both retrieval paths emit one identical shape:
    ``value={chunks, rendered_text}``, ``confidence`` = max similarity, and an
    :class:`EvidenceRef` per chunk carrying its account scope and provenance.
    """
    rendered = "\n\n".join(
        f"Source {i + 1} (sim={c.similarity:.2f}): {c.text or c.context_window or ''}"
        for i, c in enumerate(chunks)
    )
    confidence = max((c.similarity for c in chunks), default=0.0)
    evidence_refs = [
        EvidenceRef(
            source_type="knowledge_base",
            source_id=c.document_id,
            row_id=str(c.metadata.get("chunk_id")),
            account_scope={
                "workspace_id": workspace_id,
                "client_id": client_id,
                "kb_id": kb.kb_id,
            },
            metadata={
                "similarity": c.similarity,
                "source_uri": c.source_uri,
                "start_pos": c.metadata.get("start_pos"),
                "end_pos": c.metadata.get("end_pos"),
            },
        )
        for c in chunks
    ]
    return ContextField(
        key=key,
        source="knowledge_base",
        confidence=confidence,
        freshness_seconds=freshness_seconds,
        value={
            "chunks": [c.model_dump(mode="json") for c in chunks],
            "rendered_text": rendered,
        },
        metadata={
            "kb_id": kb.kb_id,
            "kb_name": kb.name,
            "client_id": client_id,
            "query": query,
        },
        evidence_refs=evidence_refs,
    )


__all__ = ["KnowledgeBase", "KnowledgeBaseAdapter", "build_kb_context_field"]
