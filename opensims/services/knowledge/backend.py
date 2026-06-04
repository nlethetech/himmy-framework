"""Knowledge kernel: the Postgres + pgvector backend (requires [postgres,knowledge]).

This module is import-safe without ``asyncpg`` / ``pgvector`` / a running database:
every import is lazy and guarded, and the DDL builders are plain functions that
return strings. :class:`PgVectorKnowledgeBackend` wraps an ``asyncpg`` pool and
implements the documented production retrieval path:

- :func:`build_knowledge_schema_ddl` / :meth:`create_knowledge_schema` pick the
  embedding column type (``vector(N)`` / ``halfvec(N)``) and index method
  (``ivfflat`` / ``hnsw``) from ``vector_dim`` (pgvector's 2000/4000-dim ceilings),
  with explicit overrides + ``ivfflat_lists`` / ``hnsw_m`` / ``hnsw_ef_construction``
  tuning knobs.
- the actual column type is auto-detected from ``pg_attribute`` after each call so a
  reconnect against a pre-existing schema does not silently miscast queries.
- ``knowledge_bases`` / ``knowledge_documents`` / ``knowledge_chunks`` persistence
  with cascade-delete and cosine top-k search (``<=>`` distance -> ``1 - distance``).

Its integration tests are docker-gated (a real ``pgvector/pgvector`` container) and
skip when ``OPENSIMS_TEST_POSTGRES_DSN`` is unset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from opensims.core.errors import OpenSimsError
from opensims.services.knowledge.models import (
    KnowledgeBaseRecord,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievedChunk,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: pgvector indexable dimension ceilings (see pgvector docs).
VECTOR_MAX_DIM = 2000
HALFVEC_MAX_DIM = 4000


@runtime_checkable
class KnowledgeBackendProtocol(Protocol):
    """The persistence + retrieval contract the KnowledgeBase delegates to.

    A backend owns KB/document/chunk storage and cosine search; the in-memory
    KnowledgeBase uses its own dicts when no backend is injected.
    """

    async def create_kb(
        self, *, workspace_id: str, client_id: str, name: str, vector_dim: int
    ) -> KnowledgeBaseRecord:
        """Create a KB unique on ``(workspace_id, client_id, name)``."""
        ...

    async def get_kb(self, kb_id: str) -> KnowledgeBaseRecord | None:
        """Return a KB record by id, or None."""
        ...

    async def resolve_kb(
        self, *, workspace_id: str, client_id: str, name: str
    ) -> KnowledgeBaseRecord | None:
        """Resolve a KB by its scope keys."""
        ...

    async def delete_kb(self, kb_id: str) -> bool:
        """Cascade-delete a KB and its documents/chunks."""
        ...

    async def persist_documents(
        self,
        kb_id: str,
        documents: list[KnowledgeDocument],
        chunks: list[KnowledgeChunk],
    ) -> None:
        """Persist a batch of documents and their embedded chunks."""
        ...

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """Delete a document and its chunks; return success."""
        ...

    async def search(
        self,
        kb_id: str,
        query_vec: list[float],
        *,
        top_k: int,
        similarity_threshold: float | None,
        metadata_filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Cosine top-k search over a KB's chunks."""
        ...


def resolve_column_and_index(
    vector_dim: int,
    *,
    embedding_column_type: str | None = None,
    index_method: str | None = None,
) -> tuple[str, str | None]:
    """Pick the embedding column type and index method for ``vector_dim``.

    The defaults follow pgvector's indexable ceilings: ``vector(N)`` + ``ivfflat``
    at or below 2000 dims, ``halfvec(N)`` + ``hnsw`` above (and up to 4000). Beyond
    4000 dims nothing is indexable, so the index method is ``None`` (sequential
    scan). Explicit ``embedding_column_type`` / ``index_method`` override the
    inference; an out-of-range explicit pairing raises a clear error.
    """
    if vector_dim <= 0:
        raise OpenSimsError("vector_dim must be a positive integer.")

    col = (embedding_column_type or "").lower() or None
    idx = (index_method or "").lower() or None

    if col is None:
        col = "vector" if vector_dim <= VECTOR_MAX_DIM else "halfvec"
    if col not in ("vector", "halfvec"):
        raise OpenSimsError(
            f"embedding_column_type must be 'vector' or 'halfvec', got {col!r}."
        )

    # Enforce pgvector's dimension ceilings for the chosen column type.
    if col == "vector" and vector_dim > VECTOR_MAX_DIM:
        raise OpenSimsError(
            f"vector({vector_dim}) exceeds the {VECTOR_MAX_DIM}-dim ceiling; "
            "use embedding_column_type='halfvec'."
        )
    if col == "halfvec" and vector_dim > HALFVEC_MAX_DIM:
        # Above 4000 dims nothing is indexable — fall back to a sequential scan.
        return col, None

    if idx is None:
        idx = "ivfflat" if col == "vector" else "hnsw"
    if idx not in ("ivfflat", "hnsw"):
        raise OpenSimsError(f"index_method must be 'ivfflat' or 'hnsw', got {idx!r}.")
    return col, idx


def build_knowledge_schema_ddl(
    *,
    vector_dim: int,
    embedding_column_type: str | None = None,
    index_method: str | None = None,
    ivfflat_lists: int = 100,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
) -> tuple[str, str, str]:
    """Return ``(extension_sql, tables_sql, index_sql)`` for the knowledge schema.

    The column type is sized to ``vector_dim`` and the index method is tuned by the
    ``ivfflat_lists`` / ``hnsw_*`` knobs. Returned as separate statements so callers
    can run them in order (extension, tables, then the embedding index).
    """
    col, idx = resolve_column_and_index(
        vector_dim,
        embedding_column_type=embedding_column_type,
        index_method=index_method,
    )
    col_type = f"{col}({vector_dim})"

    extension_sql = "CREATE EXTENSION IF NOT EXISTS vector;"

    tables_sql = f"""
CREATE TABLE IF NOT EXISTS knowledge_bases (
    kb_id        TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    client_id    TEXT NOT NULL,
    name         TEXT NOT NULL,
    vector_dim   INTEGER NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    UNIQUE (workspace_id, client_id, name)
);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id  TEXT PRIMARY KEY,
    kb_id        TEXT NOT NULL REFERENCES knowledge_bases (kb_id) ON DELETE CASCADE,
    title        TEXT,
    source_uri   TEXT,
    text         TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb
);
CREATE INDEX IF NOT EXISTS knowledge_documents_kb_id_idx
    ON knowledge_documents (kb_id);
-- Backs the replace-by-source upsert in persist_documents.
CREATE INDEX IF NOT EXISTS knowledge_documents_source_idx
    ON knowledge_documents (kb_id, source_uri);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT NOT NULL
        REFERENCES knowledge_documents (document_id) ON DELETE CASCADE,
    kb_id       TEXT NOT NULL REFERENCES knowledge_bases (kb_id) ON DELETE CASCADE,
    text        TEXT,
    start_pos   INTEGER NOT NULL DEFAULT 0,
    end_pos     INTEGER NOT NULL DEFAULT 0,
    embedding   {col_type},
    chunk_kind  TEXT NOT NULL DEFAULT 'text',
    image_uri   TEXT,
    caption     TEXT,
    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb
);
CREATE INDEX IF NOT EXISTS knowledge_chunks_kb_id_idx
    ON knowledge_chunks (kb_id);
CREATE INDEX IF NOT EXISTS knowledge_chunks_document_id_idx
    ON knowledge_chunks (document_id);
""".strip()

    if idx == "ivfflat":
        ops = "vector_cosine_ops" if col == "vector" else "halfvec_cosine_ops"
        index_sql = (
            "CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx "
            f"ON knowledge_chunks USING ivfflat (embedding {ops}) "
            f"WITH (lists = {int(ivfflat_lists)});"
        )
    elif idx == "hnsw":
        ops = "vector_cosine_ops" if col == "vector" else "halfvec_cosine_ops"
        index_sql = (
            "CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx "
            f"ON knowledge_chunks USING hnsw (embedding {ops}) "
            f"WITH (m = {int(hnsw_m)}, ef_construction = {int(hnsw_ef_construction)});"
        )
    else:
        # No indexable method (dims > 4000) — sequential scan only.
        index_sql = ""

    return extension_sql, tables_sql, index_sql


def _vector_literal(vec: list[float]) -> str:
    """Render a Python float list as a pgvector text literal ``[a,b,c]``."""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


class PgVectorKnowledgeBackend:
    """Postgres + pgvector implementation of :class:`KnowledgeBackendProtocol`.

    Construct with a live ``asyncpg`` pool (e.g.
    ``PostgresStorageService.connect(dsn)._pool``), call
    :meth:`create_knowledge_schema` once, then wire it into a
    :class:`~opensims.services.knowledge.service.KnowledgeBase` via
    ``KnowledgeBase(..., backend=backend)``.
    """

    def __init__(self, pool: Any) -> None:
        """Wrap a live asyncpg pool. The pool must already be connected."""
        if pool is None:
            raise OpenSimsError(
                "PgVectorKnowledgeBackend requires a live asyncpg pool."
            )
        self._pool = pool
        #: Detected ``(column_type, vector_dim)`` after create_knowledge_schema.
        self._column_type: str | None = None
        self._vector_dim: int | None = None

    # -------------------------------------------------------------- schema/DDL
    async def create_knowledge_schema(
        self,
        *,
        vector_dim: int,
        embedding_column_type: str | None = None,
        index_method: str | None = None,
        ivfflat_lists: int = 100,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
    ) -> None:
        """Create the pgvector knowledge schema, then auto-detect the column type.

        Runs ``CREATE EXTENSION``, the table DDL, and the embedding index in order,
        then reads ``pg_attribute`` so subsequent queries cast vectors to the column
        type the database actually committed to (guarding reconnects against an
        existing schema).
        """
        extension_sql, tables_sql, index_sql = build_knowledge_schema_ddl(
            vector_dim=vector_dim,
            embedding_column_type=embedding_column_type,
            index_method=index_method,
            ivfflat_lists=ivfflat_lists,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(extension_sql)
            await conn.execute(tables_sql)
            if index_sql:
                await conn.execute(index_sql)
        await self._detect_column_type()

    async def _detect_column_type(self) -> None:
        """Auto-detect the embedding column's type + dimension from pg_attribute."""
        sql = """
        SELECT format_type(a.atttypid, a.atttypmod) AS coltype
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'knowledge_chunks'
          AND a.attname = 'embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql)
        if row is None:
            return
        coltype = str(row["coltype"])  # e.g. "vector(1536)" or "halfvec(3072)"
        base = coltype.split("(", 1)[0].strip().lower()
        self._column_type = base
        if "(" in coltype:
            try:
                self._vector_dim = int(coltype.split("(", 1)[1].rstrip(")"))
            except ValueError:  # pragma: no cover - defensive
                self._vector_dim = None

    def _cast(self) -> str:
        """Return the SQL cast for a parameter to the detected embedding type."""
        col = self._column_type or "vector"
        return f"::{col}"

    # ---------------------------------------------------------------- kb crud
    async def create_kb(
        self, *, workspace_id: str, client_id: str, name: str, vector_dim: int
    ) -> KnowledgeBaseRecord:
        """Insert a KB row; raise on a duplicate ``(workspace, client, name)``."""
        record = KnowledgeBaseRecord(
            workspace_id=workspace_id,
            client_id=client_id,
            name=name,
            vector_dim=vector_dim,
        )
        sql = """
        INSERT INTO knowledge_bases
            (kb_id, workspace_id, client_id, name, vector_dim, metadata)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (workspace_id, client_id, name) DO NOTHING
        RETURNING kb_id
        """
        async with self._pool.acquire() as conn:
            inserted = await conn.fetchval(
                sql,
                record.kb_id,
                workspace_id,
                client_id,
                name,
                vector_dim,
                record.metadata,
            )
        if inserted is None:
            raise OpenSimsError(
                f"KnowledgeBase {name!r} already exists for "
                f"({workspace_id}, {client_id})."
            )
        return record

    async def get_kb(self, kb_id: str) -> KnowledgeBaseRecord | None:
        """Load a KB by id."""
        sql = "SELECT * FROM knowledge_bases WHERE kb_id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, kb_id)
        return self._row_to_kb(row) if row is not None else None

    async def resolve_kb(
        self, *, workspace_id: str, client_id: str, name: str
    ) -> KnowledgeBaseRecord | None:
        """Resolve a KB by its scope keys."""
        sql = (
            "SELECT * FROM knowledge_bases "
            "WHERE workspace_id = $1 AND client_id = $2 AND name = $3"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, workspace_id, client_id, name)
        return self._row_to_kb(row) if row is not None else None

    async def delete_kb(self, kb_id: str) -> bool:
        """Cascade-delete a KB (its documents/chunks go via ON DELETE CASCADE)."""
        sql = "DELETE FROM knowledge_bases WHERE kb_id = $1"
        async with self._pool.acquire() as conn:
            status = await conn.execute(sql, kb_id)
        return status.endswith(" 1")

    # -------------------------------------------------------------- documents
    async def persist_documents(
        self,
        kb_id: str,
        documents: list[KnowledgeDocument],
        chunks: list[KnowledgeChunk],
    ) -> None:
        """Insert documents + chunks in one transaction (vectors cast to the col).

        Replace-by-source: any prior document(s) sharing an incoming ``source_uri``
        are deleted first (cascading to their chunks), so re-ingesting a changed or
        unchanged source replaces it rather than doubling the index.
        """
        cast = self._cast()
        sources = {d.source_uri for d in documents if d.source_uri is not None}
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for src in sources:
                    await conn.execute(
                        "DELETE FROM knowledge_documents "
                        "WHERE kb_id = $1 AND source_uri = $2",
                        kb_id,
                        src,
                    )
                for doc in documents:
                    await conn.execute(
                        """
                        INSERT INTO knowledge_documents
                            (document_id, kb_id, title, source_uri, text,
                             content_hash, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (document_id) DO NOTHING
                        """,
                        doc.document_id,
                        kb_id,
                        doc.title,
                        doc.source_uri,
                        doc.text,
                        doc.content_hash,
                        doc.metadata,
                    )
                for ch in chunks:
                    await conn.execute(
                        f"""
                        INSERT INTO knowledge_chunks
                            (chunk_id, document_id, kb_id, text, start_pos, end_pos,
                             embedding, chunk_kind, image_uri, caption, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7{cast}, $8, $9, $10, $11)
                        ON CONFLICT (chunk_id) DO NOTHING
                        """,
                        ch.chunk_id,
                        ch.document_id,
                        kb_id,
                        ch.text,
                        ch.start_pos,
                        ch.end_pos,
                        _vector_literal(ch.embedding),
                        ch.chunk_kind,
                        ch.image_uri,
                        ch.caption,
                        ch.metadata,
                    )

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """Delete a document (chunks cascade) within a KB."""
        sql = "DELETE FROM knowledge_documents WHERE document_id = $1 AND kb_id = $2"
        async with self._pool.acquire() as conn:
            status = await conn.execute(sql, document_id, kb_id)
        return status.endswith(" 1")

    # ----------------------------------------------------------------- search
    async def search(
        self,
        kb_id: str,
        query_vec: list[float],
        *,
        top_k: int,
        similarity_threshold: float | None,
        metadata_filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Cosine top-k over a KB's chunks (``1 - (<=> distance)`` = similarity).

        ``similarity_threshold=None`` drops non-positive similarities (matching the
        in-memory path); an explicit float applies ``sim >= threshold``.
        ``metadata_filters`` is pushed into SQL via JSONB containment (``@>``).
        """
        cast = self._cast()
        params: list[Any] = [_vector_literal(query_vec), kb_id]
        where = ["c.kb_id = $2"]
        if metadata_filters:
            import json

            params.append(json.dumps(metadata_filters))
            where.append(f"c.metadata @> ${len(params)}::jsonb")

        # Similarity cutoff pushed into SQL so the index does the filtering.
        if similarity_threshold is None:
            where.append("(1 - (c.embedding <=> $1" + cast + ")) > 0.0")
        else:
            params.append(float(similarity_threshold))
            where.append(f"(1 - (c.embedding <=> $1{cast})) >= ${len(params)}")

        params.append(int(top_k))
        limit_param = f"${len(params)}"

        sql = f"""
        SELECT
            c.chunk_id, c.document_id, c.text, c.start_pos, c.end_pos,
            c.chunk_kind, c.image_uri, c.caption, c.metadata,
            d.source_uri AS doc_source_uri, d.text AS doc_text,
            (1 - (c.embedding <=> $1{cast})) AS similarity
        FROM knowledge_chunks c
        JOIN knowledge_documents d ON d.document_id = c.document_id
        WHERE {" AND ".join(where)}
        ORDER BY c.embedding <=> $1{cast} ASC
        LIMIT {limit_param}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results: list[RetrievedChunk] = []
        for row in rows:
            similarity = float(row["similarity"])
            results.append(
                RetrievedChunk(
                    text=row["text"],
                    similarity=similarity,
                    context_window=self._window(
                        row["doc_text"], row["start_pos"], row["end_pos"], row["text"]
                    ),
                    document_id=row["document_id"],
                    source_uri=row["doc_source_uri"],
                    chunk_kind=row["chunk_kind"],
                    metadata={
                        **self._jsonb(row["metadata"]),
                        "chunk_id": row["chunk_id"],
                        "start_pos": row["start_pos"],
                        "end_pos": row["end_pos"],
                    },
                )
            )
        return results

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _window(
        doc_text: str | None, start: int, end: int, fallback: str | None
    ) -> str | None:
        """Padded window of the parent doc around the chunk span (mirrors in-mem)."""
        if doc_text is None:
            return fallback
        pad = 200
        s = max(0, start - pad)
        e = min(len(doc_text), end + pad)
        return doc_text[s:e]

    @staticmethod
    def _jsonb(value: Any) -> dict[str, Any]:
        """Decode a JSONB column value (codec may return a dict or a str)."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            import json

            try:
                return json.loads(value)
            except (ValueError, TypeError):  # pragma: no cover - defensive
                return {}
        return dict(value)

    def _row_to_kb(self, row: Any) -> KnowledgeBaseRecord:
        """Map a knowledge_bases row to a KnowledgeBaseRecord."""
        return KnowledgeBaseRecord(
            kb_id=row["kb_id"],
            workspace_id=row["workspace_id"],
            client_id=row["client_id"],
            name=row["name"],
            vector_dim=row["vector_dim"],
            metadata=self._jsonb(row["metadata"]),
        )


__all__ = [
    "KnowledgeBackendProtocol",
    "PgVectorKnowledgeBackend",
    "build_knowledge_schema_ddl",
    "resolve_column_and_index",
    "VECTOR_MAX_DIM",
    "HALFVEC_MAX_DIM",
]
