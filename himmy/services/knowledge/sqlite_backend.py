"""Knowledge kernel: a durable, file-backed SQLite knowledge backend (offline).

The desktop/offline twin of :class:`~himmy.services.knowledge.backend.PgVectorKnowledgeBackend`:
it implements the same :class:`~himmy.services.knowledge.backend.KnowledgeBackendProtocol`
(plus the optional :class:`~himmy.services.knowledge.backend.LexicalSearchProtocol`) over a
single stdlib ``sqlite3`` file — no Postgres, no ``pgvector``, no server. Wire it into a
:class:`~himmy.services.knowledge.service.KnowledgeBase` to make the index DURABLE: the
embeddings + chunks + the embedder fingerprint are persisted to disk, so an app restart
LOADS the existing index instead of re-embedding the whole corpus.

Why this exists (Daybook / any embedded async server): the in-memory KnowledgeBase
(``backend=None``) keeps everything in process-local dicts, so every launch re-embeds the
corpus (a ~2-minute CPU job for ~1M chars). Pointing the host at a disk path lets that cost
be paid ONCE ever — the second launch resolves the same KB, finds the same documents by
``(source_uri, content_hash)`` identity, and skips re-embedding (the service-layer
content-hash dedup, see :meth:`list_document_identities`).

Storage shape mirrors the Postgres schema (``knowledge_bases`` / ``knowledge_documents`` /
``knowledge_chunks``) so records round-trip identically: the full chunk metadata (including
the ``embedder_fingerprint`` stamped at ingest) rides in a JSON ``metadata`` column and the
embedding rides in a JSON ``embedding`` column. There is NO approximate vector index — search
is brute-force cosine in Python over the KB's chunks, which is the right tradeoff for desktop
KB sizes (thousands of chunks, single-digit-ms search) and keeps the file a plain ``.db`` with
no extension to install.

The blocking ``sqlite3`` work runs in :func:`asyncio.to_thread` (and a process lock serializes
writes) so the async surface never blocks the event loop — exactly the discipline the in-memory
embedder path now follows for the CPU-bound embed itself.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from himmy.core.errors import HimmyError
from himmy.core.sqlite_util import connect_hardened
from himmy.services.knowledge.models import (
    KnowledgeBaseRecord,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievedChunk,
)


@dataclass(frozen=True, slots=True)
class _CachedChunk:
    """One chunk of a KB's cached dense-search snapshot with JSON already decoded.

    Holds exactly the row fields the dense search paths consume — the decoded ``embedding``
    float list and ``metadata`` dict (the load-bearing ``json.loads``), plus the scalar
    columns :meth:`SqliteKnowledgeBackend._row_to_result` needs to hydrate a
    :class:`RetrievedChunk` without a second SQL round-trip. Immutable so a cached snapshot
    is safe to read concurrently across threads once published.
    """

    chunk_id: str
    document_id: str
    text: str | None
    start_pos: int
    end_pos: int
    chunk_kind: str
    image_uri: str | None
    caption: str | None
    doc_source_uri: str | None
    doc_text: str | None
    embedding: list[float]
    metadata: dict[str, Any]

#: Idempotent schema for the SQLite knowledge backend, a 1:1 shape-mirror of the
#: Postgres knowledge tables (see :func:`himmy.services.knowledge.backend.build_knowledge_schema_ddl`)
#: minus the pgvector-only embedding column type / ANN index: the embedding rides in a JSON
#: ``embedding`` column and search is brute-force cosine in Python, so no vector extension is
#: required. ``metadata`` is JSON text (the embedder fingerprint travels here). The
#: ``(kb_id, source_uri)`` index backs the replace-by-source upsert + the dedup identity scan.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    kb_id        TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    client_id    TEXT NOT NULL,
    name         TEXT NOT NULL,
    vector_dim   INTEGER NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    UNIQUE (workspace_id, client_id, name)
);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id  TEXT PRIMARY KEY,
    kb_id        TEXT NOT NULL REFERENCES knowledge_bases (kb_id) ON DELETE CASCADE,
    title        TEXT,
    source_uri   TEXT,
    text         TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS knowledge_documents_kb_id_idx
    ON knowledge_documents (kb_id);
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
    embedding   TEXT NOT NULL DEFAULT '[]',
    chunk_kind  TEXT NOT NULL DEFAULT 'text',
    image_uri   TEXT,
    caption     TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS knowledge_chunks_kb_id_idx
    ON knowledge_chunks (kb_id);
CREATE INDEX IF NOT EXISTS knowledge_chunks_document_id_idx
    ON knowledge_chunks (document_id);
"""

#: Additive, idempotent FTS5 lexical index for the optional lexical leg of hybrid
#: retrieval. Kept SEPARATE from the base schema (like the Postgres ``text_tsv`` DDL) so a
#: backend without it stays a valid dense-only store; :meth:`SqliteKnowledgeBackend.lexical_available`
#: reports whether the table exists. FTS5 is a stdlib SQLite module on virtually every build;
#: when it is absent the create is a no-op and the KB degrades to dense-only retrieval.
_LEXICAL_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5 ("
    "chunk_id UNINDEXED, kb_id UNINDEXED, text)"
)


#: Default upper bound on the number of DISTINCT ``kb_id`` matrices held in the decoded
#: vector cache (an LRU by kb_id). Overridable via ``HIMMY_KB_VECTOR_CACHE_MAX_KBS``; set to
#: ``0`` (or any non-positive value) to DISABLE the cache entirely and always decode from
#: SQLite (the original, cache-free behaviour). Bounding by kb-count keeps a process that
#: fans over many knowledge bases from pinning every decoded matrix in memory forever.
_DEFAULT_VECTOR_CACHE_MAX_KBS = 32


def _vector_cache_max_kbs() -> int:
    """Resolve the vector-cache kb-bound from ``HIMMY_KB_VECTOR_CACHE_MAX_KBS``.

    Returns the default when unset/garbage; a non-positive value disables caching (the
    backend then decodes every embedding from SQLite on each search, byte-identically to the
    pre-cache behaviour).
    """
    raw = os.environ.get("HIMMY_KB_VECTOR_CACHE_MAX_KBS")
    if raw is None or raw == "":
        return _DEFAULT_VECTOR_CACHE_MAX_KBS
    try:
        return int(raw)
    except ValueError:  # pragma: no cover - defensive
        return _DEFAULT_VECTOR_CACHE_MAX_KBS


class _VectorCacheState:
    """Decoded-vector cache state ANCHORED to a shared SQLite connection, not an instance.

    :meth:`SqliteStorageService.knowledge_backend` is a FACTORY: it hands out a NEW
    :class:`SqliteKnowledgeBackend` per call, all fronting the SAME connection. If each
    kept its own cache + generation counter, a mutation through backend A would invalidate
    only A's cache, and a warm backend B over the same connection would serve deleted/old
    vectors forever. Storing the cache dict, its lock, and the commit-generation counter on
    a per-CONNECTION holder (looked up in :data:`_VECTOR_CACHE_STATES`) makes every backend
    built from one connection share the SAME cache, so any instance's committed mutation
    invalidates the snapshot every sibling would read.
    """

    __slots__ = ("cache", "cache_lock", "data_version", "gen")

    def __init__(self) -> None:
        self.cache: OrderedDict[str, list[_CachedChunk]] = OrderedDict()
        self.cache_lock = threading.Lock()
        #: Monotonic commit-generation, bumped under the backend's write lock at every
        #: committed chunk mutation by ANY backend sharing this connection.
        self.gen = 0
        #: Last observed ``PRAGMA data_version`` for the shared connection. SQLite bumps it
        #: whenever ANOTHER connection commits to the file (a second process, sidecar, or
        #: raw ``sqlite3`` write); a change means the whole cache may be stale, so it is
        #: dropped. ``None`` until the first read. Guarded by ``cache_lock``.
        self.data_version: int | None = None


#: Per-connection vector-cache state, so multiple :class:`SqliteKnowledgeBackend`\\ s over
#: one shared connection coordinate one cache + generation counter (see
#: :class:`_VectorCacheState`). ``sqlite3.Connection`` is neither weakly-referenceable nor
#: attribute-settable, so we key by ``id(conn)`` and STRONG-reference the connection in the
#: value tuple ``(conn, state)`` — pinning it guarantees the id cannot be reused (and thus
#: collide) while the entry lives. :meth:`SqliteKnowledgeBackend.close` evicts the entry
#: for a connection this backend OWNS; a shared connection's entry lives as long as its
#: owning storage service (matching that lifetime). Guarded by the states lock on lookup.
_VECTOR_CACHE_STATES: dict[int, tuple[sqlite3.Connection, _VectorCacheState]] = {}
_VECTOR_CACHE_STATES_LOCK = threading.Lock()


def _vector_cache_state_for(conn: sqlite3.Connection) -> _VectorCacheState:
    """Return the shared cache state for ``conn``, creating it once (thread-safe)."""
    with _VECTOR_CACHE_STATES_LOCK:
        entry = _VECTOR_CACHE_STATES.get(id(conn))
        if entry is None:
            entry = (conn, _VectorCacheState())
            _VECTOR_CACHE_STATES[id(conn)] = entry
        return entry[1]


def _drop_vector_cache_state(conn: sqlite3.Connection) -> None:
    """Evict a connection's shared cache state (called when an OWNED connection closes)."""
    with _VECTOR_CACHE_STATES_LOCK:
        _VECTOR_CACHE_STATES.pop(id(conn), None)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (0.0 when degenerate).

    Identical semantics to the in-memory KnowledgeBase ``_cosine`` so disk-backed and
    in-memory search rank the same way: an empty/length-mismatched/zero-norm pair scores
    0.0 rather than raising or producing a NaN.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SqliteKnowledgeBackend:
    """File-backed SQLite implementation of :class:`KnowledgeBackendProtocol`.

    Construct with a filesystem ``path`` (its parent directory is auto-created) and wire it
    into a :class:`~himmy.services.knowledge.service.KnowledgeBase` via
    ``KnowledgeBase(..., backend=SqliteKnowledgeBackend(path))``. The KB/document/chunk
    persistence and cosine search live on disk, so the index survives process restarts and a
    re-ingest of the same corpus skips re-embedding (the KnowledgeBase reads the persisted
    ``(source_uri, content_hash)`` identities via :meth:`list_document_identities`).

    Pass an already-open ``SqliteStorageService`` connection + lock via ``connection=`` /
    ``lock=`` to SHARE one ``.db`` file with the rest of the storage surface (this is what
    :meth:`himmy.services.storage.sqlite.SqliteStorageService.knowledge_backend` does); the
    knowledge tables simply coexist with the storage tables.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        connection: sqlite3.Connection | None = None,
        lock: threading.Lock | None = None,
        enable_lexical: bool = True,
    ) -> None:
        """Open (or reuse) a SQLite database and apply the knowledge schema.

        ``path`` is used when no ``connection`` is supplied; its parent directory is created
        when missing so a default like ``~/.daybook/knowledge.db`` works on first run.
        ``:memory:`` is accepted for ephemeral use. Pass ``connection`` (+ its ``lock``) to
        reuse a host ``SqliteStorageService`` connection so the knowledge tables live in the
        same file. ``enable_lexical`` provisions the FTS5 table for the hybrid lexical leg
        (best-effort: a SQLite build without FTS5 silently degrades to dense-only).
        """
        if connection is not None:
            self._conn = connection
            self._lock = lock or threading.Lock()
            self._owns_conn = False
            self._path = ":shared:"
        else:
            self._path = str(path)
            if self._path != ":memory:":
                Path(self._path).expanduser().resolve().parent.mkdir(
                    parents=True, exist_ok=True
                )
            self._conn = connect_hardened(self._path)
            self._conn.row_factory = sqlite3.Row
            self._lock = threading.Lock()
            self._owns_conn = True
        self._conn.row_factory = sqlite3.Row
        self._lexical_enabled = enable_lexical
        self._lexical_ready = False
        # Decoded-vector cache: per-``kb_id`` snapshot of the KB's chunks with the embedding
        # (and metadata) JSON already decoded, so a repeat dense search skips re-paying the
        # high-dim ``json.loads`` that dwarfs the cosine math. An ``OrderedDict`` gives an LRU
        # bounded by :func:`_vector_cache_max_kbs` distinct kb_ids (so many KBs can't grow
        # memory without bound). Guarded by its OWN lock (never the write lock) so concurrent
        # searches don't serialize on cache reads; every mutation point that changes a KB's
        # chunks (:meth:`persist_documents`, :meth:`delete_document`, :meth:`delete_kb`, and
        # the KB-wide chunk purge) invalidates that kb_id precisely so a stale matrix is never
        # served. Keyed strictly by ``kb_id`` (a global PRIMARY KEY, unique across
        # tenant/workspace even in a shared ``.db``), so KB A never serves KB B's matrix.
        # Cache state is anchored to the CONNECTION (shared across every backend built from
        # the same connection), NOT to this instance — see :class:`_VectorCacheState`. Two
        # backends over one shared ``.db`` therefore coordinate one cache + generation
        # counter, so a mutation through one invalidates the snapshot the other would read.
        self._vec_state = _vector_cache_state_for(self._conn)
        self._vec_cache_max_kbs = _vector_cache_max_kbs()
        # Last ``PRAGMA data_version`` this backend observed per kb_id snapshot. SQLite bumps
        # ``data_version`` whenever ANOTHER connection (a second process, a sidecar, a raw
        # ``sqlite3`` write) commits to the file, so folding it into the snapshot-validity
        # check makes an externally-written KB re-decode instead of serving a stale matrix.
        self._vec_data_version = 0
        # The commit-generation counter (``self._vec_state.gen``) is bumped under
        # ``self._lock`` at every committed chunk mutation (persist/delete-document/
        # delete-kb). A snapshot captures the generation ATOMICALLY with its row fetch (same
        # ``self._lock`` hold); publication into the cache only happens if the generation
        # has not advanced since. This closes the fetch->publish TOCTOU where an
        # invalidation lands after the fetch but before the publish (a bare pop would be a
        # no-op on the not-yet-published entry, then the stale snapshot would be published
        # and served indefinitely). For a SHARED connection ``self._lock`` is the shared
        # write lock, so sibling backends' bumps are seen coherently.
        self._apply_schema()

    @property
    def path(self) -> str:
        """The filesystem path of the backing database (``:shared:`` for a reused conn)."""
        return self._path

    @property
    def _vec_cache(self) -> OrderedDict[str, list[_CachedChunk]]:
        """The shared decoded-vector cache for this backend's connection (see the state).

        A property so multiple backends over one connection observe the SAME mapping and
        tests can introspect it, while the storage lives on the per-connection
        :class:`_VectorCacheState` (never a per-instance dict).
        """
        return self._vec_state.cache

    async def close(self) -> None:
        """Close the connection if this backend owns it (a shared conn is left alone)."""
        if self._owns_conn:
            # Evict the per-connection cache state BEFORE closing so its ``id`` can't be
            # reused by a later connection while a stale entry lingers.
            _drop_vector_cache_state(self._conn)
            await asyncio.to_thread(self._conn.close)

    # ------------------------------------------------------------- sync primitives
    def _apply_schema(self) -> None:
        """Create the knowledge tables (+ optional FTS5 index) under the write lock."""
        with self._lock:
            try:
                self._conn.executescript(_SCHEMA)
                if self._lexical_enabled:
                    try:
                        self._conn.execute(_LEXICAL_SCHEMA)
                        self._lexical_ready = True
                    except sqlite3.OperationalError:
                        # SQLite built without FTS5 — degrade to dense-only retrieval.
                        self._lexical_ready = False
                self._conn.commit()
            except BaseException:
                self._rollback_quietly()
                raise

    def _rollback_quietly(self) -> None:
        """Roll back the shared connection, swallowing rollback-time errors."""
        try:
            self._conn.rollback()
        except sqlite3.Error:  # pragma: no cover - rollback on a dead connection
            pass

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cast("sqlite3.Row | None", cur.fetchone())

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return list(cur.fetchall())

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
        inserted = await asyncio.to_thread(self._create_kb_sync, record)
        if not inserted:
            raise HimmyError(
                f"KnowledgeBase {name!r} already exists for "
                f"({workspace_id}, {client_id})."
            )
        return record

    def _create_kb_sync(self, record: KnowledgeBaseRecord) -> bool:
        """The locked insert-if-absent for a KB (returns False on a dup scope)."""
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO knowledge_bases "
                    "(kb_id, workspace_id, client_id, name, vector_dim, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (workspace_id, client_id, name) DO NOTHING",
                    (
                        record.kb_id,
                        record.workspace_id,
                        record.client_id,
                        record.name,
                        record.vector_dim,
                        json.dumps(record.metadata),
                    ),
                )
                self._conn.commit()
                return cur.rowcount == 1
            except BaseException:
                self._rollback_quietly()
                raise

    async def get_kb(self, kb_id: str) -> KnowledgeBaseRecord | None:
        """Load a KB by id."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT * FROM knowledge_bases WHERE kb_id = ?",
            (kb_id,),
        )
        return self._row_to_kb(row) if row is not None else None

    async def resolve_kb(
        self, *, workspace_id: str, client_id: str, name: str
    ) -> KnowledgeBaseRecord | None:
        """Resolve a KB by its scope keys."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT * FROM knowledge_bases "
            "WHERE workspace_id = ? AND client_id = ? AND name = ?",
            (workspace_id, client_id, name),
        )
        return self._row_to_kb(row) if row is not None else None

    async def delete_kb(self, kb_id: str) -> bool:
        """Cascade-delete a KB and its documents/chunks (FKs are not enforced; manual)."""
        return await asyncio.to_thread(self._delete_kb_sync, kb_id)

    def _delete_kb_sync(self, kb_id: str) -> bool:
        with self._lock:
            try:
                self._delete_chunks_for_kb(kb_id)
                self._conn.execute(
                    "DELETE FROM knowledge_documents WHERE kb_id = ?", (kb_id,)
                )
                cur = self._conn.execute(
                    "DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,)
                )
                self._conn.commit()
                deleted = cur.rowcount == 1
                # Bump under the write lock so any snapshot that captured the pre-commit
                # generation (and thus pre-commit rows) will refuse to publish afterwards.
                self._vec_state.gen += 1
            except BaseException:
                self._rollback_quietly()
                raise
        # Committed: the KB (and all its chunks) are gone — evict its cached matrix.
        self._invalidate_vector_cache(kb_id)
        return deleted

    # -------------------------------------------------------------- documents
    async def persist_documents(
        self,
        kb_id: str,
        documents: list[KnowledgeDocument],
        chunks: list[KnowledgeChunk],
    ) -> None:
        """Insert documents + chunks in one transaction.

        Replace-by-source: any prior document(s) sharing an incoming ``source_uri`` are
        deleted first (cascading to their chunks + FTS rows), so re-ingesting a changed
        source replaces it rather than doubling the index — matching the Postgres backend.
        """
        await asyncio.to_thread(self._persist_documents_sync, kb_id, documents, chunks)

    def _persist_documents_sync(
        self,
        kb_id: str,
        documents: list[KnowledgeDocument],
        chunks: list[KnowledgeChunk],
    ) -> None:
        sources = {d.source_uri for d in documents if d.source_uri is not None}
        with self._lock:
            try:
                for src in sources:
                    doomed = self._conn.execute(
                        "SELECT document_id FROM knowledge_documents "
                        "WHERE kb_id = ? AND source_uri = ?",
                        (kb_id, src),
                    ).fetchall()
                    for row in doomed:
                        self._delete_document_rows(kb_id, row["document_id"])
                for doc in documents:
                    self._conn.execute(
                        "INSERT INTO knowledge_documents "
                        "(document_id, kb_id, title, source_uri, text, content_hash, "
                        "metadata) VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT (document_id) DO NOTHING",
                        (
                            doc.document_id,
                            kb_id,
                            doc.title,
                            doc.source_uri,
                            doc.text,
                            doc.content_hash,
                            json.dumps(doc.metadata),
                        ),
                    )
                for ch in chunks:
                    self._conn.execute(
                        "INSERT INTO knowledge_chunks "
                        "(chunk_id, document_id, kb_id, text, start_pos, end_pos, "
                        "embedding, chunk_kind, image_uri, caption, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT (chunk_id) DO NOTHING",
                        (
                            ch.chunk_id,
                            ch.document_id,
                            kb_id,
                            ch.text,
                            ch.start_pos,
                            ch.end_pos,
                            json.dumps([float(v) for v in ch.embedding]),
                            ch.chunk_kind,
                            ch.image_uri,
                            ch.caption,
                            json.dumps(ch.metadata),
                        ),
                    )
                    if self._lexical_ready and ch.text:
                        self._conn.execute(
                            "INSERT INTO knowledge_chunks_fts (chunk_id, kb_id, text) "
                            "VALUES (?, ?, ?)",
                            (ch.chunk_id, kb_id, ch.text),
                        )
                self._conn.commit()
                # Bump under the write lock so an in-flight snapshot that read the old rows
                # (with the old generation) is refused publication after this commit.
                self._vec_state.gen += 1
            except BaseException:
                self._rollback_quietly()
                raise
        # Committed: the KB's chunk set changed — drop its cached matrix so the next search
        # rebuilds from the new rows (never a stale snapshot).
        self._invalidate_vector_cache(kb_id)

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """Delete a document (chunks cascade) within a KB."""
        return await asyncio.to_thread(
            self._delete_document_locked, kb_id, document_id
        )

    def _delete_document_locked(self, kb_id: str, document_id: str) -> bool:
        with self._lock:
            try:
                deleted = self._delete_document_rows(kb_id, document_id)
                self._conn.commit()
                # Bump under the write lock so a concurrent snapshot that captured the
                # pre-commit generation cannot publish its now-stale rows.
                self._vec_state.gen += 1
            except BaseException:
                self._rollback_quietly()
                raise
        # Committed: invalidate even when nothing matched is harmless (a miss is a no-op);
        # when a chunk was removed this is what prevents a stale cached matrix.
        self._invalidate_vector_cache(kb_id)
        return deleted

    def _delete_document_rows(self, kb_id: str, document_id: str) -> bool:
        """Delete a document + its chunks (+ FTS rows). Caller holds the lock + commits."""
        if self._lexical_ready:
            self._conn.execute(
                "DELETE FROM knowledge_chunks_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM knowledge_chunks WHERE document_id = ?)",
                (document_id,),
            )
        self._conn.execute(
            "DELETE FROM knowledge_chunks WHERE document_id = ?", (document_id,)
        )
        cur = self._conn.execute(
            "DELETE FROM knowledge_documents WHERE document_id = ? AND kb_id = ?",
            (document_id, kb_id),
        )
        return cur.rowcount == 1

    def _delete_chunks_for_kb(self, kb_id: str) -> None:
        """Delete all chunks (+ FTS rows) for a KB. Caller holds the lock + commits."""
        if self._lexical_ready:
            self._conn.execute(
                "DELETE FROM knowledge_chunks_fts WHERE kb_id = ?", (kb_id,)
            )
        self._conn.execute("DELETE FROM knowledge_chunks WHERE kb_id = ?", (kb_id,))

    async def list_document_identities(
        self, kb_id: str
    ) -> list[tuple[str | None, str]]:
        """Return persisted ``(source_uri, content_hash)`` identities for a KB.

        This is the hook that makes a restart skip re-embedding: the
        :class:`~himmy.services.knowledge.service.KnowledgeBase` reads these identities at
        ingest and, for any input whose ``(source_uri, content_hash)`` already exists,
        reuses the stored document instead of re-embedding it (the same content-hash dedup
        the in-memory path does over its dicts). Returning ``[]`` (or omitting this method
        on a backend) keeps the prior always-re-embed behaviour.
        """
        rows = await asyncio.to_thread(
            self._fetchall,
            "SELECT source_uri, content_hash FROM knowledge_documents WHERE kb_id = ?",
            (kb_id,),
        )
        return [(row["source_uri"], row["content_hash"] or "") for row in rows]

    async def get_document(
        self, kb_id: str, source_uri: str | None, content_hash: str
    ) -> KnowledgeDocument | None:
        """Re-hydrate a stored document by its ``(source_uri, content_hash)`` identity.

        Used by the dedup path to return the EXISTING document for an unchanged input so the
        caller hands back a real persisted record (with its original ``document_id``), not a
        freshly-minted one. ``source_uri=None`` matches a uri-less raw-text document.
        """
        if source_uri is None:
            sql = (
                "SELECT * FROM knowledge_documents "
                "WHERE kb_id = ? AND source_uri IS NULL AND content_hash = ?"
            )
            params: tuple[Any, ...] = (kb_id, content_hash)
        else:
            sql = (
                "SELECT * FROM knowledge_documents "
                "WHERE kb_id = ? AND source_uri = ? AND content_hash = ?"
            )
            params = (kb_id, source_uri, content_hash)
        row = await asyncio.to_thread(self._fetchone, sql, params)
        return self._row_to_document(row) if row is not None else None

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
        """Brute-force cosine top-k over a KB's chunks (mirrors the in-memory path).

        ``similarity_threshold=None`` drops non-positive similarities (the in-memory and
        pgvector default); an explicit float applies ``sim >= threshold``.
        ``metadata_filters`` is exact-equality over the chunk's top-level metadata, matching
        the in-memory ``search``.
        """
        return await asyncio.to_thread(
            self._search_sync,
            kb_id,
            query_vec,
            top_k,
            similarity_threshold,
            metadata_filters,
        )

    def _search_sync(
        self,
        kb_id: str,
        query_vec: list[float],
        top_k: int,
        similarity_threshold: float | None,
        metadata_filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        chunks = self._cached_chunks(kb_id)

        def _keep(sim: float) -> bool:
            if similarity_threshold is None:
                return sim > 0.0
            return sim >= similarity_threshold

        scored: list[tuple[float, _CachedChunk]] = []
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
            results.append(self._cached_to_result(chunk, sim))
        return results

    # --------------------------------------------------------------- lexical
    async def lexical_available(self, kb_id: str) -> bool:
        """True when the FTS5 lexical index exists for this build."""
        return self._lexical_ready

    async def search_lexical(
        self,
        kb_id: str,
        query: str,
        *,
        top_k: int,
        metadata_filters: dict[str, Any] | None,
    ) -> list[tuple[str, float]]:
        """FTS5 top-k ``(chunk_id, score)`` over the lexical index, best first.

        SQLite FTS5 ``bm25()`` returns a DISTANCE (lower is better); it is negated here so
        a larger returned score means a better match, matching the ``(chunk_id, score)``
        contract the hybrid RRF fusion expects. ``metadata_filters`` is applied after the
        match (the dense leg pushes the same equality), mirroring the in-memory lexical leg.
        """
        if not self._lexical_ready:
            return []
        return await asyncio.to_thread(
            self._search_lexical_sync, kb_id, query, top_k, metadata_filters
        )

    def _search_lexical_sync(
        self,
        kb_id: str,
        query: str,
        top_k: int,
        metadata_filters: dict[str, Any] | None,
    ) -> list[tuple[str, float]]:
        # A blank / punctuation-only query yields no MATCH terms; FTS5 raises on an empty
        # query string, so guard it and return nothing (dense-only for that turn).
        if not query or not query.strip():
            return []
        pool = top_k * 2 if metadata_filters else top_k
        try:
            rows = self._fetchall(
                "SELECT f.chunk_id AS chunk_id, bm25(knowledge_chunks_fts) AS score "
                "FROM knowledge_chunks_fts f "
                "WHERE f.kb_id = ? AND knowledge_chunks_fts MATCH ? "
                "ORDER BY score ASC LIMIT ?",
                (kb_id, self._fts_query(query), pool),
            )
        except sqlite3.OperationalError:
            # A query that does not parse as an FTS5 expression — degrade to dense-only.
            return []
        hits = [(str(row["chunk_id"]), -float(row["score"])) for row in rows]
        if not metadata_filters:
            return hits[:top_k]
        kept: list[tuple[str, float]] = []
        for chunk_id, score in hits:
            meta_row = self._fetchone(
                "SELECT metadata FROM knowledge_chunks WHERE chunk_id = ?",
                (chunk_id,),
            )
            if meta_row is None:
                continue
            meta = self._json(meta_row["metadata"])
            if any(meta.get(k) != v for k, v in metadata_filters.items()):
                continue
            kept.append((chunk_id, score))
        return kept[:top_k]

    @staticmethod
    def _fts_query(query: str) -> str:
        """Render a free-text query as a safe FTS5 OR-of-terms MATCH expression.

        Each alphanumeric token is double-quoted (so it is a literal term, never an FTS5
        operator) and joined with ``OR`` — the lexical leg wants "any of these words", which
        RRF then fuses with the dense leg. Punctuation/operators in the raw query can never
        reach the FTS5 parser this way.
        """
        import re

        tokens = re.findall(r"[A-Za-z0-9]+", query)
        if not tokens:
            return '""'
        return " OR ".join(f'"{t}"' for t in tokens)

    async def get_chunk(self, kb_id: str, chunk_id: str) -> RetrievedChunk | None:
        """Re-hydrate one chunk by id (used by the hybrid pipeline's fusion cut)."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT c.chunk_id, c.document_id, c.text, c.start_pos, c.end_pos, "
            "c.embedding, c.chunk_kind, c.image_uri, c.caption, c.metadata, "
            "d.source_uri AS doc_source_uri, d.text AS doc_text "
            "FROM knowledge_chunks c "
            "JOIN knowledge_documents d ON d.document_id = c.document_id "
            "WHERE c.kb_id = ? AND c.chunk_id = ?",
            (kb_id, chunk_id),
        )
        if row is None:
            return None
        return self._row_to_result(row, 0.0, self._json(row["metadata"]))

    async def dense_ranked(
        self,
        kb_id: str,
        query_vec: list[float],
        *,
        top_k: int,
        metadata_filters: dict[str, Any] | None,
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, cosine_similarity)`` for the dense leg of fusion.

        Like :meth:`search` but ids+scores only (no re-hydration) and no threshold, so the
        fused candidate pool can be wide; the hybrid pipeline re-hydrates survivors via
        :meth:`get_chunk`.
        """
        return await asyncio.to_thread(
            self._dense_ranked_sync, kb_id, query_vec, top_k, metadata_filters
        )

    def _dense_ranked_sync(
        self,
        kb_id: str,
        query_vec: list[float],
        top_k: int,
        metadata_filters: dict[str, Any] | None,
    ) -> list[tuple[str, float]]:
        chunks = self._cached_chunks(kb_id)
        scored: list[tuple[str, float]] = []
        for chunk in chunks:
            if metadata_filters and any(
                chunk.metadata.get(k) != v for k, v in metadata_filters.items()
            ):
                continue
            sim = _cosine(query_vec, chunk.embedding)
            if sim > 0.0:
                scored.append((chunk.chunk_id, sim))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]

    # ------------------------------------------------------------- vector cache
    def _cached_chunks(self, kb_id: str) -> list[_CachedChunk]:
        """Return the KB's decoded-chunk snapshot, building + caching it on a miss.

        The returned list is the CACHED object; it is treated as immutable by all readers
        (dense search never mutates it, only scores over it), so it is safe to hand the same
        list to concurrent searches. When caching is disabled (bound <= 0) a fresh snapshot
        is built and returned WITHOUT caching, preserving the exact pre-cache decode path.
        """
        if self._vec_cache_max_kbs <= 0:
            _gen, snapshot = self._build_chunk_snapshot(kb_id)
            return snapshot
        # Drop the whole cache if ANOTHER connection committed to the file since we last
        # looked (cross-process / raw-SQL write) — cheap ``PRAGMA data_version`` check.
        self._drop_cache_on_external_write()
        cache = self._vec_state.cache
        with self._vec_state.cache_lock:
            cached = cache.get(kb_id)
            if cached is not None:
                cache.move_to_end(kb_id)
                return cached
        # Build OUTSIDE the cache lock (the SELECT + JSON decode can be slow and must not
        # serialize other KBs' cache reads); the write lock still guards the SQL fetch AND the
        # generation read, so ``gen`` and ``snapshot`` describe the SAME committed state.
        gen, snapshot = self._build_chunk_snapshot(kb_id)
        with self._vec_state.cache_lock:
            # A concurrent builder may have populated it first — prefer the published entry so
            # every reader shares one snapshot (both reflect the same committed rows).
            existing = cache.get(kb_id)
            if existing is not None:
                cache.move_to_end(kb_id)
                return existing
            # Only publish if no committed mutation intervened between the fetch and now.
            # If the generation advanced, an invalidation already fired (a no-op pop, since
            # we hadn't published yet) and this snapshot is stale — return it uncached so the
            # NEXT search rebuilds from the newer rows. This closes the fetch->publish TOCTOU
            # that would otherwise pin a stale matrix indefinitely.
            if self._current_vec_gen() == gen:
                cache[kb_id] = snapshot
                cache.move_to_end(kb_id)
                while len(cache) > self._vec_cache_max_kbs:
                    cache.popitem(last=False)
        return snapshot

    def _drop_cache_on_external_write(self) -> None:
        """Clear the shared cache when another CONNECTION has committed since last checked.

        ``PRAGMA data_version`` is a per-connection counter SQLite increments whenever the
        database file is modified by a DIFFERENT connection (another process, a sidecar
        worker, or a manual ``sqlite3`` write) — it does NOT move for this connection's own
        writes (those are tracked precisely by ``self._vec_state.gen``). Reading it is a
        cheap pragma. On a change we conservatively drop the ENTIRE cache so a KB written
        externally re-decodes fresh rows instead of serving a stale matrix — restoring the
        read-your-writes-across-processes property the cache-free path had. Guarded by the
        write lock (the pragma is a read on the shared connection) + the cache lock.
        """
        with self._lock:
            row = self._conn.execute("PRAGMA data_version").fetchone()
        current = int(row[0]) if row is not None else 0
        with self._vec_state.cache_lock:
            if self._vec_state.data_version is None:
                self._vec_state.data_version = current
                return
            if current != self._vec_state.data_version:
                self._vec_state.cache.clear()
                self._vec_state.data_version = current

    def _current_vec_gen(self) -> int:
        """Read the commit-generation counter under the write lock (atomic w.r.t. bumps)."""
        with self._lock:
            return self._vec_state.gen

    def _build_chunk_snapshot(self, kb_id: str) -> tuple[int, list[_CachedChunk]]:
        """Fetch + decode a KB's chunks, returning ``(generation, snapshot)`` atomically.

        The commit generation is read in the SAME ``self._lock`` critical section as the row
        fetch, so the returned ``generation`` exactly describes the committed state these rows
        came from. :meth:`_cached_chunks` then refuses to publish the snapshot if the
        generation has since advanced, so an invalidation that lands between fetch and publish
        can never leave a stale matrix cached. This is the one place the ``embedding`` /
        ``metadata`` JSON is decoded; the cache amortises it across repeat searches. Ordering
        is left to the callers' sort, so the row order here does not affect results.
        """
        with self._lock:
            gen = self._vec_state.gen
            cur = self._conn.execute(
                "SELECT c.chunk_id, c.document_id, c.text, c.start_pos, c.end_pos, "
                "c.embedding, c.chunk_kind, c.image_uri, c.caption, c.metadata, "
                "d.source_uri AS doc_source_uri, d.text AS doc_text "
                "FROM knowledge_chunks c "
                "JOIN knowledge_documents d ON d.document_id = c.document_id "
                "WHERE c.kb_id = ?",
                (kb_id,),
            )
            rows = list(cur.fetchall())
        return gen, [
            _CachedChunk(
                chunk_id=str(row["chunk_id"]),
                document_id=row["document_id"],
                text=row["text"],
                start_pos=row["start_pos"],
                end_pos=row["end_pos"],
                chunk_kind=row["chunk_kind"],
                image_uri=row["image_uri"],
                caption=row["caption"],
                doc_source_uri=row["doc_source_uri"],
                doc_text=row["doc_text"],
                embedding=self._json_list(row["embedding"]),
                metadata=self._json(row["metadata"]),
            )
            for row in rows
        ]

    def _invalidate_vector_cache(self, kb_id: str) -> None:
        """Drop the cached decoded snapshot for ``kb_id`` (precise, single-KB eviction).

        Called from every mutation that changes a KB's chunk set (AFTER it bumped
        ``self._vec_state.gen`` under the write lock) so the next search rebuilds from
        committed rows. Because the cache is anchored to the shared connection, this evicts
        the entry that EVERY backend over that connection would read — not just this
        instance's — closing the cross-instance staleness. The generation guard in
        :meth:`_cached_chunks` covers the complementary in-flight-build case (a bare pop
        would no-op there). Cheap and idempotent; other KBs' snapshots are untouched.
        """
        with self._vec_state.cache_lock:
            self._vec_state.cache.pop(kb_id, None)

    # ---------------------------------------------------------------- helpers
    def _row_to_result(
        self, row: sqlite3.Row, similarity: float, meta: dict[str, Any]
    ) -> RetrievedChunk:
        """Map a chunk-join row to a RetrievedChunk (mirrors the pgvector mapping)."""
        return RetrievedChunk(
            text=row["text"],
            similarity=similarity,
            context_window=self._window(
                row["doc_text"], row["start_pos"], row["end_pos"], row["text"]
            ),
            document_id=row["document_id"],
            source_uri=row["doc_source_uri"],
            chunk_kind=row["chunk_kind"],
            metadata={
                **meta,
                "chunk_id": row["chunk_id"],
                "start_pos": row["start_pos"],
                "end_pos": row["end_pos"],
            },
        )

    def _cached_to_result(
        self, chunk: _CachedChunk, similarity: float
    ) -> RetrievedChunk:
        """Map a cached decoded chunk to a RetrievedChunk (parity with :meth:`_row_to_result`).

        Produces byte-identical output to the row-based mapping — the cache only pre-decodes
        the JSON, it does not change what a hit looks like.
        """
        return RetrievedChunk(
            text=chunk.text,
            similarity=similarity,
            context_window=self._window(
                chunk.doc_text, chunk.start_pos, chunk.end_pos, chunk.text
            ),
            document_id=chunk.document_id,
            source_uri=chunk.doc_source_uri,
            chunk_kind=chunk.chunk_kind,
            metadata={
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
            },
        )

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
    def _json(value: Any) -> dict[str, Any]:
        """Decode a JSON text metadata column to a dict (empty on null/garbage)."""
        if not value:
            return {}
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _json_list(value: Any) -> list[float]:
        """Decode a JSON text embedding column to a float list (empty on garbage)."""
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return []
        if not isinstance(decoded, list):
            return []
        return [float(v) for v in decoded]

    def _row_to_kb(self, row: sqlite3.Row) -> KnowledgeBaseRecord:
        """Map a knowledge_bases row to a KnowledgeBaseRecord."""
        return KnowledgeBaseRecord(
            kb_id=row["kb_id"],
            workspace_id=row["workspace_id"],
            client_id=row["client_id"],
            name=row["name"],
            vector_dim=row["vector_dim"],
            metadata=self._json(row["metadata"]),
        )

    def _row_to_document(self, row: sqlite3.Row) -> KnowledgeDocument:
        """Map a knowledge_documents row to a KnowledgeDocument."""
        return KnowledgeDocument(
            document_id=row["document_id"],
            kb_id=row["kb_id"],
            title=row["title"],
            source_uri=row["source_uri"],
            text=row["text"],
            content_hash=row["content_hash"] or "",
            metadata=self._json(row["metadata"]),
        )


__all__ = ["SqliteKnowledgeBackend"]
