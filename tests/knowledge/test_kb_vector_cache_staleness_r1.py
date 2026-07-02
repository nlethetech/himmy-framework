"""Red-team round-1 regressions for KB vector-cache STALENESS (eff-p1 safe-r1).

The decoded-vector cache must not serve stale/deleted vectors when the SAME ``.db`` is
mutated through a DIFFERENT backend instance over one shared connection, or by an EXTERNAL
connection (second process / raw SQL). These lock the fixes:

* the cache + generation counter are anchored to the CONNECTION, so a mutation through
  backend A invalidates the snapshot a warm sibling backend B would read;
* a ``PRAGMA data_version`` change (another connection committed) drops the cache so an
  externally-written KB re-decodes fresh rows.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from himmy.core.sqlite_util import connect_hardened
from himmy.services.knowledge.models import KnowledgeChunk, KnowledgeDocument
from himmy.services.knowledge.sqlite_backend import SqliteKnowledgeBackend
from tests.conftest import run_async


def _make_kb(backend: SqliteKnowledgeBackend, name: str = "kb") -> str:
    rec = run_async(
        backend.create_kb(workspace_id="ws", client_id="cl", name=name, vector_dim=3)
    )
    return rec.kb_id


def _ingest(
    backend: SqliteKnowledgeBackend,
    kb_id: str,
    *,
    doc_id: str,
    source_uri: str,
    text: str,
    embedding: list[float],
) -> None:
    doc = KnowledgeDocument(
        document_id=doc_id,
        kb_id=kb_id,
        title=text,
        source_uri=source_uri,
        text=text,
        content_hash=source_uri,
    )
    chunk = KnowledgeChunk(
        chunk_id=f"{doc_id}-c0",
        document_id=doc_id,
        kb_id=kb_id,
        text=text,
        start_pos=0,
        end_pos=len(text),
        embedding=embedding,
    )
    run_async(backend.persist_documents(kb_id, [doc], [chunk]))


def _search(backend: SqliteKnowledgeBackend, kb: str, q: list[float]) -> list[str]:
    hits = run_async(
        backend.search(kb, q, top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    return [r.text for r in hits]


# -------------------------------------------- two backends over ONE connection


def test_two_backends_one_connection_share_cache_invalidation() -> None:
    """B's warm cache must reflect a delete committed through sibling backend A.

    Regression: the cache was per-INSTANCE, so A's delete invalidated only A and warm B
    served the deleted chunk forever. Anchored to the connection, A's mutation invalidates
    the snapshot B reads.
    """
    conn = connect_hardened(":memory:")
    conn.row_factory = sqlite3.Row
    lock = threading.Lock()
    backend_a = SqliteKnowledgeBackend(connection=conn, lock=lock, enable_lexical=False)
    backend_b = SqliteKnowledgeBackend(connection=conn, lock=lock, enable_lexical=False)

    kb = _make_kb(backend_a)
    _ingest(backend_a, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])

    # Warm B's read of the shared cache.
    assert _search(backend_b, kb, [1.0, 0.0, 0.0]) == ["alpha"]
    # Both instances share ONE cache dict.
    assert backend_a._vec_cache is backend_b._vec_cache

    # Mutate through A.
    run_async(backend_a.delete_document(kb, "d1"))

    # B must NOT serve the deleted chunk.
    assert _search(backend_b, kb, [1.0, 0.0, 0.0]) == []
    run_async(backend_a.close())


def test_two_backends_one_connection_new_doc_visible_to_sibling() -> None:
    """A doc persisted through A appears in B's next search (no stale hidden matrix)."""
    conn = connect_hardened(":memory:")
    conn.row_factory = sqlite3.Row
    lock = threading.Lock()
    backend_a = SqliteKnowledgeBackend(connection=conn, lock=lock, enable_lexical=False)
    backend_b = SqliteKnowledgeBackend(connection=conn, lock=lock, enable_lexical=False)

    kb = _make_kb(backend_a)
    _ingest(backend_a, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    assert _search(backend_b, kb, [0.0, 1.0, 0.0]) == []  # warms B, nothing on y

    _ingest(backend_a, kb, doc_id="d2", source_uri="s2", text="beta",
            embedding=[0.0, 1.0, 0.0])
    assert _search(backend_b, kb, [0.0, 1.0, 0.0]) == ["beta"]
    run_async(backend_a.close())


# --------------------------------------------------- external / cross-process write


def test_external_connection_write_drops_stale_cache(tmp_path: Path) -> None:
    """A commit by ANOTHER connection (data_version bump) invalidates the cache.

    Models a second process / sidecar / raw ``sqlite3`` write to the shared ``.db``: the
    cache-free path re-read rows each search, so it observed the external write; the cache
    must re-decode instead of serving the deleted chunk.
    """
    db = str(tmp_path / "kb.db")
    backend = SqliteKnowledgeBackend(db, enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    assert _search(backend, kb, [1.0, 0.0, 0.0]) == ["alpha"]  # warms the cache

    # A SEPARATE connection deletes the chunk (external writer).
    ext = connect_hardened(db)
    try:
        ext.execute("DELETE FROM knowledge_chunks WHERE kb_id = ?", (kb,))
        ext.execute("DELETE FROM knowledge_documents WHERE kb_id = ?", (kb,))
        ext.commit()
    finally:
        ext.close()

    # The next search must observe the external delete, not the stale cached matrix.
    assert _search(backend, kb, [1.0, 0.0, 0.0]) == []
    run_async(backend.close())


def test_owned_connection_own_writes_do_not_spurious_clear() -> None:
    """Our OWN commits (same connection) don't trip the data_version full-clear.

    ``PRAGMA data_version`` only moves for OTHER connections, so a normal own-mutation path
    still uses the precise per-kb invalidation (not a blunt whole-cache clear), keeping the
    efficiency win for other KBs intact.
    """
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb_a = _make_kb(backend, "a")
    kb_b = _make_kb(backend, "b")
    _ingest(backend, kb_a, doc_id="a1", source_uri="sa", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    _ingest(backend, kb_b, doc_id="b1", source_uri="sb", text="beta",
            embedding=[0.0, 1.0, 0.0])
    # Warm both.
    assert _search(backend, kb_a, [1.0, 0.0, 0.0]) == ["alpha"]
    assert _search(backend, kb_b, [0.0, 1.0, 0.0]) == ["beta"]
    assert kb_a in backend._vec_cache and kb_b in backend._vec_cache

    # Mutate kb_a only; kb_b's cached snapshot must survive (precise invalidation).
    _ingest(backend, kb_a, doc_id="a2", source_uri="sa2", text="alpha2",
            embedding=[1.0, 0.0, 0.0])
    assert kb_b in backend._vec_cache  # not blown away by a spurious full clear
    run_async(backend.close())


def test_close_evicts_shared_cache_state() -> None:
    """Closing an owned backend evicts its per-connection state (no id-reuse leak)."""
    from himmy.services.knowledge import sqlite_backend as sb

    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    conn_id = id(backend._conn)
    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    _search(backend, kb, [1.0, 0.0, 0.0])
    assert conn_id in sb._VECTOR_CACHE_STATES
    run_async(backend.close())
    assert conn_id not in sb._VECTOR_CACHE_STATES
