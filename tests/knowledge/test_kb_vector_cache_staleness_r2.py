"""Red-team round-2 regressions for KB vector-cache leaks + external-write TOCTOU.

Two confirmed bugs from the vector-cache efficiency work:

* ``SqliteStorageService.close()`` never evicted the per-connection cache state, so a
  knowledge backend built from a storage service (``_owns_conn=False``, whose own close is
  a no-op for eviction) leaked a closed connection + decoded matrices from the process-wide
  ``_VECTOR_CACHE_STATES`` forever — an unbounded leak under storage-service churn.
* the fetch->publish generation guard only tracked THIS connection's own writes (``gen``);
  an EXTERNAL connection committing a DELETE between the snapshot build and the publish was
  invisible (it never moves ``gen``), pinning a stale/deleted matrix for a search window.
  The fix folds ``PRAGMA data_version`` into the publish check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from himmy.core.sqlite_util import connect_hardened
from himmy.services.knowledge import sqlite_backend as sb
from himmy.services.knowledge.models import KnowledgeChunk, KnowledgeDocument
from himmy.services.knowledge.sqlite_backend import SqliteKnowledgeBackend
from himmy.services.storage.sqlite import SqliteStorageService
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


# ------------------------------------- bug #5: storage-service close evicts cache state


def test_storage_service_close_evicts_shared_cache_state(tmp_path: Path) -> None:
    """A knowledge backend from a storage service must not leak its cache entry on close.

    The backend shares the service's connection (``_owns_conn=False``), so its OWN close is
    a no-op for eviction. The storage service is the true owner and must drop the entry.
    """
    db = str(tmp_path / "kb.db")
    svc = SqliteStorageService(db)
    backend = svc.knowledge_backend(enable_lexical=False)
    conn_id = id(backend._conn)

    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    _search(backend, kb, [1.0, 0.0, 0.0])  # warms the cache -> inserts the state entry
    assert conn_id in sb._VECTOR_CACHE_STATES

    run_async(svc.close())
    # The service is the connection owner: closing it must reclaim the cache state, else
    # the closed connection + decoded matrices stay pinned forever.
    assert conn_id not in sb._VECTOR_CACHE_STATES


def test_repeated_storage_service_churn_does_not_accumulate_state(
    tmp_path: Path,
) -> None:
    """Create/close many storage services -> no residual cache-state entries from them."""
    seen_conn_ids: list[int] = []
    for i in range(8):
        db = str(tmp_path / f"kb_{i}.db")
        svc = SqliteStorageService(db)
        backend = svc.knowledge_backend(enable_lexical=False)
        seen_conn_ids.append(id(backend._conn))
        kb = _make_kb(backend)
        _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
                embedding=[1.0, 0.0, 0.0])
        _search(backend, kb, [1.0, 0.0, 0.0])
        run_async(svc.close())
    # None of the closed services' connections may still pin a cache-state entry.
    for conn_id in seen_conn_ids:
        assert conn_id not in sb._VECTOR_CACHE_STATES


# ------------------- bug #6: external write in the fetch->publish window is not masked


def test_external_delete_in_fetch_publish_window_not_published(tmp_path: Path) -> None:
    """An EXTERNAL commit racing the in-flight build must NOT be masked by a stale publish.

    Round-1 only closed the OWN-write TOCTOU (``gen``). Here a DIFFERENT connection deletes
    the chunk AFTER the snapshot is fetched but BEFORE it is published; that bumps
    ``PRAGMA data_version`` (NOT ``gen``), so without the data_version publish-check the
    deleted matrix would be pinned and served for a search window. The two-signal guard must
    refuse to publish it, so the very NEXT search reflects the delete.
    """
    db = str(tmp_path / "kb.db")
    backend = SqliteKnowledgeBackend(db, enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    _search(backend, kb, [1.0, 0.0, 0.0])  # warm
    backend._invalidate_vector_cache(kb)  # force the build->publish path next time

    real_build = backend._build_chunk_snapshot
    fired = {"done": False}

    def build_then_external_delete(kb_id: str) -> Any:
        # Reads the OLD (pre-delete) rows + old gen + old data_version.
        gen, data_version, snapshot = real_build(kb_id)
        if not fired["done"]:
            fired["done"] = True
            # An EXTERNAL connection deletes the chunk in the fetch->publish window. This
            # bumps the file's data_version (visible to our connection) but NOT our ``gen``.
            ext = connect_hardened(db)
            try:
                ext.execute("DELETE FROM knowledge_chunks WHERE kb_id = ?", (kb_id,))
                ext.execute("DELETE FROM knowledge_documents WHERE kb_id = ?", (kb_id,))
                ext.commit()
            finally:
                ext.close()
        return gen, data_version, snapshot

    backend._build_chunk_snapshot = build_then_external_delete  # type: ignore[method-assign]
    # This search builds the stale (pre-delete) snapshot; the external delete lands
    # mid-flight. The data_version publish-check must refuse to cache it.
    run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None,
                       metadata_filters=None)
    )
    backend._build_chunk_snapshot = real_build  # type: ignore[method-assign]

    # The stale/deleted snapshot must NOT have been cached. The next search rebuilds from
    # the newer (post-delete) committed state and returns nothing.
    after = _search(backend, kb, [1.0, 0.0, 0.0])
    assert after == [], (
        "an external delete in the fetch->publish window must not be masked by a stale publish"
    )
    run_async(backend.close())


def test_build_snapshot_returns_gen_and_data_version(tmp_path: Path) -> None:
    """``_build_chunk_snapshot`` returns ``(gen, data_version, snapshot)`` atomically."""
    db = str(tmp_path / "kb.db")
    backend = SqliteKnowledgeBackend(db, enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    gen, data_version, snapshot = backend._build_chunk_snapshot(kb)
    assert isinstance(gen, int)
    assert isinstance(data_version, int)
    assert [c.text for c in snapshot] == ["alpha"]
    run_async(backend.close())
