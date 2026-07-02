"""Red-team round-3 regressions for KB vector-cache lifecycle + version tracking.

Two confirmed bugs from the vector-cache efficiency work:

* An OWNED-connection ``SqliteKnowledgeBackend`` (``path=...``) garbage-collected WITHOUT an
  awaited ``close()`` leaked forever: the process-global ``_VECTOR_CACHE_STATES`` strong-
  references its connection, so the connection could never self-collect and no ``__del__``
  could run. The fix registers a ``weakref.finalize`` that evicts the entry when the backend
  is dropped, releasing the pinned fd + decoded matrices.
* ``_vec_state.data_version`` was advanced ONLY in ``_drop_cache_on_external_write``, never
  on the publish path. If an external write raced the in-flight build, a snapshot was
  published at the newer version while the observed baseline stayed stale — so the NEXT
  search's drop-check saw a version mismatch and cleared the WHOLE cache on a false positive.
  The fix syncs the baseline forward (monotonically) on publish.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from himmy.core.sqlite_util import connect_hardened
from himmy.services.knowledge import sqlite_backend as sb
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


# --------------------------- bug: owned backend dropped without close() leaks cache state


def test_owned_backend_dropped_without_close_evicts_via_finalizer(
    tmp_path: Path,
) -> None:
    """A path-owning backend GC'd WITHOUT close() must not pin its cache-state entry.

    The global ``_VECTOR_CACHE_STATES`` strong-references the connection, so nothing else
    can reclaim it — the weakref.finalize on the backend is the only self-healing path.
    """
    db = str(tmp_path / "kb.db")
    backend = SqliteKnowledgeBackend(db, enable_lexical=False)
    conn_id = id(backend._conn)
    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    _search(backend, kb, [1.0, 0.0, 0.0])  # warm -> inserts the state entry
    assert conn_id in sb._VECTOR_CACHE_STATES

    # Drop the ONLY strong reference to the backend WITHOUT awaiting close().
    del backend
    gc.collect()

    assert conn_id not in sb._VECTOR_CACHE_STATES, (
        "an owned backend dropped without close() must evict its cache state via finalize"
    )


def test_many_owned_backends_dropped_without_close_do_not_accumulate(
    tmp_path: Path,
) -> None:
    """Churning many path-owning backends without close() leaves no residual entries."""
    seen: list[int] = []
    for i in range(12):
        backend = SqliteKnowledgeBackend(
            str(tmp_path / f"kb_{i}.db"), enable_lexical=False
        )
        seen.append(id(backend._conn))
        kb = _make_kb(backend)
        _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
                embedding=[1.0, 0.0, 0.0])
        _search(backend, kb, [1.0, 0.0, 0.0])
        del backend
        gc.collect()
    residual = [cid for cid in seen if cid in sb._VECTOR_CACHE_STATES]
    assert residual == [], f"leaked cache-state entries: {residual}"


def test_explicit_close_still_evicts_and_cancels_finalizer(tmp_path: Path) -> None:
    """The normal close() path still evicts deterministically (finalizer detached)."""
    db = str(tmp_path / "kb.db")
    backend = SqliteKnowledgeBackend(db, enable_lexical=False)
    conn_id = id(backend._conn)
    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    _search(backend, kb, [1.0, 0.0, 0.0])
    assert conn_id in sb._VECTOR_CACHE_STATES
    assert backend._cache_finalizer is not None and backend._cache_finalizer.alive

    run_async(backend.close())
    assert conn_id not in sb._VECTOR_CACHE_STATES
    assert not backend._cache_finalizer.alive  # detached by close()


# ------------------- bug: data_version not advanced on publish -> spurious full clears


def test_publish_advances_observed_data_version(tmp_path: Path) -> None:
    """A snapshot built at a NEWER external version syncs the observed baseline forward.

    An external write lands in the fetch->publish window so the snapshot is built at a newer
    ``data_version`` than the baseline. The publish must advance ``_vec_state.data_version``
    to that newer value, otherwise the NEXT drop-check clears the whole cache on a false
    positive (nothing changed since the snapshot was built).
    """
    db = str(tmp_path / "kb.db")
    backend = SqliteKnowledgeBackend(db, enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    _search(backend, kb, [1.0, 0.0, 0.0])  # warm; sets an initial baseline
    backend._invalidate_vector_cache(kb)  # force the build->publish path next search

    real_build = backend._build_chunk_snapshot
    fired = {"done": False}

    def external_touch_then_build(kb_id: str) -> Any:
        # Reproduce the bug's precondition: the drop-check has ALREADY run (baseline == V0,
        # no clear). NOW an unrelated external commit bumps the file data_version to V1
        # BEFORE the build fetches its snapshot, so the snapshot is (correctly) built and
        # published at V1 while the observed baseline still reads V0. The publish must sync
        # the baseline forward to V1, else the next drop-check false-clears the whole cache.
        if not fired["done"]:
            fired["done"] = True
            ext = connect_hardened(db)
            try:
                ext.execute("CREATE TABLE IF NOT EXISTS _unrelated (id INTEGER)")
                ext.execute("INSERT INTO _unrelated (id) VALUES (1)")
                ext.commit()
            finally:
                ext.close()
        return real_build(kb_id)

    backend._build_chunk_snapshot = external_touch_then_build  # type: ignore[method-assign]
    _search(backend, kb, [1.0, 0.0, 0.0])  # builds + publishes at the newer version
    backend._build_chunk_snapshot = real_build  # type: ignore[method-assign]

    # The published snapshot must be cached AND the observed baseline advanced to it, so a
    # subsequent drop-check sees NO change and keeps the cache warm (no spurious clear).
    cached = backend._vec_cache.get(kb)
    assert cached is not None, "the snapshot built at the newer version must be published"
    published_version = backend._vec_state.data_version
    assert published_version is not None

    # A follow-up search with NO further external write must NOT clear the cache.
    same_obj = backend._vec_cache.get(kb)
    _search(backend, kb, [1.0, 0.0, 0.0])
    assert backend._vec_cache.get(kb) is same_obj, (
        "no external write happened, so the cached matrix must survive (no false-positive clear)"
    )
    assert backend._vec_state.data_version == published_version
    run_async(backend.close())


def test_data_version_baseline_never_goes_backwards(tmp_path: Path) -> None:
    """Publish advances the baseline with ``max`` — a concurrent HIGHER value is not lowered.

    The drop-check normally resets the baseline to the observed version, so to isolate the
    publish path's monotonic ``max`` we no-op the drop-check and force a higher baseline than
    the snapshot's version; the publish must keep the higher value (never regress it).
    """
    db = str(tmp_path / "kb.db")
    backend = SqliteKnowledgeBackend(db, enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(backend, kb, doc_id="d1", source_uri="s1", text="alpha",
            embedding=[1.0, 0.0, 0.0])
    _search(backend, kb, [1.0, 0.0, 0.0])
    backend._invalidate_vector_cache(kb)
    # Suppress the drop-check's own baseline reset so the publish's max() is what's exercised.
    backend._drop_cache_on_external_write = lambda: None  # type: ignore[method-assign]
    with backend._vec_state.cache_lock:
        backend._vec_state.data_version = 10**9  # a higher concurrent observation
    _search(backend, kb, [1.0, 0.0, 0.0])  # builds at a small version, publishes
    assert backend._vec_state.data_version is not None
    assert backend._vec_state.data_version >= 10**9, (
        "publish must not regress a higher observed baseline (monotonic max)"
    )
    run_async(backend.close())
