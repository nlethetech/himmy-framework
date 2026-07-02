"""Tests for the decoded-vector cache on :class:`SqliteKnowledgeBackend`.

The dense-search paths (:meth:`search` / :meth:`dense_ranked`) used to ``json.loads`` every
stored embedding on EVERY query. These tests pin the efficiency win (eff-p1 ``kb-vector-cache``)
*and* its correctness invariants:

* a repeat dense search does NOT re-decode the embeddings (JSON-decode spy),
* a mutation (:meth:`persist_documents` / :meth:`delete_document` / :meth:`delete_kb`)
  invalidates precisely so the next search reflects the change — never a stale matrix,
* different ``kb_id`` snapshots are isolated (KB A never serves KB B's vectors),
* the cache is bounded (an LRU by kb_id; disable-able via env),
* concurrent search + mutate is safe (no crash, no stale-after-mutate result).
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from himmy.services.knowledge import sqlite_backend as sb
from himmy.services.knowledge.models import KnowledgeChunk, KnowledgeDocument
from himmy.services.knowledge.sqlite_backend import SqliteKnowledgeBackend
from tests.conftest import run_async


def _make_kb(backend: SqliteKnowledgeBackend, name: str = "kb") -> str:
    rec = run_async(
        backend.create_kb(
            workspace_id="ws", client_id="cl", name=name, vector_dim=3
        )
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
    chunk_id: str | None = None,
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
        chunk_id=chunk_id or f"{doc_id}-c0",
        document_id=doc_id,
        kb_id=kb_id,
        text=text,
        start_pos=0,
        end_pos=len(text),
        embedding=embedding,
    )
    run_async(backend.persist_documents(kb_id, [doc], [chunk]))


class _DecodeSpy:
    """Counts calls to :func:`json.loads` on the module under test."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real = json.loads

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(sb.json, "loads", counted)


def test_repeat_search_does_not_redecode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second identical dense search re-uses the cached matrix (no json.loads)."""
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="d1", source_uri="s1", text="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    _ingest(
        backend, kb, doc_id="d2", source_uri="s2", text="beta",
        embedding=[0.0, 1.0, 0.0],
    )

    query = [1.0, 0.0, 0.0]
    first = run_async(
        backend.search(kb, query, top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert [r.text for r in first] == ["alpha"]

    # Spy AFTER the warming search so we measure only the repeat path.
    spy = _DecodeSpy(monkeypatch)
    second = run_async(
        backend.search(kb, query, top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert spy.count == 0, "repeat search must not re-decode embeddings"

    # ...and the result is byte-identical to the cold search.
    assert [r.model_dump() for r in second] == [r.model_dump() for r in first]


def test_dense_ranked_shares_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fusion dense leg uses the same cache — a warm call decodes nothing."""
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="d1", source_uri="s1", text="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    # Warm the cache via search, then dense_ranked must not re-decode.
    run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    spy = _DecodeSpy(monkeypatch)
    ranked = run_async(
        backend.dense_ranked(kb, [1.0, 0.0, 0.0], top_k=5, metadata_filters=None)
    )
    assert ranked and ranked[0][0] == "d1-c0"
    assert spy.count == 0


def test_persist_invalidates_no_stale_results() -> None:
    """After persisting a new doc, the next search reflects it (no stale matrix)."""
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="d1", source_uri="s1", text="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    warm = run_async(
        backend.search(kb, [0.0, 1.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert [r.text for r in warm] == []  # nothing points at the y-axis yet

    _ingest(
        backend, kb, doc_id="d2", source_uri="s2", text="beta",
        embedding=[0.0, 1.0, 0.0],
    )
    after = run_async(
        backend.search(kb, [0.0, 1.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert [r.text for r in after] == ["beta"], "cache must not hide the new chunk"


def test_persist_replace_by_source_invalidates() -> None:
    """Re-ingesting a changed source replaces it AND refreshes the cache."""
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="d1", source_uri="s1", text="old",
        embedding=[1.0, 0.0, 0.0],
    )
    run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    # Same source_uri, new content/embedding -> replace-by-source.
    _ingest(
        backend, kb, doc_id="d1b", source_uri="s1", text="new",
        embedding=[1.0, 0.0, 0.0],
    )
    after = run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert [r.text for r in after] == ["new"]


def test_delete_document_invalidates_no_stale_results() -> None:
    """Deleting a document drops it from the very next search."""
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="d1", source_uri="s1", text="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    _ingest(
        backend, kb, doc_id="d2", source_uri="s2", text="beta",
        embedding=[0.0, 1.0, 0.0],
    )
    run_async(
        backend.search(kb, [1.0, 1.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert run_async(backend.delete_document(kb, "d1")) is True
    after = run_async(
        backend.search(kb, [1.0, 1.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert [r.text for r in after] == ["beta"], "deleted chunk must not survive in cache"


def test_delete_kb_invalidates() -> None:
    """Deleting a KB clears its cached matrix (no lingering vectors)."""
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="d1", source_uri="s1", text="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert run_async(backend.delete_kb(kb)) is True
    assert kb not in backend._vec_cache


def test_different_kb_isolated() -> None:
    """KB A never serves KB B's vectors — cache is keyed strictly by kb_id."""
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb_a = _make_kb(backend, name="A")
    kb_b = _make_kb(backend, name="B")
    _ingest(
        backend, kb_a, doc_id="a1", source_uri="sa", text="apple",
        embedding=[1.0, 0.0, 0.0],
    )
    _ingest(
        backend, kb_b, doc_id="b1", source_uri="sb", text="banana",
        embedding=[1.0, 0.0, 0.0],
    )
    ra = run_async(
        backend.search(kb_a, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    rb = run_async(
        backend.search(kb_b, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert [r.text for r in ra] == ["apple"]
    assert [r.text for r in rb] == ["banana"]
    # A mutation on B must not touch A's cached snapshot.
    a_snapshot = backend._vec_cache[kb_a]
    _ingest(
        backend, kb_b, doc_id="b2", source_uri="sb2", text="cherry",
        embedding=[0.0, 1.0, 0.0],
    )
    assert backend._vec_cache[kb_a] is a_snapshot


def test_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache evicts the least-recently-used kb_id past its bound."""
    monkeypatch.setenv("HIMMY_KB_VECTOR_CACHE_MAX_KBS", "2")
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kbs = []
    for i in range(3):
        kb = _make_kb(backend, name=f"kb{i}")
        _ingest(
            backend, kb, doc_id=f"d{i}", source_uri=f"s{i}", text=f"t{i}",
            embedding=[1.0, 0.0, 0.0],
        )
        run_async(
            backend.search(kb, [1.0, 0.0, 0.0], top_k=1, similarity_threshold=None, metadata_filters=None)
        )
        kbs.append(kb)
    assert len(backend._vec_cache) == 2
    # The first (LRU) kb was evicted; the two most recent remain.
    assert kbs[0] not in backend._vec_cache
    assert kbs[1] in backend._vec_cache
    assert kbs[2] in backend._vec_cache


def test_cache_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-positive bound disables caching (results still correct, always decoded)."""
    monkeypatch.setenv("HIMMY_KB_VECTOR_CACHE_MAX_KBS", "0")
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="d1", source_uri="s1", text="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert len(backend._vec_cache) == 0

    spy = _DecodeSpy(monkeypatch)
    res = run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert [r.text for r in res] == ["alpha"]
    assert spy.count > 0, "disabled cache must decode from SQLite each search"


def test_concurrent_search_and_mutate_safe() -> None:
    """Hammer search + persist/delete from many threads: no crash, no stale-after-final."""
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    for i in range(6):
        _ingest(
            backend, kb, doc_id=f"seed{i}", source_uri=f"seed{i}",
            text=f"seed{i}", embedding=[1.0, 0.0, 0.0],
        )

    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher() -> None:
        try:
            while not stop.is_set():
                run_async(
                    backend.search(
                        kb, [1.0, 0.0, 0.0], top_k=10,
                        similarity_threshold=None, metadata_filters=None,
                    )
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def mutator() -> None:
        try:
            for i in range(30):
                _ingest(
                    backend, kb, doc_id=f"m{i}", source_uri=f"m{i}",
                    text=f"m{i}", embedding=[0.0, 1.0, 0.0],
                )
                run_async(backend.delete_document(kb, f"m{i}"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    searchers = [threading.Thread(target=searcher) for _ in range(4)]
    mutators = [threading.Thread(target=mutator) for _ in range(2)]
    for t in searchers + mutators:
        t.start()
    for t in mutators:
        t.join()
    stop.set()
    for t in searchers:
        t.join()

    assert not errors, f"concurrent access raised: {errors}"

    # After all mutations, the final search must reflect committed state (only seeds remain
    # on the x-axis; every m* was deleted) — proving no stale matrix outlived the mutations.
    final = run_async(
        backend.search(
            kb, [1.0, 0.0, 0.0], top_k=20, similarity_threshold=None, metadata_filters=None
        )
    )
    assert sorted(r.text for r in final) == [f"seed{i}" for i in range(6)]


def test_fetch_publish_window_no_stale_publish() -> None:
    """A commit landing in the fetch->publish gap must NOT be masked by a stale publish.

    This exercises the exact TOCTOU the generation guard closes: a search fetches the OLD
    committed rows (and old generation), then — BEFORE it publishes — a mutation commits new
    rows and invalidates (a no-op pop, since nothing is published yet). Without the guard the
    search would then publish its stale snapshot and serve it indefinitely. We drive the
    interleaving deterministically by committing a query-ALIGNED chunk from inside the
    snapshot builder (after it read the old state), and assert the next fresh search sees it.
    """
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="seed", source_uri="seed", text="seed",
        embedding=[1.0, 0.0, 0.0],
    )
    # Warm, then evict so the next search takes the build->publish path (not a warm hit).
    run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    backend._invalidate_vector_cache(kb)

    real_build = backend._build_chunk_snapshot
    fired = {"done": False}

    def build_then_mutate(kb_id: str) -> Any:
        gen, snapshot = real_build(kb_id)  # reads OLD committed rows + old generation
        if not fired["done"]:
            fired["done"] = True
            # Commit a NEW, query-aligned chunk in the fetch->publish window. This bumps
            # _vec_gen and invalidates (no-op pop) BEFORE the caller publishes `snapshot`.
            _ingest(
                backend, kb, doc_id="late", source_uri="late", text="late",
                embedding=[1.0, 0.0, 0.0],
            )
        return gen, snapshot

    backend._build_chunk_snapshot = build_then_mutate  # type: ignore[method-assign]
    # This search built the stale snapshot, then the late chunk committed mid-flight.
    run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    backend._build_chunk_snapshot = real_build  # type: ignore[method-assign]

    # The stale snapshot must NOT have been cached over the newer committed state. The next
    # search rebuilds and MUST reflect the late chunk.
    after = run_async(
        backend.search(kb, [1.0, 0.0, 0.0], top_k=5, similarity_threshold=None, metadata_filters=None)
    )
    assert sorted(r.text for r in after) == ["late", "seed"], (
        "a commit in the fetch->publish window must not be masked by a stale publish"
    )


def test_concurrent_aligned_mutate_seen_at_barrier() -> None:
    """Query-aligned transient chunks: every committed chunk is observable, no stale mask.

    Unlike the orthogonal-vector stress, these transient chunks share the query axis, so a
    stale matrix (missing a committed chunk, or retaining a deleted one) would be DETECTABLE
    in the result. We commit a batch, assert a fresh search sees them all, delete them, assert
    a fresh search sees none — hammered concurrently with background searchers.
    """
    backend = SqliteKnowledgeBackend(":memory:", enable_lexical=False)
    kb = _make_kb(backend)
    _ingest(
        backend, kb, doc_id="base", source_uri="base", text="base",
        embedding=[1.0, 0.0, 0.0],
    )

    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher() -> None:
        try:
            while not stop.is_set():
                run_async(
                    backend.search(
                        kb, [1.0, 0.0, 0.0], top_k=50,
                        similarity_threshold=None, metadata_filters=None,
                    )
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    searchers = [threading.Thread(target=searcher) for _ in range(4)]
    for t in searchers:
        t.start()
    try:
        for rnd in range(15):
            ids = [f"r{rnd}_{j}" for j in range(4)]
            for cid in ids:
                _ingest(
                    backend, kb, doc_id=cid, source_uri=cid, text=cid,
                    embedding=[1.0, 0.0, 0.0],
                )
            seen = run_async(
                backend.search(
                    kb, [1.0, 0.0, 0.0], top_k=50,
                    similarity_threshold=None, metadata_filters=None,
                )
            )
            texts = {r.text for r in seen}
            assert set(ids) <= texts, f"round {rnd}: committed chunks missing (stale): {set(ids) - texts}"
            for cid in ids:
                run_async(backend.delete_document(kb, cid))
            seen2 = run_async(
                backend.search(
                    kb, [1.0, 0.0, 0.0], top_k=50,
                    similarity_threshold=None, metadata_filters=None,
                )
            )
            texts2 = {r.text for r in seen2}
            assert not (set(ids) & texts2), f"round {rnd}: deleted chunks lingered (stale): {set(ids) & texts2}"
    finally:
        stop.set()
        for t in searchers:
            t.join()

    assert not errors, f"concurrent access raised: {errors}"
    final = run_async(
        backend.search(
            kb, [1.0, 0.0, 0.0], top_k=50, similarity_threshold=None, metadata_filters=None
        )
    )
    assert [r.text for r in final] == ["base"]
