"""Regression tests for the KnowledgeBase deployment ticket (found in Daybook).

Two acceptance behaviours are pinned here:

* **Fix A — non-blocking embed.** ``FastEmbedEmbedder`` must run its (GIL-releasing)
  ONNX work in a worker thread via ``asyncio.to_thread`` so a host async server's event
  loop stays responsive during an index build, instead of freezing for the whole build.
  We exercise the *real* ``FastEmbedEmbedder`` code path with an injected stand-in model
  whose ``embed`` sleeps (``time.sleep`` releases the GIL exactly like onnxruntime), so
  the test is fast and deterministic yet would FAIL if the ``to_thread`` offload were
  removed (the embed would then run on the loop thread and stall a concurrent heartbeat).

* **Fix B — disk persistence.** A ``SqliteKnowledgeBackend`` must persist the index so an
  app restart LOADS it from disk instead of re-embedding the corpus. We simulate a restart
  by reopening the same ``.db`` file and assert (1) a re-ingest of the identical corpus
  spends ZERO document embeddings, and (2) search still returns the correct cited chunk,
  identical to the freshly-built index.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from himmy.services.knowledge import DeterministicEmbedder, KnowledgeBase
from himmy.services.knowledge.local_embedders import FastEmbedEmbedder
from himmy.services.knowledge.sqlite_backend import SqliteKnowledgeBackend
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async

# --------------------------------------------------------------------------- Fix A


class _GilReleasingModel:
    """Stand-in for fastembed's ``TextEmbedding`` whose calls sleep (releasing the GIL).

    Records the thread each call ran on so the test can assert the work happened OFF the
    event-loop thread. ``time.sleep`` mimics onnxruntime releasing the GIL during native
    inference, which is the precise property that makes ``asyncio.to_thread`` free the loop.
    """

    def __init__(self, *, dim: int = 8, delay: float = 0.4) -> None:
        self.dim = dim
        self.delay = delay
        self.embed_thread: threading.Thread | None = None
        self.query_thread: threading.Thread | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_thread = threading.current_thread()
        time.sleep(self.delay)
        return [[0.1] * self.dim for _ in texts]

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        self.query_thread = threading.current_thread()
        time.sleep(self.delay)
        return [[0.1] * self.dim for _ in texts]


def test_fastembed_embed_documents_runs_off_the_event_loop() -> None:
    """A real FastEmbedEmbedder.embed_documents must not block the loop during a build."""

    async def scenario() -> None:
        emb = FastEmbedEmbedder(dim=8)
        model = _GilReleasingModel(delay=0.4)
        emb._impl = model  # inject so no real model is downloaded
        loop_thread = threading.current_thread()

        gaps: list[float] = []
        running = [True]

        async def heartbeat() -> None:
            loop = asyncio.get_running_loop()
            prev = loop.time()
            while running[0]:
                await asyncio.sleep(0.02)
                now = loop.time()
                gaps.append(now - prev)
                prev = now

        hb = asyncio.create_task(heartbeat())
        vecs = await emb.embed_documents(["alpha", "beta", "gamma"])
        running[0] = False
        await hb

        assert len(vecs) == 3 and len(vecs[0]) == 8
        # The synchronous embed ran on a worker thread, NOT the event-loop thread.
        assert model.embed_thread is not None
        assert model.embed_thread is not loop_thread
        # The loop stayed responsive during the ~0.4s embed (no ~0.4s stall).
        assert max(gaps) < 0.2, f"event loop stalled {max(gaps):.3f}s during embed"

    run_async(scenario())


def test_fastembed_embed_query_runs_off_the_event_loop() -> None:
    """embed_query is also offloaded so per-query embedding never blocks the loop."""

    async def scenario() -> None:
        emb = FastEmbedEmbedder(dim=8)
        model = _GilReleasingModel(delay=0.3)
        emb._impl = model
        loop_thread = threading.current_thread()
        vec = await emb.embed_query("a query")
        assert len(vec) == 8
        assert model.query_thread is not None
        assert model.query_thread is not loop_thread

    run_async(scenario())


# --------------------------------------------------------------------------- Fix B


class _CountingEmbedder(DeterministicEmbedder):
    """DeterministicEmbedder that counts document-embed calls (fast, no network)."""

    def __init__(self) -> None:
        super().__init__(dim=64)
        self.doc_calls = 0
        self.doc_texts = 0

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        self.doc_calls += 1
        self.doc_texts += len(texts)
        return await super().embed_documents(texts)


_CORPUS = [
    ("doc://mitochondria", "Mitochondria are the powerhouse of the cell."),
    ("doc://photosynthesis", "Photosynthesis converts sunlight into chemical energy."),
    ("doc://black-holes", "A black hole is a region of spacetime with extreme gravity."),
]
_QUERY = "powerhouse of the cell"


async def _close(obj: Any) -> None:
    """Best-effort close of a backend/storage (awaits close/aclose if present)."""
    for name in ("close", "aclose"):
        fn = getattr(obj, name, None)
        if fn is not None:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            return


async def _ingest_corpus(kb: KnowledgeBase, kb_id: str) -> None:
    for uri, text in _CORPUS:
        await kb.ingest_text(kb_id, text, source_uri=uri)


def test_restart_reingest_skips_reembedding(tmp_path) -> None:
    """Reopening the same .db and re-ingesting the identical corpus embeds NOTHING."""
    db = str(tmp_path / "knowledge.db")

    async def scenario() -> None:
        # --- cold build (first launch) ---
        emb1 = _CountingEmbedder()
        backend1 = SqliteKnowledgeBackend(db)
        kb1 = KnowledgeBase(storage=StorageService(), embedder=emb1, backend=backend1)
        rec = await kb1.create_kb(
            workspace_id="w", client_id="c", name="kb", vector_dim=64
        )
        await _ingest_corpus(kb1, rec.kb_id)
        assert emb1.doc_calls > 0  # the cold build actually embedded
        cold = await kb1.search(rec.kb_id, _QUERY, top_k=3)
        assert cold and "powerhouse" in (cold[0].text or "")
        await _close(backend1)

        # --- restart (reopen the same file in a fresh KB) ---
        emb2 = _CountingEmbedder()
        backend2 = SqliteKnowledgeBackend(db)
        kb2 = KnowledgeBase(storage=StorageService(), embedder=emb2, backend=backend2)
        resolved = await kb2.resolve_kb(workspace_id="w", client_id="c", name="kb")
        assert resolved is not None and resolved.kb_id == rec.kb_id
        await _ingest_corpus(kb2, resolved.kb_id)
        assert emb2.doc_calls == 0  # loaded from disk — NOTHING re-embedded
        warm = await kb2.search(resolved.kb_id, _QUERY, top_k=3)
        assert warm and "powerhouse" in (warm[0].text or "")
        # Search fidelity is preserved across the restart.
        assert [r.text for r in warm] == [r.text for r in cold]
        await _close(backend2)

    run_async(scenario())


def test_restart_search_loads_persisted_index_without_reingest(tmp_path) -> None:
    """After a restart, a plain search returns correct results loaded from disk."""
    db = str(tmp_path / "knowledge.db")

    async def scenario() -> None:
        emb1 = _CountingEmbedder()
        backend1 = SqliteKnowledgeBackend(db)
        kb1 = KnowledgeBase(storage=StorageService(), embedder=emb1, backend=backend1)
        rec = await kb1.create_kb(
            workspace_id="w", client_id="c", name="kb", vector_dim=64
        )
        await _ingest_corpus(kb1, rec.kb_id)
        cold = await kb1.search(rec.kb_id, _QUERY, top_k=3)
        await _close(backend1)

        # Reopen and search WITHOUT re-ingesting — the index must come from disk.
        emb2 = _CountingEmbedder()
        backend2 = SqliteKnowledgeBackend(db)
        kb2 = KnowledgeBase(storage=StorageService(), embedder=emb2, backend=backend2)
        resolved = await kb2.resolve_kb(workspace_id="w", client_id="c", name="kb")
        assert resolved is not None
        warm = await kb2.search(resolved.kb_id, _QUERY, top_k=3)
        assert warm and "powerhouse" in (warm[0].text or "")
        assert [r.text for r in warm] == [r.text for r in cold]
        assert emb2.doc_calls == 0  # a search never re-embeds documents
        await _close(backend2)

    run_async(scenario())


def test_shared_storage_service_persists_index_across_restart(tmp_path) -> None:
    """The production wiring (SqliteStorageService.knowledge_backend) also persists."""
    db = str(tmp_path / "app.db")

    async def scenario() -> None:
        emb1 = _CountingEmbedder()
        storage1 = SqliteStorageService(db)
        kb1 = KnowledgeBase(
            storage=storage1, embedder=emb1, backend=storage1.knowledge_backend()
        )
        rec = await kb1.create_kb(
            workspace_id="w", client_id="c", name="kb", vector_dim=64
        )
        await _ingest_corpus(kb1, rec.kb_id)
        assert emb1.doc_calls > 0
        await _close(storage1)

        emb2 = _CountingEmbedder()
        storage2 = SqliteStorageService(db)
        kb2 = KnowledgeBase(
            storage=storage2, embedder=emb2, backend=storage2.knowledge_backend()
        )
        resolved = await kb2.resolve_kb(workspace_id="w", client_id="c", name="kb")
        assert resolved is not None and resolved.kb_id == rec.kb_id
        await _ingest_corpus(kb2, resolved.kb_id)
        assert emb2.doc_calls == 0  # restart loaded the index from the shared .db
        warm = await kb2.search(resolved.kb_id, _QUERY, top_k=3)
        assert warm and "powerhouse" in (warm[0].text or "")
        await _close(storage2)

    run_async(scenario())
