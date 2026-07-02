"""Red-team r1: concurrent cold-start must build a shared native session EXACTLY ONCE.

``functools.cache`` runs the wrapped loader OUTSIDE its own lock, so two threads that miss
the cache for the same model at cold start would each construct a heavy ONNX session (only
to discard one) — defeating the "save the duplicate load" goal. The build lock around the
cache lookup+build in ``_model_impl`` makes first-load load-exactly-once. These tests count
constructions of a fake session under a barrier so no real ONNX model is built.
"""

from __future__ import annotations

import threading

import pytest

from himmy.services.knowledge import local_embedders
from himmy.services.knowledge.local_embedders import FastEmbedEmbedder
from himmy.services.knowledge.retrieval import reranker as rr


class _CountingBuild:
    """A fake native-session class whose __init__ is counted and briefly blocks.

    The block widens the race window so two un-serialised builders would both be inside
    __init__ concurrently; under the build lock only one construction ever happens.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        self._lock = threading.Lock()

    def make(self, **_kwargs: object) -> object:
        with self._lock:
            self.count += 1
        # Simulate the slow native load so concurrent missers would overlap.
        threading.Event().wait(0.05)
        return object()


def _run_concurrent(fn: object, n: int = 8) -> None:
    barrier = threading.Barrier(n)

    def _worker() -> None:
        barrier.wait()
        fn()  # type: ignore[operator]

    threads = [threading.Thread(target=_worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_text_embedding_built_once_under_concurrent_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first embeds for one model build the TextEmbedding session once."""
    local_embedders.reset_embedder_cache()
    counter = _CountingBuild(monkeypatch)

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", counter.make)
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: True)

    def _load() -> None:
        FastEmbedEmbedder(model="model-x", dim=3)._model_impl()

    _run_concurrent(_load)
    assert counter.count == 1  # exactly one native session built despite the race
    local_embedders.reset_embedder_cache()


def test_cross_encoder_built_once_under_concurrent_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first reranks for one model build the cross-encoder session once."""
    rr.reset_reranker_cache()
    counter = _CountingBuild(monkeypatch)

    from fastembed.rerank import cross_encoder as ce

    monkeypatch.setattr(ce, "TextCrossEncoder", counter.make)

    def _load() -> None:
        rr.FastEmbedReranker(model="rr-x")._model_impl()

    _run_concurrent(_load)
    assert counter.count == 1
    rr.reset_reranker_cache()
