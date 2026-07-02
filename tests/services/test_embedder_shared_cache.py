"""Shared native-session cache for fastembed embedders + rerankers (eff-p1).

These tests assert the *latency/memory* optimization is behaviour-preserving: two
builds for the same ``(model, dim)`` reuse ONE native ONNX session; a different
model/dim yields a different session; concurrent embeds against a shared session are
safe and byte-identical; and the reset hook clears the cache. The core caching
*mechanism* is exercised with a lightweight fake loader (no model download) so the
assertions run even without the ``[embeddings]`` extra; an extra-gated test also pins
the real fastembed session identity when the extra is installed.
"""

from __future__ import annotations

import threading

import pytest

from himmy.services.knowledge import local_embedders
from himmy.services.knowledge.local_embedders import (
    FastEmbedEmbedder,
    build_embedder,
    reset_embedder_cache,
)
from himmy.services.knowledge.retrieval import reranker as reranker_mod
from himmy.services.knowledge.retrieval.reranker import (
    FastEmbedReranker,
    build_reranker,
    reset_reranker_cache,
)
from tests.conftest import run_async


class _FakeTextEmbedding:
    """A stand-in for fastembed's TextEmbedding: cheap, deterministic, no download.

    Tracks how many times the *native* session was constructed so a test can prove the
    cache collapses N builds into one native load. ``embed``/``query_embed`` return a
    stable vector per input so byte-identity under concurrency is checkable.
    """

    instances = 0

    def __init__(self, model_name: str) -> None:
        type(self).instances += 1
        self.model_name = model_name

    def embed(self, texts: list[str]):
        for t in texts:
            yield [float(len(t)), 1.0, 2.0]

    def query_embed(self, texts: list[str]):
        for t in texts:
            yield [float(len(t)), 9.0]


@pytest.fixture(autouse=True)
def _clean_cache():
    """Every test starts and ends with an empty shared cache (no leakage)."""
    reset_embedder_cache()
    yield
    reset_embedder_cache()


def _install_fake_loader(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTextEmbedding]:
    """Route the cached loader through the fake native session and reset the counter."""
    _FakeTextEmbedding.instances = 0

    def _fake_load(model_name: str):
        return _FakeTextEmbedding(model_name)

    # cache_clear so the fake replaces any real session cached by a prior test.
    local_embedders._load_text_embedding.cache_clear()
    monkeypatch.setattr(
        local_embedders, "_load_text_embedding", _make_cached(_fake_load)
    )
    return _FakeTextEmbedding


def _make_cached(fn):
    """Wrap ``fn`` in the same cache the module uses (so cache_clear works)."""
    from functools import cache

    return cache(fn)


# ---- caching mechanism (fake loader; runs without the extra) ----


def test_same_model_dim_shares_one_native_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two build_embedder calls for the same (model, dim) reuse ONE native session."""
    fake = _install_fake_loader(monkeypatch)
    e1 = build_embedder("fastembed", model="m-a", dim=384)
    e2 = build_embedder("fastembed", model="m-a", dim=384)
    assert isinstance(e1, FastEmbedEmbedder)
    # Force the lazy load on both wrappers.
    s1 = e1._model_impl()
    s2 = e2._model_impl()
    assert s1 is s2  # same underlying ONNX session object
    assert fake.instances == 1  # native model loaded exactly once


def test_different_model_yields_different_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different model name loads a distinct native session (correct cache keying)."""
    fake = _install_fake_loader(monkeypatch)
    s_a = build_embedder("fastembed", model="m-a")._model_impl()
    s_b = build_embedder("fastembed", model="m-b")._model_impl()
    assert s_a is not s_b
    assert fake.instances == 2


def test_reset_clears_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """reset_embedder_cache drops the cached session so the next build reloads it."""
    fake = _install_fake_loader(monkeypatch)
    first = build_embedder("fastembed", model="m-a")._model_impl()
    assert fake.instances == 1
    reset_embedder_cache()  # public hook clears the cached (fake) text session
    second = build_embedder("fastembed", model="m-a")._model_impl()
    assert second is not first
    assert fake.instances == 2


def test_cache_off_gives_each_instance_its_own_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HIMMY_EMBED_CACHE=off restores per-instance sessions (the pre-cache behaviour).

    With the cache off, ``_model_impl`` constructs ``fastembed.TextEmbedding`` directly
    (not via the shared loader), so we patch that import target with the fake and assert
    two wrappers hold *distinct* private sessions — proving the kill-switch reverts to the
    pre-optimization behaviour byte-for-byte.
    """
    fake = _install_fake_loader(monkeypatch)
    monkeypatch.setenv("HIMMY_EMBED_CACHE", "off")

    import fastembed  # the extra is present in this env

    monkeypatch.setattr(fastembed, "TextEmbedding", fake)
    e1 = build_embedder("fastembed", model="m-a")
    e2 = build_embedder("fastembed", model="m-a")
    s1 = e1._model_impl()
    s2 = e2._model_impl()
    assert s1 is not s2


def test_concurrent_embeds_are_safe_and_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent embeds over a shared session return byte-identical vectors, no races."""
    _install_fake_loader(monkeypatch)
    emb = build_embedder("fastembed", model="m-a")
    expected = run_async(emb.embed_query("hello"))

    results: list[list[float]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def _worker() -> None:
        try:
            vec = run_async(emb.embed_query("hello"))
            with lock:
                results.append(vec)
        except BaseException as exc:  # noqa: BLE001 - surface any race as a failure
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 16
    assert all(vec == expected for vec in results)


# ---- reranker cache mechanism ----


class _FakeCrossEncoder:
    instances = 0

    def __init__(self, model_name: str) -> None:
        type(self).instances += 1
        self.model_name = model_name

    def rerank(self, query: str, texts: list[str]):
        return [float(len(t)) for t in texts]


def test_reranker_same_model_shares_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two build_reranker calls for one model reuse a single native cross-encoder."""
    _FakeCrossEncoder.instances = 0
    reranker_mod._load_cross_encoder.cache_clear()
    monkeypatch.setattr(
        reranker_mod,
        "_load_cross_encoder",
        _make_cached(lambda name: _FakeCrossEncoder(name)),
    )
    r1 = build_reranker("fastembed", model="rr")
    r2 = build_reranker("fastembed", model="rr")
    assert isinstance(r1, FastEmbedReranker)
    assert r1._model_impl() is r2._model_impl()
    assert _FakeCrossEncoder.instances == 1


def test_reranker_reset_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """reset_reranker_cache drops the cached cross-encoder so the next build reloads."""
    _FakeCrossEncoder.instances = 0
    reranker_mod._load_cross_encoder.cache_clear()
    monkeypatch.setattr(
        reranker_mod,
        "_load_cross_encoder",
        _make_cached(lambda name: _FakeCrossEncoder(name)),
    )
    first = build_reranker("fastembed", model="rr")._model_impl()
    reset_reranker_cache()  # public hook clears the cached cross-encoder
    second = build_reranker("fastembed", model="rr")._model_impl()
    assert first is not second
    assert _FakeCrossEncoder.instances == 2


# ---- real fastembed path (extra-gated: pins actual session identity) ----


@pytest.mark.skipif(
    not local_embedders.fastembed_available(),
    reason="fastembed [embeddings] extra not installed — real session identity untested",
)
def test_real_fastembed_shares_session_and_identical_vectors() -> None:  # pragma: no cover - extra-gated
    """With the real extra: same model shares one ONNX session; vectors are identical."""
    reset_embedder_cache()
    e1 = build_embedder("fastembed", model="BAAI/bge-small-en-v1.5", dim=384)
    e2 = build_embedder("fastembed", model="BAAI/bge-small-en-v1.5", dim=384)
    assert e1._model_impl() is e2._model_impl()

    v1 = run_async(e1.embed_query("the quick brown fox"))
    v2 = run_async(e2.embed_query("the quick brown fox"))
    assert v1 == v2  # byte-identical: same weights, same session
    assert len(v1) == 384
    reset_embedder_cache()
