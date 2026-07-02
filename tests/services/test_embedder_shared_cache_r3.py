"""Red-team round-3 regressions for the shared-embedder cache (eff-p1 safe-r3).

Two confirmed hardening gaps from the shared native-session work:

* the ``functools.cache`` on ``_load_text_embedding`` / ``_load_cross_encoder`` was
  UNBOUNDED — every distinct model name ever seen pinned its heavy ONNX session forever, so
  RSS grew without bound in a process reconfigured across many models. The fix bounds them
  with a small ``lru_cache`` (env-overridable) that evicts the least-recently-used session.
* a large SHARED-session ``embed_documents`` held ``_TEXT_EMBED_LOCK`` for the WHOLE batch,
  so a huge ingest head-of-line-blocked a concurrent latency-critical ``embed_query`` on the
  same session. The fix releases + re-acquires the lock per sub-batch (byte-identical output,
  since fastembed embeds each text independently).
"""

from __future__ import annotations

import threading

import pytest

from himmy.services.knowledge import local_embedders
from himmy.services.knowledge.local_embedders import (
    build_embedder,
    reset_embedder_cache,
)
from himmy.services.knowledge.retrieval import reranker as reranker_mod
from tests.conftest import run_async


class _FakeTextEmbedding:
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
    reset_embedder_cache()
    yield
    reset_embedder_cache()


# ---------------------------------------------- bug: unbounded session cache -> LRU bound


def test_text_embedding_cache_is_bounded_lru() -> None:
    """The text-embedding session cache evicts LRU instead of pinning every model forever."""
    info = local_embedders._load_text_embedding.cache_info()
    assert info.maxsize is not None and info.maxsize > 0, (
        "the native-session cache must be a BOUNDED lru_cache, not unbounded functools.cache"
    )


def test_cross_encoder_cache_is_bounded_lru() -> None:
    """The cross-encoder session cache is likewise bounded."""
    info = reranker_mod._load_cross_encoder.cache_info()
    assert info.maxsize is not None and info.maxsize > 0


def test_distinct_models_evict_lru_and_bound_resident_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading more distinct models than the bound keeps only ``maxsize`` sessions resident.

    Reproduces the leak precondition (many distinct model names in one process) and asserts
    the cache does NOT grow past its bound — the least-recently-used session is dropped.
    """
    from functools import lru_cache

    _FakeTextEmbedding.instances = 0
    bound = 3
    cached = lru_cache(maxsize=bound)(lambda name: _FakeTextEmbedding(name))
    monkeypatch.setattr(local_embedders, "_load_text_embedding", cached)

    for i in range(bound + 5):
        build_embedder("fastembed", model=f"m-{i}")._model_impl()

    info = cached.cache_info()
    assert info.currsize == bound, (
        f"resident sessions must be bounded to {bound}, got {info.currsize}"
    )


# ------------------------------- bug: shared-session embed_documents holds lock whole batch


def test_shared_embed_documents_releases_lock_per_subbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large shared-session batch must NOT hold the inference lock for its whole duration.

    We drive a batch larger than the sub-batch size and count how many DISTINCT times the
    shared lock was acquired during the embed. More than one acquisition proves the lock is
    released between sub-batches (so a concurrent embed_query can interleave), instead of the
    old single whole-batch hold that head-of-line-blocked co-resident embeds.
    """
    _FakeTextEmbedding.instances = 0
    from functools import lru_cache

    monkeypatch.setattr(
        local_embedders,
        "_load_text_embedding",
        lru_cache(maxsize=8)(lambda name: _FakeTextEmbedding(name)),
    )
    monkeypatch.setenv("HIMMY_EMBED_SUBBATCH", "4")

    acquisitions = {"n": 0}
    real_lock = threading.Lock()

    class _CountingLock:
        def __enter__(self) -> object:
            acquisitions["n"] += 1
            return real_lock.__enter__()

        def __exit__(self, *exc: object) -> object:
            return real_lock.__exit__(*exc)

    # Replace the module lock so both the returned guard and the ``guard is _TEXT_EMBED_LOCK``
    # sub-batch check reference this counting object.
    monkeypatch.setattr(local_embedders, "_TEXT_EMBED_LOCK", _CountingLock())

    emb = build_embedder("fastembed", model="m-a")
    texts = [f"t{i}" for i in range(10)]  # 10 texts / sub-batch 4 -> 3 sub-batches
    vecs = run_async(emb.embed_documents(texts))

    assert acquisitions["n"] >= 3, (
        "shared-session batch must acquire the lock per sub-batch (released in between)"
    )
    # And the output is byte-identical to a single-pass embed of the same texts.
    assert vecs == [[float(len(t)), 1.0, 2.0] for t in texts]


def test_subbatch_output_identical_to_single_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-batching must be OUTPUT-preserving: same vectors, same order, as one pass."""
    from functools import lru_cache

    monkeypatch.setattr(
        local_embedders,
        "_load_text_embedding",
        lru_cache(maxsize=8)(lambda name: _FakeTextEmbedding(name)),
    )
    emb = build_embedder("fastembed", model="m-a")
    texts = [f"doc-{i}" for i in range(7)]

    monkeypatch.setenv("HIMMY_EMBED_SUBBATCH", "0")  # single pass
    single = run_async(emb.embed_documents(texts))
    monkeypatch.setenv("HIMMY_EMBED_SUBBATCH", "2")  # sub-batched
    batched = run_async(emb.embed_documents(texts))

    assert single == batched


def test_private_session_never_subbatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A private session (kill-switch off) embeds in ONE pass regardless of the sub-batch env.

    It never takes the shared lock, so the pre-cache single-pass decode path is preserved
    exactly; sub-batching only applies to the SHARED session's lock granularity.
    """
    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTextEmbedding)
    monkeypatch.setenv("HIMMY_EMBED_CACHE", "off")
    monkeypatch.setenv("HIMMY_EMBED_SUBBATCH", "1")

    calls = {"embed": 0}

    class _CountingEmbed(_FakeTextEmbedding):
        def embed(self, texts: list[str]):
            calls["embed"] += 1
            yield from super().embed(texts)

    monkeypatch.setattr(fastembed, "TextEmbedding", _CountingEmbed)
    emb = build_embedder("fastembed", model="m-a")
    run_async(emb.embed_documents([f"t{i}" for i in range(6)]))
    assert calls["embed"] == 1, "private session must embed the whole batch in one pass"
