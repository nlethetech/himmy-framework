"""Cross-encoder reranking — the highest-precision lever in the pipeline.

Bi-encoder retrieval (dense vectors) and BM25 score a query and a chunk
*independently*; a cross-encoder instead jointly encodes ``(query, chunk)`` and
emits a single relevance score, which is far more precise but too expensive to run
over a whole corpus. So it runs only over the already-fused top-N candidates and
re-orders them before the final ``top_k`` cut.

:class:`FastEmbedReranker` wraps fastembed's local ONNX ``TextCrossEncoder`` behind
the EXISTING ``[embeddings]`` extra, exactly like
:class:`~himmy.services.knowledge.local_embedders.FastEmbedEmbedder`: the module
imports without fastembed installed and raises a clear :class:`HimmyError` only when
a reranker is actually *used* without the extra. Reranking is opt-in
(``RetrievalConfig.rerank=False`` by default), so the zero-config offline path never
imports or downloads a model.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import threading
from functools import cache
from typing import Any, Protocol, runtime_checkable

from himmy.config.flags import env_falsy
from himmy.core.errors import HimmyError

#: A small, fast ONNX cross-encoder that fastembed can fetch on first use.
DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

#: Kill-switch for the shared cross-encoder cache (default ON). Shares the embedder's
#: :data:`~himmy.services.knowledge.local_embedders._EMBED_CACHE_ENV` toggle so one
#: ``HIMMY_EMBED_CACHE=off`` disables *all* shared native sessions coherently.
_EMBED_CACHE_ENV = "HIMMY_EMBED_CACHE"


def _rerank_cache_enabled() -> bool:
    """Whether the shared cross-encoder cache is active (default ON, env-overridable)."""
    return not env_falsy(_EMBED_CACHE_ENV)


#: Guards concurrent ``rerank()`` calls that share one native cross-encoder session, for
#: the same reason as the embedder's lock: onnxruntime inference is thread-safe but
#: fastembed's wrapper carries mutable per-instance state, so we serialise inference over
#: a shared session to keep scores byte-identical under concurrency.
#:
#: Acquired *only* when the impl is the shared cached session (see
#: :meth:`FastEmbedReranker._impl_and_guard`): a private per-instance session — an injected
#: ``self._impl`` or a ``HIMMY_EMBED_CACHE=off`` build — is not shared and runs under
#: :class:`contextlib.nullcontext`, so the kill-switch restores pre-cache parallelism.
_CROSS_ENCODER_LOCK = threading.Lock()


@cache
def _load_cross_encoder(model_name: str) -> Any:
    """Load (and cache process-wide) the heavy fastembed ``TextCrossEncoder`` session.

    Keyed by ``model_name`` only (a reranker carries no ``dim``), so every
    :class:`FastEmbedReranker` for the same model reuses one ONNX session instead of
    reloading the native cross-encoder per build — hundreds of ms + tens–hundreds MB
    saved per duplicate. Wrapped by :func:`functools.cache`; raises a clear
    :class:`HimmyError` when the ``[embeddings]`` extra is absent (``functools.cache``
    does not memoise the raised exception, so a later install can succeed).
    """
    try:
        from fastembed.rerank.cross_encoder import (  # type: ignore
            TextCrossEncoder,
        )
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise HimmyError(
            "FastEmbedReranker requires the [embeddings] extra "
            "(pip install 'himmy[embeddings]')."
        ) from exc
    return TextCrossEncoder(model_name=model_name)


def reset_reranker_cache() -> None:
    """Clear the shared cross-encoder cache (test hook / operator escape hatch).

    Mirrors :func:`~himmy.services.knowledge.local_embedders.reset_embedder_cache` for
    the rerank session so tests can assert cache identity/invalidation in isolation.
    """
    _load_cross_encoder.cache_clear()


@runtime_checkable
class RerankerProtocol(Protocol):
    """The reranking contract the hybrid pipeline optionally calls.

    Implementations re-score fused candidates by jointly attending to the query and
    each chunk's text, returning ``(chunk_id, score)`` pairs (higher = more
    relevant). The pipeline sorts by the returned score; absolute scale is
    irrelevant.
    """

    async def rerank(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Re-score ``(chunk_id, text)`` candidates against ``query``."""
        ...


def fastembed_rerank_available() -> bool:
    """True when fastembed (the [embeddings] extra) is importable, no side effects."""
    try:
        return importlib.util.find_spec("fastembed") is not None
    except (ImportError, ValueError):  # pragma: no cover - exotic import machinery
        return False


class FastEmbedReranker:
    """A local ONNX cross-encoder reranker via ``fastembed`` ([embeddings] extra).

    Import-safe without fastembed installed; the model is constructed lazily on the
    first :meth:`rerank` and a clear :class:`HimmyError` is raised if the extra is
    missing. The synchronous fastembed call is run in a worker thread so it never
    blocks the event loop.
    """

    def __init__(self, *, model: str = DEFAULT_RERANKER_MODEL) -> None:
        """Configure the cross-encoder model name (loaded on first use)."""
        self.model = model
        self._impl: Any = None

    def _model_impl(self) -> Any:
        """Return the cross-encoder, sharing one native session per model name.

        When the shared cache is enabled (the default; see :data:`_EMBED_CACHE_ENV`),
        the session comes from the process-wide :func:`_load_cross_encoder` cache so
        every :class:`FastEmbedReranker` for the same model reuses one ONNX session; with
        the cache off, each instance lazily builds and holds its own private session.

        An explicitly pre-set ``self._impl`` (e.g. a test-injected fake) always wins over
        the shared cache, so per-instance overrides keep working unchanged.
        """
        if self._impl is not None:
            return self._impl
        if _rerank_cache_enabled():
            return _load_cross_encoder(self.model)
        if self._impl is None:
            try:
                from fastembed.rerank.cross_encoder import (  # type: ignore
                    TextCrossEncoder,
                )
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise HimmyError(
                    "FastEmbedReranker requires the [embeddings] extra "
                    "(pip install 'himmy[embeddings]')."
                ) from exc
            self._impl = TextCrossEncoder(model_name=self.model)
        return self._impl

    def _impl_and_guard(self) -> tuple[Any, Any]:
        """Return the cross-encoder impl and the lock to serialise inference under.

        The impl is *shared* only when this instance holds no private session
        (``self._impl is None``) **and** the cache is enabled — exactly when
        :meth:`_model_impl` returns the process-wide :func:`_load_cross_encoder` session,
        which is then serialised under :data:`_CROSS_ENCODER_LOCK`. A private session (an
        injected ``self._impl`` or a ``HIMMY_EMBED_CACHE=off`` build) is unique to this
        instance and parallelises under :class:`contextlib.nullcontext`. The shared flag is
        captured *before* :meth:`_model_impl` populates ``self._impl``.
        """
        shared = self._impl is None and _rerank_cache_enabled()
        impl = self._model_impl()
        guard = _CROSS_ENCODER_LOCK if shared else contextlib.nullcontext()
        return impl, guard

    async def rerank(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Cross-encode ``(query, text)`` for each candidate and rank by score."""
        if not candidates:
            return []
        texts = [text for (_, text) in candidates]

        def _score() -> list[float]:
            # Load the cross-encoder (first use) AND score inside the worker thread:
            # both the native model load and rerank() are GIL-releasing ONNX work, so
            # keeping them off the event loop preserves host-loop responsiveness. When
            # the session is shared, serialise inference so concurrent rerank() calls
            # cannot race on fastembed's mutable state — scores stay byte-identical; a
            # private session (kill-switch off, or injected) parallelises freely.
            impl, guard = self._impl_and_guard()
            # fastembed's rerank() yields one score per document, in input order.
            with guard:
                return [float(s) for s in impl.rerank(query, texts)]

        scores = await asyncio.to_thread(_score)
        ranked = [
            (candidates[i][0], scores[i])
            for i in range(min(len(candidates), len(scores)))
        ]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked


def build_reranker(
    name: str = "fastembed", *, model: str | None = None
) -> RerankerProtocol:
    """Build a reranker by backend name (mirrors :func:`build_embedder`).

    Currently only ``"fastembed"`` (a local ONNX cross-encoder behind the
    ``[embeddings]`` extra) is supported; the returned object is import-safe and
    fetches its model only on first use. Additional wire-format rerankers (Cohere,
    Voyage) can follow the ``build_openai_compatible_embedder`` pattern later.
    """
    if name == "fastembed":
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        return FastEmbedReranker(**kwargs)
    raise HimmyError(f"unknown reranker {name!r} (expected: fastembed)")


__all__ = [
    "RerankerProtocol",
    "FastEmbedReranker",
    "build_reranker",
    "fastembed_rerank_available",
    "reset_reranker_cache",
    "DEFAULT_RERANKER_MODEL",
]
