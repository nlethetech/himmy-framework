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
import importlib.util
from typing import Any, Protocol, runtime_checkable

from himmy.core.errors import HimmyError

#: A small, fast ONNX cross-encoder that fastembed can fetch on first use.
DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


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
        """Lazily construct the fastembed cross-encoder, raising if absent."""
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

    async def rerank(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Cross-encode ``(query, text)`` for each candidate and rank by score."""
        if not candidates:
            return []
        impl = self._model_impl()
        texts = [text for (_, text) in candidates]

        def _score() -> list[float]:
            # fastembed's rerank() yields one score per document, in input order.
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
    "DEFAULT_RERANKER_MODEL",
]
