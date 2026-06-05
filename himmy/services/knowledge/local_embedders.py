"""Real local embedders: Ollama (HTTP), fastembed (ONNX), + an embedder factory.

The default :class:`~himmy.services.knowledge.embedder.DeterministicEmbedder` only
matches on exact token overlap; these give genuine semantic recall for the knowledge
and memory subsystems while staying local/keyless:

* :class:`OllamaEmbedder` — calls a local Ollama server's ``/api/embeddings`` over
  httpx (zero new dependencies; an injectable ``transport`` keeps tests offline);
* :class:`FastEmbedEmbedder` — a self-contained ONNX model via the optional
  ``embeddings`` extra (no server, downloads a small model on first use).

:func:`build_embedder` selects one by name so config/CLI can switch backends without
code. All satisfy :class:`~himmy.services.knowledge.embedder.EmbedderProtocol`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from himmy.core.errors import HimmyError
from himmy.services.knowledge.embedder import (
    DeterministicEmbedder,
    EmbedderProtocol,
)

OllamaTransport = Callable[[str, dict[str, Any]], Any]


class OllamaEmbedder:
    """Embed text via a local Ollama server (``/api/embeddings``), keyless."""

    supports_images: bool = False

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dim: int = 768,
        transport: OllamaTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Configure the model, server URL, embedding dim, and (test) transport."""
        self.model = model
        self.dim = dim
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout

    async def _embed_one(self, text: str) -> list[float]:
        data = await self._post(
            "/api/embeddings", {"model": self.model, "prompt": text}
        )
        vec = data.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise HimmyError("Ollama returned an empty embedding")
        return [float(x) for x in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each document (Ollama embeds one prompt per call)."""
        return [await self._embed_one(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return await self._embed_one(text)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            result = self._transport(path, payload)
            if isinstance(result, Awaitable):
                return await result  # type: ignore[no-any-return]
            return result  # type: ignore[no-any-return]
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._base_url + path, json=payload)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]


class FastEmbedEmbedder:
    """Embed text with a local ONNX model via ``fastembed`` (the [embeddings] extra)."""

    supports_images: bool = False

    def __init__(
        self, *, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384
    ) -> None:
        """Configure the model name and its embedding dimension (lazy-loaded)."""
        self.model = model
        self.dim = dim
        self._impl: Any = None

    def _model_impl(self) -> Any:
        """Lazily construct the fastembed model, raising a clear error if absent."""
        if self._impl is None:
            try:
                from fastembed import TextEmbedding  # type: ignore
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise HimmyError(
                    "FastEmbedEmbedder requires the [embeddings] extra "
                    "(pip install 'himmy[embeddings]')."
                ) from exc
            self._impl = TextEmbedding(model_name=self.model)
        return self._impl

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (fastembed is sync; vectors are numpy arrays)."""
        impl = self._model_impl()
        return [list(map(float, vec)) for vec in impl.embed(texts)]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        impl = self._model_impl()
        return [float(x) for x in next(iter(impl.query_embed([text])))]


#: Default embedding dimension per backend (used when a dim is not configured).
_DEFAULT_DIMS = {
    "deterministic": 64,
    "nepali": 64,
    "ollama": 768,
    "fastembed": 384,
    "openai": 1536,
}


def default_dim_for(name: str) -> int:
    """Return the conventional embedding dimension for a backend name."""
    return _DEFAULT_DIMS.get(name, 64)


def build_embedder(
    name: str = "deterministic",
    *,
    model: str | None = None,
    dim: int | None = None,
    base_url: str | None = None,
) -> EmbedderProtocol:
    """Build an embedder by backend name (``deterministic|ollama|fastembed|openai``)."""
    if name == "deterministic":
        return DeterministicEmbedder(dim=dim or _DEFAULT_DIMS["deterministic"])
    if name == "ollama":
        kwargs: dict[str, Any] = {"dim": dim or _DEFAULT_DIMS["ollama"]}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbedder(**kwargs)
    if name == "fastembed":
        kwargs = {"dim": dim or _DEFAULT_DIMS["fastembed"]}
        if model:
            kwargs["model"] = model
        return FastEmbedEmbedder(**kwargs)
    if name == "nepali":
        # Cross-script (Devanagari/Roman) folding over the offline embedder.
        from himmy.nepal.language import NepaliEmbedder

        return NepaliEmbedder(dim=dim or _DEFAULT_DIMS["deterministic"])
    if name == "openai":
        from himmy.services.knowledge.embedder import build_openai_compatible_embedder

        return build_openai_compatible_embedder(
            model=model, base_url=base_url, dimensions=dim
        )
    raise HimmyError(
        f"unknown embedder {name!r} "
        "(expected deterministic | ollama | fastembed | openai)"
    )


__all__ = [
    "OllamaEmbedder",
    "FastEmbedEmbedder",
    "build_embedder",
    "default_dim_for",
]
