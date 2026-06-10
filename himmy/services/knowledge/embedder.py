"""Knowledge kernel: embedders — the deterministic offline default + provider paths.

The default :class:`DeterministicEmbedder` is hash-based, fixed-dimension, and
network-free so tests and examples run offline. The provider-backed embedders
(:func:`build_openai_compatible_embedder`, :class:`OpenAIMultimodalEmbeddingModel`)
are import-safe: they construct without ``openai``/``pydantic_ai`` installed and
talk to a real provider only when actually used (lazy, guarded imports). Their
integration tests are extra/network-gated and skip offline.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from typing import Any, Protocol, runtime_checkable

from himmy.core.errors import HimmyError


@runtime_checkable
class EmbedderProtocol(Protocol):
    """The minimal async embedding contract the KnowledgeBase depends on."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (asymmetric document-side path)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (asymmetric query-side path)."""
        ...


#: Reserved chunk-metadata key recording which embedder produced a chunk's vector.
EMBEDDER_FINGERPRINT_KEY = "embedder_fingerprint"


def embedder_fingerprint(embedder: Any) -> str:
    """Stable identity fingerprint of an embedder (class + model + dimension).

    Hashes the salient configuration — the embedder's class name, its model id
    (``model`` / ``model_name``, when set), and its output dimension (``dim`` /
    ``dimensions``, when set) — into a short stable hex token. The KnowledgeBase
    stamps this on every chunk at ingest (under :data:`EMBEDDER_FINGERPRINT_KEY`)
    and compares it at search time, so swapping embedding models is *detected*
    instead of silently scoring garbage similarities against incompatible old
    vectors. A custom embedder can bypass the inference by exposing a non-empty
    string ``fingerprint`` attribute, which is used verbatim.
    """
    explicit = getattr(embedder, "fingerprint", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    model = getattr(embedder, "model", None) or getattr(embedder, "model_name", None)
    dim = getattr(embedder, "dim", None)
    if dim is None:
        dim = getattr(embedder, "dimensions", None)
    raw = "|".join(
        (
            type(embedder).__name__,
            str(model) if model is not None else "",
            str(dim) if dim is not None else "",
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def embedder_is_multimodal(embedder: Any) -> bool:
    """Return True only when an embedder explicitly declares image capability.

    A text-only embedder (including :class:`DeterministicEmbedder`) returns False so
    :meth:`KnowledgeBase.ingest_image` refuses it rather than producing silent
    garbage image embeddings. Multimodal embedders advertise the capability via a
    truthy ``supports_images`` or ``is_multimodal`` attribute.
    """
    return bool(
        getattr(embedder, "supports_images", False)
        or getattr(embedder, "is_multimodal", False)
    )


class DeterministicEmbedder:
    """Offline, network-free embedder producing stable normalized hash vectors.

    The same text always maps to the same unit vector of dimension ``dim``. It is
    the default for tests and local development — no provider, no keys, no quota.
    """

    #: Text-only — :func:`embedder_is_multimodal` returns False for this embedder.
    supports_images: bool = False

    def __init__(self, *, dim: int = 64) -> None:
        """Create an embedder producing ``dim``-dimensional unit vectors."""
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        """Map text to a deterministic normalized bag-of-tokens hash vector.

        Each token is hashed to one bucket with a hashed sign (the hashing-trick).
        Shared tokens between two texts add aligned signal, so cosine similarity
        reflects lexical overlap — a query word present in a document scores
        positively. This keeps the embedder offline and deterministic while making
        it a usable retrieval default.
        """
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-alphanumerics; fall back to the raw text."""
        import re

        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        return tokens or [(text or "").lower()]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents deterministically."""
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string deterministically."""
        return self._embed(text)


def _require_openai() -> Any:
    """Import the openai SDK lazily; raise a clear error if the extra is missing."""
    try:
        import openai  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise HimmyError(
            "OpenAI-compatible embedder requires the [knowledge] extra "
            "(pip install 'himmy[knowledge]')."
        ) from exc
    return openai


class _OpenAICompatibleEmbedder:
    """A real embedder for any OpenAI-wire-format embedding endpoint.

    Reads ``OPENAI_COMPATIBLE_API_KEY`` / ``OPENAI_COMPATIBLE_BASE_URL`` /
    ``OPENAI_COMPATIBLE_EMBED_MODEL`` (explicit kwargs override). Constructs without
    ``openai`` installed (import is lazy); on first ``embed_*`` it calls the
    provider's ``embeddings.create`` in provider-safe batches with bounded retry.
    The query / document paths are asymmetric so providers that support an
    input-type hint (Cohere, Voyage via OpenAI-compat shims) get the right side.
    """

    #: Text-only by default (multimodal lives in OpenAIMultimodalEmbeddingModel).
    supports_images: bool = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int = 256,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        dimensions: int | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        self.model = model or os.environ.get("OPENAI_COMPATIBLE_EMBED_MODEL")
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.dimensions = dimensions
        self._client: Any | None = None

    def _client_or_build(self) -> Any:
        """Return a cached AsyncOpenAI client, building it lazily on first use."""
        if self._client is not None:
            return self._client
        openai = _require_openai()
        if not self.model:
            raise HimmyError(
                "OpenAI-compatible embedder needs a model "
                "(OPENAI_COMPATIBLE_EMBED_MODEL or model=...)."
            )
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        try:
            self._client = openai.AsyncOpenAI(**kwargs)
        except Exception as exc:  # normalize SDK config errors (missing key, etc.)
            raise HimmyError(
                "OpenAI-compatible embedder is not configured: set "
                "OPENAI_COMPATIBLE_API_KEY (or pass api_key=...) and a base_url/"
                f"model. ({exc})"
            ) from exc
        return self._client

    async def _create(self, inputs: list[str]) -> list[list[float]]:
        """Call ``embeddings.create`` once with bounded exponential-backoff retry."""
        client = self._client_or_build()
        extra: dict[str, Any] = {}
        if self.dimensions is not None:
            extra["dimensions"] = self.dimensions
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await client.embeddings.create(
                    model=self.model, input=inputs, **extra
                )
                # Provider returns items in request order, but sort defensively.
                items = sorted(resp.data, key=lambda d: d.index)
                return [list(item.embedding) for item in items]
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                if attempt + 1 >= self.max_retries:
                    break
                await asyncio.sleep(self.retry_base_delay * (2**attempt))
        raise HimmyError(
            f"OpenAI-compatible embedding call failed after {self.max_retries} "
            f"attempts: {last_exc}"
        ) from last_exc

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Split into provider-safe batches and concatenate results in order."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(await self._create(texts[i : i + self.batch_size]))
        return out

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents via the configured endpoint (requires the extra)."""
        if not texts:
            return []
        return await self._embed_in_batches(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query via the configured endpoint (requires the extra)."""
        result = await self._create([text])
        return result[0]


def build_openai_compatible_embedder(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    batch_size: int = 256,
    max_retries: int = 3,
    dimensions: int | None = None,
) -> _OpenAICompatibleEmbedder:
    """Build a real embedder for any OpenAI-wire-format endpoint.

    Reads ``OPENAI_COMPATIBLE_API_KEY`` / ``OPENAI_COMPATIBLE_BASE_URL`` /
    ``OPENAI_COMPATIBLE_EMBED_MODEL`` by default; any explicit kwarg overrides the
    corresponding env var. The naming is deliberately separate from
    ``OPENAI_API_KEY`` so the ``.env`` does not lie about where traffic goes. The
    returned object is import-safe (the ``openai`` import is deferred to first use)
    and embeds for real once the ``[knowledge]`` extra is installed.
    """
    return _OpenAICompatibleEmbedder(
        api_key=api_key,
        base_url=base_url,
        model=model,
        batch_size=batch_size,
        max_retries=max_retries,
        dimensions=dimensions,
    )


# A marker prefix used to encode image inputs in the multimodal text channel so a
# plain ``list[str]`` carries both modalities through one ``embed_documents`` call.
_IMAGE_MARKER = "image://"


class OpenAIMultimodalEmbeddingModel:
    """A multimodal (text + image) embedding model on the OpenAI-compatible wire.

    Inputs are a ``list[str]``: a plain string is text; a string beginning with
    ``image://`` (or any ``http(s)://`` URL pointing at an image, or a ``data:``
    URI) is an image reference. This lets a single ``embed_documents`` call mix both
    modalities, matching how :meth:`KnowledgeBase.ingest_image` feeds image URIs.

    In production this is wrapped by ``pydantic_ai.Embedder`` for provider batching,
    retry, and OTel; the wrapper is exposed via :meth:`as_pydantic_ai_embedder`.
    It reads the same ``OPENAI_COMPATIBLE_*`` env vars, constructs without
    ``openai``/``pydantic_ai`` installed, and calls the provider only on use.
    """

    #: Declares image capability so KnowledgeBase.ingest_image accepts it.
    supports_images: bool = True
    is_multimodal: bool = True

    def __init__(
        self,
        model_name: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Record the multimodal model id (defaults to OPENAI_COMPATIBLE_EMBED_MODEL)."""
        self.model_name = model_name or os.environ.get("OPENAI_COMPATIBLE_EMBED_MODEL")
        # Reuse the OpenAI-compatible client plumbing for the text channel + auth.
        self._delegate = _OpenAICompatibleEmbedder(
            api_key=api_key,
            base_url=base_url,
            model=self.model_name,
        )

    @staticmethod
    def _is_image_ref(item: str) -> bool:
        """True when ``item`` looks like an image reference, not plain text."""
        lowered = item.lower()
        return (
            lowered.startswith(_IMAGE_MARKER)
            or lowered.startswith("data:image")
            or lowered.startswith("http://")
            or lowered.startswith("https://")
        )

    def _to_input(self, item: str) -> dict[str, Any] | str:
        """Encode one item for the multimodal endpoint (image dict or raw text)."""
        if self._is_image_ref(item):
            uri = item[len(_IMAGE_MARKER) :] if item.startswith(_IMAGE_MARKER) else item
            return {"type": "image_url", "image_url": {"url": uri}}
        return item

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed mixed text/image documents through the multimodal endpoint."""
        if not texts:
            return []
        client = self._delegate._client_or_build()
        encoded = [self._to_input(t) for t in texts]
        last_exc: Exception | None = None
        for attempt in range(self._delegate.max_retries):
            try:
                resp = await client.embeddings.create(
                    model=self.model_name, input=encoded
                )
                items = sorted(resp.data, key=lambda d: d.index)
                return [list(item.embedding) for item in items]
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                if attempt + 1 >= self._delegate.max_retries:
                    break
                await asyncio.sleep(self._delegate.retry_base_delay * (2**attempt))
        raise HimmyError(
            f"Multimodal embedding call failed after {self._delegate.max_retries} "
            f"attempts: {last_exc}"
        ) from last_exc

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single (text or image) query through the multimodal endpoint."""
        result = await self.embed_documents([text])
        return result[0]

    def as_pydantic_ai_embedder(self) -> Any:
        """Wrap this model in a ``pydantic_ai.Embedder`` (requires [providers]).

        The wrapper gives the multimodal model provider-side batching, retry, and
        OpenTelemetry for free. Import is lazy so this module stays import-safe.
        """
        try:
            from pydantic_ai import Embedder  # type: ignore
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise HimmyError(
                "as_pydantic_ai_embedder requires the [providers] extra "
                "(pip install 'himmy[providers]')."
            ) from exc
        # ``self`` is a himmy embedder adapter satisfying pydantic-ai's EmbeddingModel
        # protocol structurally; its type union (literal model names) can't express that.
        return Embedder(self)  # type: ignore[arg-type]


__all__ = [
    "EMBEDDER_FINGERPRINT_KEY",
    "EmbedderProtocol",
    "DeterministicEmbedder",
    "build_openai_compatible_embedder",
    "OpenAIMultimodalEmbeddingModel",
    "embedder_fingerprint",
    "embedder_is_multimodal",
]
