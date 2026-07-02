"""Real local embedders: Ollama (HTTP), fastembed (ONNX), + an embedder factory.

The default :class:`~himmy.services.knowledge.embedder.DeterministicEmbedder` only
matches on exact token overlap; these give genuine semantic recall for the knowledge
and memory subsystems while staying local/keyless:

* :class:`OllamaEmbedder` — calls a local Ollama server's ``/api/embeddings`` over
  httpx (zero new dependencies; an injectable ``transport`` keeps tests offline);
* :class:`FastEmbedEmbedder` — a self-contained ONNX model via the optional
  ``embeddings`` extra (no server, downloads a small model on first use).

:func:`build_embedder` selects one by name so config/CLI can switch backends without
code. The ``"auto"`` name asks for *genuine semantic* retrieval when it is locally
available and degrades gracefully otherwise: it prefers ``fastembed`` (if the optional
dep is importable, no network), then a local Ollama **that has the configured embedding
model actually pulled** (``HIMMY_OLLAMA_EMBED_MODEL``, default ``qwen3-embedding``) — a
fast, fail-closed localhost probe of ``/api/tags``, never a hard dependency — and
finally falls back to the offline :class:`DeterministicEmbedder`. The Ollama leg is
gated on the model being present (not just the server being up) so a reachable Ollama
with no embed model degrades to deterministic instead of 404-ing at embed time. So
zero-config callers that opt into ``"auto"`` still import and run with no keys, no new
required deps, and no working network — yet a user who has pulled an Ollama embedding
model gets genuine local semantic recall automatically.

All embedders satisfy :class:`~himmy.services.knowledge.embedder.EmbedderProtocol`.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import threading
from collections.abc import Awaitable, Callable
from functools import cache
from typing import Any

from himmy.config.flags import env_falsy
from himmy.core.errors import HimmyError
from himmy.services.knowledge.embedder import (
    DeterministicEmbedder,
    EmbedderProtocol,
)

OllamaTransport = Callable[[str, dict[str, Any]], Any]

#: Kill-switch for the shared native-session cache (default ON). Set
#: ``HIMMY_EMBED_CACHE=off`` (``0``/``false``/``no``) to force each embedder to build
#: its own private ONNX session — the pre-cache behaviour — e.g. to isolate a suspected
#: cross-instance state bug. Read per call so tests/operators can flip it at runtime.
_EMBED_CACHE_ENV = "HIMMY_EMBED_CACHE"


def _embed_cache_enabled() -> bool:
    """Whether the shared native-session cache is active (default ON, env-overridable)."""
    return not env_falsy(_EMBED_CACHE_ENV)


#: Serialises access to :func:`_load_text_embedding`'s cache *construction* and guards
#: concurrent ``.embed()`` calls that share one native session. fastembed drives
#: onnxruntime, whose ``InferenceSession.run`` is thread-safe, but fastembed's own
#: ``embed()`` keeps mutable per-instance batching state, so once a session is *shared*
#: across embedder wrappers we serialise inference to keep outputs byte-identical and
#: avoid data races. The lock is process-wide and re-entrant-free (no nested embeds).
#:
#: Acquired *only* when the impl is the shared cached session (see
#: :meth:`FastEmbedEmbedder._impl_and_guard`): a private per-instance session — either a
#: test-injected ``self._impl`` or the ``HIMMY_EMBED_CACHE=off`` build — is not shared and
#: runs under :class:`contextlib.nullcontext`, so flipping the kill-switch restores the
#: pre-cache parallel throughput, not just the pre-cache vectors.
_TEXT_EMBED_LOCK = threading.Lock()


@cache
def _load_text_embedding(model_name: str) -> Any:
    """Load (and cache process-wide) the heavy fastembed ``TextEmbedding`` session.

    Keyed by ``model_name`` only: the wrapper's ``dim`` is metadata that never changes
    the native weights, so two :class:`FastEmbedEmbedder`\\ s for the same model share
    one ONNX session (hundreds of ms + tens–hundreds MB saved per duplicate build)
    while producing byte-identical vectors. Wrapped by :func:`functools.cache` so the
    first caller pays the load and every later caller gets the same object. Raises a
    clear :class:`HimmyError` when the ``[embeddings]`` extra is absent (never caches the
    failure — ``functools.cache`` does not memoise exceptions).
    """
    try:
        from fastembed import TextEmbedding  # type: ignore
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise HimmyError(
            "FastEmbedEmbedder requires the [embeddings] extra "
            "(pip install 'himmy[embeddings]')."
        ) from exc
    return TextEmbedding(model_name=model_name)


def reset_embedder_cache() -> None:
    """Clear the shared native-session caches (test hook / operator escape hatch).

    Drops every cached :func:`_load_text_embedding` (and, via the reranker module, every
    cached cross-encoder) so the next build reloads the native model — used by tests to
    assert cache identity/invalidation without leaking sessions between cases. Safe to
    call at any time; a subsequent build simply pays the one-time load again.
    """
    _load_text_embedding.cache_clear()
    from himmy.services.knowledge.retrieval import reranker as _reranker

    _reranker.reset_reranker_cache()


class OllamaEmbedder:
    """Embed text via a local Ollama server (``/api/embeddings``), keyless."""

    supports_images: bool = False

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
        dim: int | None = None,
        transport: OllamaTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Configure the model, server URL, embedding dim, and (test) transport.

        ``model``/``dim`` default to the configured Ollama embedding model
        (``HIMMY_OLLAMA_EMBED_MODEL``, default ``qwen3-embedding`` at 4096-d) and its
        native dimension, so a bare ``OllamaEmbedder()`` matches what ``"auto"`` selects.
        """
        model = model or default_ollama_embed_model()
        self.model = model
        self.dim = dim if dim is not None else default_ollama_embed_dim(model)
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
    """Embed text with a local ONNX model via ``fastembed`` (the [embeddings] extra).

    Both the one-time model load and every inference call are CPU-bound ONNX work.
    fastembed/onnxruntime release the GIL while running inference (and during the
    native model load), so running them in a worker thread via ``asyncio.to_thread``
    genuinely frees the caller's event loop — a host async server (FastAPI/uvicorn)
    that builds a KnowledgeBase from inside its running loop stays responsive instead
    of freezing for the duration of the build. This mirrors :class:`FastEmbedReranker`.
    """

    supports_images: bool = False

    def __init__(
        self, *, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384
    ) -> None:
        """Configure the model name and its embedding dimension (lazy-loaded)."""
        self.model = model
        self.dim = dim
        self._impl: Any = None

    def _model_impl(self) -> Any:
        """Return the fastembed model, sharing one native session per model name.

        The native ``TextEmbedding`` constructor downloads (first time) and
        initialises an ONNX session — heavy, GIL-releasing work — so callers run
        this inside ``asyncio.to_thread`` to avoid blocking the event loop. When the
        shared cache is enabled (the default; see :data:`_EMBED_CACHE_ENV`), the session
        is fetched from the process-wide :func:`_load_text_embedding` cache so every
        :class:`FastEmbedEmbedder` for the same model reuses one ONNX session instead of
        reloading the native model per build. With the cache off, each instance builds
        and holds its own private session (the pre-cache behaviour).

        An explicitly pre-set ``self._impl`` (e.g. a test-injected fake session) always
        wins over the shared cache, so per-instance overrides keep working unchanged.
        """
        if self._impl is not None:
            return self._impl
        if _embed_cache_enabled():
            return _load_text_embedding(self.model)
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

    def _impl_and_guard(self) -> tuple[Any, Any]:
        """Return the fastembed impl and the lock to serialise inference under.

        The impl is *shared* only when this instance holds no private session
        (``self._impl is None``) **and** the cache is enabled — that is exactly the
        combination that makes :meth:`_model_impl` return the process-wide
        :func:`_load_text_embedding` session. In that case inference is serialised under
        :data:`_TEXT_EMBED_LOCK`; a private session (test-injected ``self._impl`` or a
        ``HIMMY_EMBED_CACHE=off`` build) is unique to this instance, so it parallelises
        under :class:`contextlib.nullcontext` — the pre-cache behaviour. The shared flag is
        captured *before* :meth:`_model_impl` populates ``self._impl`` so the two never
        disagree.
        """
        shared = self._impl is None and _embed_cache_enabled()
        impl = self._model_impl()
        guard = _TEXT_EMBED_LOCK if shared else contextlib.nullcontext()
        return impl, guard

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (fastembed is sync; vectors are numpy arrays).

        Model load + inference are GIL-releasing ONNX work, run off the event loop.
        When the session is shared across instances, inference is serialised under
        :data:`_TEXT_EMBED_LOCK` so concurrent embeds against one session cannot race on
        fastembed's mutable batching state — outputs stay byte-identical. A private
        session (kill-switch off, or an injected impl) parallelises freely.
        """

        def _embed() -> list[list[float]]:
            impl, guard = self._impl_and_guard()
            with guard:
                return [list(map(float, vec)) for vec in impl.embed(texts)]

        return await asyncio.to_thread(_embed)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (off the event loop; see :meth:`embed_documents`)."""

        def _embed() -> list[float]:
            impl, guard = self._impl_and_guard()
            with guard:
                return [float(x) for x in next(iter(impl.query_embed([text])))]

        return await asyncio.to_thread(_embed)


#: Default embedding dimension per backend (used when a dim is not configured).
_DEFAULT_DIMS = {
    "deterministic": 64,
    "nepali": 64,
    "ollama": 4096,
    "fastembed": 384,
    "openai": 1536,
}

#: Native embedding dim per known Ollama embedding model (used when no dim is given).
_OLLAMA_EMBED_DIMS = {
    "qwen3-embedding": 4096,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "bge-m3": 1024,
}

#: The default Ollama embedding model. ``qwen3-embedding`` is a strong multilingual
#: local model; override with ``HIMMY_OLLAMA_EMBED_MODEL``.
_DEFAULT_OLLAMA_EMBED_MODEL = "qwen3-embedding"


def default_ollama_embed_model() -> str:
    """The configured Ollama embedding model (``HIMMY_OLLAMA_EMBED_MODEL`` or default)."""
    return os.environ.get("HIMMY_OLLAMA_EMBED_MODEL") or _DEFAULT_OLLAMA_EMBED_MODEL


def default_ollama_embed_dim(model: str | None = None) -> int:
    """Native dim for an Ollama embed model: ``HIMMY_OLLAMA_EMBED_DIM`` → known map → 4096."""
    env = os.environ.get("HIMMY_OLLAMA_EMBED_DIM")
    if env:
        try:
            return int(env)
        except ValueError:  # pragma: no cover - defensive
            pass
    name = (model or default_ollama_embed_model()).split(":", 1)[0]
    return _OLLAMA_EMBED_DIMS.get(name, _DEFAULT_DIMS["ollama"])


#: Default base URL for the local Ollama server (also the reachability probe target).
_DEFAULT_OLLAMA_URL = "http://localhost:11434"


def default_dim_for(name: str) -> int:
    """Return the conventional embedding dimension for a backend name.

    ``"auto"`` reports the dim of whichever backend it would actually resolve to
    *right now* (see :func:`resolve_auto_backend`), so a KB's ``vector_dim`` always
    matches the embedder it is paired with.
    """
    if name == "auto":
        return _DEFAULT_DIMS.get(resolve_auto_backend(), 64)
    return _DEFAULT_DIMS.get(name, 64)


def fastembed_available() -> bool:
    """True when the ``fastembed`` optional dep is importable (no import side effects).

    Uses :func:`importlib.util.find_spec` so probing for availability never triggers
    fastembed's heavier import or model download — that is deferred to first embed.
    """
    try:
        return importlib.util.find_spec("fastembed") is not None
    except (ImportError, ValueError):  # pragma: no cover - exotic import machinery
        return False


def ollama_reachable(
    base_url: str = _DEFAULT_OLLAMA_URL, *, timeout: float = 0.25
) -> bool:
    """True when a local Ollama server answers at ``base_url`` within ``timeout``.

    A fast, fail-closed probe: any connection error, timeout, missing ``httpx``, or
    non-2xx/3xx response returns False so a reachable server is *preferred* but never
    *required*. The default short timeout keeps the offline path snappy — when nothing
    is listening the OS refuses the connection immediately. Localhost-only by default.
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a core dependency
        return False
    url = base_url.rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=timeout)
    except Exception:  # noqa: BLE001 - any probe failure means "not reachable"
        return False
    return response.status_code < 400


def ollama_embed_model_available(
    model: str | None = None,
    base_url: str = _DEFAULT_OLLAMA_URL,
    *,
    timeout: float = 0.5,
) -> bool:
    """True when a local Ollama server has the embedding ``model`` actually pulled.

    A fast, fail-closed probe of ``/api/tags`` (the model registry): selecting Ollama for
    ``"auto"`` requires the embed model to be *present*, not merely the server to be up —
    otherwise the first embed call 404s. Any connection error, timeout, missing ``httpx``,
    or absent model returns False so the offline ``deterministic`` fallback is used. The
    match is tag-insensitive (``qwen3-embedding`` matches ``qwen3-embedding:latest``).
    """
    want = (model or default_ollama_embed_model()).split(":", 1)[0]
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a core dependency
        return False
    try:
        response = httpx.get(base_url.rstrip("/") + "/api/tags", timeout=timeout)
    except Exception:  # noqa: BLE001 - any probe failure means "not available"
        return False
    if response.status_code >= 400:
        return False
    try:
        models = response.json().get("models", [])
    except Exception:  # noqa: BLE001 - malformed body ⇒ treat as unavailable
        return False
    return any(str(m.get("name", "")).split(":", 1)[0] == want for m in models)


def resolve_auto_backend(*, ollama_base_url: str | None = None) -> str:
    """Resolve ``"auto"`` to a concrete backend name, preferring real local semantics.

    Order: ``fastembed`` (local ONNX, when the ``[embeddings]`` extra is installed) → a
    local Ollama **that has the configured embedding model pulled** → the offline
    ``deterministic`` fallback. The Ollama leg is gated on
    :func:`ollama_embed_model_available` (not merely :func:`ollama_reachable`) so a
    reachable server without the embed model degrades to deterministic instead of 404-ing
    at embed time — making ``"auto"`` both robust and genuinely semantic when the user has
    pulled an Ollama embedding model. The result is a plain backend name so callers can
    both build the embedder and look up its conventional dim coherently.
    """
    if fastembed_available():
        return "fastembed"
    base = ollama_base_url or _DEFAULT_OLLAMA_URL
    if ollama_reachable(base) and ollama_embed_model_available(base_url=base):
        return "ollama"
    return "deterministic"


def build_embedder(
    name: str = "deterministic",
    *,
    model: str | None = None,
    dim: int | None = None,
    base_url: str | None = None,
) -> EmbedderProtocol:
    """Build an embedder by backend name.

    Names: ``auto | deterministic | ollama | fastembed | openai | nepali``.
    ``"auto"`` resolves to a real local embedder when one is available and otherwise
    falls back to the offline :class:`DeterministicEmbedder` (see
    :func:`resolve_auto_backend`). When ``"auto"`` resolves to a real backend, an
    explicit ``dim`` is ignored unless it matches that backend's native dimension —
    a foreign dim would silently mis-size the vector store.
    """
    if name == "auto":
        resolved = resolve_auto_backend(ollama_base_url=base_url)
        # Only forward an explicit dim when it matches the resolved backend's native
        # dim; otherwise honour the backend's own dimension (e.g. fastembed's 384).
        resolved_dim = dim if dim == _DEFAULT_DIMS.get(resolved) else None
        return build_embedder(
            resolved, model=model, dim=resolved_dim, base_url=base_url
        )
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
        "(expected auto | deterministic | ollama | fastembed | openai)"
    )


__all__ = [
    "OllamaEmbedder",
    "FastEmbedEmbedder",
    "build_embedder",
    "default_dim_for",
    "fastembed_available",
    "ollama_reachable",
    "resolve_auto_backend",
    "reset_embedder_cache",
]
