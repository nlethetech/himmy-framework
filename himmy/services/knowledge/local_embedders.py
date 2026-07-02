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
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any

from himmy.config.flags import env_falsy
from himmy.core.errors import HimmyError
from himmy.services.knowledge.embedder import (
    DeterministicEmbedder,
    EmbedderProtocol,
)

OllamaTransport = Callable[[str, dict[str, Any]], Any]

#: Max in-flight ``/api/embeddings`` round-trips per :meth:`OllamaEmbedder.embed_documents`
#: batch. Ollama embeds ONE prompt per HTTP call, so a batch is otherwise a serial chain of
#: awaited round-trips; a bounded ``asyncio.gather`` overlaps up to this many at once so a
#: many-chunk ingest completes in ~ceil(n/cap) round-trips of latency instead of n. Output is
#: byte-identical: ``gather`` preserves input order and each prompt is embedded independently,
#: so the concurrency only changes timing, never the vectors or their order. The default (8)
#: is a modest fan-out that helps a slow/remote server without stampeding a tiny local one; a
#: misconfigured value (non-int) falls back to the default and ``<= 0`` forces the fully serial
#: path — a bad knob never breaks embed. Read per call so tests/operators can flip it live.
_OLLAMA_EMBED_CONCURRENCY_ENV = "HIMMY_OLLAMA_EMBED_CONCURRENCY"
_DEFAULT_OLLAMA_EMBED_CONCURRENCY = 8


def _ollama_embed_concurrency() -> int:
    """Max concurrent Ollama embed round-trips per batch (env → int, else default 8).

    A misconfigured value (non-int) falls back to the default; ``0`` or negative forces the
    fully serial path (one in-flight request), so a bad knob never breaks embed.
    """
    raw = os.environ.get(_OLLAMA_EMBED_CONCURRENCY_ENV)
    if raw is None:
        return _DEFAULT_OLLAMA_EMBED_CONCURRENCY
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_OLLAMA_EMBED_CONCURRENCY

#: Kill-switch for the shared native-session cache (default ON). Set
#: ``HIMMY_EMBED_CACHE=off`` (``0``/``false``/``no``) to force each embedder to build
#: its own private ONNX session — the pre-cache behaviour — e.g. to isolate a suspected
#: cross-instance state bug. Read per call so tests/operators can flip it at runtime.
_EMBED_CACHE_ENV = "HIMMY_EMBED_CACHE"


def _embed_cache_enabled() -> bool:
    """Whether the shared native-session cache is active (default ON, env-overridable)."""
    return not env_falsy(_EMBED_CACHE_ENV)


#: Sub-batch size for a SHARED-session ``embed_documents``: the inference lock is released
#: and re-acquired every this-many texts so a large ingest cannot monopolise the one shared
#: ONNX session for its whole duration, letting a latency-critical concurrent ``embed_query``
#: (or another tenant's small embed) interleave between sub-batches instead of head-of-line
#: blocking behind thousands of docs. Output stays byte-identical: fastembed embeds each text
#: independently, so ``embed([a,b,c])`` == ``embed([a,b]) + embed([c])`` per vector; only the
#: lock-hold granularity changes. ``0``/negative/non-int ⇒ no sub-batching (hold the lock for
#: the whole batch, the prior behaviour). A PRIVATE session (kill-switch off / injected impl)
#: never takes the lock, so it is embedded in ONE pass regardless — the exact pre-cache path.
_EMBED_SUBBATCH_ENV = "HIMMY_EMBED_SUBBATCH"
_DEFAULT_EMBED_SUBBATCH = 64


def _embed_subbatch_size() -> int:
    """Texts per lock-hold for a shared-session batch embed (env → int, else default 64).

    A misconfigured value (non-int) falls back to the default; ``0`` or negative disables
    sub-batching (single lock-hold over the whole batch), so a bad knob never breaks embed.
    """
    raw = os.environ.get(_EMBED_SUBBATCH_ENV)
    if raw is None:
        return _DEFAULT_EMBED_SUBBATCH
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_EMBED_SUBBATCH
    return value


#: Serialises the one-time CONSTRUCTION of a shared :func:`_load_text_embedding` session
#: so two threads that miss the cache for the same model at cold start don't each build a
#: heavy ONNX session (only to discard one). ``functools.cache`` runs the wrapped function
#: OUTSIDE its own lock, so without this the "save the duplicate load" goal would be
#: defeated on concurrent first use; :meth:`_model_impl` holds this lock across the cache
#: lookup+build to make first-load load-exactly-once. Distinct from the inference guard
#: below (a build must not block an in-flight embed on another model, and vice versa).
_TEXT_EMBED_BUILD_LOCK = threading.Lock()

#: Guards concurrent ``.embed()`` calls that share one native session. fastembed drives
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


def _session_cache_maxsize(env_var: str, default: int = 4) -> int:
    """LRU bound for a native-session cache (``env_var`` → positive int, else ``default``).

    Bounds how many DISTINCT model names' heavy ONNX sessions stay resident at once, so a
    long-lived process that is reconfigured across many embedder/reranker models (multi-
    tenant specs, a rotating ``HIMMY_EMBEDDER_MODEL``) evicts the least-recently-used
    native session instead of pinning every model ever seen for the process lifetime. The
    default (4) comfortably covers the common 1–2 concurrent models while capping RSS; a
    misconfigured value (non-int, ``<= 0``) falls back to the default rather than failing —
    a bad knob must never take the embedder path down.
    """
    raw = os.environ.get(env_var, "")
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


#: LRU bound (distinct model names) for the shared text-embedding session cache. Read ONCE
#: at import so the ``lru_cache`` maxsize is fixed for the process; override with
#: ``HIMMY_EMBED_SESSION_CACHE_MAX`` before import (env at process start).
_TEXT_EMBED_CACHE_MAX = _session_cache_maxsize("HIMMY_EMBED_SESSION_CACHE_MAX")


@lru_cache(maxsize=_TEXT_EMBED_CACHE_MAX)
def _load_text_embedding(model_name: str) -> Any:
    """Load (and cache process-wide) the heavy fastembed ``TextEmbedding`` session.

    Keyed by ``model_name`` only: the wrapper's ``dim`` is metadata that never changes
    the native weights, so two :class:`FastEmbedEmbedder`\\ s for the same model share
    one ONNX session (hundreds of ms + tens–hundreds MB saved per duplicate build)
    while producing byte-identical vectors. Wrapped by a BOUNDED :func:`functools.lru_cache`
    (see :data:`_TEXT_EMBED_CACHE_MAX`) so the first caller for a model pays the load and
    every later caller gets the same object, while a process reconfigured across MANY
    distinct model names evicts the least-recently-used native session instead of pinning
    every model ever seen (unbounded ``functools.cache`` did the latter — RSS grew without
    bound). An in-flight embed holds its own reference to the session object, so an LRU
    eviction of the cache ENTRY cannot free a session another thread is still using — the
    object simply lives until its last referent drops (the pre-existing single-instance
    lifetime semantics). Raises a clear :class:`HimmyError` when the ``[embeddings]`` extra
    is absent (never caches the failure — an errored call is not memoised).
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
    call at any time; a subsequent build simply pays the one-time load again. Also clears
    the per-process auto-backend decision memo (see :func:`reset_auto_backend_cache`) so
    the next ``"auto"`` build re-probes Ollama.
    """
    _load_text_embedding.cache_clear()
    reset_auto_backend_cache()
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
        """Embed each document (Ollama embeds one prompt per call).

        Ollama embeds one prompt per HTTP round-trip, so a batch fans out under a
        bounded :func:`asyncio.gather` — up to :func:`_ollama_embed_concurrency` requests
        in flight at once (``HIMMY_OLLAMA_EMBED_CONCURRENCY``, default 8) — instead of a
        serial chain of awaited calls. ``gather`` preserves input order and each prompt is
        embedded independently, so the returned vectors are byte-identical and in the SAME
        order as ``texts``; only the wall-clock latency drops. A cap ``<= 0`` (or a single
        text) runs the fully serial path — the prior behaviour.
        """
        cap = _ollama_embed_concurrency()
        if cap <= 0 or len(texts) <= 1:
            return [await self._embed_one(t) for t in texts]
        semaphore = asyncio.Semaphore(cap)

        async def _bounded(text: str) -> list[float]:
            async with semaphore:
                return await self._embed_one(text)

        return list(await asyncio.gather(*(_bounded(t) for t in texts)))

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

    def _model_impl(self, cache_enabled: bool | None = None) -> Any:
        """Return the fastembed model, sharing one native session per model name.

        The native ``TextEmbedding`` constructor downloads (first time) and
        initialises an ONNX session — heavy, GIL-releasing work — so callers run
        this inside ``asyncio.to_thread`` to avoid blocking the event loop. When the
        shared cache is enabled (the default; see :data:`_EMBED_CACHE_ENV`), the session
        is fetched from the process-wide :func:`_load_text_embedding` cache so every
        :class:`FastEmbedEmbedder` for the same model reuses one ONNX session instead of
        reloading the native model per build. With the cache off, each instance builds
        and holds its own private session (the pre-cache behaviour).

        ``cache_enabled`` lets the caller pass a SINGLE observation of
        :func:`_embed_cache_enabled` so the shared-vs-private decision here and the
        guard-selection in :meth:`_impl_and_guard` are made from the same read — closing
        a TOCTOU where a runtime env flip between two independent reads could pair the
        shared session with a no-op lock. When ``None`` the flag is read here (callers
        outside the guard path are unaffected).

        An explicitly pre-set ``self._impl`` (e.g. a test-injected fake session) always
        wins over the shared cache, so per-instance overrides keep working unchanged.
        """
        if self._impl is not None:
            return self._impl
        if cache_enabled is None:
            cache_enabled = _embed_cache_enabled()
        if cache_enabled:
            # Hold the build lock across the cache lookup+build so a concurrent cold-start
            # for the same model loads the ONNX session EXACTLY ONCE (``functools.cache``
            # runs the loader outside its own lock, so two missers would otherwise each
            # build a session and discard one). After warm-up this is an uncontended
            # dict-hit; the lock is released before any inference runs.
            with _TEXT_EMBED_BUILD_LOCK:
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

        THREAD-SAFETY: :func:`_embed_cache_enabled` reads the env LIVE, and operators/tests
        may flip ``HIMMY_EMBED_CACHE`` at runtime. The flag is therefore read EXACTLY ONCE
        here and threaded into :meth:`_model_impl`, so the guard decision (shared → lock)
        and the impl decision (shared → process-wide session) can never diverge across two
        independent reads — which would otherwise pair the shared session with a no-op
        ``nullcontext`` and let concurrent embeds race on fastembed's mutable state.
        """
        cache_enabled = _embed_cache_enabled()
        shared = self._impl is None and cache_enabled
        impl = self._model_impl(cache_enabled)
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
            # A PRIVATE session (guard is a nullcontext) is unique to this instance and never
            # contends, so embed the whole batch in ONE pass — the exact pre-cache path. A
            # SHARED session (guard is _TEXT_EMBED_LOCK) is serialised process-wide, so hold
            # the lock only per sub-batch and release between: a huge ingest can no longer
            # head-of-line-block a concurrent latency-critical embed_query on the same session.
            subbatch = _embed_subbatch_size() if guard is _TEXT_EMBED_LOCK else 0
            if subbatch <= 0 or len(texts) <= subbatch:
                with guard:
                    return [list(map(float, vec)) for vec in impl.embed(texts)]
            out: list[list[float]] = []
            for start in range(0, len(texts), subbatch):
                chunk = texts[start : start + subbatch]
                with guard:
                    out.extend(list(map(float, vec)) for vec in impl.embed(chunk))
            return out

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


#: Kill-switch for the per-process auto-backend decision cache (default ON). Set
#: ``HIMMY_BACKEND_PROBE_CACHE=off`` (``0``/``false``/``no``) to re-probe Ollama on every
#: :func:`resolve_auto_backend` call — the pre-cache behaviour — e.g. when a long-lived
#: process must notice an Ollama server that came up (or pulled the embed model) mid-run
#: without an explicit :func:`reset_auto_backend_cache`. Read per call.
_BACKEND_PROBE_CACHE_ENV = "HIMMY_BACKEND_PROBE_CACHE"

#: TTL (seconds) after which a NON-terminal auto-backend decision is re-probed, so a
#: process that resolved ``deterministic``/``ollama`` before Ollama was fully up (or its
#: embed model pulled) notices the change without a restart. ``fastembed`` is terminal
#: (importability can't change mid-process) and is cached forever. The default hides the
#: per-turn repeat-probe cost (a turn's 1–4 calls fall well inside one window) while
#: bounding staleness to a few seconds; ``HIMMY_BACKEND_PROBE_TTL`` overrides, ``0`` (or
#: negative/non-numeric) disables the TTL entirely (cache-forever, the prior behaviour).
_BACKEND_PROBE_TTL_ENV = "HIMMY_BACKEND_PROBE_TTL"
_DEFAULT_BACKEND_PROBE_TTL = 10.0

#: Backend decisions that can never improve by re-probing, so they are cached with no TTL.
_TERMINAL_BACKENDS = frozenset({"fastembed"})

#: Process-wide memo of the resolved auto-backend, keyed by the NORMALISED Ollama base
#: URL (see :func:`_normalise_base_url`) and storing ``(decision, monotonic_deadline)``
#: where ``deadline`` is ``None`` for a terminal decision. The decision is stable for the
#: process lifetime — ``fastembed`` importability does not change turn-to-turn — yet
#: ``build_runtime_for_spec`` re-runs it 1–4× *per turn*, each doing blocking ~0.25s +
#: ~0.5s HTTP probes. Caching keyed on ``(normalised base_url, embed model)`` keeps
#: different servers AND different embed models independent (the Ollama leg is
#: model-dependent: it checks the *configured* embed model is pulled) while de-duping
#: trivially-different spellings of the SAME server. A model-blind key would serve an
#: ``ollama`` decision made for a pulled model when a different, un-pulled model is later
#: configured against the same server — building an OllamaEmbedder that 404s at first embed.
_AUTO_BACKEND_CACHE: dict[tuple[str, str], tuple[str, float | None]] = {}

#: Guards :data:`_AUTO_BACKEND_CACHE` so concurrent turns racing on the first probe
#: cannot corrupt the dict or run redundant probes; the actual probe (network I/O) runs
#: *outside* the lock so a slow/hung Ollama can never block unrelated base_urls.
_AUTO_BACKEND_LOCK = threading.Lock()


def _backend_probe_cache_enabled() -> bool:
    """Whether the auto-backend decision cache is active (default ON, env-overridable)."""
    return not env_falsy(_BACKEND_PROBE_CACHE_ENV)


def _backend_probe_ttl() -> float:
    """Re-probe TTL (seconds) for a non-terminal decision; ``0`` ⇒ cache forever.

    A misconfigured value (non-float, negative) falls back to caching-forever (``0``)
    rather than failing — a bad knob must never take the embedder path down.
    """
    raw = os.environ.get(_BACKEND_PROBE_TTL_ENV)
    if raw is None:
        return _DEFAULT_BACKEND_PROBE_TTL
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def _normalise_base_url(base: str) -> str:
    """Canonicalise an Ollama base URL so equivalent spellings share ONE cache entry.

    Strips trailing slashes and lowercases the scheme+host (URLs are case-insensitive
    there) while leaving any path/port untouched — matching the ``base_url.rstrip("/")``
    the network layer itself applies before hitting ``/api/tags``, so
    ``http://localhost:11434`` and ``http://localhost:11434/`` no longer each pay their
    own blocking probe. Purely a cache-key dedupe; the resolved backend is identical.
    """
    trimmed = base.rstrip("/")
    scheme, sep, rest = trimmed.partition("://")
    if not sep:
        return trimmed
    host, slash, path = rest.partition("/")
    return f"{scheme.lower()}://{host.lower()}{slash}{path}"


def reset_auto_backend_cache() -> None:
    """Forget every memoised auto-backend decision (test hook / operator escape hatch).

    Clears :data:`_AUTO_BACKEND_CACHE` so the next :func:`resolve_auto_backend` re-probes
    Ollama — used by tests to assert the probe fires exactly once per base_url and by a
    runtime that reconfigures its Ollama endpoint (or after pulling an embed model) so the
    stale ``deterministic`` decision is not pinned for the process lifetime.
    """
    with _AUTO_BACKEND_LOCK:
        _AUTO_BACKEND_CACHE.clear()


def _probe_auto_backend(base: str, embed_model: str) -> str:
    """Actually probe for the concrete auto-backend for a normalised Ollama ``base`` URL.

    The uncached inner decision (see :func:`resolve_auto_backend` for the caching layer):
    ``fastembed`` → a local Ollama **with the configured ``embed_model`` pulled** →
    ``deterministic``. The Ollama leg is probed for the SPECIFIC ``embed_model`` that will
    actually be embedded with, so switching to an un-pulled model degrades to deterministic
    instead of 404-ing at embed time.
    """
    if fastembed_available():
        return "fastembed"
    if ollama_reachable(base) and ollama_embed_model_available(
        model=embed_model, base_url=base
    ):
        return "ollama"
    return "deterministic"


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

    The decision is **memoised per process, keyed on ``(NORMALISED Ollama base URL,
    configured embed model)``** (default ON; disable with ``HIMMY_BACKEND_PROBE_CACHE=off``).
    Within a turn ``build_runtime_for_spec`` calls this several times; the memo collapses
    those to at most one blocking probe per window — a pure latency win with a
    byte-identical result. Equivalent spellings of the same server (trailing slash, host
    case) share one entry (see :func:`_normalise_base_url`). The embed model is part of the
    key because the Ollama leg is model-dependent: a decision made while one model is pulled
    must NOT be served when a different (possibly un-pulled) model is later configured
    against the same server — that would build an OllamaEmbedder that 404s at first embed
    instead of degrading to deterministic. A ``fastembed`` decision is terminal (cached
    forever). A ``deterministic``/``ollama`` decision is re-probed after
    ``HIMMY_BACKEND_PROBE_TTL`` seconds (default 10; ``0`` ⇒ cache forever) so a
    long-lived process that started before Ollama was up (or before the embed model was
    pulled) UPGRADES to semantic embeddings without a restart, instead of pinning the
    degraded decision for the process lifetime. :func:`reset_auto_backend_cache` (also
    fired by :func:`reset_embedder_cache`) clears the memo outright for tests/reconfig.
    """
    base = _normalise_base_url(ollama_base_url or _DEFAULT_OLLAMA_URL)
    embed_model = default_ollama_embed_model()
    key = (base, embed_model)
    if not _backend_probe_cache_enabled():
        return _probe_auto_backend(base, embed_model)
    now = time.monotonic()
    cached = _AUTO_BACKEND_CACHE.get(key)
    if cached is not None:
        decision, deadline = cached
        if deadline is None or now < deadline:
            return decision
    # Probe outside the lock so a slow Ollama never blocks unrelated base_urls. A benign
    # race just re-probes (identical result); the last writer wins under the lock. A
    # terminal (``fastembed``) decision is pinned (deadline ``None``); a non-terminal one
    # carries a TTL deadline so liveness changes are eventually noticed.
    resolved = _probe_auto_backend(base, embed_model)
    ttl = _backend_probe_ttl()
    deadline = None if (resolved in _TERMINAL_BACKENDS or ttl <= 0) else now + ttl
    with _AUTO_BACKEND_LOCK:
        _AUTO_BACKEND_CACHE[key] = (resolved, deadline)
        return resolved


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
    "reset_auto_backend_cache",
    "reset_embedder_cache",
]
