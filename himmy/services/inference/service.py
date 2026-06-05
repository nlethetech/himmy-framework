"""Inference kernel: the async ``InferenceService`` entry point.

Wraps a client manager with the things production needs on top of raw model
calls: a per-attempt timeout ceiling, bounded retries on retryable error codes
only, ordered batching with bounded concurrency, an optional response cache, a
streaming entry point, and lifecycle event emission.

Service-level failure contract (INF-1/INF-2/INF-10): ``run`` MUST NEVER raise for
provider/manager errors. Any exception a client manager raises is normalized to a
``FAILED`` :class:`InferenceResponse` carrying a typed :class:`InferenceError`, so
retries, latency stamping, and the ``INFERENCE_FAILED`` event always fire and a
single bad request can never kill a batch.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from himmy.core.errors import HimmyError
from himmy.services.inference.cache import InferenceCache, NoopInferenceCache
from himmy.services.inference.models import (
    RETRYABLE_ERROR_CODES,
    BatchInferenceRequest,
    BatchInferenceResponse,
    InferenceError,
    InferenceErrorCode,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.core.events import EventSink
    from himmy.services.inference.client_manager import ClientManager

from himmy.core.events import EventType, RunEvent


class StreamDelta(BaseModel):
    """A single streamed chunk of an inference response.

    Emitted by :meth:`InferenceService.run_stream`. ``delta`` is the incremental
    text since the previous chunk; ``done`` marks the final chunk, which also
    carries the fully-materialized :class:`InferenceResponse` in ``response``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    delta: str = ""
    index: int = 0
    done: bool = False
    response: InferenceResponse | None = None


# Exception types (by name) the providers may raise that map to known codes. We
# match by class name so we never import provider SDKs on the offline path.
_ERROR_NAME_MAP: dict[str, tuple[InferenceErrorCode, bool]] = {
    "TimeoutError": (InferenceErrorCode.TIMEOUT, True),
    "ConnectTimeout": (InferenceErrorCode.TIMEOUT, True),
    "ReadTimeout": (InferenceErrorCode.TIMEOUT, True),
    "ConnectError": (InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
    "ConnectionError": (InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
    "RemoteProtocolError": (InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
    "UnexpectedModelBehavior": (InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
}


def _normalize_exception(exc: BaseException) -> InferenceError:
    """Map an arbitrary provider/manager exception to a typed InferenceError.

    Unknown exceptions become a non-retryable ``UNKNOWN`` error; a small set of
    transport-ish exception class names map to retryable transient codes. We key
    on the class name to avoid importing provider SDKs.
    """
    code, retryable = _ERROR_NAME_MAP.get(
        type(exc).__name__, (InferenceErrorCode.UNKNOWN, False)
    )
    return InferenceError(
        code=code, message=str(exc) or type(exc).__name__, retryable=retryable
    )


class InferenceService:
    """The single async entry point for model calls across the framework."""

    def __init__(
        self,
        client_manager: ClientManager,
        *,
        max_retries: int = 2,
        default_timeout_seconds: float = 90.0,
        retry_base_delay_seconds: float = 0.2,
        retry_jitter_seconds: float = 0.1,
        event_sink: EventSink | None = None,
        timeout_grace_seconds: float = 0.25,
        timeout_grace_factor: float = 1.05,
        cache: InferenceCache | None = None,
    ) -> None:
        """Configure the client manager, retry/timeout policy, cache, and sink.

        ``timeout_grace_*`` (INF-3/INF-11) make the hard ``asyncio.wait_for``
        ceiling proportional: ``ceiling = timeout * factor + grace``, so a
        sub-second per-request timeout is no longer dominated by a fixed +1.0s.
        ``cache`` (INF-9) is a pluggable response cache honored only when a
        request opts in via ``generation_params['use_cache']``; the default is a
        no-op so behavior is unchanged unless wired.
        """
        self._client_manager = client_manager
        self._max_retries = max_retries
        self._default_timeout_seconds = default_timeout_seconds
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_jitter_seconds = retry_jitter_seconds
        self._event_sink = event_sink
        self._timeout_grace_seconds = timeout_grace_seconds
        self._timeout_grace_factor = timeout_grace_factor
        self._cache: InferenceCache = cache or NoopInferenceCache()

    def _ceiling(self, timeout: float) -> float:
        """Proportional hard ceiling for the outer ``asyncio.wait_for`` guard."""
        return timeout * self._timeout_grace_factor + self._timeout_grace_seconds

    @staticmethod
    def _wants_cache(request: InferenceRequest) -> bool:
        """True when the request opted into the response cache."""
        return bool(request.generation_params.get("use_cache"))

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        """Run a single request with a timeout ceiling and bounded retries.

        Only :data:`RETRYABLE_ERROR_CODES` (rate limits, timeouts, transient
        provider outages) are retried; AUTH/QUOTA/INVALID_REQUEST bubble up as a
        FAILED response. This method never raises for provider/manager errors —
        every failure is normalized into a FAILED :class:`InferenceResponse`.
        """
        await self._emit(
            EventType.INFERENCE_REQUESTED,
            request_id=request.request_id,
            payload={"model_key": request.model_key},
        )
        started = time.perf_counter()

        # --- response cache (INF-9): serve identical opted-in requests warm ----
        use_cache = self._wants_cache(request)
        cache_key: str | None = None
        if use_cache:
            cache_key = self._cache.key_for(request)
            cached = await self._cache.get(cache_key)
            if cached is not None:
                hit = cached.model_copy(deep=True)
                hit.request_id = request.request_id
                hit.latency_ms = (time.perf_counter() - started) * 1000.0
                hit.metadata = {**hit.metadata, "cache_hit": True}
                await self._emit(
                    EventType.INFERENCE_SUCCEEDED,
                    request_id=request.request_id,
                    latency_ms=hit.latency_ms,
                    cost=0.0,
                    payload={
                        "input_tokens": hit.input_tokens,
                        "output_tokens": hit.output_tokens,
                        "cache_hit": True,
                    },
                )
                return hit

        timeout = request.timeout_seconds or self._default_timeout_seconds
        ceiling = self._ceiling(timeout)
        last_response: InferenceResponse | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._run_once(request), timeout=ceiling
                )
            except TimeoutError:
                response = InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.TIMEOUT,
                        message=f"inference exceeded {ceiling:.2f}s ceiling",
                        retryable=True,
                    ),
                )

            last_response = response
            if response.status == InferenceStatus.SUCCESS:
                response.latency_ms = (time.perf_counter() - started) * 1000.0
                if use_cache and cache_key is not None:
                    await self._cache.set(cache_key, response)
                await self._emit(
                    EventType.INFERENCE_SUCCEEDED,
                    request_id=request.request_id,
                    latency_ms=response.latency_ms,
                    cost=response.cost,
                    payload={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                )
                return response

            # Failure: retry only when retryable and attempts remain.
            if (
                attempt < self._max_retries
                and response.error is not None
                and response.error.retryable
                and response.error.code in RETRYABLE_ERROR_CODES
            ):
                await asyncio.sleep(self._backoff_delay(attempt))
                continue
            break

        # Exhausted: stamp latency and emit the failure.
        assert last_response is not None
        last_response.latency_ms = (time.perf_counter() - started) * 1000.0
        await self._emit(
            EventType.INFERENCE_FAILED,
            request_id=request.request_id,
            latency_ms=last_response.latency_ms,
            error=(last_response.error.message if last_response.error else "unknown"),
        )
        return last_response

    async def _run_once(self, request: InferenceRequest) -> InferenceResponse:
        """Single delegation to the client manager with error normalization.

        This is the service-level boundary (INF-1) that guarantees the manager's
        ``generate`` can raise ANY exception without escaping ``run``:
        ``NotImplementedError`` and ``HimmyError`` (e.g. a forced-tool request with no
        bound tool) map to a non-retryable INVALID_REQUEST; any other exception is
        normalized to a typed FAILED response. Forced-tool (``ResponseFormat.TOOL``) is
        normalized here so every provider drives it through the standard tool path.
        """
        from himmy.services.inference.client_manager import normalize_forced_tool

        try:
            return await self._client_manager.generate(normalize_forced_tool(request))
        except asyncio.CancelledError:  # honor cancellation / timeout cleanly
            raise
        except NotImplementedError as exc:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                error=InferenceError(
                    code=InferenceErrorCode.INVALID_REQUEST,
                    message=str(exc) or "not implemented",
                    retryable=False,
                ),
            )
        except HimmyError as exc:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                error=InferenceError(
                    code=InferenceErrorCode.INVALID_REQUEST,
                    message=str(exc),
                    retryable=False,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - normalize ALL provider errors
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                error=_normalize_exception(exc),
            )

    async def run_batch(self, batch: BatchInferenceRequest) -> BatchInferenceResponse:
        """Run a batch with bounded concurrency, preserving input order.

        Failure-isolated (INF-2): one raising/failing request can never abort the
        batch. Each request goes through ``run`` (which never raises); as a
        belt-and-suspenders guard, any escaped exception is still converted to a
        FAILED response before tallying.
        """
        semaphore = asyncio.Semaphore(max(1, batch.max_concurrency))
        started = time.perf_counter()

        async def _guarded(req: InferenceRequest) -> InferenceResponse:
            async with semaphore:
                try:
                    return await self.run(req)
                except Exception as exc:  # noqa: BLE001 - never let one kill the batch
                    return InferenceResponse(
                        request_id=req.request_id,
                        status=InferenceStatus.FAILED,
                        error=_normalize_exception(exc),
                    )

        responses = await asyncio.gather(
            *(_guarded(req) for req in batch.requests), return_exceptions=False
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        success = sum(1 for r in responses if r.status == InferenceStatus.SUCCESS)
        return BatchInferenceResponse(
            responses=list(responses),
            success_count=success,
            failure_count=len(responses) - success,
            elapsed_ms=elapsed_ms,
        )

    async def run_stream(
        self, request: InferenceRequest, *, chunk_size: int = 24
    ) -> AsyncIterator[StreamDelta]:
        """Stream an inference response as incremental text deltas (INF-7).

        If the client manager exposes ``generate_stream`` (the pydantic-ai path
        backed by ``agent.run_stream``), deltas come straight from the provider.
        Otherwise this falls back to a deterministic, offline-testable chunking of
        the buffered ``run`` output. The terminal chunk has ``done=True`` and
        carries the fully-materialized :class:`InferenceResponse`.
        """
        manager_stream = getattr(self._client_manager, "generate_stream", None)
        if manager_stream is not None:
            index = 0
            final: InferenceResponse | None = None
            async for piece in manager_stream(request):
                if isinstance(piece, InferenceResponse):
                    final = piece
                    continue
                yield StreamDelta(
                    request_id=request.request_id, delta=str(piece), index=index
                )
                index += 1
            if final is None:
                final = await self.run(request)
            yield StreamDelta(
                request_id=request.request_id,
                delta="",
                index=index,
                done=True,
                response=final,
            )
            return

        # Offline fallback: buffer via run(), then emit deterministic chunks.
        response = await self.run(request)
        text = response.output_text or ""
        index = 0
        for start in range(0, len(text), max(1, chunk_size)):
            piece = text[start : start + max(1, chunk_size)]
            yield StreamDelta(request_id=request.request_id, delta=piece, index=index)
            index += 1
        yield StreamDelta(
            request_id=request.request_id,
            delta="",
            index=index,
            done=True,
            response=response,
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter for retry attempt ``attempt`` (0-based)."""
        base = self._retry_base_delay_seconds * (2**attempt)
        return base + random.uniform(0.0, self._retry_jitter_seconds)

    async def _emit(self, event_type: EventType, **kwargs: object) -> None:
        """Emit a lifecycle event to the sink if one is configured (best-effort)."""
        if self._event_sink is None:
            return
        try:
            await self._event_sink.append_event(
                RunEvent(event_type=event_type, **kwargs)  # type: ignore[arg-type]
            )
        except Exception:  # noqa: BLE001 - observability must never break inference
            pass
