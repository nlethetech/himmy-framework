"""Inference kernel: a pluggable response cache (INF-9).

``LLMConfig.use_cache`` is a real cost-control lever: identical opted-in requests
should not re-hit the provider. The cache is keyed by a stable hash of the
request's caching-relevant surface (model_key, messages, response_format,
output_json_schema, workflow, and the non-cache generation params) so two requests
that would yield the same model call share a result.

The default :class:`NoopInferenceCache` does nothing — behavior is unchanged unless
a real cache is wired into :class:`InferenceService`. :class:`InMemoryTTLCache` is a
process-local, TTL-bounded implementation usable offline and in tests.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Protocol, runtime_checkable

from himmy.services.inference.models import InferenceRequest, InferenceResponse


def compute_cache_key(request: InferenceRequest) -> str:
    """Compute a stable cache key from a request's caching-relevant surface.

    ``request_id`` (random) and ``use_cache`` itself are excluded so identical
    logical calls collide; everything that changes the model's output is included.
    """
    gen = {
        k: v for k, v in sorted(request.generation_params.items()) if k != "use_cache"
    }
    payload = {
        "model_key": request.model_key,
        "route_override": request.route_override,
        "response_format": (
            request.response_format.value if request.response_format else None
        ),
        "output_json_schema": request.output_json_schema,
        "workflow": (
            request.workflow.model_dump() if request.workflow is not None else None
        ),
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
            }
            for m in request.messages
        ],
        "tool_names": sorted(t.name for t in request.bound_tools),
        "generation_params": gen,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@runtime_checkable
class InferenceCache(Protocol):
    """The pluggable response-cache contract used by :class:`InferenceService`."""

    def key_for(self, request: InferenceRequest) -> str:
        """Return a stable cache key for ``request``."""
        ...

    async def get(self, key: str) -> InferenceResponse | None:
        """Return a cached response for ``key`` or ``None`` (incl. on expiry)."""
        ...

    async def set(self, key: str, response: InferenceResponse) -> None:
        """Store ``response`` under ``key`` (only successful responses are cached)."""
        ...


class NoopInferenceCache:
    """The default cache: never stores, never hits. Keeps behavior unchanged."""

    def key_for(self, request: InferenceRequest) -> str:
        """Compute a key (cheap) even though nothing is stored."""
        return compute_cache_key(request)

    async def get(self, key: str) -> InferenceResponse | None:
        """Always a miss."""
        return None

    async def set(self, key: str, response: InferenceResponse) -> None:
        """No-op."""
        return None


class InMemoryTTLCache:
    """A process-local, TTL-bounded response cache (offline-friendly).

    Only SUCCESS responses are stored. Entries expire after ``ttl_seconds``; a
    soft ``max_entries`` bound evicts the oldest entries to cap memory.
    """

    def __init__(self, *, ttl_seconds: float = 300.0, max_entries: int = 1024) -> None:
        """Configure the TTL and the soft entry cap."""
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._store: dict[str, tuple[float, InferenceResponse]] = {}

    def key_for(self, request: InferenceRequest) -> str:
        """Return the stable cache key for ``request``."""
        return compute_cache_key(request)

    async def get(self, key: str) -> InferenceResponse | None:
        """Return a live cached response or ``None`` when missing/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, response = entry
        if time.monotonic() >= expires_at:
            self._store.pop(key, None)
            return None
        return response.model_copy(deep=True)

    async def set(self, key: str, response: InferenceResponse) -> None:
        """Store a SUCCESS response with a TTL; evict oldest past the cap."""
        from himmy.services.inference.models import InferenceStatus

        if response.status != InferenceStatus.SUCCESS:
            return
        if len(self._store) >= self._max_entries and key not in self._store:
            # Evict the soonest-to-expire entry to keep the cap.
            oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            self._store.pop(oldest, None)
        self._store[key] = (
            time.monotonic() + self._ttl_seconds,
            response.model_copy(deep=True),
        )
