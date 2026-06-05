"""Token-bucket rate limiting for the BFF (WS3.2).

A real limiter behind the existing ``set_rate_limiter`` hook: each caller gets a token
bucket (``capacity`` = burst, refilled at ``rate``/``window`` per second). Over budget →
HTTP 429 with a ``Retry-After``. Keyed by the authenticated principal (so it's per-caller
/ per-tenant) and falling back to the client IP for the anonymous/offline path.

Off by default (no ``HIMMY_RATE_LIMIT`` → no limiter → unchanged behavior). The in-memory
bucket store fits a single process; for multi-replica deployments swap a shared store
(e.g. Redis) behind the same ``__call__`` interface — the limiter is the only thing the
BFF holds.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from fastapi import Request


def default_key(request: Request) -> str:
    """Rate-limit key: the authenticated principal, else the client IP."""
    from himmy.api.auth.base import client_ip
    from himmy.api.auth.context import get_principal

    principal = get_principal(request)
    if not principal.all_tenants or principal.auth_method != "anonymous":
        if principal.subject and principal.subject != "anonymous":
            return f"sub:{principal.subject}"
    return f"ip:{client_ip(request) or 'unknown'}"


class TokenBucketRateLimiter:
    """A per-key token-bucket limiter usable as the ``set_rate_limiter`` hook."""

    def __init__(
        self,
        *,
        rate: float,
        capacity: float,
        window: float = 1.0,
        key_func: Callable[[Request], str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Refill ``rate`` tokens per ``window`` seconds, up to ``capacity`` (burst)."""
        if rate <= 0 or capacity <= 0 or window <= 0:
            raise ValueError("rate, capacity and window must be positive")
        self._refill_per_sec = rate / window
        self._capacity = capacity
        self._key_func = key_func or default_key
        self._clock = clock or time.monotonic
        self._buckets: dict[str, tuple[float, float]] = {}

    def __call__(self, request: Request) -> None:
        """Consume one token for the request's key; raise 429 when exhausted."""
        key = self._key_func(request)
        now = self._clock()
        tokens, last = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._refill_per_sec)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            retry = max(1, int((1.0 - tokens) / self._refill_per_sec + 0.999))
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry)},
            )
        self._buckets[key] = (tokens - 1.0, now)


def build_rate_limiter() -> TokenBucketRateLimiter | None:
    """Build a limiter from ``HIMMY_RATE_LIMIT`` env, or ``None`` when unset (off)."""
    import os

    raw = os.environ.get("HIMMY_RATE_LIMIT")
    if not raw:
        return None
    rate = float(raw)
    window = float(os.environ.get("HIMMY_RATE_WINDOW", "1"))
    capacity = float(os.environ.get("HIMMY_RATE_BURST", raw))
    return TokenBucketRateLimiter(rate=rate, capacity=capacity, window=window)


__all__ = ["TokenBucketRateLimiter", "build_rate_limiter", "default_key"]
