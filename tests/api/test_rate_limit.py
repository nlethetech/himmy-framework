"""WS3.2 — token-bucket rate limiting (per-key, 429 + Retry-After)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.ratelimit import TokenBucketRateLimiter


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Req:
    """Minimal request stand-in carrying a fixed rate-limit key."""

    def __init__(self, key: str) -> None:
        self._key = key


def _key(req: _Req) -> str:
    return req._key


def test_bucket_allows_burst_then_blocks() -> None:
    clock = _Clock()
    limiter = TokenBucketRateLimiter(
        rate=1, capacity=2, window=1.0, key_func=_key, clock=clock
    )
    req = _Req("a")
    limiter(req)  # token 2 -> 1
    limiter(req)  # token 1 -> 0
    with pytest.raises(HTTPException) as exc:
        limiter(req)  # empty -> 429
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_bucket_refills_over_time() -> None:
    clock = _Clock()
    limiter = TokenBucketRateLimiter(
        rate=1, capacity=1, window=1.0, key_func=_key, clock=clock
    )
    req = _Req("a")
    limiter(req)
    with pytest.raises(HTTPException):
        limiter(req)
    clock.t += 1.0  # one window later → one token back
    limiter(req)  # allowed again


def test_keys_are_isolated() -> None:
    clock = _Clock()
    limiter = TokenBucketRateLimiter(
        rate=1, capacity=1, window=1.0, key_func=_key, clock=clock
    )
    limiter(_Req("a"))
    limiter(_Req("b"))  # different key → its own bucket, allowed
    with pytest.raises(HTTPException):
        limiter(_Req("a"))


# ---------------------------------------------------------------- BFF integration
def test_rate_limit_enforced_through_the_bff() -> None:
    app = create_app(ApiContainer.build_default())
    app.state.rate_limiter = TokenBucketRateLimiter(rate=1, capacity=2, window=60.0)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    resp = client.get("/health")
    assert resp.status_code == 429
    assert "retry-after" in {k.lower() for k in resp.headers}


def test_offline_default_has_no_limiter() -> None:
    client = TestClient(create_app(ApiContainer.build_default()))
    for _ in range(20):
        assert client.get("/health").status_code == 200
