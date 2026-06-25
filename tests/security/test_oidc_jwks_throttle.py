"""Regression: OIDC JWKS force-refetch is throttled + single-flighted (cache stampede).

Friend-audit finding #1: a bearer token with an unknown ``kid`` makes ``_signing_key``
call ``JwksCache.get(force=True)``, which used to issue an outbound JWKS GET every time.
An unauthenticated flood of random-``kid`` tokens → a flood of outbound fetches (a DoS on
this server and the IdP). The fix rate-limits forced refetches (``min_refetch_interval``)
and coalesces concurrent ones behind a single-flight lock.
"""

from __future__ import annotations

import asyncio

import pytest

from himmy.api.auth.base import AuthError
from himmy.api.auth.oidc import JwksCache


class _FakeResp:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


class _FakeClient:
    calls = 0

    def __init__(self, *a, **k) -> None:
        pass

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *a) -> bool:
        return False

    async def get(self, _url: str) -> _FakeResp:
        _FakeClient.calls += 1
        return _FakeResp({"keys": [{"kid": "k1"}]})


@pytest.fixture(autouse=True)
def _patch_httpx(monkeypatch) -> None:
    _FakeClient.calls = 0
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


def test_force_refetch_is_throttled() -> None:
    """A burst of forced refetches within the cooldown triggers only ONE outbound fetch."""
    cache = JwksCache("http://idp/jwks", ttl=3600.0, min_refetch_interval=30.0)
    asyncio.run(cache.get(force=True))
    assert _FakeClient.calls == 1
    for _ in range(50):  # unknown-kid spam
        asyncio.run(cache.get(force=True))
    assert _FakeClient.calls == 1  # the stampede is blocked


def test_force_refetch_allowed_after_cooldown() -> None:
    """With the cooldown elapsed, a forced refetch is permitted (rotation still works)."""
    cache = JwksCache("http://idp/jwks", ttl=0.0, min_refetch_interval=0.0)
    asyncio.run(cache.get(force=True))
    assert _FakeClient.calls == 1
    asyncio.run(cache.get(force=True))
    assert _FakeClient.calls == 2


def test_concurrent_forces_coalesce() -> None:
    """20 concurrent forced refetches collapse into ONE outbound fetch (single-flight)."""
    cache = JwksCache("http://idp/jwks", ttl=3600.0, min_refetch_interval=30.0)

    async def hammer() -> None:
        await asyncio.gather(*[cache.get(force=True) for _ in range(20)])

    asyncio.run(hammer())
    assert _FakeClient.calls == 1


class _FailingClient:
    """A fake httpx client simulating a DOWN/erroring IdP (the worst case for the throttle)."""

    calls = 0

    def __init__(self, *a, **k) -> None:
        pass

    async def __aenter__(self) -> _FailingClient:
        return self

    async def __aexit__(self, *a) -> bool:
        return False

    async def get(self, _url: str) -> _FakeResp:
        _FailingClient.calls += 1
        raise RuntimeError("idp unreachable")


def test_failing_idp_sequential_flood_is_bounded(monkeypatch) -> None:
    """Unknown-kid flood while the IdP is DOWN must not hammer it (cache stays empty)."""
    import httpx

    _FailingClient.calls = 0
    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)
    cache = JwksCache("http://idp/jwks", ttl=3600.0, min_refetch_interval=30.0)
    for _ in range(1000):
        with pytest.raises((RuntimeError, AuthError)):
            asyncio.run(cache.get(force=True))
    # Only the FIRST attempt reaches the IdP; the rest are throttled (empty cache → raise).
    assert _FailingClient.calls == 1


def test_failing_idp_concurrent_flood_is_bounded(monkeypatch) -> None:
    """200 concurrent forced refetches against a DOWN IdP collapse to ONE outbound call."""
    import httpx

    _FailingClient.calls = 0
    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)
    cache = JwksCache("http://idp/jwks", ttl=3600.0, min_refetch_interval=30.0)

    async def hammer() -> list:
        return await asyncio.gather(
            *[cache.get(force=True) for _ in range(200)], return_exceptions=True
        )

    results = asyncio.run(hammer())
    assert all(isinstance(r, Exception) for r in results)
    assert _FailingClient.calls == 1
