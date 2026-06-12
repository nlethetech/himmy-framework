"""X-Forwarded-For is honored only from a configured trusted proxy.

Trusting XFF unconditionally lets any client rotate the header to land in a fresh
rate-limit bucket every request (bypass) and to spoof the audit source IP. The header
is honored only when the transport peer is in ``HIMMY_TRUSTED_PROXIES``; otherwise the
real ``request.client.host`` is used, and the real peer is always retained for audit.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request

from himmy.api.auth.base import client_ip, peer_ip


def _req(peer: str | None, xff: str | None = None) -> Request:
    """A minimal Starlette request with a given transport peer + optional XFF."""
    headers: list[tuple[bytes, bytes]] = []
    if xff is not None:
        headers.append((b"x-forwarded-for", xff.encode()))
    scope: dict[str, Any] = {
        "type": "http",
        "headers": headers,
        "client": (peer, 12345) if peer else None,
    }
    return Request(scope)


def test_xff_ignored_when_peer_untrusted(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_TRUSTED_PROXIES", raising=False)
    r = _req("203.0.113.9", "1.2.3.4, 5.6.7.8")
    assert client_ip(r) == "203.0.113.9"  # the real peer, NOT the forged header


def test_xff_honored_when_peer_trusted(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_TRUSTED_PROXIES", "203.0.113.9")
    r = _req("203.0.113.9", "1.2.3.4, 5.6.7.8")
    assert client_ip(r) == "1.2.3.4"  # first hop behind the trusted proxy


def test_xff_ignored_when_peer_not_in_allowlist(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_TRUSTED_PROXIES", "10.0.0.1")
    r = _req("203.0.113.9", "1.2.3.4")
    assert client_ip(r) == "203.0.113.9"


def test_trusted_peer_without_xff_uses_peer(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_TRUSTED_PROXIES", "203.0.113.9")
    r = _req("203.0.113.9")
    assert client_ip(r) == "203.0.113.9"


def test_peer_ip_is_always_the_real_peer(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_TRUSTED_PROXIES", "203.0.113.9")
    r = _req("203.0.113.9", "1.2.3.4")
    # client_ip follows the trusted XFF, but peer_ip keeps the real peer for audit.
    assert client_ip(r) == "1.2.3.4"
    assert peer_ip(r) == "203.0.113.9"


def test_rate_limit_key_not_bypassable_via_xff(monkeypatch: Any) -> None:
    """The default rate-limit key keys on the real peer, so XFF rotation cannot bypass."""
    monkeypatch.delenv("HIMMY_TRUSTED_PROXIES", raising=False)
    from himmy.api.ratelimit import default_key

    same_peer_a = _req("198.51.100.7", "9.9.9.9")
    same_peer_b = _req("198.51.100.7", "8.8.8.8")
    # Two requests from the same peer with different forged XFF must share a key.
    assert default_key(same_peer_a) == default_key(same_peer_b)
    assert default_key(same_peer_a) == "ip:198.51.100.7"
