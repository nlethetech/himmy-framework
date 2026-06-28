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
    # A conformant proxy APPENDS the real client; the right-most non-trusted entry
    # (here ``5.6.7.8``) is the genuine client, NOT the attacker-controlled left-most.
    r = _req("203.0.113.9", "1.2.3.4, 5.6.7.8")
    assert client_ip(r) == "5.6.7.8"


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


def test_leftmost_xff_forgery_does_not_mint_fresh_key(monkeypatch: Any) -> None:
    """A client behind a trusted proxy cannot rotate the LEFT-most XFF to bypass.

    Regression for the leftmost-XFF trust pitfall: a real proxy appends the genuine
    client to the right, so two requests from the SAME client that prepend a different
    forged left-most token must still resolve to the same client IP (one bucket), not a
    fresh bucket each time.
    """
    monkeypatch.setenv("HIMMY_TRUSTED_PROXIES", "203.0.113.9")
    from himmy.api.ratelimit import default_key

    # Same genuine client (198.51.100.7), incrementing forged left-most entries.
    forged_a = _req("203.0.113.9", "10.0.0.1, 198.51.100.7")
    forged_b = _req("203.0.113.9", "10.0.0.2, 198.51.100.7")
    assert client_ip(forged_a) == "198.51.100.7"
    assert client_ip(forged_b) == "198.51.100.7"
    assert default_key(forged_a) == default_key(forged_b) == "ip:198.51.100.7"


def test_multiple_trusted_proxy_hops_are_peeled(monkeypatch: Any) -> None:
    """Trailing trusted-proxy hops are peeled; the right-most non-trusted is the client."""
    monkeypatch.setenv("HIMMY_TRUSTED_PROXIES", "203.0.113.9, 203.0.113.10")
    # client -> edge(203.0.113.10) -> inner(203.0.113.9 = peer). Both proxies appended.
    r = _req("203.0.113.9", "198.51.100.7, 203.0.113.10")
    assert client_ip(r) == "198.51.100.7"


def test_all_forwarded_entries_trusted_falls_back_to_peer(monkeypatch: Any) -> None:
    """When every XFF entry is a trusted proxy, fall back to the real peer."""
    monkeypatch.setenv("HIMMY_TRUSTED_PROXIES", "203.0.113.9, 203.0.113.10")
    r = _req("203.0.113.9", "203.0.113.10")
    assert client_ip(r) == "203.0.113.9"
