"""Tests for HttpxFetcher's SSRF redirect guard + response size cap (mock transport).

These cover the audit finding that ``web_fetch`` followed redirects past the SSRF
guard (a 302 to 169.254.169.254 / loopback / RFC1918 reached the model) and that
fetched bodies were uncapped (a multi-GB response could exhaust memory).
"""

from __future__ import annotations

import httpx
import pytest

from himmy.connectors.fetcher import HttpxFetcher, ResponseTooLarge


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_redirect_target_is_passed_through_the_guard() -> None:
    """Every redirect hop's Location is re-validated by the guard before following."""
    seen: list[str] = []

    def guard(url: str) -> str:
        seen.append(url)
        return url

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "http://example.org/final"})
        return httpx.Response(200, text="final body")

    fetcher = HttpxFetcher(transport=_transport(handler), guard=guard, retries=0)
    assert fetcher.get_text("http://example.org/start") == "final body"
    # The guard ran on BOTH the initial URL and the redirect target.
    assert seen == ["http://example.org/start", "http://example.org/final"]


def test_redirect_to_blocked_host_is_refused_by_guard() -> None:
    """A 302 to an internal address is rejected — the guard raises before the hop."""

    def guard(url: str) -> str:
        if "169.254.169.254" in url:
            raise RuntimeError("blocked internal address")
        return url

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            # Classic SSRF-via-redirect: bounce to the cloud metadata endpoint.
            return httpx.Response(
                302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}
            )
        return httpx.Response(200, text="SECRET CREDS")  # must never be reached

    fetcher = HttpxFetcher(transport=_transport(handler), guard=guard, retries=0)
    with pytest.raises(RuntimeError, match="blocked internal address"):
        fetcher.get_text("http://example.org/start")


def test_no_guard_still_follows_redirects() -> None:
    """Without a guard the fetcher still follows redirects (legacy behavior)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(301, headers={"Location": "http://example.org/b"})
        return httpx.Response(200, text="landed")

    fetcher = HttpxFetcher(transport=_transport(handler), retries=0)
    assert fetcher.get_text("http://example.org/a") == "landed"


def test_relative_redirect_location_is_resolved() -> None:
    """A relative Location header resolves against the current URL before guarding."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/dir/start":
            return httpx.Response(302, headers={"Location": "next"})
        return httpx.Response(200, text="ok")

    fetcher = HttpxFetcher(
        transport=_transport(handler), guard=seen.append, retries=0
    )
    assert fetcher.get_text("http://example.org/dir/start") == "ok"
    assert seen == ["http://example.org/dir/start", "http://example.org/dir/next"]


def test_redirect_loop_is_bounded() -> None:
    """An endless redirect chain is bounded instead of looping forever."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://example.org/loop"})

    fetcher = HttpxFetcher(transport=_transport(handler), retries=0)
    with pytest.raises(httpx.TooManyRedirects):
        fetcher.get_text("http://example.org/loop")


def test_oversize_text_response_is_rejected() -> None:
    """A body over max_bytes raises ResponseTooLarge instead of loading into memory."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000)

    fetcher = HttpxFetcher(transport=_transport(handler), retries=0, max_bytes=1024)
    with pytest.raises(ResponseTooLarge):
        fetcher.get_text("http://example.org/big")


def test_oversize_bytes_response_is_rejected() -> None:
    """The byte path is capped too (feeds / spreadsheets)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"y" * 5000)

    fetcher = HttpxFetcher(transport=_transport(handler), retries=0, max_bytes=1024)
    with pytest.raises(ResponseTooLarge):
        fetcher.get_bytes("http://example.org/big")


def test_body_at_the_cap_is_allowed() -> None:
    """A body exactly at the cap is fine — the cap rejects only strictly-larger ones."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"z" * 1024)

    fetcher = HttpxFetcher(transport=_transport(handler), retries=0, max_bytes=1024)
    assert fetcher.get_bytes("http://example.org/ok") == b"z" * 1024


def test_per_request_auth_header_is_sent_on_the_request() -> None:
    """A per-request Authorization header rides on the GET (the connector-auth path)."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, text="ok")

    fetcher = HttpxFetcher(transport=_transport(handler), retries=0)
    fetcher.get_text(
        "http://example.org/api", headers={"Authorization": "Bearer tok"}
    )
    assert seen == ["Bearer tok"]


def test_auth_header_is_dropped_on_cross_origin_redirect() -> None:
    """A credential is NEVER replayed to a different host across a redirect hop."""
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get("Authorization")))
        if request.url.host == "api.github.com":
            # A (guarded, still public) redirect to a THIRD-PARTY host.
            return httpx.Response(
                302, headers={"Location": "http://evil.example/collect"}
            )
        return httpx.Response(200, text="ok")

    fetcher = HttpxFetcher(transport=_transport(handler), retries=0)
    fetcher.get_text(
        "http://api.github.com/issues", headers={"Authorization": "Bearer SECRET"}
    )
    # First hop carries the token; the cross-origin hop must NOT.
    assert ("api.github.com", "Bearer SECRET") in seen
    assert ("evil.example", None) in seen
    assert all(host != "evil.example" or auth is None for host, auth in seen)


def test_auth_header_is_kept_on_same_origin_redirect() -> None:
    """A same-origin redirect (e.g. a trailing-slash bounce) keeps the credential."""
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("Authorization")))
        if request.url.path == "/issues":
            return httpx.Response(302, headers={"Location": "/issues/"})
        return httpx.Response(200, text="ok")

    fetcher = HttpxFetcher(transport=_transport(handler), retries=0)
    fetcher.get_text(
        "http://api.github.com/issues", headers={"Authorization": "Bearer SECRET"}
    )
    # Same scheme/host/port → the credential is preserved across the hop.
    assert seen == [("/issues", "Bearer SECRET"), ("/issues/", "Bearer SECRET")]
