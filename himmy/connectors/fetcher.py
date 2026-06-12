"""Nepal connectors: the HTTP fetch seam (independent, injectable, no VPS).

Connectors fetch directly from the public source. The :class:`Fetcher` protocol is
the injection point so the whole connector layer is testable offline: production
uses :class:`HttpxFetcher`; tests pass a fixture-backed fetcher and never touch the
network. The live fetcher retries transient failures (429/5xx/transport errors)
with exponential backoff + jitter, honoring ``Retry-After`` when the server sends
one, so a single rate-limit blip never kills a batch.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import urljoin, urlsplit

if TYPE_CHECKING:  # pragma: no cover - typing only (httpx stays a lazy import)
    import httpx

#: A polite, identifiable default User-Agent.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; HimmyBot/0.1; "
    "+https://github.com/nlethetech/himmy-framework)"
)

#: HTTP statuses worth retrying: rate limits and transient server errors.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

#: HTTP statuses that carry a ``Location`` redirect.
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

#: Cap on redirect hops followed for a single request (httpx's own default).
_MAX_REDIRECTS = 10

#: Default ceiling on a fetched body (text or bytes) — a sane DoS guard so a
#: multi-GB / chunked-forever response cannot exhaust memory. Callers that need a
#: different bound pass ``max_bytes``.
DEFAULT_MAX_BYTES = 8 * 1024 * 1024

#: A hook run on each URL before it is fetched (the initial URL AND every redirect
#: hop). It raises to refuse the URL; its return value is ignored. This is the SSRF
#: seam: callers pass :func:`himmy.toolkit._net.guard_url` so a 3xx to an internal
#: address is rejected before the hop is followed.
UrlGuard = Callable[[str], object]

#: Request headers that carry a credential and so MUST NOT be replayed onto a redirect
#: target on a *different* origin (scheme/host/port). A guarded redirect can still send
#: the request to another host (it just has to be a public one); forwarding the
#: ``Authorization`` header there would leak the credential to a third party — the same
#: rule browsers and httpx apply on a cross-origin redirect.
_SENSITIVE_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})


def _same_origin(a: str, b: str) -> bool:
    """Whether two URLs share scheme + host + (explicit-or-default) port."""
    pa, pb = urlsplit(a), urlsplit(b)
    default = {"http": 80, "https": 443}
    return (
        pa.scheme == pb.scheme
        and (pa.hostname or "").lower() == (pb.hostname or "").lower()
        and (pa.port or default.get(pa.scheme)) == (pb.port or default.get(pb.scheme))
    )


def _strip_sensitive_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop credential-bearing headers (used when a redirect crosses origins)."""
    return {k: v for k, v in headers.items() if k.lower() not in _SENSITIVE_HEADERS}


class ResponseTooLarge(Exception):
    """A fetched body exceeded the fetcher's ``max_bytes`` ceiling."""

    def __init__(self, url: str, limit: int) -> None:
        self.url = url
        self.limit = limit
        super().__init__(
            f"response body from {url!r} exceeds the {limit}-byte safety cap"
        )


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


@runtime_checkable
class Fetcher(Protocol):
    """Fetches a URL's body as text or bytes (the only network dependency).

    An implementation MAY accept an optional ``headers`` keyword to carry per-request
    headers (an authenticated GET needs this — see :class:`HttpxFetcher`). It is kept
    optional so a minimal fetcher (a test double, a fixture) can implement just
    ``get_text(url)``; :func:`get_json` probes for header support and degrades cleanly.
    """

    def get_text(self, url: str) -> str:
        """Return the response body as decoded text."""
        ...

    def get_bytes(self, url: str) -> bytes:
        """Return the response body as raw bytes (feeds, spreadsheets)."""
        ...


class HttpxFetcher:
    """The default :class:`Fetcher` backed by ``httpx`` (a core dependency).

    Transient failures — 429, 5xx, and transport errors (DNS, resets, timeouts) —
    are retried up to ``retries`` times with exponential backoff + jitter; a
    server-sent ``Retry-After`` is honored (capped at ``backoff_max``). Pass
    ``retries=0`` to restore single-attempt behavior.

    Redirects are followed *manually* (httpx ``follow_redirects=False``) so a
    ``guard`` hook can re-validate every hop's ``Location`` before it is followed —
    closing the SSRF gap where a public URL 302s to ``169.254.169.254`` / loopback /
    an RFC1918 host past a one-shot URL check. Bodies are capped at ``max_bytes``
    (streamed) so an unbounded response cannot exhaust memory.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 30.0,
        retries: int = 2,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        guard: UrlGuard | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        """Configure the User-Agent, per-request timeout, and the retry policy.

        ``retries`` is the number of *additional* attempts after the first (so the
        default of 2 means at most 3 requests); ``transport`` is a test seam for
        injecting an ``httpx`` mock transport. ``guard`` (when set) is invoked on
        the request URL and on every redirect target before following — raising to
        refuse it (the SSRF defense). ``max_bytes`` caps the body size.
        """
        self._headers = {"User-Agent": user_agent}
        self._timeout = timeout
        self._retries = max(0, retries)
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._transport = transport
        self._guard = guard
        self._max_bytes = max_bytes

    def _backoff_delay(self, attempt: int, retry_after: str | None) -> float:
        """Seconds to wait before retry ``attempt`` (0-based), honoring Retry-After."""
        server_delay = _parse_retry_after(retry_after)
        if server_delay is not None:
            return min(server_delay, self._backoff_max)
        base = min(self._backoff_base * (2.0**attempt), self._backoff_max)
        return base + random.uniform(0.0, base * 0.25)

    def _read_capped(self, response: Any, url: str) -> bytes:
        """Stream ``response`` into memory, refusing a body over ``max_bytes``.

        Reading incrementally means a multi-GB / never-ending body is rejected as
        soon as it crosses the cap, not after the whole thing lands in memory.
        """
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > self._max_bytes:
                response.close()
                raise ResponseTooLarge(url, self._max_bytes)
            chunks.append(chunk)
        return b"".join(chunks)

    def _request_once(
        self, client: Any, url: str, headers: dict[str, str] | None = None
    ) -> Any:
        """Issue one GET (no auto-redirect) and read its capped body into memory.

        Each redirect hop's ``Location`` is re-validated through ``guard`` before
        it is followed, so the SSRF check applies to the WHOLE chain, not just the
        model-supplied URL. ``headers`` (per-request, e.g. an ``Authorization`` from a
        connector's auth spec) ride on the request, but any credential-bearing header
        is DROPPED before a redirect that crosses to a different origin — so a guarded
        3xx to another public host can't exfiltrate the token. Returns an httpx
        ``Response`` with ``_himmy_body`` set.
        """
        import httpx

        current = url
        hop_headers = dict(headers or {})
        for _ in range(_MAX_REDIRECTS + 1):
            if self._guard is not None:
                self._guard(current)
            with client.stream("GET", current, headers=hop_headers) as response:
                if (
                    response.status_code in _REDIRECT_STATUSES
                    and "location" in response.headers
                ):
                    target = urljoin(current, response.headers["location"])
                    if not _same_origin(current, target):
                        # Cross-origin hop: never replay the credential to a new host.
                        hop_headers = _strip_sensitive_headers(hop_headers)
                    current = target
                    continue
                response._himmy_body = self._read_capped(response, current)
                return response
        raise httpx.TooManyRedirects(
            f"exceeded {_MAX_REDIRECTS} redirects fetching {url!r}",
            request=httpx.Request("GET", url),
        )

    def _get(
        self,
        url: str,
        *,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        import httpx

        effective_timeout = self._timeout if timeout is None else timeout
        request_headers = dict(headers) if headers else None
        for attempt in range(self._retries + 1):
            try:
                with httpx.Client(
                    headers=self._headers,
                    timeout=effective_timeout,
                    follow_redirects=False,
                    transport=self._transport,
                ) as client:
                    response = self._request_once(client, url, request_headers)
            except httpx.TransportError:
                if attempt >= self._retries:
                    raise
                time.sleep(self._backoff_delay(attempt, None))
                continue
            if response.status_code in RETRYABLE_STATUSES and attempt < self._retries:
                time.sleep(
                    self._backoff_delay(attempt, response.headers.get("Retry-After"))
                )
                continue
            response.raise_for_status()
            return response
        raise AssertionError("unreachable: the retry loop returns or raises")

    def get_text(
        self,
        url: str,
        *,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        """Fetch ``url`` and return its decoded text body.

        ``timeout`` overrides the fetcher's default per-request timeout. ``headers``
        are per-request headers (e.g. an ``Authorization`` from a connector's auth
        spec); credential-bearing ones are dropped on a cross-origin redirect.
        """
        response = self._get(url, timeout=timeout, headers=headers)
        body: bytes = response._himmy_body
        return body.decode(response.encoding or "utf-8", "replace")

    def get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        """Fetch ``url`` and return its raw byte body.

        ``timeout`` overrides the fetcher's default per-request timeout. ``headers``
        are per-request headers, dropped on a cross-origin redirect when sensitive.
        """
        return bytes(self._get(url, timeout=timeout, headers=headers)._himmy_body)


def get_json(
    fetcher: Fetcher, url: str, *, headers: Mapping[str, str] | None = None
) -> Any:
    """Fetch ``url`` via ``fetcher`` and parse the body as JSON.

    ``headers`` (e.g. a connector's ``Authorization``) are forwarded when the
    ``fetcher`` accepts them — :class:`HttpxFetcher` does, carrying them on the GET and
    dropping any credential-bearing header on a cross-origin redirect. A minimal
    :class:`Fetcher` whose ``get_text`` takes only the URL is still supported (the
    headers are simply not applied) so existing fixtures keep working; an
    *authenticated* GET must therefore go through a header-aware fetcher.
    """
    if headers:
        try:
            return json.loads(fetcher.get_text(url, headers=headers))  # type: ignore[call-arg]
        except TypeError:
            # A header-unaware Fetcher (e.g. a test double) — fall back to the URL-only
            # call rather than crash; callers needing auth use HttpxFetcher.
            pass
    return json.loads(fetcher.get_text(url))


__all__ = [
    "Fetcher",
    "HttpxFetcher",
    "ResponseTooLarge",
    "DEFAULT_USER_AGENT",
    "DEFAULT_MAX_BYTES",
    "RETRYABLE_STATUSES",
    "get_json",
]
