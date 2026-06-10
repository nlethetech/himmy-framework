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
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only (httpx stays a lazy import)
    import httpx

#: A polite, identifiable default User-Agent.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; HimmyBot/0.1; "
    "+https://github.com/nlethetech/himmy-framework)"
)

#: HTTP statuses worth retrying: rate limits and transient server errors.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


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
    """Fetches a URL's body as text or bytes (the only network dependency)."""

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
    ) -> None:
        """Configure the User-Agent, per-request timeout, and the retry policy.

        ``retries`` is the number of *additional* attempts after the first (so the
        default of 2 means at most 3 requests); ``transport`` is a test seam for
        injecting an ``httpx`` mock transport.
        """
        self._headers = {"User-Agent": user_agent}
        self._timeout = timeout
        self._retries = max(0, retries)
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._transport = transport

    def _backoff_delay(self, attempt: int, retry_after: str | None) -> float:
        """Seconds to wait before retry ``attempt`` (0-based), honoring Retry-After."""
        server_delay = _parse_retry_after(retry_after)
        if server_delay is not None:
            return min(server_delay, self._backoff_max)
        base = min(self._backoff_base * (2.0**attempt), self._backoff_max)
        return base + random.uniform(0.0, base * 0.25)

    def _get(self, url: str, *, timeout: float | None = None) -> Any:
        import httpx

        effective_timeout = self._timeout if timeout is None else timeout
        for attempt in range(self._retries + 1):
            try:
                with httpx.Client(
                    headers=self._headers,
                    timeout=effective_timeout,
                    follow_redirects=True,
                    transport=self._transport,
                ) as client:
                    response = client.get(url)
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

    def get_text(self, url: str, *, timeout: float | None = None) -> str:
        """Fetch ``url`` and return its decoded text body.

        ``timeout`` overrides the fetcher's default per-request timeout.
        """
        return str(self._get(url, timeout=timeout).text)

    def get_bytes(self, url: str, *, timeout: float | None = None) -> bytes:
        """Fetch ``url`` and return its raw byte body.

        ``timeout`` overrides the fetcher's default per-request timeout.
        """
        return bytes(self._get(url, timeout=timeout).content)


def get_json(fetcher: Fetcher, url: str) -> Any:
    """Fetch ``url`` via ``fetcher`` and parse the body as JSON."""
    return json.loads(fetcher.get_text(url))


__all__ = [
    "Fetcher",
    "HttpxFetcher",
    "DEFAULT_USER_AGENT",
    "RETRYABLE_STATUSES",
    "get_json",
]
