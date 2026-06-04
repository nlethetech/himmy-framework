"""Nepal connectors: the HTTP fetch seam (independent, injectable, no VPS).

Connectors fetch directly from the public source. The :class:`Fetcher` protocol is
the injection point so the whole connector layer is testable offline: production
uses :class:`HttpxFetcher`; tests pass a fixture-backed fetcher and never touch the
network.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

#: A polite, identifiable default User-Agent.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; HimmyBot/0.1; "
    "+https://github.com/nlethetech/himmy-framework)"
)


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
    """The default :class:`Fetcher` backed by ``httpx`` (a core dependency)."""

    def __init__(
        self, *, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 30.0
    ) -> None:
        """Configure the User-Agent and per-request timeout."""
        self._headers = {"User-Agent": user_agent}
        self._timeout = timeout

    def _get(self, url: str) -> Any:
        import httpx

        with httpx.Client(
            headers=self._headers, timeout=self._timeout, follow_redirects=True
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response

    def get_text(self, url: str) -> str:
        """Fetch ``url`` and return its decoded text body."""
        return str(self._get(url).text)

    def get_bytes(self, url: str) -> bytes:
        """Fetch ``url`` and return its raw byte body."""
        return bytes(self._get(url).content)


def get_json(fetcher: Fetcher, url: str) -> Any:
    """Fetch ``url`` via ``fetcher`` and parse the body as JSON."""
    return json.loads(fetcher.get_text(url))


__all__ = ["Fetcher", "HttpxFetcher", "DEFAULT_USER_AGENT", "get_json"]
