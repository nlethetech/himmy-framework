"""A fixture-backed Fetcher so connector tests never touch the network."""

from __future__ import annotations


class FixtureFetcher:
    """Returns canned content for any URL (or routed by URL substring).

    ``routes`` is a list of ``(substring, content)`` pairs checked in order;
    content may be ``str`` or ``bytes`` and is coerced to the requested form.
    """

    def __init__(
        self,
        *,
        text: str = "",
        data: bytes = b"",
        routes: list[tuple[str, str | bytes]] | None = None,
    ) -> None:
        self._text = text
        self._data = data
        self._routes = list(routes or [])

    def _match(self, url: str) -> str | bytes | None:
        for substring, content in self._routes:
            if substring in url:
                return content
        return None

    def get_text(self, url: str) -> str:
        content = self._match(url)
        content = self._text if content is None else content
        return (
            content.decode("utf-8", "replace")
            if isinstance(content, bytes)
            else content
        )

    def get_bytes(self, url: str) -> bytes:
        content = self._match(url)
        content = self._data if content is None else content
        return content.encode("utf-8") if isinstance(content, str) else content
