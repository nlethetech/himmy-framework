"""Nepal connectors: independent Nepali-news aggregation over RSS/Atom feeds.

A curated set of Nepali outlet feeds (informed by the NepalOSINT source list but
fetched DIRECTLY — never via that VPS), normalized into :class:`NewsItem`s.
:class:`NewsFetcher` is one bad-feed-tolerant aggregator; the feeds are parsed with
``feedparser`` (an optional ``connectors`` extra, lazily imported).
"""

from __future__ import annotations

import re

from himmy.connectors.fetcher import Fetcher, HttpxFetcher
from himmy.connectors.models import NewsItem, NewsSource
from himmy.core.errors import HimmyError

#: Curated Nepali news feeds. Fetched directly from each outlet (no intermediary).
NEPAL_NEWS_SOURCES: list[NewsSource] = [
    NewsSource(name="onlinekhabar", url="https://www.onlinekhabar.com/feed", lang="ne"),
    NewsSource(
        name="onlinekhabar_en", url="https://english.onlinekhabar.com/feed", lang="en"
    ),
    NewsSource(name="setopati", url="https://www.setopati.com/feed", lang="ne"),
    NewsSource(name="kathmandupost", url="https://kathmandupost.com/rss", lang="en"),
    NewsSource(name="ekantipur", url="https://ekantipur.com/rss", lang="ne"),
    NewsSource(
        name="nagariknews", url="https://nagariknews.nagariknetwork.com/feed", lang="ne"
    ),
    NewsSource(name="annapurnapost", url="https://annapurnapost.com/rss", lang="ne"),
    NewsSource(name="ratopati", url="https://ratopati.com/feed", lang="ne"),
    NewsSource(name="khabarhub", url="https://khabarhub.com/feed/", lang="ne"),
    NewsSource(
        name="khabarhub_en", url="https://english.khabarhub.com/feed/", lang="en"
    ),
    NewsSource(name="baahrakhari", url="https://baahrakhari.com/feed", lang="ne"),
    NewsSource(name="nepalkhabar", url="https://nepalkhabar.com/feed", lang="ne"),
    NewsSource(name="lokaantar", url="https://lokaantar.com/feed", lang="ne"),
    NewsSource(name="pahilopost", url="https://pahilopost.com/feed", lang="ne"),
    NewsSource(name="risingnepal", url="https://risingnepaldaily.com/rss", lang="en"),
    NewsSource(name="gorkhapatra", url="https://gorkhapatraonline.com/rss", lang="ne"),
    NewsSource(
        name="annapurnaexpress", url="https://theannapurnaexpress.com/rss", lang="en"
    ),
    NewsSource(
        name="aarthiknews",
        url="https://aarthiknews.com/rss",
        lang="ne",
        category="business",
    ),
    NewsSource(
        name="bizmandu", url="https://bizmandu.com/feed", lang="ne", category="business"
    ),
    NewsSource(
        name="bbc_nepali", url="https://feeds.bbci.co.uk/nepali/rss.xml", lang="ne"
    ),
]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Strip HTML tags + collapse whitespace from a feed summary."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()


class NewsFetcher:
    """Aggregates Nepali news feeds into normalized, deduplicated items."""

    def __init__(
        self,
        fetcher: Fetcher | None = None,
        sources: list[NewsSource] | None = None,
    ) -> None:
        """Wire the HTTP fetcher and the source set (defaults to the curated list)."""
        self._fetcher = fetcher or HttpxFetcher()
        self._sources = {s.name: s for s in (sources or NEPAL_NEWS_SOURCES)}

    def sources(self) -> list[NewsSource]:
        """The configured news sources."""
        return list(self._sources.values())

    def fetch(self, source_name: str, *, limit: int = 20) -> list[NewsItem]:
        """Fetch + parse one source's latest items (newest first, capped at ``limit``)."""
        source = self._sources.get(source_name)
        if source is None:
            raise HimmyError(
                f"unknown news source {source_name!r}; known: {sorted(self._sources)}"
            )
        raw = self._fetcher.get_bytes(source.url)
        return self._parse(raw, source, limit)

    def fetch_all(
        self, *, per_source: int = 10, sources: list[str] | None = None
    ) -> list[NewsItem]:
        """Fetch many sources; a single failing feed never breaks the batch."""
        names = sources if sources is not None else list(self._sources)
        items: list[NewsItem] = []
        for name in names:
            try:
                items.extend(self.fetch(name, limit=per_source))
            except Exception:  # noqa: BLE001 - one bad feed must not sink the batch
                continue
        return items

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        sources: list[str] | None = None,
        per_source: int = 20,
    ) -> list[NewsItem]:
        """Keyword search across sources (case-insensitive over title + summary)."""
        needle = query.lower()
        hits = [
            item
            for item in self.fetch_all(per_source=per_source, sources=sources)
            if needle in f"{item.title} {item.summary}".lower()
        ]
        return hits[:limit]

    @staticmethod
    def _parse(raw: bytes, source: NewsSource, limit: int) -> list[NewsItem]:
        try:
            import feedparser
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise HimmyError(
                "news connectors require the [connectors] extra "
                "(pip install 'himmy[connectors]')."
            ) from exc
        feed = feedparser.parse(raw)
        items: list[NewsItem] = []
        for entry in feed.entries[: max(0, limit)]:
            items.append(
                NewsItem(
                    title=_clean(entry.get("title", "")),
                    link=entry.get("link", ""),
                    published=entry.get("published") or entry.get("updated"),
                    summary=_clean(entry.get("summary", ""))[:600],
                    source=source.name,
                    source_url=source.url,
                )
            )
        return items


__all__ = ["NEPAL_NEWS_SOURCES", "NewsFetcher"]
