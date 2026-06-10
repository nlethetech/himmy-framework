"""Tests for the Nepali news aggregator (offline, against a real-feed fixture)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from himmy.connectors import NewsFetcher
from himmy.connectors._feeds import MAX_FEED_BYTES, MAX_FEED_ENTRIES
from himmy.core.errors import HimmyError
from tests.connectors._fixtures import FixtureFetcher

_FIX = Path(__file__).parent / "fixtures"


def _fetcher() -> FixtureFetcher:
    return FixtureFetcher(data=(_FIX / "onlinekhabar_feed.xml").read_bytes())


class _FlakyFetcher(FixtureFetcher):
    """A FixtureFetcher that raises for URLs containing any ``fail`` substring."""

    def __init__(self, *, data: bytes, fail: list[str]) -> None:
        super().__init__(data=data)
        self._fail = fail

    def get_bytes(self, url: str) -> bytes:
        for marker in self._fail:
            if marker in url:
                raise ConnectionError(f"boom: {url}")
        return super().get_bytes(url)


def test_sources_include_known_outlets() -> None:
    """The curated source list covers the major outlets."""
    names = {s.name for s in NewsFetcher().sources()}
    assert {"onlinekhabar", "setopati", "kathmandupost", "bbc_nepali"} <= names


def test_fetch_parses_normalized_items() -> None:
    """A feed parses into items with title/link and the source stamped on."""
    news = NewsFetcher(fetcher=_fetcher())
    items = news.fetch("onlinekhabar_en", limit=5)
    assert 1 <= len(items) <= 5
    assert all(item.title and item.link for item in items)
    assert items[0].source == "onlinekhabar_en"
    assert "<" not in items[0].summary  # HTML stripped


def test_search_filters_by_keyword() -> None:
    """search() returns only items whose title/summary contain the query."""
    news = NewsFetcher(fetcher=_fetcher())
    items = news.fetch("onlinekhabar_en", limit=12)
    word = max(items[0].title.split(), key=len)  # a token guaranteed present
    hits = news.search(word, sources=["onlinekhabar_en"], per_source=12, limit=20)
    assert hits
    assert all(word.lower() in f"{h.title} {h.summary}".lower() for h in hits)


def test_unknown_source_raises() -> None:
    """Fetching an unconfigured source is a clear error."""
    with pytest.raises(HimmyError):
        NewsFetcher(fetcher=_fetcher()).fetch("not-a-real-source")


def test_fetch_all_detailed_reports_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing feed is named (with its error) instead of vanishing silently."""
    flaky = _FlakyFetcher(
        data=(_FIX / "onlinekhabar_feed.xml").read_bytes(), fail=["setopati"]
    )
    news = NewsFetcher(fetcher=flaky)
    with caplog.at_level(logging.WARNING, logger="himmy.connectors.news"):
        result = news.fetch_all_detailed(
            per_source=2, sources=["onlinekhabar_en", "setopati"]
        )
    assert result.items  # the good feed still came through
    assert not result.complete
    (failure,) = result.failures
    assert failure.source == "setopati"
    assert "boom" in failure.error
    assert any("setopati" in record.message for record in caplog.records)


def test_fetch_all_detailed_complete_when_all_succeed() -> None:
    """No failures -> complete=True and items from every source."""
    news = NewsFetcher(fetcher=_fetcher())
    result = news.fetch_all_detailed(
        per_source=2, sources=["onlinekhabar_en", "setopati"]
    )
    assert result.complete
    assert result.failures == []
    assert {i.source for i in result.items} == {"onlinekhabar_en", "setopati"}


def test_fetch_all_keeps_returning_a_plain_list() -> None:
    """The existing fetch_all shape is unchanged (failures still skipped)."""
    flaky = _FlakyFetcher(
        data=(_FIX / "onlinekhabar_feed.xml").read_bytes(), fail=["setopati"]
    )
    items = NewsFetcher(fetcher=flaky).fetch_all(
        per_source=2, sources=["onlinekhabar_en", "setopati"]
    )
    assert isinstance(items, list)
    assert items
    assert all(i.source == "onlinekhabar_en" for i in items)


def test_fetch_rejects_unparseable_feed() -> None:
    """A non-feed body (error page) raises a clear error instead of parsing to []."""
    news = NewsFetcher(fetcher=FixtureFetcher(data=b'{"error": "blocked by WAF"}'))
    with pytest.raises(HimmyError, match="not parseable"):
        news.fetch("onlinekhabar_en")


def test_fetch_rejects_oversized_feed() -> None:
    """A feed body over the size bound is refused before feedparser sees it."""
    news = NewsFetcher(fetcher=FixtureFetcher(data=b"x" * (MAX_FEED_BYTES + 1)))
    with pytest.raises(HimmyError, match="safety bound"):
        news.fetch("onlinekhabar_en")


def test_fetch_caps_entries_regardless_of_limit() -> None:
    """Even a huge limit cannot pull more than MAX_FEED_ENTRIES from one feed."""
    items_xml = "".join(
        f"<item><title>headline {i}</title><link>https://x/{i}</link></item>"
        for i in range(MAX_FEED_ENTRIES + 50)
    )
    raw = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        f"<title>big</title>{items_xml}</channel></rss>"
    ).encode()
    news = NewsFetcher(fetcher=FixtureFetcher(data=raw))
    items = news.fetch("onlinekhabar_en", limit=10_000)
    assert len(items) == MAX_FEED_ENTRIES
