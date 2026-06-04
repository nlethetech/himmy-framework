"""Tests for the Nepali news aggregator (offline, against a real-feed fixture)."""

from __future__ import annotations

from pathlib import Path

from himmy.connectors import NewsFetcher
from tests.connectors._fixtures import FixtureFetcher

_FIX = Path(__file__).parent / "fixtures"


def _fetcher() -> FixtureFetcher:
    return FixtureFetcher(data=(_FIX / "onlinekhabar_feed.xml").read_bytes())


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
    import pytest

    from himmy.core.errors import HimmyError

    with pytest.raises(HimmyError):
        NewsFetcher(fetcher=_fetcher()).fetch("not-a-real-source")
