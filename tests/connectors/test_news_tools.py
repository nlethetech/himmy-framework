"""News connector exposed as agent tools — exercised fully offline via an RSS fixture."""

from __future__ import annotations

from pathlib import Path

from himmy.connectors.models import NewsSource
from himmy.connectors.news import NewsFetcher
from himmy.connectors.news_tools import NEWS_TOOL_NAMES, register_news_tools
from himmy.services.tools.registry import ToolRegistry
from tests.conftest import run_async

FIXTURE = Path("tests/connectors/fixtures/onlinekhabar_feed.xml")


class _FixtureFetcher:
    """Returns the local RSS fixture for any URL (no network)."""

    def get_text(self, url: str) -> str:
        return FIXTURE.read_text(encoding="utf-8")

    def get_bytes(self, url: str) -> bytes:
        return FIXTURE.read_bytes()


def _registry() -> ToolRegistry:
    fetcher = NewsFetcher(
        fetcher=_FixtureFetcher(),
        sources=[NewsSource(name="ok", url="http://example/feed", lang="en")],
    )
    registry = ToolRegistry()
    register_news_tools(registry, fetcher=fetcher)
    return registry


def test_pack_registers_the_three_news_tools() -> None:
    assert {d.name for d in _registry().list()} == set(NEWS_TOOL_NAMES)


def test_news_sources_lists_feeds() -> None:
    out = run_async(_registry().handler_for("news_sources")({}))
    assert out["count"] == 1
    assert out["sources"][0]["name"] == "ok"


def test_news_fetch_returns_articles() -> None:
    out = run_async(_registry().handler_for("news_fetch")({"source": "ok", "limit": 5}))
    assert out["count"] >= 1
    assert all({"title", "link"} <= set(item) for item in out["items"])


def test_news_search_filters_by_keyword() -> None:
    out = run_async(_registry().handler_for("news_search")({"query": "india"}))
    assert out["count"] >= 1
    # every hit really contains the keyword in its title or summary
    assert all("india" in f"{i['title']} {i['summary']}".lower() for i in out["items"])


def test_news_search_unknown_keyword_is_empty() -> None:
    out = run_async(
        _registry().handler_for("news_search")({"query": "zzz-no-such-topic"})
    )
    assert out["count"] == 0
