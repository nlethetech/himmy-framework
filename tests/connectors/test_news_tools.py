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


def test_news_search_unknown_keyword_falls_back_to_latest() -> None:
    """An unmatched keyword must NOT hand the agent an empty dead end: the tool
    returns the latest headlines and says so honestly in ``note`` (found live:
    'recent political news Nepal' returned count=0 and the agent floundered)."""
    out = run_async(
        _registry().handler_for("news_search")({"query": "zzz-no-such-topic"})
    )
    assert out["count"] > 0
    assert "latest" in out.get("note", "")


def test_news_search_stopword_query_returns_latest() -> None:
    """A query of pure stopwords ('latest news today') is a browse, not a
    filter — it returns the latest headlines."""
    out = run_async(
        _registry().handler_for("news_search")({"query": "latest news today"})
    )
    assert out["count"] > 0


def test_news_search_stopwords_do_not_starve_real_keywords() -> None:
    """THE regression: 'recent political news of Nepal' must match articles
    about Nepal even though no headline contains the words 'recent' or 'news'."""
    out = run_async(
        _registry().handler_for("news_search")(
            {"query": "recent political news of Nepal"}
        )
    )
    assert out["count"] > 0
    assert "note" not in out or "latest" not in out["note"]
