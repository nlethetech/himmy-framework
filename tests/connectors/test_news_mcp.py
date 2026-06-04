"""Tests for the news MCP server, driven offline via Himmy's own MCPClient."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from himmy.services.mcp import MCPClient, MCPServerSpec
from tests.conftest import run_async

_FIX = Path(__file__).parent / "fixtures"


def _spec() -> MCPServerSpec:
    return MCPServerSpec(
        command=sys.executable,
        args=["-m", "himmy.connectors.news_mcp_server"],
        env={"HIMMY_NEWS_FIXTURE": str(_FIX / "onlinekhabar_feed.xml")},
    )


def test_news_mcp_server_lists_and_fetches() -> None:
    """The server completes the handshake, lists tools, and returns parsed news."""

    async def scenario() -> Any:
        client = await MCPClient.connect(_spec())
        try:
            tools = await client.list_tools()
            sources = await client.call_tool("list_sources", {})
            news = await client.call_tool(
                "fetch_news", {"source": "onlinekhabar_en", "limit": 5}
            )
            return tools, sources, news
        finally:
            await client.aclose()

    tools, sources, news = run_async(scenario())
    assert {"list_sources", "fetch_news", "search_news"} <= {t.name for t in tools}
    assert json.loads(sources.text)["sources"]
    items = json.loads(news.text)["items"]
    assert items and items[0]["title"] and items[0]["source"] == "onlinekhabar_en"
