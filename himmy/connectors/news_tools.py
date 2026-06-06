"""Connector tools: expose the RSS/Atom news fetcher as agent-callable tools.

The :class:`~himmy.connectors.news.NewsFetcher` connector was previously only reachable as
a standalone MCP server. These wrap it as ordinary local tools so any himmy agent can read
the news directly via ``tool_packs: [news]`` — ``news_sources`` (what feeds exist),
``news_fetch`` (latest from one feed), and ``news_search`` (keyword across feeds). The
fetcher's blocking I/O is run off the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from himmy.connectors.news import NewsFetcher
from himmy.services.tools.registry import ToolRegistry, register_local_tool

#: The tools this connector registers (the `news` toolkit pack advertises these).
NEWS_TOOL_NAMES = ("news_sources", "news_search", "news_fetch")


def register_news_tools(
    registry: ToolRegistry,
    *,
    fetcher: NewsFetcher | None = None,
    requires_approval: bool = False,
) -> list[str]:
    """Register ``news_sources``, ``news_search``, ``news_fetch`` onto ``registry``."""
    news = fetcher or NewsFetcher()

    async def _sources(_args: dict[str, Any]) -> dict[str, Any]:
        sources = await asyncio.to_thread(news.sources)
        return {"count": len(sources), "sources": [s.model_dump() for s in sources]}

    async def _search(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        limit = max(1, min(int(args.get("limit", 10)), 50))
        srcs = args.get("sources")
        items = await asyncio.to_thread(
            lambda: news.search(query, limit=limit, sources=srcs)
        )
        return {
            "query": query,
            "count": len(items),
            "items": [i.model_dump() for i in items],
        }

    async def _fetch(args: dict[str, Any]) -> dict[str, Any]:
        source = str(args["source"])
        limit = max(1, min(int(args.get("limit", 10)), 50))
        items = await asyncio.to_thread(lambda: news.fetch(source, limit=limit))
        return {
            "source": source,
            "count": len(items),
            "items": [i.model_dump() for i in items],
        }

    register_local_tool(
        registry,
        name="news_sources",
        handler=_sources,
        description="List the available news feeds (name, url, language, category).",
        args_json_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        requires_approval=requires_approval,
        metadata={"backend": "news"},
    )
    register_local_tool(
        registry,
        name="news_search",
        handler=_search,
        description=(
            "Search recent headlines across the news feeds by keyword (matches the "
            "article title + summary). Returns the matching articles."
        ),
        args_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "keyword to search for"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "optional feed names to limit the search to",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        requires_approval=requires_approval,
        metadata={"backend": "news"},
    )
    register_local_tool(
        registry,
        name="news_fetch",
        handler=_fetch,
        description="Fetch the latest headlines from one named news feed.",
        args_json_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "a feed name"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
            },
            "required": ["source"],
            "additionalProperties": False,
        },
        requires_approval=requires_approval,
        metadata={"backend": "news"},
    )
    return list(NEWS_TOOL_NAMES)


__all__ = ["register_news_tools", "NEWS_TOOL_NAMES"]
