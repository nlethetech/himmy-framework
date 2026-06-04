"""Himmy Nepal-news MCP server (stdio JSON-RPC) — independent, no VPS.

A standalone Model Context Protocol server exposing Nepali news over the curated
feeds. Any MCP client (Himmy's ``MCPClient``, Claude Desktop, …) can consume it::

    python -m himmy.connectors.news_mcp_server

Tools: ``list_sources``, ``fetch_news`` (one source), ``search_news`` (keyword
across sources). Set ``HIMMY_NEWS_FIXTURE`` to a local RSS file to run fully
offline (used by the tests) — the fetcher then returns that file for any source.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from himmy.connectors.fetcher import Fetcher, HttpxFetcher
from himmy.connectors.news import NewsFetcher

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_sources",
        "description": "List the available Nepali news sources.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "fetch_news",
        "description": "Fetch the latest items from one news source.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["source"],
        },
    },
    {
        "name": "search_news",
        "description": "Keyword search across sources (title + summary).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]


class _FixtureFetcher:
    """Returns a single local RSS file for any URL (offline test mode)."""

    def __init__(self, path: str) -> None:
        self._path = path

    def get_text(self, url: str) -> str:
        return self.get_bytes(url).decode("utf-8", "replace")

    def get_bytes(self, url: str) -> bytes:
        with open(self._path, "rb") as handle:
            return handle.read()


def _build_fetcher() -> Fetcher:
    fixture = os.environ.get("HIMMY_NEWS_FIXTURE")
    return _FixtureFetcher(fixture) if fixture else HttpxFetcher()


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _result(message_id: Any, result: dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "id": message_id, "result": result})


def _tool_result(
    message_id: Any, payload: dict[str, Any], *, is_error: bool = False
) -> None:
    _result(
        message_id,
        {
            "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
            "isError": is_error,
        },
    )


def _call(message_id: Any, params: dict[str, Any], fetcher: Fetcher) -> None:
    name = params.get("name")
    args = params.get("arguments") or {}
    news = NewsFetcher(fetcher=fetcher)
    try:
        if name == "list_sources":
            _tool_result(
                message_id, {"sources": [s.model_dump() for s in news.sources()]}
            )
        elif name == "fetch_news":
            items = news.fetch(str(args["source"]), limit=int(args.get("limit", 20)))
            _tool_result(message_id, {"items": [i.model_dump() for i in items]})
        elif name == "search_news":
            items = news.search(str(args["query"]), limit=int(args.get("limit", 20)))
            _tool_result(message_id, {"items": [i.model_dump() for i in items]})
        else:
            _tool_result(message_id, {"error": f"unknown tool {name!r}"}, is_error=True)
    except Exception as exc:  # noqa: BLE001 - report tool errors in-band (MCP semantics)
        _tool_result(
            message_id, {"error": f"{type(exc).__name__}: {exc}"}, is_error=True
        )


def main() -> None:
    """Serve the MCP protocol over stdin/stdout until EOF."""
    fetcher = _build_fetcher()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method")
        message_id = message.get("id")
        if method == "initialize":
            _result(
                message_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "himmy-news", "version": "0.1.0"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _result(message_id, {"tools": TOOLS})
        elif method == "tools/call":
            _call(message_id, message.get("params") or {}, fetcher)
        elif message_id is not None:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            )


if __name__ == "__main__":
    main()
