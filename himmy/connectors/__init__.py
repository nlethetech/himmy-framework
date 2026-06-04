"""Nepal connectors: independent, fetch-direct sources (no VPS).

Nepali **news** (curated outlet RSS feeds; also a standalone MCP server via
``python -m himmy.connectors.news_mcp_server``) and **NRB** (foreign-exchange API +
monthly macroeconomic reports + Excel workbook parsing). Every connector fetches
directly from the public source through the injectable :class:`Fetcher` seam, so
the whole layer is testable offline. Live fetching needs the ``connectors`` extra
(``pip install 'himmy[connectors]'`` — feedparser + openpyxl).
"""

from __future__ import annotations

from himmy.connectors.fetcher import DEFAULT_USER_AGENT, Fetcher, HttpxFetcher
from himmy.connectors.models import (
    ForexRate,
    MacroReport,
    NewsItem,
    NewsSource,
    Workbook,
)
from himmy.connectors.news import NEPAL_NEWS_SOURCES, NewsFetcher
from himmy.connectors.nrb import NRB_FOREX_API, NRB_MACRO_FEED, NRBClient
from himmy.connectors.nrb_tools import register_nrb_tools

__all__ = [
    "Fetcher",
    "HttpxFetcher",
    "DEFAULT_USER_AGENT",
    "NewsSource",
    "NewsItem",
    "ForexRate",
    "MacroReport",
    "Workbook",
    "NEPAL_NEWS_SOURCES",
    "NewsFetcher",
    "NRBClient",
    "NRB_FOREX_API",
    "NRB_MACRO_FEED",
    "register_nrb_tools",
]
