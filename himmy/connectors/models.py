"""Nepal connectors: shared typed models (news, NRB forex + macro)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NewsSource(BaseModel):
    """One Nepali news feed: a name, its RSS/Atom URL, language, and topic."""

    name: str
    url: str
    lang: str = "ne"  # "ne" (Nepali) | "en" (English)
    category: str = "general"


class NewsItem(BaseModel):
    """One normalized news article from a feed."""

    title: str
    link: str
    published: str | None = None
    summary: str = ""
    source: str = ""
    source_url: str = ""


class FeedFailure(BaseModel):
    """One news source that failed during an aggregate fetch (and why)."""

    source: str
    error: str


class NewsFetchResult(BaseModel):
    """An aggregate news fetch: the items plus any per-source failures.

    ``items`` is exactly what :meth:`NewsFetcher.fetch_all` returns; ``failures``
    tells the caller which sources were skipped (so "no failures" can be asserted
    rather than assumed).
    """

    items: list[NewsItem] = Field(default_factory=list)
    failures: list[FeedFailure] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True when every requested source was fetched successfully."""
        return not self.failures


class ForexRate(BaseModel):
    """One NRB foreign-exchange rate for a currency on a date (per ``unit``)."""

    date: str
    currency_iso3: str
    currency_name: str
    unit: int = 1
    buy: float | None = None
    sell: float | None = None


class MacroReport(BaseModel):
    """One NRB 'Current Macroeconomic & Financial Situation' monthly report."""

    title: str
    url: str
    published: str | None = None
    language: str = "unknown"  # nepali | english | tables | unknown
    period: str | None = None  # e.g. "nine-months 2025-26"


class Workbook(BaseModel):
    """A parsed spreadsheet: sheet name -> rows (each row a list of cells)."""

    source_url: str = ""
    sheets: dict[str, list[list[Any]]] = Field(default_factory=dict)

    def sheet_names(self) -> list[str]:
        """The worksheet names in the workbook."""
        return list(self.sheets)


__all__ = [
    "NewsSource",
    "NewsItem",
    "FeedFailure",
    "NewsFetchResult",
    "ForexRate",
    "MacroReport",
    "Workbook",
]
