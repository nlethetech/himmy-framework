"""Nepal connectors: Nepal Rastra Bank (NRB) — forex + macroeconomic reports.

Fetches DIRECTLY from NRB's public surface (no intermediary):

* :meth:`NRBClient.forex` / :meth:`latest_forex` — the public forex JSON API
  (``/api/forex/v1/rates``), normalized into :class:`ForexRate`s.
* :meth:`list_macro_reports` — the monthly "Current Macroeconomic & Financial
  Situation" reports, listed via NRB's category RSS feed (the page itself is
  JS-rendered; the feed is the reliable, structured index).
* :meth:`fetch_macro_workbook` / :meth:`parse_workbook` — download a report's Excel
  workbook (when the report page links one) and parse every sheet into rows with
  ``openpyxl``. ``parse_workbook`` works on any ``.xlsx`` bytes, so the Excel
  capability is fully exercisable offline.
"""

from __future__ import annotations

import io
import re
from datetime import date

from himmy.connectors.fetcher import Fetcher, HttpxFetcher, get_json
from himmy.connectors.models import ForexRate, MacroReport, Workbook
from himmy.core.errors import HimmyError

NRB_FOREX_API = "https://www.nrb.org.np/api/forex/v1/rates"
NRB_MACRO_FEED = "https://www.nrb.org.np/category/current-macroeconomic-situation/feed/"

# A link to a downloadable spreadsheet on the NRB site.
_XLS_RE = re.compile(r'https?://[^\s"\'<>]+?\.(?:xlsx|xls)', re.IGNORECASE)
_PERIOD_RE = re.compile(
    r"(annual|first|two|three|four|five|six|seven|eight|nine|ten|eleven)"
    r"[ -]?months?[^0-9]*([0-9]{4}[\.\-/][0-9]{2,4})",
    re.IGNORECASE,
)


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _detect_language(title: str) -> str:
    low = title.lower()
    if "tables" in low:
        return "tables"
    if "english" in low:
        return "english"
    if "nepali" in low:
        return "nepali"
    return "unknown"


def _detect_period(title: str) -> str | None:
    match = _PERIOD_RE.search(title)
    if match:
        return f"{match.group(1).lower()}-months {match.group(2)}"
    return None


def _find_workbook_link(html: str) -> str | None:
    match = _XLS_RE.search(html)
    return match.group(0) if match else None


class NRBClient:
    """A direct client for NRB forex + macroeconomic-situation reports."""

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        """Wire the HTTP fetcher (defaults to the live ``HttpxFetcher``)."""
        self._fetcher = fetcher or HttpxFetcher()

    # ------------------------------------------------------------------- forex
    def forex(
        self, from_date: str, to_date: str | None = None, *, per_page: int = 100
    ) -> list[ForexRate]:
        """Return per-currency buy/sell rates for a date (or ``from``..``to`` range).

        Dates are ISO ``YYYY-MM-DD``. Each currency's ``unit`` is the quantity the
        buy/sell price is quoted for (e.g. INR is quoted per 100). NRB caps
        ``per_page`` (the number of days per page) at 100; larger values return
        nothing, so the default and effective maximum is 100.
        """
        per_page = min(per_page, 100)
        to_date = to_date or from_date
        url = (
            f"{NRB_FOREX_API}?from={from_date}&to={to_date}&per_page={per_page}&page=1"
        )
        data = get_json(self._fetcher, url)
        rates: list[ForexRate] = []
        for day in (data.get("data") or {}).get("payload", []) or []:
            day_date = day.get("date", "")
            for entry in day.get("rates", []) or []:
                currency = entry.get("currency", {}) or {}
                rates.append(
                    ForexRate(
                        date=day_date,
                        currency_iso3=currency.get("iso3", ""),
                        currency_name=currency.get("name", ""),
                        unit=int(currency.get("unit", 1) or 1),
                        buy=_to_float(entry.get("buy")),
                        sell=_to_float(entry.get("sell")),
                    )
                )
        return rates

    def latest_forex(self) -> list[ForexRate]:
        """Convenience: today's published rates."""
        return self.forex(date.today().isoformat())

    # ------------------------------------------------------ macroeconomic data
    def list_macro_reports(self, *, limit: int = 20) -> list[MacroReport]:
        """List the latest 'Current Macroeconomic & Financial Situation' reports.

        Sourced from NRB's category RSS feed (the HTML listing is JS-rendered).
        Each report comes in Nepali / English / Tables variants — the ``language``
        field distinguishes them, and ``period`` parses the data window.
        """
        raw = self._fetcher.get_bytes(NRB_MACRO_FEED)
        try:
            import feedparser
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise HimmyError(
                "NRB macro reports require the [connectors] extra "
                "(pip install 'himmy[connectors]')."
            ) from exc
        feed = feedparser.parse(raw)
        reports: list[MacroReport] = []
        for entry in feed.entries[: max(0, limit)]:
            title = entry.get("title", "")
            reports.append(
                MacroReport(
                    title=title,
                    url=entry.get("link", ""),
                    published=entry.get("published") or entry.get("updated"),
                    language=_detect_language(title),
                    period=_detect_period(title),
                )
            )
        return reports

    def fetch_macro_workbook(self, report_url: str) -> Workbook | None:
        """Download + parse a report's Excel workbook, if one is linked.

        Returns the parsed :class:`Workbook`, or None when the report page exposes
        no downloadable ``.xlsx``/``.xls`` (some are JS-only — pass a direct
        workbook URL to :meth:`parse_workbook` in that case).
        """
        html = self._fetcher.get_text(report_url)
        link = _find_workbook_link(html)
        if link is None:
            return None
        workbook = self.parse_workbook(self._fetcher.get_bytes(link))
        workbook.source_url = link
        return workbook

    @staticmethod
    def parse_workbook(data: bytes) -> Workbook:
        """Parse ``.xlsx`` bytes into ``{sheet name: rows}`` (every cell, as-is)."""
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise HimmyError(
                "Excel parsing requires the [connectors] extra "
                "(pip install 'himmy[connectors]')."
            ) from exc
        book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sheets: dict[str, list[list[object]]] = {}
        for sheet in book.worksheets:
            sheets[sheet.title] = [
                list(row) for row in sheet.iter_rows(values_only=True)
            ]
        book.close()
        return Workbook(sheets=sheets)


__all__ = ["NRBClient", "NRB_FOREX_API", "NRB_MACRO_FEED"]
