"""Tests for the NRB connector: forex, macro report listing, Excel parsing."""

from __future__ import annotations

import io
from pathlib import Path

import openpyxl

from himmy.connectors import NRBClient
from tests.connectors._fixtures import FixtureFetcher

_FIX = Path(__file__).parent / "fixtures"


def test_forex_parses_real_response_shape() -> None:
    """The forex JSON fixture parses into per-currency rates (INR quoted per 100)."""
    client = NRBClient(
        fetcher=FixtureFetcher(text=(_FIX / "nrb_forex.json").read_text())
    )
    rates = client.forex("2024-01-01")
    assert len(rates) >= 10
    inr = next(r for r in rates if r.currency_iso3 == "INR")
    assert inr.unit == 100
    assert inr.buy and inr.sell
    assert inr.date == "2024-01-01"


def test_macro_reports_listed_from_feed() -> None:
    """The macro category feed parses into typed reports with language/period."""
    client = NRBClient(
        fetcher=FixtureFetcher(data=(_FIX / "nrb_macro_feed.xml").read_bytes())
    )
    reports = client.list_macro_reports(limit=10)
    assert reports
    assert all(r.url.startswith("https://www.nrb.org.np") for r in reports)
    assert {r.language for r in reports} & {"english", "nepali", "tables"}


def _xlsx(rows: list[list[object]], sheet: str = "Macro") -> bytes:
    book = openpyxl.Workbook()
    ws = book.active
    ws.title = sheet
    for row in rows:
        ws.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def test_parse_workbook_roundtrips_sheet_rows() -> None:
    """parse_workbook reads every sheet into rows (proves the Excel capability)."""
    data = _xlsx([["Indicator", "Value"], ["Inflation", 4.2], ["Remittance", 1200]])
    workbook = NRBClient.parse_workbook(data)
    assert workbook.sheet_names() == ["Macro"]
    assert workbook.sheets["Macro"][0] == ["Indicator", "Value"]
    assert workbook.sheets["Macro"][1][0] == "Inflation"


def test_fetch_macro_workbook_finds_link_and_parses() -> None:
    """A report page linking an .xlsx is downloaded + parsed end-to-end."""
    xlsx = _xlsx([["a", "b"], [1, 2]])
    xls_url = "https://www.nrb.org.np/contents/uploads/2025/06/macro-tables.xlsx"
    html = f'<html><body><a href="{xls_url}">Download tables</a></body></html>'
    fetcher = FixtureFetcher(routes=[(".xlsx", xlsx), ("/red/", html)])
    workbook = NRBClient(fetcher=fetcher).fetch_macro_workbook(
        "https://www.nrb.org.np/red/some-tables-report/"
    )
    assert workbook is not None
    assert workbook.source_url == xls_url
    assert workbook.sheet_names()


def test_fetch_macro_workbook_returns_none_without_link() -> None:
    """A JS-only report page (no static .xlsx) yields None, not an error."""
    fetcher = FixtureFetcher(text="<html><body>no spreadsheet here</body></html>")
    assert NRBClient(fetcher=fetcher).fetch_macro_workbook("https://x/red/y/") is None
