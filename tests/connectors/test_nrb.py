"""Tests for the NRB connector: forex, macro report listing, Excel parsing."""

from __future__ import annotations

import io
import json
from pathlib import Path

import openpyxl
import pytest

from himmy.connectors import NRBClient
from himmy.core.errors import HimmyError
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


def test_forex_schema_drift_raises_clear_error() -> None:
    """A restructured payload fails LOUDLY (HimmyError), not with KeyErrors/emptiness."""
    drifted = [
        {"data": {"payload": {"renamed": []}}},  # payload no longer a list
        {"data": {"days": []}},  # payload key renamed
        {"results": []},  # data envelope renamed
        {"data": {"payload": ["not-a-day-object"]}},  # day rows no longer objects
    ]
    for body in drifted:
        client = NRBClient(fetcher=FixtureFetcher(text=json.dumps(body)))
        with pytest.raises(HimmyError, match="schema drift"):
            client.forex("2024-01-01")


def test_forex_tolerates_value_level_noise() -> None:
    """Unparseable buy/sell and a null unit degrade per-value, not per-batch."""
    body = {
        "data": {
            "payload": [
                {
                    "date": "2024-01-01",
                    "rates": [
                        {
                            "currency": {
                                "iso3": "USD",
                                "name": "US Dollar",
                                "unit": None,
                            },
                            "buy": "N/A",
                            "sell": "132.50",
                        }
                    ],
                }
            ]
        }
    }
    client = NRBClient(fetcher=FixtureFetcher(text=json.dumps(body)))
    (rate,) = client.forex("2024-01-01")
    assert rate.currency_iso3 == "USD"
    assert rate.unit == 1  # null unit defaults
    assert rate.buy is None  # unparseable price -> None, not a crash
    assert rate.sell == 132.50


def test_macro_reports_unparseable_feed_raises() -> None:
    """An error page where the RSS feed should be is a clear error, not []."""
    client = NRBClient(
        fetcher=FixtureFetcher(data=b'{"error": "rate limited, try later"}')
    )
    with pytest.raises(HimmyError, match="not parseable"):
        client.list_macro_reports()


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
    """A plain page with no spreadsheet (and not itself a workbook) yields None."""
    fetcher = FixtureFetcher(text="<html><body>no spreadsheet here</body></html>")
    assert NRBClient(fetcher=fetcher).fetch_macro_workbook("https://x/red/y/") is None


def test_fetch_macro_workbook_parses_direct_excel() -> None:
    """A report URL that serves the .xlsx directly is parsed (NRB 'Tables' case)."""
    xlsx = _xlsx([["Indicator", "Value"], ["CPI", 105.2]])
    report_url = "https://www.nrb.org.np/red/macro-tables-9m/"
    workbook = NRBClient(fetcher=FixtureFetcher(data=xlsx)).fetch_macro_workbook(
        report_url
    )
    assert workbook is not None
    assert workbook.source_url == report_url  # the report URL was the workbook
    assert workbook.sheets["Macro"][1][0] == "CPI"


def test_fetch_latest_macro_workbook_auto_discovers() -> None:
    """With no URL, it lists reports, picks the Tables variant, and parses it."""
    feed = (_FIX / "nrb_macro_feed.xml").read_bytes()
    xlsx = _xlsx([["a", "b"], [1, 2]])
    fetcher = FixtureFetcher(
        routes=[
            ("current-macroeconomic-situation/feed", feed),
            ("tables-based-on", xlsx),
        ]
    )
    workbook = NRBClient(fetcher=fetcher).fetch_latest_macro_workbook()
    assert workbook is not None
    assert workbook.sheet_names()


# --------------------------------------------------------- zip-bomb (DoS) guard


def _zip_bomb(declared_uncompressed: int) -> bytes:
    """A tiny zip (PK magic) whose member DECLARES a huge uncompressed size.

    Highly compressible (all-zero) content makes the stored bytes tiny while the
    central-directory ``file_size`` is large — exactly the shape of a zip bomb, so
    the guard must reject it on the declared size without inflating anything.
    """
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/sharedStrings.xml", b"\x00" * declared_uncompressed)
    payload = buffer.getvalue()
    assert payload[:4] == b"PK\x03\x04"
    return payload


def test_parse_workbook_rejects_zip_bomb() -> None:
    """A small .xlsx declaring a huge decompressed member is refused before parsing."""
    from himmy.connectors import nrb

    bomb = _zip_bomb(64 * 1024)  # stays tiny compressed; 64KB uncompressed
    assert len(bomb) < 4096  # the compressed download cap would NOT have caught it
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nrb, "_MAX_WORKBOOK_UNCOMPRESSED_BYTES", 1024)
        with pytest.raises(HimmyError, match="zip bomb"):
            NRBClient.parse_workbook(bomb)


def test_parse_workbook_zip_bomb_member_guard_independent_of_total() -> None:
    """A single oversize member trips the guard even on its own (per-member bound)."""
    from himmy.connectors import nrb

    bomb = _zip_bomb(8 * 1024)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nrb, "_MAX_WORKBOOK_UNCOMPRESSED_BYTES", 2048)
        with pytest.raises(HimmyError, match="zip bomb"):
            nrb._guard_decompressed_size(bomb)


def test_parse_workbook_allows_normal_workbook_under_cap() -> None:
    """A legitimate (small) workbook still parses — the guard is a ceiling, not a floor."""
    data = _xlsx([["Indicator", "Value"], ["CPI", 105.2]])
    workbook = NRBClient.parse_workbook(data)  # default cap is generous
    assert workbook.sheets["Macro"][1][0] == "CPI"
