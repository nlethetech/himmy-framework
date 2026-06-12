"""Tests for the NRB ToolService tools (run through the tool pipeline)."""

from __future__ import annotations

from pathlib import Path

from himmy.connectors import NRBClient, register_nrb_tools
from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from tests.conftest import run_async
from tests.connectors._fixtures import FixtureFetcher

_FIX = Path(__file__).parent / "fixtures"


def test_nrb_tools_register_and_forex_runs() -> None:
    """register_nrb_tools wires the tools; nrb_forex runs through ToolService."""
    client = NRBClient(
        fetcher=FixtureFetcher(text=(_FIX / "nrb_forex.json").read_text())
    )
    registry = ToolRegistry()
    names = register_nrb_tools(registry, client=client)
    assert set(names) == {"nrb_forex", "nrb_macro_reports", "nrb_macro_workbook"}

    service = ToolService(registry)
    result = run_async(
        service.execute(
            ToolInvocation(tool_name="nrb_forex", args={"from_date": "2024-01-01"})
        )
    )
    assert result.outcome == "success"
    assert result.result["count"] >= 10
    assert any(r["currency_iso3"] == "INR" for r in result.result["rates"])


def test_nrb_macro_workbook_guards_model_supplied_url() -> None:
    """A model-supplied report_url pointing at an internal address is refused.

    Without the guard the connector would fetch the URL directly (and scan its body
    for linked spreadsheets), making nrb_macro_workbook an SSRF read primitive.
    """
    fetched: list[str] = []

    class _SpyFetcher:
        def get_text(self, url: str) -> str:  # pragma: no cover - never reached
            fetched.append(url)
            return ""

        def get_bytes(self, url: str) -> bytes:  # pragma: no cover - never reached
            fetched.append(url)
            return b""

    client = NRBClient(fetcher=_SpyFetcher())
    registry = ToolRegistry()
    register_nrb_tools(registry, client=client)

    service = ToolService(registry)
    result = run_async(
        service.execute(
            ToolInvocation(
                tool_name="nrb_macro_workbook",
                args={"report_url": "http://169.254.169.254/latest/meta-data/"},
            )
        )
    )
    assert result.outcome == "failed"
    assert fetched == []  # the guard fired BEFORE any fetch happened
