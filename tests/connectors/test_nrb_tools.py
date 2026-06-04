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
