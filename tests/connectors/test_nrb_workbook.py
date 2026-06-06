"""nrb_macro_workbook is agent-safe: lists sheets by default, reads one sheet on request,
and serializes Excel date cells (the bug that broke the full-workbook dump)."""

from __future__ import annotations

import datetime
import json

from himmy.connectors.models import Workbook
from himmy.connectors.nrb_tools import register_nrb_tools
from himmy.services.tools.registry import ToolRegistry
from tests.conftest import run_async


class _FakeNRB:
    """A stand-in NRB client returning a tiny workbook with Excel date cells."""

    def _wb(self) -> Workbook:
        return Workbook(
            source_url="http://nrb/tables.xlsx",
            sheets={
                "Content": [["Title", "macro tables"]],
                "1.SMIs": [
                    ["date", "value"],
                    [datetime.datetime(2026, 1, 15), 42.0],
                    [datetime.date(2026, 2, 1), 43.5],
                ],
            },
        )

    def fetch_latest_macro_workbook(self) -> Workbook:
        return self._wb()

    def fetch_macro_workbook(self, url: str) -> Workbook:
        return self._wb()


def _handler():
    registry = ToolRegistry()
    register_nrb_tools(registry, client=_FakeNRB())
    return registry.handler_for("nrb_macro_workbook")


def test_lists_sheet_names_by_default_no_heavy_payload() -> None:
    out = run_async(_handler()({}))
    assert out["found"] is True
    assert out["sheets"] == ["Content", "1.SMIs"]
    assert out["sheet_count"] == 2
    assert "rows" not in out  # the full data is NOT dumped by default
    json.dumps(out)  # the result is JSON-serializable


def test_reads_one_sheet_and_serializes_date_cells() -> None:
    out = run_async(_handler()({"sheet": "1.SMIs"}))
    assert out["sheet"] == "1.SMIs"
    assert out["row_count"] == 3
    blob = json.dumps(out["rows"])  # must not raise (the datetime bug)
    assert "2026-01-15" in blob and "2026-02-01" in blob


def test_max_rows_caps_but_reports_true_total() -> None:
    out = run_async(_handler()({"sheet": "1.SMIs", "max_rows": 1}))
    assert len(out["rows"]) == 1
    assert out["row_count"] == 3  # the real total, not the capped slice


def test_unknown_sheet_is_reported() -> None:
    out = run_async(_handler()({"sheet": "does-not-exist"}))
    assert "error" in out
    assert "rows" not in out
