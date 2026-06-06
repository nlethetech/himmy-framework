"""Nepal connectors: register NRB capabilities as Himmy ToolService tools.

Each tool runs through the normal tool pipeline (validation / approval / events /
lineage). Network calls are sync, so handlers offload them with
``asyncio.to_thread`` to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any

from himmy.connectors.nrb import NRBClient
from himmy.services.tools.registry import ToolRegistry, register_local_tool


def _json_safe(cell: Any) -> Any:
    """Make one spreadsheet cell JSON-serializable (Excel date cells → ISO strings)."""
    if isinstance(cell, (datetime.datetime, datetime.date)):
        return cell.isoformat()
    return cell


def _safe_rows(rows: list[list[Any]], limit: int) -> list[list[Any]]:
    return [[_json_safe(c) for c in row] for row in rows[:limit]]


def register_nrb_tools(
    registry: ToolRegistry,
    *,
    client: NRBClient | None = None,
    requires_approval: bool = False,
) -> list[str]:
    """Register ``nrb_forex``, ``nrb_macro_reports``, ``nrb_macro_workbook``."""
    nrb = client or NRBClient()

    async def _forex(args: dict[str, Any]) -> dict[str, Any]:
        from_date = str(args.get("from_date") or args.get("date") or "")
        to_date = args.get("to_date")
        rates = await asyncio.to_thread(
            nrb.forex, from_date, str(to_date) if to_date else None
        )
        return {"count": len(rates), "rates": [r.model_dump() for r in rates]}

    async def _reports(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit", 20))
        reports = await asyncio.to_thread(lambda: nrb.list_macro_reports(limit=limit))
        return {"count": len(reports), "reports": [r.model_dump() for r in reports]}

    async def _workbook(args: dict[str, Any]) -> dict[str, Any]:
        report_url = args.get("report_url")
        if report_url:
            workbook = await asyncio.to_thread(
                nrb.fetch_macro_workbook, str(report_url)
            )
        else:
            # No URL: auto-discover the latest 'Tables' macro workbook.
            workbook = await asyncio.to_thread(nrb.fetch_latest_macro_workbook)
        if workbook is None:
            return {"found": False, "source_url": None, "sheets": []}

        # The full workbook (dozens of sheets, thousands of cells) is far too big for a
        # model context — and Excel date cells aren't JSON-serializable. So return the
        # sheet NAMES by default, and only one requested sheet's rows (capped + made
        # JSON-safe) when ``sheet`` is given.
        result: dict[str, Any] = {
            "found": True,
            "source_url": workbook.source_url,
            "sheets": workbook.sheet_names(),
            "sheet_count": len(workbook.sheets),
        }
        sheet = args.get("sheet")
        if sheet:
            rows = workbook.sheets.get(str(sheet))
            if rows is None:
                result["error"] = f"unknown sheet {sheet!r}; see `sheets`"
            else:
                max_rows = max(1, min(int(args.get("max_rows", 30)), 200))
                result["sheet"] = str(sheet)
                result["row_count"] = len(rows)
                result["rows"] = _safe_rows(rows, max_rows)
        return result

    register_local_tool(
        registry,
        name="nrb_forex",
        handler=_forex,
        description="NRB foreign-exchange buy/sell rates for a date (ISO YYYY-MM-DD).",
        args_json_schema={
            "type": "object",
            "properties": {
                "from_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                "to_date": {"type": "string", "description": "optional range end"},
            },
            "required": ["from_date"],
        },
        requires_approval=requires_approval,
        metadata={"backend": "nrb"},
    )
    register_local_tool(
        registry,
        name="nrb_macro_reports",
        handler=_reports,
        description="List NRB monthly Current Macroeconomic & Financial Situation reports.",
        args_json_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
        requires_approval=requires_approval,
        metadata={"backend": "nrb"},
    )
    register_local_tool(
        registry,
        name="nrb_macro_workbook",
        handler=_workbook,
        description=(
            "Open NRB's macro 'Tables' Excel workbook. Returns the list of sheet names "
            "(call again with `sheet` to read one sheet's rows). Omit report_url to "
            "auto-discover the latest workbook. (Heavy: downloads + parses a real .xlsx.)"
        ),
        args_json_schema={
            "type": "object",
            "properties": {
                "report_url": {
                    "type": "string",
                    "description": "optional; defaults to the latest Tables report",
                },
                "sheet": {
                    "type": "string",
                    "description": "a sheet name to read rows from (omit to just list)",
                },
                "max_rows": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 30,
                    "description": "cap on rows returned for `sheet`",
                },
            },
        },
        requires_approval=requires_approval,
        metadata={"backend": "nrb"},
    )
    return ["nrb_forex", "nrb_macro_reports", "nrb_macro_workbook"]


__all__ = ["register_nrb_tools"]
