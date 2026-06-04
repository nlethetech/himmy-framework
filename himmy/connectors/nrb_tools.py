"""Nepal connectors: register NRB capabilities as Himmy ToolService tools.

Each tool runs through the normal tool pipeline (validation / approval / events /
lineage). Network calls are sync, so handlers offload them with
``asyncio.to_thread`` to keep the event loop free.
"""

from __future__ import annotations

import asyncio
from typing import Any

from himmy.connectors.nrb import NRBClient
from himmy.services.tools.registry import ToolRegistry, register_local_tool


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
        report_url = str(args.get("report_url", ""))
        workbook = await asyncio.to_thread(nrb.fetch_macro_workbook, report_url)
        return {
            "found": workbook is not None,
            "workbook": workbook.model_dump() if workbook is not None else None,
        }

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
        description="Download + parse a macro report's Excel workbook into sheet rows.",
        args_json_schema={
            "type": "object",
            "properties": {"report_url": {"type": "string"}},
            "required": ["report_url"],
        },
        requires_approval=requires_approval,
        metadata={"backend": "nrb"},
    )
    return ["nrb_forex", "nrb_macro_reports", "nrb_macro_workbook"]


__all__ = ["register_nrb_tools"]
