"""Nepal pack: Bikram Sambat dates, NPR/Devanagari formatting, transliteration, NRB forex.

Makes Himmy's Nepal modules agent-facing. `nepali_date` converts between AD and the
Bikram Sambat calendar (needs the `nepal` extra); `nepali_format` renders NPR amounts in
the Nepali digit-grouping system and Devanagari numerals; `nepali_transliterate` folds
Devanagari↔Roman; and NRB foreign-exchange tools come from the existing connector.
"""

from __future__ import annotations

import datetime
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig

_DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def _to_devanagari(text: str) -> str:
    """Map ASCII digits in ``text`` to Devanagari numerals."""
    return text.translate(_DEVANAGARI_DIGITS)


def _group_nepali(integer: str) -> str:
    """Group an integer string in the South-Asian system (last 3, then pairs)."""
    if len(integer) <= 3:
        return integer
    head, tail = integer[:-3], integer[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


_DATE_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": "An AD date YYYY-MM-DD; omit for today.",
        }
    },
    "additionalProperties": False,
}

_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {"amount": {"type": "number", "description": "An NPR amount."}},
    "required": ["amount"],
    "additionalProperties": False,
}

_TRANSLIT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def register_nepal_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register the Nepal tools (calendar / formatting / transliteration / NRB forex)."""

    def nepali_date(args: dict[str, Any]) -> dict[str, Any]:
        from himmy.nepal.calendar import ad_to_bs, nepali_fiscal_year

        raw = args.get("date")
        ad = datetime.date.fromisoformat(str(raw)) if raw else datetime.date.today()
        bs = ad_to_bs(ad)
        return {
            "ad": ad.isoformat(),
            "bs": f"{bs.year}-{bs.month:02d}-{bs.day:02d}",
            "bs_month_en": bs.month_name("en"),
            "bs_month_ne": bs.month_name("ne"),
            "fiscal_year": nepali_fiscal_year(ad),
        }

    def nepali_format(args: dict[str, Any]) -> dict[str, Any]:
        amount = float(args["amount"])
        whole = f"{abs(amount):,.2f}".replace(",", "")
        integer, _, frac = whole.partition(".")
        grouped = _group_nepali(integer)
        sign = "-" if amount < 0 else ""
        npr = f"{sign}रू {grouped}.{frac}"
        return {
            "npr": npr,
            "grouped": f"{sign}{grouped}.{frac}",
            "devanagari": _to_devanagari(npr),
        }

    def nepali_transliterate(args: dict[str, Any]) -> dict[str, Any]:
        from himmy.nepal.language import normalize_nepali, transliterate

        text = str(args["text"])
        return {
            "text": text,
            "roman": transliterate(text),
            "normalized": normalize_nepali(text),
        }

    register_local_tool(
        registry,
        name="nepali_date",
        handler=nepali_date,
        description="Convert an AD date (or today) to the Bikram Sambat calendar + FY.",
        args_json_schema=_DATE_SCHEMA,
        metadata={"pack": "nepal"},
    )
    register_local_tool(
        registry,
        name="nepali_format",
        handler=nepali_format,
        description="Format an NPR amount (Nepali grouping + Devanagari numerals).",
        args_json_schema=_FORMAT_SCHEMA,
        metadata={"pack": "nepal"},
    )
    register_local_tool(
        registry,
        name="nepali_transliterate",
        handler=nepali_transliterate,
        description="Transliterate/normalize text across Devanagari and Roman scripts.",
        args_json_schema=_TRANSLIT_SCHEMA,
        metadata={"pack": "nepal"},
    )

    from himmy.connectors.nrb_tools import register_nrb_tools

    register_nrb_tools(registry)


#: Tool names the nepal pack provides (for the catalog / `himmy tools`).
NEPAL_TOOL_NAMES = (
    "nepali_date",
    "nepali_format",
    "nepali_transliterate",
    "nrb_forex",
    "nrb_macro_reports",
    "nrb_macro_workbook",
)


__all__ = ["register_nepal_pack", "NEPAL_TOOL_NAMES"]
