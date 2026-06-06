"""Utils pack: ``calculator`` (safe arithmetic) and ``current_time``.

``calculator`` evaluates a math expression via a restricted ``ast`` walk — numbers and
the arithmetic operators only, with no names, calls, or attribute access — so it is
safe against the injection that a bare ``eval`` would invite. ``current_time`` returns
the current instant in a requested timezone with optional day/hour offsets.
"""

from __future__ import annotations

import ast
import operator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST node, rejecting anything unsafe."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("expression contains an unsupported or unsafe term")


def safe_calculate(expression: str) -> float:
    """Evaluate an arithmetic ``expression`` (``+ - * / // % **`` and unary signs)."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree)


_CALC_SCHEMA = {
    "type": "object",
    "properties": {"expression": {"type": "string", "description": "Math expression."}},
    "required": ["expression"],
    "additionalProperties": False,
}

_TIME_SCHEMA = {
    "type": "object",
    "properties": {
        "timezone": {"type": "string", "description": "IANA tz name, default UTC."},
        "offset_days": {"type": "number", "default": 0},
        "offset_hours": {"type": "number", "default": 0},
    },
    "additionalProperties": False,
}


def register_utils_pack(registry: ToolRegistry, config: Any) -> None:
    """Register the ``calculator`` and ``current_time`` tools (no config needed)."""

    def calculator(args: dict[str, Any]) -> dict[str, Any]:
        expression = str(args["expression"])
        return {"expression": expression, "result": safe_calculate(expression)}

    def current_time(args: dict[str, Any]) -> dict[str, Any]:
        tz_name = str(args.get("timezone", "UTC"))
        now = datetime.now(_resolve_tz(tz_name)) + timedelta(
            days=float(args.get("offset_days", 0)),
            hours=float(args.get("offset_hours", 0)),
        )
        return {
            "iso": now.isoformat(),
            "timezone": tz_name,
            "weekday": now.strftime("%A"),
            "unix": now.timestamp(),
        }

    register_local_tool(
        registry,
        name="calculator",
        read_only=True,
        handler=calculator,
        description="Evaluate a math expression (+, -, *, /, //, %, **).",
        args_json_schema=_CALC_SCHEMA,
        metadata={"pack": "utils"},
    )
    register_local_tool(
        registry,
        name="current_time",
        read_only=True,
        handler=current_time,
        description="Return the current time in a timezone, with optional offsets.",
        args_json_schema=_TIME_SCHEMA,
        metadata={"pack": "utils"},
    )


def _resolve_tz(name: str) -> timezone | Any:
    """Resolve an IANA timezone name; fall back to UTC when unavailable."""
    if name.upper() == "UTC":
        return UTC
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return UTC


__all__ = ["register_utils_pack", "safe_calculate"]
