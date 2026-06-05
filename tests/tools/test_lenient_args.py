"""Tier 1.2/1.4 — tolerant tool-arg coercion + schema-in-error self-correction."""

from __future__ import annotations

from typing import Any

from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async

_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "string"},
        "y": {"type": "string"},  # optional
    },
    "required": ["x"],
    "additionalProperties": False,
}


def _service(**over: Any) -> ToolService:
    reg = ToolRegistry()
    register_local_tool(
        reg,
        name="t",
        handler=lambda args: {"seen": args},
        args_json_schema=_SCHEMA,
    )
    return ToolService(reg, **over)


def _call(svc: ToolService, **args: Any):
    return run_async(svc.execute(ToolInvocation(tool_name="t", args=args)))


def test_strips_hallucinated_unknown_key() -> None:
    res = _call(_service(), x="hi", phantom="junk")
    assert res.outcome == "success"
    assert res.result["seen"] == {"x": "hi"}  # phantom dropped before the handler


def test_drops_null_valued_optional() -> None:
    res = _call(_service(), x="hi", y=None)
    assert res.outcome == "success"
    assert "y" not in res.result["seen"]  # null optional treated as omitted


def test_keeps_valid_args_untouched() -> None:
    res = _call(_service(), x="hi", y="there")
    assert res.result["seen"] == {"x": "hi", "y": "there"}


def test_required_still_enforced_with_schema_hint() -> None:
    res = _call(_service(), y="only-optional")  # missing required x
    assert res.outcome != "success"
    assert "expected args:" in (res.error_message or "")
    assert "x (string)*" in (res.error_message or "")  # required marked with *


def test_strict_mode_rejects_unknown_key() -> None:
    res = _call(_service(lenient_args=False), x="hi", phantom="junk")
    assert res.outcome != "success"  # opt-out keeps strict validation
