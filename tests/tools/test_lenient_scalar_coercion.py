"""Lossless scalar arg-coercion on the lenient path (QUICK WIN 4).

Small local models (Ollama / Claude-CLI) routinely stringify scalars (``"5"`` for an
``integer``, ``"true"`` for a ``boolean``), which the strict validator hard-fails —
making the local path strictly worse than cloud. ``_coerce_lenient_args`` now
LOSSLESSLY coerces a stringified scalar to a declared scalar type (and only then),
records the coerced keys for measurement, and leaves ambiguous/lossy values alone.
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService, _coerce_lenient_args
from tests.conftest import run_async

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "label": {"type": "string"},
        "either": {"type": ["integer", "null"]},
    },
    "required": ["count"],
    "additionalProperties": False,
}


# --------------------------------------------------------- helper-level coercion
def test_coerces_stringified_int() -> None:
    cleaned, keys = _coerce_lenient_args({"count": "5"}, _SCHEMA)
    assert cleaned["count"] == 5 and isinstance(cleaned["count"], int)
    assert keys == ["count"]


def test_coerces_stringified_float() -> None:
    cleaned, keys = _coerce_lenient_args({"count": 1, "ratio": "3.14"}, _SCHEMA)
    assert cleaned["ratio"] == 3.14 and isinstance(cleaned["ratio"], float)
    assert keys == ["ratio"]


def test_coerces_stringified_bool() -> None:
    cleaned, keys = _coerce_lenient_args({"count": 1, "flag": "true"}, _SCHEMA)
    assert cleaned["flag"] is True
    assert keys == ["flag"]
    cleaned2, _ = _coerce_lenient_args({"count": 1, "flag": "false"}, _SCHEMA)
    assert cleaned2["flag"] is False


def test_coerces_into_union_scalar_type() -> None:
    cleaned, keys = _coerce_lenient_args({"count": 1, "either": "7"}, _SCHEMA)
    assert cleaned["either"] == 7
    assert "either" in keys


# --------------------------------------------------------- left alone (safe)
def test_string_typed_field_is_never_coerced() -> None:
    """A ``"5"`` against a string field stays a string (target type is string)."""
    cleaned, keys = _coerce_lenient_args({"count": 1, "label": "5"}, _SCHEMA)
    assert cleaned["label"] == "5"
    assert keys == []


def test_lossy_int_is_left_alone() -> None:
    """``"5.0"``/``" 5"``/``"05"`` do not round-trip to an int losslessly -> untouched."""
    for raw in ("5.0", " 5", "05", "+5", "0x5"):
        cleaned, keys = _coerce_lenient_args({"count": raw}, _SCHEMA)
        assert cleaned["count"] == raw
        assert keys == []


def test_non_canonical_float_is_left_alone() -> None:
    """``"1e3"`` does not equal ``repr(1000.0)`` -> left as-is (no silent reshape)."""
    cleaned, keys = _coerce_lenient_args({"count": 1, "ratio": "1e3"}, _SCHEMA)
    assert cleaned["ratio"] == "1e3"
    assert keys == []


def test_non_canonical_bool_is_left_alone() -> None:
    cleaned, keys = _coerce_lenient_args({"count": 1, "flag": "True"}, _SCHEMA)
    assert cleaned["flag"] == "True"
    assert keys == []


def test_already_typed_scalar_is_untouched() -> None:
    cleaned, keys = _coerce_lenient_args({"count": 5, "flag": True}, _SCHEMA)
    assert cleaned == {"count": 5, "flag": True}
    assert keys == []


# --------------------------------------------------------- end-to-end + metadata
def _service() -> ToolService:
    reg = ToolRegistry()
    register_local_tool(
        reg,
        name="t",
        handler=lambda args: {"seen": args},
        args_json_schema=_SCHEMA,
    )
    return ToolService(reg)


def test_end_to_end_coercion_passes_strict_validation() -> None:
    """A stringified scalar now passes validation and reaches the handler as a real scalar."""
    svc = _service()
    res = run_async(
        svc.execute(ToolInvocation(tool_name="t", args={"count": "5", "flag": "true"}))
    )
    assert res.outcome == "success"
    assert res.result["seen"] == {"count": 5, "flag": True}


def test_coerced_keys_recorded_in_metadata() -> None:
    svc = _service()
    res = run_async(
        svc.execute(ToolInvocation(tool_name="t", args={"count": "5", "ratio": "2.5"}))
    )
    assert res.outcome == "success"
    assert set(res.metadata.get("coerced_args", [])) == {"count", "ratio"}


def test_no_coercion_leaves_metadata_clean() -> None:
    svc = _service()
    res = run_async(svc.execute(ToolInvocation(tool_name="t", args={"count": 5})))
    assert res.outcome == "success"
    assert "coerced_args" not in res.metadata


def test_strict_mode_disables_coercion() -> None:
    """With ``lenient_args=False`` a stringified scalar still hard-fails (unchanged)."""
    reg = ToolRegistry()
    register_local_tool(
        reg, name="t", handler=lambda args: {"seen": args}, args_json_schema=_SCHEMA
    )
    svc = ToolService(reg, lenient_args=False)
    res = run_async(svc.execute(ToolInvocation(tool_name="t", args={"count": "5"})))
    assert res.outcome != "success"
