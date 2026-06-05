"""Tests for the utils pack: calculator (safe) + current_time."""

from __future__ import annotations

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.utils import register_utils_pack, safe_calculate


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_utils_pack(registry, ToolkitConfig())
    return registry


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2+2*10", 22.0),
        ("(1+2)*3", 9.0),
        ("2**8", 256.0),
        ("-5 + 3", -2.0),
        ("7 % 3", 1.0),
    ],
)
def test_calculator_arithmetic(expr: str, expected: float) -> None:
    assert (
        _registry().handler_for("calculator")({"expression": expr})["result"]
        == expected
    )


@pytest.mark.parametrize(
    "expr",
    ["__import__('os')", "1 + name", "len([1])", "().__class__"],
)
def test_calculator_rejects_unsafe(expr: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        safe_calculate(expr)


def test_current_time_utc() -> None:
    out = _registry().handler_for("current_time")({})
    assert out["timezone"] == "UTC"
    assert "T" in out["iso"]
    assert isinstance(out["unix"], float)


def test_current_time_offset_changes_instant() -> None:
    handler = _registry().handler_for("current_time")
    now = handler({})["unix"]
    later = handler({"offset_hours": 1})["unix"]
    assert later > now
