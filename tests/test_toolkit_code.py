"""Tests for the code pack: run_python via the SubprocessSandbox."""

from __future__ import annotations

from himmy.services.sandbox.models import SandboxLimits
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.code import register_code_pack
from himmy.toolkit.config import ToolkitConfig
from tests.conftest import run_async


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    # Generous wall/cpu limits so a cold interpreter start under heavy parallel
    # test load cannot time the run out (the limits themselves aren't under test).
    limits = SandboxLimits(cpu_seconds=30, timeout_seconds=30)
    register_code_pack(registry, ToolkitConfig(sandbox_limits=limits))
    return registry


def test_run_python_executes_and_captures_stdout() -> None:
    handler = _registry().handler_for("run_python")
    out = run_async(handler({"code": "print(6 * 7)"}))
    assert out["ok"] is True
    assert out["exit_code"] == 0
    assert out["stdout"].strip() == "42"


def test_run_python_is_approval_gated() -> None:
    assert _registry().get("run_python").requires_approval is True
