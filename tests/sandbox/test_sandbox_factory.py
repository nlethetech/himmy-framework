"""WS2.1 — sandbox backend selection (off / subprocess / container)."""

from __future__ import annotations

from himmy.services.sandbox import (
    ContainerSandbox,
    DisabledSandbox,
    SubprocessSandbox,
    build_sandbox,
)
from himmy.services.sandbox.tools import register_sandbox_tool
from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from himmy.toolkit import ToolkitConfig, register_packs
from tests.conftest import run_async


def test_build_sandbox_modes() -> None:
    assert isinstance(build_sandbox("off"), DisabledSandbox)
    assert isinstance(build_sandbox("subprocess"), SubprocessSandbox)
    assert isinstance(build_sandbox("container"), ContainerSandbox)
    # Unknown / empty falls back to the portable default (never accidentally off).
    assert isinstance(build_sandbox(""), SubprocessSandbox)
    assert isinstance(build_sandbox("bogus"), SubprocessSandbox)


def test_disabled_sandbox_refuses() -> None:
    result = run_async(DisabledSandbox().run_code("print(1)"))
    assert result.ok is False
    assert "disabled by policy" in result.stderr


def test_code_pack_off_mode_refuses_execution() -> None:
    """With HIMMY_CODE_EXEC=off the run_python tool is present but refuses to run."""
    registry = ToolRegistry()
    register_packs(registry, ["code"], ToolkitConfig(code_exec="off"))
    service = ToolService(registry)
    # The tool is approval-gated; approve, then it should still refuse (policy off).
    res = run_async(
        service.execute(
            ToolInvocation(
                tool_name="run_python",
                args={"code": "print(1)"},
                metadata={"approved": True},
            )
        )
    )
    assert res.outcome == "success"  # the tool ran; the sandbox returned a refusal
    assert res.result["ok"] is False
    assert "disabled by policy" in res.result["stderr"]


def test_code_pack_default_is_subprocess() -> None:
    """An unconfigured code pack keeps the portable subprocess backend (no breakage)."""
    sandbox = build_sandbox(ToolkitConfig().code_exec)
    assert isinstance(sandbox, SubprocessSandbox)


def test_register_sandbox_tool_is_approval_gated_for_container() -> None:
    registry = ToolRegistry()
    register_sandbox_tool(registry, build_sandbox("container"), name="run_python")
    assert registry.get("run_python").requires_approval is True
