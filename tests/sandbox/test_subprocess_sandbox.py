"""Tests for the SubprocessSandbox: isolation, limits, timeout, and the tool."""

from __future__ import annotations

import pytest

from opensims.core.errors import OpenSimsError
from opensims.services.sandbox import (
    SandboxLimits,
    SubprocessSandbox,
    register_sandbox_tool,
)
from opensims.services.tools.models import ToolInvocation
from opensims.services.tools.registry import ToolRegistry
from opensims.services.tools.service import ToolService
from tests.conftest import run_async


def test_runs_code_and_captures_stdout() -> None:
    """A clean run returns ok=True, exit 0, and the captured stdout."""
    result = run_async(SubprocessSandbox().run_code("print(2 + 2)"))
    assert result.ok is True
    assert result.exit_code == 0
    assert result.stdout.strip() == "4"
    assert result.timed_out is False


def test_stdin_is_fed_to_the_process() -> None:
    """stdin is delivered to the sandboxed code."""
    result = run_async(
        SubprocessSandbox().run_code(
            "import sys; print(sys.stdin.read().upper())", stdin="abc"
        )
    )
    assert result.stdout.strip() == "ABC"


def test_nonzero_exit_is_reported_not_raised() -> None:
    """A non-zero exit is captured in the result, not raised."""
    result = run_async(SubprocessSandbox().run_code("raise SystemExit(3)"))
    assert result.ok is False
    assert result.exit_code == 3


def test_exception_traceback_goes_to_stderr() -> None:
    """An uncaught exception lands on stderr with a non-zero exit."""
    result = run_async(SubprocessSandbox().run_code("raise ValueError('boom')"))
    assert result.ok is False
    assert result.exit_code == 1
    assert "ValueError" in result.stderr


def test_wall_clock_timeout_kills_the_process() -> None:
    """A sleep beyond the wall-clock budget is killed and flagged timed_out."""
    sandbox = SubprocessSandbox(limits=SandboxLimits(timeout_seconds=0.5))
    result = run_async(sandbox.run_code("import time; time.sleep(5)"))
    assert result.timed_out is True
    assert result.ok is False
    assert result.duration_ms < 4000  # killed early, not after the full sleep


def test_environment_is_stripped_unless_allow_listed(monkeypatch) -> None:
    """Parent env is hidden from the child unless explicitly allow-listed."""
    monkeypatch.setenv("OPENSIMS_SBX_SECRET", "top-secret")
    code = "import os; print(os.environ.get('OPENSIMS_SBX_SECRET', 'MISSING'))"

    stripped = run_async(SubprocessSandbox().run_code(code))
    assert stripped.stdout.strip() == "MISSING"

    allowed = run_async(
        SubprocessSandbox(
            limits=SandboxLimits(allow_env=["OPENSIMS_SBX_SECRET"])
        ).run_code(code)
    )
    assert allowed.stdout.strip() == "top-secret"


def test_files_are_written_into_the_working_dir() -> None:
    """Provided files are readable by relative name from the working dir."""
    result = run_async(
        SubprocessSandbox().run_code(
            "print(open('data.txt').read())", files={"data.txt": "hello-from-disk"}
        )
    )
    assert result.stdout.strip() == "hello-from-disk"


def test_output_is_truncated_to_the_limit() -> None:
    """Captured output is bounded and the truncation is flagged."""
    sandbox = SubprocessSandbox(limits=SandboxLimits(max_output_bytes=100))
    result = run_async(sandbox.run_code("print('x' * 10000)"))
    assert result.truncated is True
    assert len(result.stdout.encode("utf-8")) <= 100


def test_file_path_traversal_is_rejected() -> None:
    """A file name escaping the working dir is a hard error."""
    with pytest.raises(OpenSimsError):
        run_async(SubprocessSandbox().run_code("pass", files={"../escape.txt": "x"}))


# ----------------------------------------------------- tool integration (gated)
def test_sandbox_tool_is_approval_gated_then_runs() -> None:
    """register_sandbox_tool gates on approval, then executes through ToolService."""
    registry = ToolRegistry()
    register_sandbox_tool(registry, SubprocessSandbox())
    service = ToolService(registry)

    denied = run_async(
        service.execute(
            ToolInvocation(tool_name="run_python", args={"code": "print(2 + 2)"})
        )
    )
    assert denied.outcome == "denied"  # requires_approval defaults to True

    approved = run_async(
        service.execute(
            ToolInvocation(
                tool_name="run_python",
                args={"code": "print(2 + 2)"},
                metadata={"approved": True},
            )
        )
    )
    assert approved.outcome == "success"
    assert approved.result["ok"] is True
    assert approved.result["stdout"].strip() == "4"
