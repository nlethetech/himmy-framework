"""WS2.2 — ContainerSandbox, live-verified against a real container engine.

Skipped automatically when Docker is unavailable (e.g. CI without a daemon); on a
machine with Docker these prove the actual security envelope: egress denied,
read-only rootfs, non-root user, resource limits, and a hard wall-clock kill.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from himmy.services.sandbox import ContainerSandbox
from himmy.services.sandbox.models import SandboxLimits
from tests.conftest import run_async

_IMAGE = "python:3.12-slim"


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15
            ).returncode
            == 0
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_ok(), reason="docker engine not available")


@pytest.fixture(scope="module", autouse=True)
def _ensure_image() -> None:
    """Make sure the sandbox image is present (pull once, or skip if offline)."""
    if (
        subprocess.run(
            ["docker", "image", "inspect", _IMAGE], capture_output=True
        ).returncode
        != 0
    ):
        pull = subprocess.run(
            ["docker", "pull", _IMAGE], capture_output=True, timeout=600
        )
        if pull.returncode != 0:  # pragma: no cover - depends on network
            pytest.skip("could not pull the sandbox image")


def _sbx(**limit_kwargs: object) -> ContainerSandbox:
    limits = SandboxLimits(**limit_kwargs) if limit_kwargs else None
    # /tmp is shared with the engine VM on Docker Desktop (macOS), unlike $TMPDIR.
    return ContainerSandbox(
        image=_IMAGE, limits=limits, workdir_base="/tmp", startup_grace=25.0
    )


def test_hello_world_runs_and_captures_stdout() -> None:
    result = run_async(_sbx().run_code("print('hello from container')"))
    assert result.ok is True
    assert result.exit_code == 0
    assert "hello from container" in result.stdout


def test_runs_as_non_root_user() -> None:
    result = run_async(_sbx().run_code("import os; print(os.getuid())"))
    assert result.ok is True
    assert result.stdout.strip() == "65534"


def test_network_egress_is_denied() -> None:
    code = (
        "import socket\n"
        "socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "print('CONNECTED')\n"
    )
    result = run_async(_sbx().run_code(code))
    assert result.ok is False
    assert "CONNECTED" not in result.stdout  # the connection must have failed


def test_root_filesystem_is_read_only() -> None:
    result = run_async(_sbx().run_code("open('/oops', 'w').write('x')"))
    assert result.ok is False
    assert "read-only" in result.stderr.lower() or result.exit_code != 0


def test_input_files_are_readable() -> None:
    result = run_async(
        _sbx().run_code(
            "print(open('/work/data.txt').read().strip())",
            files={"data.txt": "the-secret-value"},
        )
    )
    assert result.ok is True
    assert result.stdout.strip() == "the-secret-value"


def test_wall_clock_timeout_kills_the_container() -> None:
    result = run_async(_sbx(timeout_seconds=2.0).run_code("while True: pass"))
    assert result.timed_out is True
    assert result.ok is False
