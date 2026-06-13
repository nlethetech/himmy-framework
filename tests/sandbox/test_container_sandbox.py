"""WS2.2 — ContainerSandbox, live-verified against a real container engine.

Skipped automatically when Docker is unavailable (e.g. CI without a daemon); on a
machine with Docker these prove the actual five enterprise guarantees per run:
egress denied, ephemeral (no bind mount + removed after), resource-capped
(memory/pids/ulimit + a hard wall-clock kill), no host access (read-only rootfs,
cap-drop, non-root), and audited (every run on the event spine).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from himmy.core.errors import HimmyError
from himmy.core.events import RunEvent
from himmy.services.sandbox import ContainerSandbox, NetworkMode, SandboxPolicy
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


class _RecordingSink:
    """A minimal in-memory EventSink that captures emitted events for assertions."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        self.events.append(event)


def _sbx(
    *, sink: _RecordingSink | None = None, **limit_kwargs: object
) -> ContainerSandbox:
    # No host path needs sharing: the workspace is an in-container tmpfs (no bind mount),
    # so this runs unmodified on Docker Desktop / macOS where $TMPDIR isn't shared.
    limits = SandboxLimits(**limit_kwargs) if limit_kwargs else None
    return ContainerSandbox(
        image=_IMAGE, limits=limits, startup_grace=25.0, event_sink=sink
    )


# ----------------------------------------------------------------- basic execution
def test_hello_world_runs_and_captures_stdout() -> None:
    result = run_async(_sbx().run_code("print('hello from container')"))
    assert result.ok is True
    assert result.exit_code == 0
    assert "hello from container" in result.stdout


def test_input_files_are_readable_by_relative_name() -> None:
    result = run_async(
        _sbx().run_code(
            "print(open('data.txt').read().strip())",
            files={"data.txt": "the-secret-value"},
        )
    )
    assert result.ok is True
    assert result.stdout.strip() == "the-secret-value"


def test_input_file_path_traversal_is_rejected() -> None:
    with pytest.raises(HimmyError):
        run_async(_sbx().run_code("pass", files={"../escape.txt": "x"}))


def test_stdin_is_handed_to_the_snippet() -> None:
    """The caller's stdin still reaches the code, after the bundle is consumed."""
    result = run_async(
        _sbx().run_code(
            "import sys; print('STDIN:[' + sys.stdin.read().strip() + ']')",
            stdin="hello-from-stdin",
        )
    )
    assert result.ok is True
    assert result.stdout.strip() == "STDIN:[hello-from-stdin]"


def test_large_code_and_files_run_via_out_of_band_stdin() -> None:
    """Regression: a >70KB snippet + a large input file must run, not fail with 255.

    The old design embedded the (double-base64-expanded) bundle in a single ``sh -c``
    argv string, so a 120KB code snippet or a 300KB file blew past the kernel's per-arg
    limit (``exec /usr/bin/sh: argument list too long`` → ok=False, exit_code=255). The
    bundle is now streamed on stdin, so both run cleanly within the documented
    ``file_size_mb`` workspace envelope.
    """
    big_code = (
        "x = '" + ("A" * 120_000) + "'\n"
        "data = open('payload.bin').read()\n"
        "print('CODELEN', len(x))\n"
        "print('FILELEN', len(data))\n"
    )
    result = run_async(_sbx().run_code(big_code, files={"payload.bin": "B" * 300_000}))
    assert result.ok is True, result.stderr
    assert result.exit_code == 0
    assert "argument list too long" not in result.stderr.lower()
    assert "CODELEN 120000" in result.stdout
    assert "FILELEN 300000" in result.stdout


def test_oversized_input_raises_clear_error_naming_the_cap() -> None:
    """Code+files beyond the file_size_mb workspace fail fast with an actionable error."""
    with pytest.raises(HimmyError) as exc:
        run_async(
            _sbx(file_size_mb=1).run_code(
                "pass", files={"big.bin": "Z" * (2 * 1024 * 1024)}
            )
        )
    assert "file_size_mb" in str(exc.value)


# ---------------------------------------------------------- (4) no host access
def test_runs_as_non_root_user() -> None:
    result = run_async(_sbx().run_code("import os; print(os.getuid())"))
    assert result.ok is True
    assert result.stdout.strip() == "65534"


def test_root_filesystem_is_read_only() -> None:
    result = run_async(_sbx().run_code("open('/oops', 'w').write('x')"))
    assert result.ok is False
    assert "read-only" in result.stderr.lower() or result.exit_code != 0


def test_no_host_bind_mount_is_used() -> None:
    """The argv must never contain a ``-v`` host bind mount (host access)."""
    sbx = _sbx()
    argv = sbx._argv(sbx.policy, "himmy-csbx-probe")
    assert "-v" not in argv
    assert "--volume" not in argv
    # The writable workspace is a tmpfs, not a host path.
    assert any("/sandbox:rw" in a for a in argv)


# ---------------------------------------------------------- (1) network: deny
def test_network_egress_is_denied_by_default() -> None:
    code = (
        "import socket\n"
        "socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "print('CONNECTED')\n"
    )
    result = run_async(_sbx().run_code(code))
    assert result.ok is False
    assert "CONNECTED" not in result.stdout  # the connection must have failed


def test_network_none_uses_no_interface() -> None:
    """Even loopback DNS/connect is impossible with --network none."""
    sbx = _sbx()
    argv = sbx._argv(sbx.policy, "himmy-csbx-probe")
    assert argv[argv.index("--network") + 1] == "none"


# ------------------------------------------------------- (3) resource limits
def test_memory_limit_is_enforced() -> None:
    """Allocating past the memory cap is OOM-killed, not allowed to succeed."""
    result = run_async(
        _sbx(memory_mb=64).run_code(
            "x = bytearray(300 * 1024 * 1024); print('ALLOCATED', len(x))"
        )
    )
    assert result.ok is False
    assert "ALLOCATED" not in result.stdout


def test_pids_limit_blocks_a_fork_bomb() -> None:
    code = (
        "import os\n"
        "for i in range(1000):\n"
        "    try:\n"
        "        os.fork()\n"
        "    except OSError:\n"
        "        print('FORK_BLOCKED')\n"
        "        break\n"
    )
    result = run_async(_sbx().run_code(code))
    assert "FORK_BLOCKED" in result.stdout


def test_ulimit_caps_open_files() -> None:
    """The nofile ulimit is applied (RLIMIT_NOFILE matches the policy soft limit)."""
    policy = SandboxPolicy()
    policy.ulimits["nofile"] = "64:64"
    sbx = ContainerSandbox(image=_IMAGE, policy=policy, startup_grace=25.0)
    result = run_async(
        sbx.run_code(
            "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE))"
        )
    )
    assert result.ok is True
    assert "64" in result.stdout


def test_wall_clock_timeout_kills_the_container() -> None:
    result = run_async(_sbx(timeout_seconds=2.0).run_code("while True: pass"))
    assert result.timed_out is True
    assert result.ok is False


# ----------------------------------------------------------- (2) ephemeral
def test_container_is_removed_after_the_run() -> None:
    """--rm removes the container; nothing with the run's name survives."""
    sbx = _sbx()
    result = run_async(sbx.run_code("print('ephemeral')"))
    assert result.ok is True
    # No himmy-csbx-* container should be left behind by a clean run.
    listed = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=himmy-csbx-",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "himmy-csbx-" not in listed.stdout


def test_workspace_does_not_persist_across_runs() -> None:
    """A file written by one run is gone in the next (fresh tmpfs per container)."""
    sbx = _sbx()
    first = run_async(sbx.run_code("open('marker.txt', 'w').write('present')"))
    assert first.ok is True
    second = run_async(
        sbx.run_code(
            "import os; print('EXISTS' if os.path.exists('marker.txt') else 'GONE')"
        )
    )
    assert second.stdout.strip() == "GONE"


# ------------------------------------------------------------- (5) audited
def test_every_execution_is_audited_on_the_event_spine() -> None:
    sink = _RecordingSink()
    result = run_async(_sbx(sink=sink).run_code("print('audited')"))
    assert result.ok is True
    types = [e.event_type.value for e in sink.events]
    assert types == ["TOOL_CALLED", "TOOL_COMPLETED"]
    start = sink.events[0]
    assert start.payload["backend"] == "container"
    assert start.payload["network"] == "none"
    assert start.payload["tool_name"] == "sandbox.run_code"


def test_audit_records_failure_outcome() -> None:
    sink = _RecordingSink()
    run_async(_sbx(sink=sink, timeout_seconds=2.0).run_code("while True: pass"))
    assert [e.event_type.value for e in sink.events] == ["TOOL_CALLED", "TOOL_FAILED"]


# --------------------------------------------------- egress allow-list (config)
def test_egress_proxy_mode_routes_through_proxy_never_bridge() -> None:
    """The mediated allow-list attaches to the isolated net + injects the proxy."""
    policy = SandboxPolicy(
        network=NetworkMode.EGRESS_PROXY,
        egress_network="himmy-egress",
        egress_proxy_url="http://egress-proxy:3128",
    )
    sbx = ContainerSandbox(image=_IMAGE, policy=policy)
    argv = sbx._argv(policy, "himmy-csbx-probe")
    assert argv[argv.index("--network") + 1] == "himmy-egress"  # never 'none'/'bridge'
    assert any("HTTPS_PROXY=http://egress-proxy:3128" in a for a in argv)
    assert any("HTTP_PROXY=http://egress-proxy:3128" in a for a in argv)


def test_egress_proxy_mode_requires_network_and_proxy() -> None:
    policy = SandboxPolicy(network=NetworkMode.EGRESS_PROXY)  # missing both
    sbx = ContainerSandbox(image=_IMAGE, policy=policy)
    with pytest.raises(HimmyError):
        sbx._argv(policy, "himmy-csbx-probe")
