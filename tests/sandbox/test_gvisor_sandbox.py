"""WS2.3 — GVisorSandbox CONTRACT tests (command/flag/policy assertions).

REQUIRES LINUX + KVM to *execute* (gVisor's ``runsc`` runtime cannot run on macOS) —
so these tests do NOT launch a runtime. They mock the runtime invocation and assert the
exact ``docker run --runtime=runsc`` command + isolation flags the backend produces, that
network is denied, that an overrunning run's ephemeral teardown is invoked, that the
resource limits are passed, and that every run is audited as ``gvisor``. The live
behaviour must be verified on a Linux host (see ``docs/enterprise/sandbox.md``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from himmy.core.events import RunEvent
from himmy.services.sandbox import ContainerSandbox, GVisorSandbox
from himmy.services.sandbox.gvisor_sandbox import RUNSC_RUNTIME
from himmy.services.sandbox.models import NetworkMode, SandboxLimits, SandboxPolicy
from tests.conftest import run_async


class _RecordingSink:
    """A minimal in-memory EventSink that captures emitted events for assertions."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        self.events.append(event)


class _FakeProc:
    """A stand-in subprocess that returns canned output without a real runtime."""

    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.pid = 4321
        self.received_stdin: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.received_stdin = input
        return self._stdout, self._stderr

    def kill(self) -> None:  # pragma: no cover - only on the timeout path
        pass

    async def wait(self) -> int:  # pragma: no cover - only on the timeout path
        return self.returncode


def _patch_exec(monkeypatch, proc: _FakeProc) -> list[list[str]]:
    """Capture every ``create_subprocess_exec`` argv; return canned ``proc`` for runs."""
    calls: list[list[str]] = []

    async def _fake_exec(*argv: str, **_kw: Any) -> _FakeProc:
        calls.append(list(argv))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return calls


# -------------------------------------------------- backend identity / runtime pinning
def test_gvisor_is_a_container_backend_under_runsc() -> None:
    """gVisor reuses the hardened container machinery, only swapping the runtime."""
    sbx = GVisorSandbox()
    assert isinstance(sbx, ContainerSandbox)
    assert sbx.runtime == RUNSC_RUNTIME
    assert sbx._backend_label() == "gvisor"


def test_argv_pins_the_runsc_runtime() -> None:
    """The produced command is ``docker run … --runtime=runsc``."""
    sbx = GVisorSandbox()
    argv = sbx._argv(sbx.policy, "himmy-csbx-probe")
    assert argv[0] == "docker"
    assert argv[1] == "run"
    assert argv[argv.index("--runtime") + 1] == "runsc"


def test_custom_runtime_name_is_honored() -> None:
    """An operator who registered runsc under a different name can point at it."""
    sbx = GVisorSandbox(runtime="runsc-ptrace")
    argv = sbx._argv(sbx.policy, "himmy-csbx-probe")
    assert argv[argv.index("--runtime") + 1] == "runsc-ptrace"


# ----------------------------------------- the five guarantees encoded in the command
def test_argv_carries_all_five_guarantees() -> None:
    policy = SandboxPolicy(limits=SandboxLimits(memory_mb=128, file_size_mb=8))
    sbx = GVisorSandbox(policy=policy)
    argv = sbx._argv(policy, "himmy-csbx-probe")
    # (1) network deny
    assert argv[argv.index("--network") + 1] == "none"
    # (2) ephemeral: --rm + tmpfs workspace, NO host bind mount
    assert "--rm" in argv
    assert any("/sandbox:rw" in a for a in argv)
    assert "-v" not in argv and "--volume" not in argv
    # (3) resource limits
    assert "--memory" in argv and "128m" in argv
    assert "--cpus" in argv
    assert "--pids-limit" in argv
    assert "--ulimit" in argv
    # (4) no host access
    assert "--read-only" in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in argv
    assert argv[argv.index("--user") + 1] == "65534:65534"
    # (5) under the stronger runtime
    assert argv[argv.index("--runtime") + 1] == "runsc"


def test_egress_proxy_mode_routes_through_proxy_never_bridge() -> None:
    """The mediated allow-list is inherited unchanged under gVisor."""
    policy = SandboxPolicy(
        network=NetworkMode.EGRESS_PROXY,
        egress_network="himmy-egress",
        egress_proxy_url="http://egress-proxy:3128",
    )
    sbx = GVisorSandbox(policy=policy)
    argv = sbx._argv(policy, "himmy-csbx-probe")
    assert argv[argv.index("--network") + 1] == "himmy-egress"  # never none/bridge
    assert any("HTTPS_PROXY=http://egress-proxy:3128" in a for a in argv)


# ------------------------------------------------- run: command, stdin, ephemeral, audit
def test_run_invokes_runsc_command_and_streams_bundle_on_stdin(monkeypatch) -> None:
    """A run shells out to ``docker run --runtime=runsc`` and feeds the bundle on stdin."""
    proc = _FakeProc(returncode=0, stdout=b"hello\n")
    calls = _patch_exec(monkeypatch, proc)
    sink = _RecordingSink()
    sbx = GVisorSandbox(event_sink=sink)

    result = run_async(sbx.run_code("print('hello')"))

    assert result.ok is True
    assert "hello" in result.stdout
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "docker" and argv[1] == "run"
    assert argv[argv.index("--runtime") + 1] == "runsc"
    # The code+files bundle is delivered out-of-band on stdin (length-prefixed), never argv.
    assert proc.received_stdin is not None
    assert proc.received_stdin.split(b"\n", 1)[0].isdigit()


def test_run_is_audited_as_gvisor(monkeypatch) -> None:
    proc = _FakeProc(returncode=0, stdout=b"ok\n")
    _patch_exec(monkeypatch, proc)
    sink = _RecordingSink()
    sbx = GVisorSandbox(event_sink=sink)

    run_async(sbx.run_code("print('ok')"))

    assert [e.event_type.value for e in sink.events] == ["TOOL_CALLED", "TOOL_COMPLETED"]
    start = sink.events[0].payload
    assert start["backend"] == "gvisor"  # NOT 'container' — the boundary is named
    assert start["runtime"] == "runsc"
    assert start["network"] == "none"
    assert start["tool_name"] == "sandbox.run_code"


def test_timeout_invokes_ephemeral_force_remove(monkeypatch) -> None:
    """An overrunning run is force-removed (ephemeral teardown), and reported timed out."""
    removed: list[list[str]] = []

    class _HangingProc(_FakeProc):
        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            raise TimeoutError

    async def _fake_exec(*argv: str, **_kw: Any) -> _FakeProc:
        if "rm" in argv:  # the force-remove teardown
            removed.append(list(argv))
            return _FakeProc(returncode=0)
        return _HangingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    sbx = GVisorSandbox(limits=SandboxLimits(timeout_seconds=0.01), startup_grace=0.0)
    result = run_async(sbx.run_code("while True: pass"))

    assert result.timed_out is True
    assert result.ok is False
    # The ephemeral teardown ran: `docker rm -f <name>`.
    assert removed and removed[0][:3] == ["docker", "rm", "-f"]


def test_resource_limits_are_passed_through(monkeypatch) -> None:
    proc = _FakeProc(returncode=0, stdout=b"")
    calls = _patch_exec(monkeypatch, proc)
    limits = SandboxLimits(memory_mb=64, file_size_mb=4, timeout_seconds=7)
    sbx = GVisorSandbox(limits=limits)

    run_async(sbx.run_code("pass"))

    argv = calls[0]
    assert "64m" in argv  # --memory
    assert any("size=4m" in a for a in argv)  # tmpfs sized to file_size_mb
    # The in-container hard wall-clock kill carries the policy timeout.
    bootstrap = argv[-1]
    assert "timeout" in bootstrap and "7" in bootstrap


# ----------------------------------------------------------------- capability detection
def test_available_false_without_runsc(monkeypatch) -> None:
    """On a host without runsc (e.g. macOS), gVisor reports itself unavailable."""

    def _which(name: str) -> str | None:
        return None if name == RUNSC_RUNTIME else "/usr/bin/docker"

    monkeypatch.setattr("himmy.services.sandbox.gvisor_sandbox.shutil.which", _which)
    assert GVisorSandbox.available() is False


def test_available_false_when_runtime_not_registered(monkeypatch) -> None:
    """runsc on PATH but NOT registered with the engine ⇒ still unavailable."""
    monkeypatch.setattr(
        "himmy.services.sandbox.gvisor_sandbox.shutil.which",
        lambda name: "/usr/bin/" + name,
    )

    class _Info:
        returncode = 0
        stdout = '{"runc":{"path":"runc"}}'  # runsc absent from the runtimes map

    monkeypatch.setattr(
        "himmy.services.sandbox.gvisor_sandbox.subprocess.run",
        lambda *a, **k: _Info(),
    )
    assert GVisorSandbox.available() is False


def test_available_true_when_runsc_present_and_registered(monkeypatch) -> None:
    monkeypatch.setattr(
        "himmy.services.sandbox.gvisor_sandbox.shutil.which",
        lambda name: "/usr/bin/" + name,
    )

    class _Info:
        returncode = 0
        stdout = '{"runc":{"path":"runc"},"runsc":{"path":"/usr/bin/runsc"}}'

    monkeypatch.setattr(
        "himmy.services.sandbox.gvisor_sandbox.subprocess.run",
        lambda *a, **k: _Info(),
    )
    assert GVisorSandbox.available() is True


def test_macos_host_reports_unavailable() -> None:
    """Concrete platform check: a developer macOS box cannot run gVisor (no runsc)."""
    import shutil as _shutil

    if _shutil.which(RUNSC_RUNTIME) is not None:  # pragma: no cover - Linux+gVisor host
        pytest.skip("runsc is installed on this host")
    assert GVisorSandbox.available() is False
