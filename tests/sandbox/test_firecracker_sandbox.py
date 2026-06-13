"""WS2.3 — FirecrackerSandbox CONTRACT tests (command/flag/policy assertions).

REQUIRES LINUX + KVM to *execute* (Firecracker is a KVM VMM; it cannot run on macOS) —
so these tests do NOT launch a microVM. They mock the runtime invocation and assert the
exact firecracker-containerd driver command + isolation flags the backend produces: the
``aws.firecracker`` runtime is pinned, the firecracker-containerd socket/namespace are
targeted, network is denied (no tap interface) or mediated through the proxy, an
overrunning run's ephemeral teardown is invoked, the vCPU/mem/pids limits are passed, and
every run is audited as ``firecracker``. Live behaviour must be verified on a Linux+KVM
host (see ``docs/enterprise/sandbox.md``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from himmy.core.events import RunEvent
from himmy.services.sandbox import FirecrackerSandbox
from himmy.services.sandbox.firecracker_sandbox import (
    DEFAULT_DRIVER,
    DEFAULT_FC_ADDRESS,
    DEFAULT_FC_NAMESPACE,
    FIRECRACKER_RUNTIME,
)
from himmy.services.sandbox.models import NetworkMode, SandboxLimits, SandboxPolicy
from tests.conftest import run_async


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        self.events.append(event)


class _FakeProc:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.pid = 9876
        self.received_stdin: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.received_stdin = input
        return self._stdout, self._stderr

    def kill(self) -> None:  # pragma: no cover - only on the timeout path
        pass

    async def wait(self) -> int:  # pragma: no cover - only on the timeout path
        return self.returncode


def _patch_exec(monkeypatch, proc: _FakeProc) -> list[list[str]]:
    calls: list[list[str]] = []

    async def _fake_exec(*argv: str, **_kw: Any) -> _FakeProc:
        calls.append(list(argv))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return calls


# ------------------------------------------------------ driver / runtime / socket plumbing
def test_argv_drives_firecracker_containerd() -> None:
    """The command targets the firecracker-containerd socket+namespace under aws.firecracker."""
    sbx = FirecrackerSandbox()
    argv = sbx._argv(sbx.policy, "himmy-fcvm-probe")
    assert argv[0] == DEFAULT_DRIVER  # nerdctl (containerd-compatible driver)
    assert argv[argv.index("--address") + 1] == DEFAULT_FC_ADDRESS
    assert argv[argv.index("--namespace") + 1] == DEFAULT_FC_NAMESPACE
    assert "run" in argv
    assert argv[argv.index("--runtime") + 1] == FIRECRACKER_RUNTIME
    assert sbx.runtime == FIRECRACKER_RUNTIME


def test_custom_socket_and_namespace_are_honored() -> None:
    sbx = FirecrackerSandbox(
        address="/run/custom/fc.sock", namespace="tenant-a", driver="ctr-fc"
    )
    argv = sbx._argv(sbx.policy, "himmy-fcvm-probe")
    assert argv[0] == "ctr-fc"
    assert argv[argv.index("--address") + 1] == "/run/custom/fc.sock"
    assert argv[argv.index("--namespace") + 1] == "tenant-a"


# ------------------------------------------ the same five guarantees, now inside a microVM
def test_argv_carries_all_five_guarantees() -> None:
    policy = SandboxPolicy(limits=SandboxLimits(memory_mb=256, file_size_mb=8))
    sbx = FirecrackerSandbox(policy=policy)
    argv = sbx._argv(policy, "himmy-fcvm-probe")
    # (1) network deny — no tap interface in the microVM
    assert argv[argv.index("--network") + 1] == "none"
    # (2) ephemeral: --rm + tmpfs workspace, NO host bind mount (no host fs in the VM)
    assert "--rm" in argv
    assert any("/sandbox:rw" in a for a in argv)
    assert "-v" not in argv and "--volume" not in argv
    # (3) resource limits sizing the microVM's vCPU/RAM + pids cap
    assert "--memory" in argv and "256m" in argv
    assert "--cpus" in argv
    assert "--pids-limit" in argv
    assert "--ulimit" in argv
    # (4) no host access
    assert "--read-only" in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in argv
    assert argv[argv.index("--user") + 1] == "65534:65534"
    # (5) one microVM per run
    assert argv[argv.index("--runtime") + 1] == FIRECRACKER_RUNTIME


def test_network_deny_by_default_no_bridge() -> None:
    sbx = FirecrackerSandbox()
    argv = sbx._argv(sbx.policy, "himmy-fcvm-probe")
    assert argv[argv.index("--network") + 1] == "none"
    assert "bridge" not in argv


def test_egress_proxy_mode_routes_through_proxy_never_bridge() -> None:
    policy = SandboxPolicy(
        network=NetworkMode.EGRESS_PROXY,
        egress_network="himmy-egress",
        egress_proxy_url="http://egress-proxy:3128",
    )
    sbx = FirecrackerSandbox(policy=policy)
    argv = sbx._argv(policy, "himmy-fcvm-probe")
    assert argv[argv.index("--network") + 1] == "himmy-egress"  # never none/bridge
    assert any("HTTPS_PROXY=http://egress-proxy:3128" in a for a in argv)
    assert any("HTTP_PROXY=http://egress-proxy:3128" in a for a in argv)


def test_egress_proxy_mode_requires_network_and_proxy() -> None:
    from himmy.core.errors import HimmyError

    policy = SandboxPolicy(network=NetworkMode.EGRESS_PROXY)  # missing both fields
    sbx = FirecrackerSandbox(policy=policy)
    with pytest.raises(HimmyError):
        sbx._argv(policy, "himmy-fcvm-probe")


# ------------------------------------------------- run: command, stdin, ephemeral, audit
def test_run_invokes_the_microvm_command_and_streams_bundle_on_stdin(monkeypatch) -> None:
    proc = _FakeProc(returncode=0, stdout=b"from-vm\n")
    calls = _patch_exec(monkeypatch, proc)
    sbx = FirecrackerSandbox()

    result = run_async(sbx.run_code("print('from-vm')"))

    assert result.ok is True
    assert "from-vm" in result.stdout
    argv = calls[0]
    assert argv[0] == DEFAULT_DRIVER
    assert argv[argv.index("--runtime") + 1] == FIRECRACKER_RUNTIME
    # No host bind mount; the bundle rides stdin (length-prefixed), not the argv.
    assert proc.received_stdin is not None
    assert proc.received_stdin.split(b"\n", 1)[0].isdigit()


def test_input_files_ride_the_stdin_bundle_not_a_host_mount(monkeypatch) -> None:
    """Input files are embedded in the stdin bundle (no host path shared into the VM)."""
    proc = _FakeProc(returncode=0, stdout=b"")
    calls = _patch_exec(monkeypatch, proc)
    sbx = FirecrackerSandbox()

    run_async(sbx.run_code("print(open('data.txt').read())", files={"data.txt": "secret"}))

    argv = calls[0]
    assert "-v" not in argv and "--volume" not in argv
    assert proc.received_stdin is not None and len(proc.received_stdin) > 0


def test_run_is_audited_as_firecracker(monkeypatch) -> None:
    proc = _FakeProc(returncode=0, stdout=b"ok\n")
    _patch_exec(monkeypatch, proc)
    sink = _RecordingSink()
    sbx = FirecrackerSandbox(event_sink=sink)

    run_async(sbx.run_code("print('ok')"))

    assert [e.event_type.value for e in sink.events] == ["TOOL_CALLED", "TOOL_COMPLETED"]
    start = sink.events[0].payload
    assert start["backend"] == "firecracker"
    assert start["runtime"] == FIRECRACKER_RUNTIME
    assert start["driver"] == DEFAULT_DRIVER
    assert start["network"] == "none"
    assert start["tool_name"] == "sandbox.run_code"


def test_failure_is_audited_as_failed(monkeypatch) -> None:
    proc = _FakeProc(returncode=1, stderr=b"boom")
    _patch_exec(monkeypatch, proc)
    sink = _RecordingSink()
    sbx = FirecrackerSandbox(event_sink=sink)

    run_async(sbx.run_code("raise SystemExit(1)"))

    assert [e.event_type.value for e in sink.events] == ["TOOL_CALLED", "TOOL_FAILED"]


def test_timeout_invokes_ephemeral_force_remove(monkeypatch) -> None:
    """An overrunning microVM is force-removed (ephemeral teardown) and reported timed out."""
    removed: list[list[str]] = []

    class _HangingProc(_FakeProc):
        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            raise TimeoutError

    async def _fake_exec(*argv: str, **_kw: Any) -> _FakeProc:
        if "rm" in argv:
            removed.append(list(argv))
            return _FakeProc(returncode=0)
        return _HangingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    sbx = FirecrackerSandbox(
        limits=SandboxLimits(timeout_seconds=0.01), startup_grace=0.0
    )
    result = run_async(sbx.run_code("while True: pass"))

    assert result.timed_out is True
    assert result.ok is False
    # Ephemeral teardown ran against the firecracker-containerd socket: `rm -f <vm>`.
    assert removed
    rm = removed[0]
    assert rm[0] == DEFAULT_DRIVER
    assert rm[rm.index("--address") + 1] == DEFAULT_FC_ADDRESS
    assert "rm" in rm and "-f" in rm


def test_timeout_exit_code_is_surfaced_as_timed_out(monkeypatch) -> None:
    """An in-guest coreutils-timeout kill (124/137) is reported as a timeout."""
    proc = _FakeProc(returncode=137, stderr=b"")
    _patch_exec(monkeypatch, proc)
    sbx = FirecrackerSandbox()
    result = run_async(sbx.run_code("while True: pass"))
    assert result.timed_out is True
    assert result.ok is False


def test_resource_limits_size_the_microvm(monkeypatch) -> None:
    proc = _FakeProc(returncode=0)
    calls = _patch_exec(monkeypatch, proc)
    limits = SandboxLimits(memory_mb=128, file_size_mb=4, timeout_seconds=9)
    sbx = FirecrackerSandbox(limits=limits)

    run_async(sbx.run_code("pass"))

    argv = calls[0]
    assert "128m" in argv  # microVM RAM
    assert any("size=4m" in a for a in argv)  # tmpfs workspace = file_size_mb
    bootstrap = argv[-1]
    assert "timeout" in bootstrap and "9" in bootstrap  # in-guest hard wall-clock kill


def test_oversized_input_raises_clear_error(monkeypatch) -> None:
    from himmy.core.errors import HimmyError

    _patch_exec(monkeypatch, _FakeProc())
    sbx = FirecrackerSandbox(limits=SandboxLimits(file_size_mb=1))
    with pytest.raises(HimmyError) as exc:
        run_async(sbx.run_code("pass", files={"big.bin": "Z" * (2 * 1024 * 1024)}))
    assert "file_size_mb" in str(exc.value)


# ----------------------------------------------------------------- capability detection
def test_available_false_without_kvm(monkeypatch) -> None:
    """No /dev/kvm (e.g. macOS) ⇒ Firecracker reports itself unavailable."""
    monkeypatch.setattr(
        "himmy.services.sandbox.firecracker_sandbox.os.path.exists",
        lambda p: False,
    )
    assert FirecrackerSandbox.available() is False


def test_available_false_without_firecracker_binary(monkeypatch) -> None:
    monkeypatch.setattr(
        "himmy.services.sandbox.firecracker_sandbox.os.path.exists",
        lambda p: True,  # KVM present
    )

    def _which(name: str) -> str | None:
        return None if name == "firecracker" else "/usr/bin/" + name

    monkeypatch.setattr(
        "himmy.services.sandbox.firecracker_sandbox.shutil.which", _which
    )
    assert FirecrackerSandbox.available() is False


def test_available_true_on_linux_kvm_with_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "himmy.services.sandbox.firecracker_sandbox.os.path.exists",
        lambda p: True,
    )
    monkeypatch.setattr(
        "himmy.services.sandbox.firecracker_sandbox.shutil.which",
        lambda name: "/usr/bin/" + name,
    )
    assert FirecrackerSandbox.available() is True


def test_macos_host_reports_unavailable() -> None:
    """Concrete platform check: a developer macOS box cannot run Firecracker (no /dev/kvm)."""
    import os as _os

    if _os.path.exists("/dev/kvm"):  # pragma: no cover - Linux+KVM host
        pytest.skip("/dev/kvm exists on this host")
    assert FirecrackerSandbox.available() is False
