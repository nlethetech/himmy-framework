"""WS2.1/2.3 — sandbox backend selection (off / subprocess / container / gvisor /
firecracker) + fail-closed posture + hardened-runtime capability detection."""

from __future__ import annotations

import pytest

from himmy.core.errors import HimmyError
from himmy.services.sandbox import (
    CODE_EXEC_MODES,
    ContainerSandbox,
    DisabledSandbox,
    FirecrackerSandbox,
    GVisorSandbox,
    NetworkMode,
    SandboxPolicy,
    SubprocessSandbox,
    build_sandbox,
)
from himmy.services.sandbox.models import SandboxLimits
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
    # The stronger Linux-only boundaries are selectable by config (capability check off so
    # construction works for inspection on any host, incl. macOS CI).
    gvisor = build_sandbox("gvisor", capability_check=False)
    assert isinstance(gvisor, GVisorSandbox)
    assert gvisor.runtime == "runsc"
    firecracker = build_sandbox("firecracker", capability_check=False)
    assert isinstance(firecracker, FirecrackerSandbox)
    assert firecracker.runtime == "aws.firecracker"
    # GVISOR / FireCracker are case-insensitive like the other modes.
    assert isinstance(build_sandbox("GVISOR", capability_check=False), GVisorSandbox)
    # Unknown / empty falls back to the portable default (never accidentally off).
    assert isinstance(build_sandbox(""), SubprocessSandbox)
    assert isinstance(build_sandbox("bogus"), SubprocessSandbox)


def test_code_exec_modes_lists_all_backends() -> None:
    assert set(CODE_EXEC_MODES) == {
        "off",
        "subprocess",
        "container",
        "gvisor",
        "firecracker",
    }


# --------------------------------------------------------- fail-closed posture
def test_server_posture_refuses_subprocess_backend() -> None:
    """A server context must never run untrusted code on the unsandboxed backend."""
    with pytest.raises(HimmyError) as exc:
        build_sandbox("subprocess", server_context=True)
    assert "subprocess" in str(exc.value)
    # The fallback (empty/unknown → subprocess) is refused too, not silently downgraded.
    with pytest.raises(HimmyError):
        build_sandbox("", server_context=True)
    with pytest.raises(HimmyError):
        build_sandbox("bogus", server_context=True)


def test_server_posture_allows_hardened_backends() -> None:
    assert isinstance(build_sandbox("container", server_context=True), ContainerSandbox)
    assert isinstance(build_sandbox("off", server_context=True), DisabledSandbox)
    # gVisor and Firecracker are hardened too — allowed in a server posture (capability
    # check forced off so this passes on a host without the Linux-only runtimes).
    assert isinstance(
        build_sandbox("gvisor", server_context=True, capability_check=False),
        GVisorSandbox,
    )
    assert isinstance(
        build_sandbox("firecracker", server_context=True, capability_check=False),
        FirecrackerSandbox,
    )


def test_subprocess_refusal_message_names_all_hardened_backends() -> None:
    """The fail-closed error points operators at every hardened option."""
    with pytest.raises(HimmyError) as exc:
        build_sandbox("subprocess", server_context=True)
    msg = str(exc.value)
    assert "container" in msg and "gvisor" in msg and "firecracker" in msg


# ----------------------------------------------- hardened-runtime capability detection
def test_capability_check_refuses_unavailable_gvisor(monkeypatch) -> None:
    """With the probe on, an unavailable runsc runtime fails to start (clear error)."""
    monkeypatch.setattr(GVisorSandbox, "available", classmethod(lambda cls, e="docker": False))
    with pytest.raises(HimmyError) as exc:
        build_sandbox("gvisor", capability_check=True)
    msg = str(exc.value)
    assert "gvisor" in msg and ("runsc" in msg or "Linux" in msg)


def test_capability_check_refuses_unavailable_firecracker(monkeypatch) -> None:
    monkeypatch.setattr(
        FirecrackerSandbox, "available", classmethod(lambda cls, d="nerdctl": False)
    )
    with pytest.raises(HimmyError) as exc:
        build_sandbox("firecracker", capability_check=True)
    msg = str(exc.value)
    assert "firecracker" in msg and ("KVM" in msg or "Linux" in msg)


def test_capability_check_passes_when_runtime_available(monkeypatch) -> None:
    monkeypatch.setattr(GVisorSandbox, "available", classmethod(lambda cls, e="docker": True))
    assert isinstance(build_sandbox("gvisor", capability_check=True), GVisorSandbox)


def test_capability_probe_auto_on_with_verify_env(monkeypatch) -> None:
    """HIMMY_SANDBOX_VERIFY_RUNTIME=1 turns the startup probe on without an explicit flag."""
    monkeypatch.setenv("HIMMY_SANDBOX_VERIFY_RUNTIME", "1")
    monkeypatch.setattr(
        FirecrackerSandbox, "available", classmethod(lambda cls, d="nerdctl": False)
    )
    with pytest.raises(HimmyError):
        build_sandbox("firecracker")


def test_capability_probe_off_by_default(monkeypatch) -> None:
    """Without the verify env, construction does NOT probe (dev/test default unchanged)."""
    monkeypatch.delenv("HIMMY_SANDBOX_VERIFY_RUNTIME", raising=False)

    def _boom(cls, *a, **k):  # pragma: no cover - must never be called
        raise AssertionError("availability must not be probed by default")

    monkeypatch.setattr(GVisorSandbox, "available", classmethod(_boom))
    assert isinstance(build_sandbox("gvisor"), GVisorSandbox)


def test_require_sandbox_env_refuses_subprocess(monkeypatch) -> None:
    """HIMMY_REQUIRE_SANDBOX=1 opts a non-server caller into the hardened posture."""
    monkeypatch.setenv("HIMMY_REQUIRE_SANDBOX", "1")
    with pytest.raises(HimmyError):
        build_sandbox("subprocess")
    assert isinstance(build_sandbox("container"), ContainerSandbox)


def test_dev_default_subprocess_unchanged(monkeypatch) -> None:
    """The dev/CLI default (no server context, no env) keeps the subprocess backend."""
    monkeypatch.delenv("HIMMY_REQUIRE_SANDBOX", raising=False)
    assert isinstance(build_sandbox("subprocess"), SubprocessSandbox)


# ------------------------------------------------------------- sandbox policy
def test_sandbox_policy_locked_down_defaults() -> None:
    """The default policy is the locked-down enterprise posture."""
    policy = SandboxPolicy()
    assert policy.network is NetworkMode.NONE
    assert policy.auto_remove is True
    assert policy.read_only_rootfs is True
    assert policy.workspace_tmpfs is True
    assert policy.drop_all_capabilities is True
    assert policy.no_new_privileges is True
    assert policy.user == "65534:65534"
    assert policy.audit is True
    assert policy.cpus == 1.0
    assert policy.pids_limit == 128
    assert "nofile" in policy.ulimits


def test_policy_argv_carries_all_five_guarantees() -> None:
    """The hardened argv encodes network-deny, ephemeral, limits, no-host, (audit)."""
    policy = SandboxPolicy(limits=SandboxLimits(memory_mb=128, file_size_mb=8))
    sbx = ContainerSandbox(policy=policy)
    argv = sbx._argv(policy, "himmy-csbx-probe")
    # (1) network deny
    assert argv[argv.index("--network") + 1] == "none"
    # (2) ephemeral: --rm + tmpfs workspace, no host bind mount
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


# --------------------------------------- out-of-band bundle delivery (no argv cap)
def test_large_code_snippet_is_not_passed_on_argv() -> None:
    """A >70KB snippet must not ride the argv (kernel MAX_ARG_STRLEN ~128KB/arg).

    Regression: the old design base64-embedded the bundle into a single ``sh -c``
    string, so a large code snippet (or input file) failed with an opaque
    ``exec /usr/bin/sh: argument list too long`` (ok=False, exit_code=255). The bundle is
    now streamed on stdin instead, so the argv stays tiny and the only argv payload is the
    fixed bootstrap.
    """
    sbx = ContainerSandbox(policy=SandboxPolicy())
    argv = sbx._argv(sbx.policy, "himmy-csbx-probe")
    # No argv element should be anywhere near the per-arg ceiling: the bundle isn't here.
    assert all(len(a) < 4096 for a in argv)
    big_code = "x = '" + ("A" * 120_000) + "'\nprint(len(x))\n"
    bundle = sbx._build_bundle(big_code, None, sbx.policy.limits)
    framed = sbx._frame_input(bundle, "user-stdin")
    # The big payload lives in the framed stdin (length-prefixed), never the argv.
    assert len(bundle) > 120_000
    assert framed.endswith(b"user-stdin")
    assert framed.split(b"\n", 1)[0] == str(len(bundle)).encode("ascii")


def test_oversized_bundle_raises_clear_himmy_error() -> None:
    """Code+files past the file_size_mb workspace envelope fail fast, naming the cap."""
    sbx = ContainerSandbox(policy=SandboxPolicy())
    limits = SandboxLimits(file_size_mb=1)
    oversize = "Z" * (2 * 1024 * 1024)  # 2 MB of code vs a 1 MB workspace
    with pytest.raises(HimmyError) as exc:
        sbx._build_bundle(oversize, None, limits)
    msg = str(exc.value)
    assert "file_size_mb" in msg
    assert "1 MB" in msg


def test_bundle_within_envelope_is_accepted() -> None:
    """A bundle comfortably under file_size_mb encodes without raising."""
    sbx = ContainerSandbox(policy=SandboxPolicy())
    limits = SandboxLimits(file_size_mb=16)
    bundle = sbx._build_bundle("print('ok')", {"data.txt": "x" * 1000}, limits)
    assert isinstance(bundle, bytes) and len(bundle) > 0


def test_toolkit_config_builds_policy_from_env(monkeypatch) -> None:
    monkeypatch.setenv("HIMMY_SANDBOX_NETWORK", "egress_proxy")
    monkeypatch.setenv("HIMMY_SANDBOX_EGRESS_NETWORK", "himmy-egress")
    monkeypatch.setenv("HIMMY_SANDBOX_EGRESS_PROXY", "http://proxy:3128")
    config = ToolkitConfig.from_env()
    policy = config.build_sandbox_policy()
    assert policy.network is NetworkMode.EGRESS_PROXY
    assert policy.egress_network == "himmy-egress"
    assert policy.egress_proxy_url == "http://proxy:3128"


def test_toolkit_config_default_policy_is_deny() -> None:
    policy = ToolkitConfig().build_sandbox_policy()
    assert policy.network is NetworkMode.NONE


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


def test_code_pack_in_server_context_refuses_subprocess() -> None:
    """A server-context code pack on the default subprocess backend fails closed."""
    registry = ToolRegistry()
    config = ToolkitConfig(code_exec="subprocess", server_context=True)
    with pytest.raises(HimmyError) as exc:
        register_packs(registry, ["code"], config)
    assert "subprocess" in str(exc.value)


def test_code_pack_in_server_context_allows_container() -> None:
    """A server-context code pack on the hardened backend registers normally."""
    registry = ToolRegistry()
    config = ToolkitConfig(code_exec="container", server_context=True)
    register_packs(registry, ["code"], config)
    assert registry.get("run_python") is not None
