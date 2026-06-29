"""Sandbox backend selection: subprocess / container / gVisor / Firecracker / off.

:func:`build_sandbox` picks the execution backend from a mode string so the ``code``
toolkit pack and any served deployment can choose their isolation posture by config
(``HIMMY_CODE_EXEC``). ``off`` returns a :class:`DisabledSandbox` that refuses to run.

The boundaries form a ladder (weakest → strongest): ``subprocess`` (resource/fault only,
not a security boundary) → ``container`` (shared host kernel, OS-level isolation) →
``gvisor`` (a user-space kernel intercepts syscalls; tiny host-kernel surface) →
``firecracker`` (a KVM-isolated microVM with its own guest kernel per run). All hardened
backends enforce the same five-guarantee :class:`SandboxPolicy`.

The :class:`SubprocessSandbox` is NOT a security boundary (it isolates resources and
faults, not a determined adversary). To make that impossible to forget in production,
the **server posture refuses it**: when ``server_context=True`` (or
``HIMMY_REQUIRE_SANDBOX=1``), ``subprocess`` is rejected — a server must run untrusted
tool/code execution under one of the hardened backends (``container`` / ``gvisor`` /
``firecracker``) or disable it with ``off``. This ties into the same fail-closed posture
as the API auth/bind checks: a default dev or test run is unchanged, but a served
deployment cannot silently run untrusted code unsandboxed.

gVisor and Firecracker REQUIRE a Linux host (and KVM, for Firecracker) — they are not
executable on macOS. :func:`build_sandbox` therefore **capability-detects** the chosen
backend at startup and raises a clear :class:`HimmyError` if the runtime is unavailable
(rather than failing opaquely on the first run), unless the caller opts out of the probe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from himmy.config.flags import env_truthy
from himmy.core.errors import HimmyError
from himmy.services.sandbox.models import SandboxResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.core.events import EventSink
    from himmy.services.sandbox.base import Sandbox
    from himmy.services.sandbox.models import SandboxLimits, SandboxPolicy

#: Recognized backend modes.
CODE_EXEC_MODES = ("off", "subprocess", "container", "gvisor", "firecracker")

#: Modes that constitute a real OS-level security boundary (server-posture safe).
_HARDENED_MODES = ("off", "container", "gvisor", "firecracker")


class DisabledSandbox:
    """A sandbox that refuses every run (code execution disabled by policy)."""

    async def run_code(
        self,
        code: str,
        *,
        stdin: str | None = None,
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Return a structured refusal instead of executing anything."""
        return SandboxResult(
            ok=False,
            exit_code=None,
            stdout="",
            stderr=(
                "code execution is disabled by policy "
                "(set HIMMY_CODE_EXEC=subprocess|container to enable)"
            ),
        )


def _require_hardened_backend(*, server_context: bool) -> bool:
    """Whether the fail-closed posture must reject the unsandboxed subprocess backend.

    True in a server context, or when the operator explicitly demands it via
    ``HIMMY_REQUIRE_SANDBOX=1`` (so a CLI driving an untrusted multi-tenant workload can
    opt in). False on the default dev/test path, leaving existing behaviour unchanged.
    """
    if server_context:
        return True
    # red-team reattack-r7: route through the canonical truthy reader so this opt-in shares
    # the posture vocabulary (``HIMMY_REQUIRE_SANDBOX=y`` is now honored, where the prior
    # ad-hoc ``("1","true","yes","on")`` tuple treated it as off and failed OPEN vs operator
    # intent on the CLI path). The server surface already returns True above unconditionally.
    return env_truthy("HIMMY_REQUIRE_SANDBOX")


def _verify_runtime_requested() -> bool:
    """Whether to probe that the chosen hardened runtime is actually present at startup.

    Off by default (so a backend can be constructed for inspection on any host, and the
    dev/test path is unchanged); turned on with ``HIMMY_SANDBOX_VERIFY_RUNTIME=1``, the
    recommended production setting so a misconfigured deployment (e.g. ``gvisor`` without
    ``runsc``, or ``firecracker`` without KVM) fails to start instead of on first run.
    """
    # red-team reattack-r7: route through the canonical truthy reader so this shares the
    # posture vocabulary (``=y`` is now honored, where the prior ad-hoc tuple ignored it).
    return env_truthy("HIMMY_SANDBOX_VERIFY_RUNTIME")


def build_sandbox(
    mode: str,
    *,
    limits: SandboxLimits | None = None,
    policy: SandboxPolicy | None = None,
    image: str = "python:3.12-slim",
    engine: str = "docker",
    server_context: bool = False,
    event_sink: EventSink | None = None,
    capability_check: bool | None = None,
) -> Sandbox:
    """Build the configured sandbox backend.

    Backends by mode:

    * ``off`` → :class:`DisabledSandbox` (refuses every run).
    * ``container`` →
      :class:`~himmy.services.sandbox.container_sandbox.ContainerSandbox` (hardened,
      shared host kernel).
    * ``gvisor`` → :class:`~himmy.services.sandbox.gvisor_sandbox.GVisorSandbox`
      (the same hardened container under the ``runsc`` user-space kernel). **Linux only.**
    * ``firecracker`` →
      :class:`~himmy.services.sandbox.firecracker_sandbox.FirecrackerSandbox`
      (a KVM-isolated microVM per run). **Linux + KVM only.**
    * anything else → the portable
      :class:`~himmy.services.sandbox.subprocess_sandbox.SubprocessSandbox` (default).

    ``server_context`` (set by every served entrypoint) flips the fail-closed posture:
    the unsandboxed ``subprocess`` backend is then **refused** with a :class:`HimmyError`
    — a server must run untrusted code under one of the hardened backends (``container`` /
    ``gvisor`` / ``firecracker``) or disable it with ``off``. The dev/CLI default
    (``server_context=False``) is unchanged.

    ``capability_check`` verifies the chosen hardened runtime is actually available on
    this host before returning, raising a clear :class:`HimmyError` if not — so a
    deployment that asks for ``gvisor`` on macOS, or ``firecracker`` without ``/dev/kvm``,
    fails to **start** rather than failing opaquely on the first run. It defaults to
    ``None`` = *auto*: the probe runs only when the operator opts in with
    ``HIMMY_SANDBOX_VERIFY_RUNTIME=1`` (the recommended production setting), so the
    dev/CLI/test default that merely constructs a backend for inspection is unchanged.
    Pass ``True``/``False`` to force it on or off regardless of the env.
    """
    normalized = (mode or "subprocess").strip().lower()
    if normalized == "off":
        return DisabledSandbox()
    if capability_check is None:
        capability_check = _verify_runtime_requested()
    if normalized == "container":
        from himmy.services.sandbox.container_sandbox import ContainerSandbox

        if capability_check and not ContainerSandbox.available(engine):
            raise HimmyError(_unavailable_message("container", engine))
        return ContainerSandbox(
            image=image,
            engine=engine,
            limits=limits,
            policy=policy,
            event_sink=event_sink,
        )
    if normalized == "gvisor":
        from himmy.services.sandbox.gvisor_sandbox import GVisorSandbox

        if capability_check and not GVisorSandbox.available(engine):
            raise HimmyError(_unavailable_message("gvisor", engine))
        return GVisorSandbox(
            image=image,
            engine=engine,
            limits=limits,
            policy=policy,
            event_sink=event_sink,
        )
    if normalized == "firecracker":
        from himmy.services.sandbox.firecracker_sandbox import FirecrackerSandbox

        if capability_check and not FirecrackerSandbox.available():
            raise HimmyError(_unavailable_message("firecracker", engine))
        return FirecrackerSandbox(
            image=image,
            limits=limits,
            policy=policy,
            event_sink=event_sink,
        )
    # Anything else resolves to the portable subprocess backend — but that is NOT a
    # security boundary, so a server posture refuses it (fail closed).
    if _require_hardened_backend(server_context=server_context):
        raise HimmyError(
            "refusing to run untrusted code on the unsandboxed 'subprocess' backend in "
            "a server/production posture: it isolates resources and faults but is NOT a "
            "security boundary against hostile code. Set HIMMY_CODE_EXEC to a hardened "
            "backend (container | gvisor | firecracker) or HIMMY_CODE_EXEC=off (disable "
            "code execution)."
        )
    from himmy.services.sandbox.subprocess_sandbox import SubprocessSandbox

    return SubprocessSandbox(limits=limits)


def _unavailable_message(mode: str, engine: str) -> str:
    """A clear, actionable error for a hardened backend whose runtime is missing.

    Names the missing runtime, the Linux(+KVM) requirement for gVisor/Firecracker, and
    the fallbacks — so a misconfigured deployment fails to start with a fixable message
    instead of an opaque runtime error on the first execution.
    """
    detail = {
        "container": (
            f"the '{engine}' container engine is not on PATH. Install Docker/Podman, or "
            "set HIMMY_CODE_EXEC=off."
        ),
        "gvisor": (
            f"gVisor is not usable: the 'runsc' runtime must be installed AND registered "
            f"with the '{engine}' engine (Linux only — runsc cannot run on macOS). "
            "Install gVisor, register the runsc runtime, then retry; or fall back to "
            "HIMMY_CODE_EXEC=container."
        ),
        "firecracker": (
            "Firecracker microVMs require a Linux host with /dev/kvm, the 'firecracker' "
            "binary, and firecracker-containerd's driver (nerdctl) on PATH — none of "
            "which exist on macOS. Verify on a Linux+KVM host, or fall back to "
            "HIMMY_CODE_EXEC=gvisor or HIMMY_CODE_EXEC=container."
        ),
    }[mode]
    return f"sandbox backend '{mode}' is configured but unavailable on this host: {detail}"


__all__ = ["build_sandbox", "DisabledSandbox", "CODE_EXEC_MODES"]
