"""Sandbox kernel: a **Firecracker microVM** isolate for hostile code (WS2.3).

REQUIRES LINUX + KVM — Firecracker is a KVM-based VMM and is NOT executable on macOS
(or any host without ``/dev/kvm``). The command/flag/policy contract is unit-tested
here; the live behaviour MUST be verified on a Linux+KVM host (see
``docs/enterprise/sandbox.md``).

Where a standard container (runc) shares the **host kernel** and gVisor interposes a
*user-space* kernel, :class:`FirecrackerSandbox` gives each run its **own microVM with a
separate guest kernel**, hardware-isolated by KVM — the strongest practical boundary for
hostile, model-authored, multi-tenant code. A guest-kernel exploit is contained inside
the VM; the host kernel is never directly reachable.

It enforces the *identical* five-guarantee
:class:`~himmy.services.sandbox.models.SandboxPolicy` as the container/gVisor backends
(network deny/allowlist, vCPU/mem/pids limits + hard wall-clock kill, ephemeral teardown,
no host filesystem, audited) and reuses this package's proven bundle/bootstrap/stdin
machinery (no host bind mount: the code+files are streamed length-prefixed on the VM's
stdin and materialized onto an in-guest tmpfs).

Production driver — **firecracker-containerd**
----------------------------------------------
The supported way to run OCI images inside Firecracker microVMs is
`firecracker-containerd <https://github.com/firecracker-microvm/firecracker-containerd>`_,
which registers a containerd runtime (``aws.firecracker``). We drive it with ``nerdctl``
(the docker-compatible containerd CLI) pointed at the firecracker-containerd socket and
namespace, so the **same hardened OCI flag set** (``--network none``, ``--memory``,
``--cpus``, ``--pids-limit``, ``--read-only``, ``--cap-drop ALL``,
``--security-opt no-new-privileges``, ``--user``, a tmpfs workspace, ``--rm``) maps onto a
microVM. The jailer (``--jailer``-equivalent confinement, cgroups, seccomp) is applied by
firecracker-containerd's runtime; the per-run vCPU/mem the VM is sized with come straight
from :class:`~himmy.services.sandbox.models.SandboxLimits`.

This is implemented from scratch (not a thin ``ContainerSandbox`` subclass) because the
driver, capability detection, socket/namespace plumbing, and audit labelling differ — but
the bundle/bootstrap/watchdog/truncation helpers are reused verbatim from
:class:`ContainerSandbox` so the execution semantics are byte-for-byte identical.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.services.sandbox.container_sandbox import (
    _TIMEOUT_EXIT_CODES,
    _WORKDIR,
    DEFAULT_IMAGE,
    ContainerSandbox,
)
from himmy.services.sandbox.models import (
    NetworkMode,
    SandboxLimits,
    SandboxPolicy,
    SandboxResult,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.core.events import EventSink

#: The containerd runtime name firecracker-containerd registers (one microVM per run).
FIRECRACKER_RUNTIME = "aws.firecracker"
#: The docker-compatible containerd CLI used to drive firecracker-containerd.
DEFAULT_DRIVER = "nerdctl"
#: firecracker-containerd's default control socket (the dedicated containerd instance).
DEFAULT_FC_ADDRESS = "/run/firecracker-containerd/containerd.sock"
#: The containerd namespace firecracker-containerd workloads live in by convention.
DEFAULT_FC_NAMESPACE = "firecracker-containerd"
#: The Linux KVM device node Firecracker requires; absent ⇒ microVMs cannot be launched.
_KVM_DEVICE = "/dev/kvm"


class FirecrackerSandbox:
    """Run Python in a throwaway Firecracker microVM (one guest kernel per run).

    Same :class:`SandboxPolicy` contract as the container/gVisor backends; the boundary
    is a KVM-isolated microVM instead of a shared-kernel container. Linux + KVM +
    firecracker-containerd only.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        driver: str = DEFAULT_DRIVER,
        runtime: str = FIRECRACKER_RUNTIME,
        address: str = DEFAULT_FC_ADDRESS,
        namespace: str = DEFAULT_FC_NAMESPACE,
        limits: SandboxLimits | None = None,
        policy: SandboxPolicy | None = None,
        startup_grace: float = 45.0,
        event_sink: EventSink | None = None,
        cpus: float | None = None,
        pids_limit: int | None = None,
        user: str | None = None,
    ) -> None:
        """Configure the microVM driver, runtime, socket/namespace, and policy.

        ``policy`` is the full enterprise contract; ``limits`` overrides only its
        resource envelope when given (it also sizes the microVM's vCPU/memory). The
        legacy ``cpus`` / ``pids_limit`` / ``user`` overrides mirror
        :class:`ContainerSandbox`. ``startup_grace`` is larger than the container
        backend's because a microVM cold-boot (guest kernel + rootfs) is slower than
        starting a container; the wall-clock *kill* is still the policy timeout.
        """
        self._image = image
        self._driver = driver
        self._runtime = runtime or FIRECRACKER_RUNTIME
        self._address = address
        self._namespace = namespace
        base = policy or SandboxPolicy()
        if limits is not None:
            base = base.with_limits(limits)
        overrides: dict[str, object] = {}
        if cpus is not None:
            overrides["cpus"] = cpus
        if pids_limit is not None:
            overrides["pids_limit"] = pids_limit
        if user is not None:
            overrides["user"] = user
        if overrides:
            base = base.model_copy(update=overrides)
        self._policy = base
        self._startup_grace = startup_grace
        self._event_sink = event_sink
        # Reuse the container backend's proven bundle/bootstrap/watchdog/truncate helpers
        # so a microVM run is byte-for-byte identical to a container run at the snippet
        # level (no host bind mount; length-prefixed stdin; tmpfs workspace).
        self._oci = ContainerSandbox(
            image=image,
            engine=driver,
            policy=base,
            runtime=runtime,
            startup_grace=startup_grace,
        )

    @property
    def policy(self) -> SandboxPolicy:
        """The enterprise isolation policy this microVM sandbox enforces per run."""
        return self._policy

    @property
    def runtime(self) -> str:
        """The containerd runtime each run is launched under (``aws.firecracker``)."""
        return self._runtime

    @classmethod
    def available(cls, driver: str = DEFAULT_DRIVER) -> bool:
        """Whether Firecracker microVMs can be launched on this host.

        Requires (1) a Linux host with the KVM device node ``/dev/kvm`` present, (2) the
        ``firecracker`` VMM binary on PATH, and (3) the containerd driver CLI (nerdctl)
        on PATH. On macOS none of these hold, so this is ``False`` — which is exactly
        what the factory's capability gate relies on to refuse Firecracker off-Linux.
        """
        if not os.path.exists(_KVM_DEVICE):
            return False
        if shutil.which("firecracker") is None:
            return False
        return shutil.which(driver) is not None

    async def run_code(
        self,
        code: str,
        *,
        stdin: str | None = None,
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute ``code`` inside a throwaway Firecracker microVM and capture its result."""
        policy = self._policy
        limits = policy.limits
        name = f"himmy-fcvm-{uuid.uuid4().hex[:16]}"
        # Same out-of-band, host-path-free delivery as the container backend.
        bundle = self._oci._build_bundle(code, files, limits)
        in_bytes = self._oci._frame_input(bundle, stdin)
        argv = self._argv(policy, name)
        await self._audit_start(name, policy, limits)
        result = await self._run(argv, name, policy, limits, in_bytes)
        await self._audit_finish(name, result)
        return result

    # ------------------------------------------------------------------ internals
    def _argv(self, policy: SandboxPolicy, name: str) -> list[str]:
        """Build the hardened ``nerdctl run`` argv that boots one microVM for this run.

        The driver is pointed at firecracker-containerd (``--address`` + ``--namespace``)
        and the run is pinned to the ``aws.firecracker`` runtime, so each invocation gets
        its own KVM-isolated microVM with a separate guest kernel. Every OCI isolation
        flag from the policy is applied exactly as the container backend applies them, so
        the five guarantees hold identically inside the VM:

        * network deny (``--network none``) or mediated egress (never a general bridge),
        * ephemeral (``--rm`` + a tmpfs workspace, no host bind mount),
        * resource limits (``--memory``/``--cpus``/``--pids-limit``/``--ulimit``) sizing
          the microVM's vCPU/RAM, plus the in-guest coreutils wall-clock SIGKILL,
        * no host access (``--read-only`` rootfs, ``--cap-drop ALL``,
          ``--security-opt no-new-privileges``, non-root ``--user``),
        * audited (emitted around this call).
        """
        limits = policy.limits
        mem = f"{max(6, limits.memory_mb)}m"
        # firecracker-containerd lives on its own containerd socket + namespace; the
        # docker-compatible driver targets them with the global --address/--namespace.
        argv = [
            self._driver,
            "--address",
            self._address,
            "--namespace",
            self._namespace,
            "run",
            "-i",
            "--runtime",
            self._runtime,
            "--name",
            name,
        ]
        if policy.auto_remove:
            argv.append("--rm")
        if policy.read_only_rootfs:
            argv.append("--read-only")
        if policy.drop_all_capabilities:
            argv += ["--cap-drop", "ALL"]
        if policy.no_new_privileges:
            argv += ["--security-opt", "no-new-privileges"]
        argv += [
            "--pids-limit",
            str(policy.pids_limit),
            "--memory",
            mem,
            "--cpus",
            str(policy.cpus),
            "--user",
            policy.user,
        ]
        for ulimit, value in policy.ulimits.items():
            argv += ["--ulimit", f"{ulimit}={value}"]
        if policy.workspace_tmpfs:
            argv += [
                "--tmpfs",
                f"{_WORKDIR}:rw,nosuid,nodev,exec,mode=1777,"
                f"size={max(1, limits.file_size_mb)}m",
            ]
        argv += ["-w", _WORKDIR]
        argv += self._network_args(policy)
        for key in limits.allow_env:
            passthrough = os.environ.get(key)
            if passthrough is not None:
                argv += ["-e", f"{key}={passthrough}"]
        argv.append(self._image)
        # The fixed bootstrap (reads the bundle off stdin, then exec's the timeout-wrapped
        # snippet) is identical to the container backend's — reuse it verbatim.
        argv += ["python", "-c", self._oci._bootstrap(limits.timeout_seconds)]
        return argv

    @staticmethod
    def _network_args(policy: SandboxPolicy) -> list[str]:
        """Network flags for the microVM: default DENY, or mediated egress (never bridge).

        Reuses the container backend's exact policy so a microVM has no tap interface at
        all under ``NetworkMode.NONE``, and under ``EGRESS_PROXY`` is forced through the
        operator-provisioned proxy — identical semantics, identical config surface.
        """
        if policy.network is NetworkMode.NONE:
            return ["--network", "none"]
        return ContainerSandbox._network_args(policy)

    async def _run(
        self,
        argv: list[str],
        name: str,
        policy: SandboxPolicy,
        limits: SandboxLimits,
        in_bytes: bytes,
    ) -> SandboxResult:
        """Boot the microVM, enforce the watchdog, and normalize the result.

        Mirrors :meth:`ContainerSandbox._run`: the framed bundle is fed on stdin, an
        outer wall-clock guard (policy timeout + microVM startup grace) force-removes an
        overrunning VM, and timeout exit codes (124/137 from the in-guest ``timeout``) are
        surfaced as ``timed_out``.
        """
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:  # pragma: no cover - infra failure, not code failure
            raise HimmyError(
                f"sandbox failed to start the Firecracker microVM via {self._driver}: {exc}"
            ) from exc

        timed_out = False
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(in_bytes),
                timeout=limits.timeout_seconds + self._startup_grace,
            )
        except TimeoutError:
            timed_out = True
            await self._force_remove(name)
            with suppress(Exception):
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            out, err = b"", b"[sandbox] killed: wall-clock timeout exceeded"

        duration_ms = (time.monotonic() - started) * 1000.0
        exit_code = proc.returncode
        if exit_code in _TIMEOUT_EXIT_CODES:
            timed_out = True
        stdout, trunc_out = ContainerSandbox._truncate(out, limits.max_output_bytes)
        stderr, trunc_err = ContainerSandbox._truncate(err, limits.max_output_bytes)
        return SandboxResult(
            ok=(not timed_out) and exit_code == 0,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_ms=duration_ms,
            truncated=trunc_out or trunc_err,
            limits=limits,
        )

    async def _force_remove(self, name: str) -> None:
        """Best-effort force-remove a microVM that overran its watchdog (ephemeral)."""
        with suppress(Exception):
            rm = await asyncio.create_subprocess_exec(
                self._driver,
                "--address",
                self._address,
                "--namespace",
                self._namespace,
                "rm",
                "-f",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(rm.wait(), timeout=10.0)

    # ------------------------------------------------------------------ audit
    async def _audit_start(
        self, name: str, policy: SandboxPolicy, limits: SandboxLimits
    ) -> None:
        """Record that a microVM execution is starting (if auditing is on)."""
        if not policy.audit or self._event_sink is None:
            return
        await self._emit(
            RunEvent(
                event_type=EventType.TOOL_CALLED,
                payload={
                    "tool_name": "sandbox.run_code",
                    "backend": "firecracker",
                    "driver": self._driver,
                    "runtime": self._runtime,
                    "image": self._image,
                    "vm": name,
                    "network": policy.network.value,
                    "limits": limits.model_dump(),
                },
            )
        )

    async def _audit_finish(self, name: str, result: SandboxResult) -> None:
        """Record the outcome of a microVM execution (if auditing is on)."""
        if not self._policy.audit or self._event_sink is None:
            return
        await self._emit(
            RunEvent(
                event_type=(
                    EventType.TOOL_COMPLETED if result.ok else EventType.TOOL_FAILED
                ),
                latency_ms=result.duration_ms,
                payload={
                    "tool_name": "sandbox.run_code",
                    "backend": "firecracker",
                    "vm": name,
                    "ok": result.ok,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                },
            )
        )

    async def _emit(self, event: RunEvent) -> None:
        """Best-effort emit; a broken audit sink must never fail a run."""
        if self._event_sink is None:
            return
        try:
            await self._event_sink.append_event(event)
        except Exception:  # pragma: no cover - defensive
            pass


__all__ = [
    "FirecrackerSandbox",
    "FIRECRACKER_RUNTIME",
    "DEFAULT_DRIVER",
    "DEFAULT_FC_ADDRESS",
    "DEFAULT_FC_NAMESPACE",
]
