"""Sandbox kernel: a **gVisor** (runsc) isolate for hostile code (WS2.3).

REQUIRES LINUX + KVM (or ptrace) — gVisor's ``runsc`` runtime is Linux-only and is
NOT executable on macOS. The command/flag/policy contract is unit-tested here; the
live behaviour must be verified on a Linux host (see ``docs/enterprise/sandbox.md``).

:class:`GVisorSandbox` is the same hardened container as
:class:`~himmy.services.sandbox.container_sandbox.ContainerSandbox` — it reuses that
backend's whole argv machinery and enforces the identical five-guarantee
:class:`~himmy.services.sandbox.models.SandboxPolicy` — but it pins the OCI **runtime**
to gVisor's ``runsc`` (``docker run --runtime=runsc``). The difference is the *boundary*,
not the policy:

* A standard runc container shares the **host kernel** — a single kernel exploit from
  hostile, model-authored code is a host compromise.
* gVisor interposes a **user-space kernel** (the Sentry) between the workload and the
  host: the application's syscalls are intercepted and serviced in user space, so the
  workload reaches only a tiny, hardened slice of the real host kernel. The host kernel
  attack surface is reduced by roughly two orders of magnitude.

Use it for **untrusted / multi-tenant** code where a shared-kernel container is too weak
but a full microVM (Firecracker) is heavier than you need. It is a drop-in for the
container backend via the same :class:`~himmy.services.sandbox.base.Sandbox` protocol; the
only change is the configured runtime, so nothing else in the stack is affected.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from himmy.services.sandbox.container_sandbox import DEFAULT_IMAGE, ContainerSandbox

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.core.events import EventSink
    from himmy.services.sandbox.models import SandboxLimits, SandboxPolicy

#: The OCI runtime name gVisor registers with Docker/containerd.
RUNSC_RUNTIME = "runsc"


class GVisorSandbox(ContainerSandbox):
    """Run Python in a hardened container under the gVisor ``runsc`` runtime.

    Identical policy + argv to :class:`ContainerSandbox`, but the container is launched
    with ``--runtime=runsc`` so the workload runs against gVisor's user-space kernel
    instead of sharing the host kernel. Linux + ``runsc`` only.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        engine: str = "docker",
        runtime: str = RUNSC_RUNTIME,
        limits: SandboxLimits | None = None,
        policy: SandboxPolicy | None = None,
        startup_grace: float = 30.0,
        event_sink: EventSink | None = None,
        cpus: float | None = None,
        pids_limit: int | None = None,
        user: str | None = None,
    ) -> None:
        """Configure the gVisor backend (defaults to the ``runsc`` runtime).

        ``runtime`` is exposed only so an operator who registered gVisor under a
        non-default name (e.g. a second ``runsc-ptrace`` runtime) can point at it; it
        defaults to ``runsc`` and is always passed to ``docker run --runtime``.
        """
        super().__init__(
            image=image,
            engine=engine,
            limits=limits,
            policy=policy,
            runtime=runtime or RUNSC_RUNTIME,
            startup_grace=startup_grace,
            event_sink=event_sink,
            cpus=cpus,
            pids_limit=pids_limit,
            user=user,
        )

    @property
    def runtime(self) -> str:
        """The OCI runtime this sandbox launches the container under (``runsc``)."""
        return self._runtime or RUNSC_RUNTIME

    @classmethod
    def available(cls, engine: str = "docker") -> bool:
        """Whether gVisor is usable: the ``runsc`` binary is on PATH **and** the engine
        has the ``runsc`` runtime registered.

        Both checks matter — ``runsc`` can be installed without being wired into the
        Docker daemon's ``runtimes`` map, in which case ``--runtime=runsc`` would fail at
        run time. We probe ``docker info`` for the registered runtime so capability
        detection matches what an actual run would do. On macOS (no ``runsc``) this is
        ``False``, which is exactly what the factory's capability gate expects.
        """
        if shutil.which(RUNSC_RUNTIME) is None:
            return False
        if shutil.which(engine) is None:
            return False
        try:
            # Fixed argv (no shell); ``engine`` is an operator-set engine name, not
            # untrusted model input — same trust boundary as the rest of the backend.
            proc = subprocess.run(  # noqa: S603 - fixed argv, operator-controlled engine
                [engine, "info", "--format", "{{json .Runtimes}}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:  # pragma: no cover - engine missing/unreachable
            return False
        if proc.returncode != 0:
            return False
        # The runtimes map is rendered as JSON keyed by runtime name; a plain substring
        # check is sufficient and avoids depending on the exact engine output shape.
        return RUNSC_RUNTIME in proc.stdout

    # The audit trail must name the *real* boundary, so a reader of the event spine can
    # tell a gVisor run from a plain container run. Everything else (the five guarantees,
    # the argv, the watchdog, ephemeral teardown) is inherited unchanged.
    def _backend_label(self) -> str:
        return "gvisor"


__all__ = ["GVisorSandbox", "RUNSC_RUNTIME"]
