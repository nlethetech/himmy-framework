"""Sandbox kernel: a hardened **container** isolate for untrusted code (WS2.2).

Where :class:`~himmy.services.sandbox.subprocess_sandbox.SubprocessSandbox` isolates
*resources and faults*, :class:`ContainerSandbox` adds an OS-level security boundary
suitable for **hostile, model-authored code**. It enforces all five enterprise
guarantees per run, driven by a :class:`~himmy.services.sandbox.models.SandboxPolicy`
(locked-down defaults):

* **Network — default DENY** (``--network none``): no interface, egress impossible.
  An optional *mediated* egress allow-list (``NetworkMode.EGRESS_PROXY``) attaches the
  container to an operator-provisioned isolated network whose only route off-host is a
  forward proxy — never a general bridge.
* **Ephemeral** (``--rm`` + a tmpfs workspace + a fresh container per run): nothing the
  code writes survives, and no two runs share state.
* **Resource limits** (``--memory`` / ``--cpus`` / ``--pids-limit`` / ``--ulimit``) plus
  a **hard wall-clock kill** (coreutils ``timeout -s KILL``) backed by an outer watchdog
  that force-removes an overrunning container.
* **No host access** (``--read-only`` rootfs, ``--cap-drop ALL``,
  ``--security-opt no-new-privileges``, a **non-root** ``--user``, and **no host bind
  mounts** — input files live on a tmpfs the run owns, streamed in out-of-band over the
  container's stdin so no host path is shared and no argv size ceiling is hit).
* **Logged** — every execution is recorded through the event/audit spine.

It implements the :class:`~himmy.services.sandbox.base.Sandbox` protocol, so it is a
drop-in for the subprocess backend. It shells out to the Docker/Podman CLI (no SDK); the
container image (default ``python:3.12-slim``) must already be present.

For an even stronger boundary on hostile multi-tenant workloads, run the same image under
gVisor (``runtime=runsc``) or a microVM (Kata/Firecracker) — pass ``runtime=``; the threat
model and trade-offs are documented in ``docs/enterprise/sandbox_backends.md``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from contextlib import suppress
from typing import TYPE_CHECKING

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.services.sandbox.models import (
    NetworkMode,
    SandboxLimits,
    SandboxPolicy,
    SandboxResult,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.core.events import EventSink

#: Default container image (must provide a ``python`` interpreter + coreutils ``timeout``).
DEFAULT_IMAGE = "python:3.12-slim"
#: Exit codes coreutils ``timeout`` uses when it had to kill the command.
_TIMEOUT_EXIT_CODES = {124, 137}
#: The writable, tmpfs-backed workspace the code runs in (no host disk, no bind mount).
_WORKDIR = "/sandbox"


class ContainerSandbox:
    """Run Python in a hardened, throwaway container (egress-denied by default)."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        engine: str = "docker",
        limits: SandboxLimits | None = None,
        policy: SandboxPolicy | None = None,
        runtime: str | None = None,
        startup_grace: float = 30.0,
        event_sink: EventSink | None = None,
        cpus: float | None = None,
        pids_limit: int | None = None,
        user: str | None = None,
    ) -> None:
        """Configure the image, engine, isolation policy, and audit sink.

        ``policy`` is the full enterprise contract (network/ephemeral/limits/host/
        audit); ``limits`` overrides only its resource envelope when given. The legacy
        ``cpus`` / ``pids_limit`` / ``user`` keyword args still work and overlay the
        policy so existing callers keep their behaviour.
        """
        self._image = image
        self._engine = engine
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
        self._runtime = runtime
        self._startup_grace = startup_grace
        self._event_sink = event_sink

    @property
    def policy(self) -> SandboxPolicy:
        """The enterprise isolation policy this sandbox enforces per run."""
        return self._policy

    def _backend_label(self) -> str:
        """The backend name recorded on the audit spine (``container``).

        Subclasses that change the *boundary* (e.g. gVisor's user-space kernel) while
        reusing this argv machinery override this so a reader of the event spine can tell
        the runs apart. The policy, argv, and watchdog stay identical.
        """
        return "container"

    @classmethod
    def available(cls, engine: str = "docker") -> bool:
        """Whether the container engine CLI is on PATH (for skip-gating tests)."""
        return shutil.which(engine) is not None

    async def run_code(
        self,
        code: str,
        *,
        stdin: str | None = None,
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute ``code`` inside a hardened container and capture its result."""
        import uuid

        policy = self._policy
        limits = policy.limits
        name = f"himmy-csbx-{uuid.uuid4().hex[:16]}"
        bundle = self._build_bundle(code, files, limits)
        argv = self._argv(policy, name)
        # The bundle is streamed in out-of-band on the container's stdin (length-prefixed
        # framing the bootstrap reads exactly), so it never rides the argv — no host path
        # is shared and the kernel's per-arg size ceiling (MAX_ARG_STRLEN) is never hit.
        # Whatever the caller passed as ``stdin`` follows the bundle and is handed to the
        # snippet untouched (the bootstrap exec's it inheriting the same fd 0).
        in_bytes = self._frame_input(bundle, stdin)
        await self._audit_start(name, policy, limits)
        result = await self._run(argv, name, policy, limits, in_bytes)
        await self._audit_finish(name, result)
        return result

    # ------------------------------------------------------------------ internals
    def _build_bundle(
        self, code: str, files: dict[str, str] | None, limits: SandboxLimits
    ) -> bytes:
        """Encode the script + input files into one self-extracting bootstrap payload.

        No host directory is ever bind-mounted (a bind mount is host access we refuse).
        Instead the files are base64-embedded in a JSON bundle the in-container bootstrap
        decodes and materializes onto the **tmpfs** workspace before exec'ing the snippet.
        Path traversal in a supplied filename is rejected before encoding.

        The bundle is delivered out-of-band (streamed on the container's stdin), so it is
        bounded only by the workspace envelope it must fit on, not by the kernel's argv
        size ceiling. We pre-validate the *materialized* size against the ``file_size_mb``
        tmpfs the run owns and raise a clear :class:`HimmyError` naming the cap, rather
        than letting the engine fail opaquely once the bundle is too large to land.
        """
        import base64
        import json

        payload: dict[str, str] = {}
        for fname, content in (files or {}).items():
            self._reject_traversal(fname)
            payload[fname] = base64.b64encode(content.encode("utf-8")).decode("ascii")
        payload["__sandbox_main__.py"] = base64.b64encode(code.encode("utf-8")).decode(
            "ascii"
        )
        bundle = base64.b64encode(json.dumps(payload).encode("utf-8"))
        self._check_bundle_size(payload, limits)
        return bundle

    @staticmethod
    def _check_bundle_size(payload: dict[str, str], limits: SandboxLimits) -> None:
        """Refuse a bundle whose files can't fit the tmpfs workspace, with a clear cap.

        The decoded files are materialized onto the ``/sandbox`` tmpfs (sized at
        ``file_size_mb``); the bootstrap also holds the JSON bundle in memory while
        decoding. Sum the decoded byte sizes (base64 → raw is ``* 3 / 4``) and compare
        against the workspace envelope so an oversized input fails fast with an actionable
        message instead of an opaque infra error mid-run.
        """
        cap_bytes = max(1, limits.file_size_mb) * 1024 * 1024
        materialized = sum((len(b64) * 3) // 4 for b64 in payload.values())
        if materialized > cap_bytes:
            raise HimmyError(
                "sandbox input bundle is too large: "
                f"{materialized} bytes of code+files exceed the {limits.file_size_mb} MB "
                f"workspace envelope (SandboxLimits.file_size_mb). Reduce the input or "
                f"raise file_size_mb."
            )

    @staticmethod
    def _frame_input(bundle: bytes, stdin: str | None) -> bytes:
        """Frame the stdin fed to the container: ``<len>\\n<bundle><user-stdin>``.

        The bootstrap reads the decimal byte length on the first line, then exactly that
        many bytes of bundle, leaving the caller's ``stdin`` (if any) in the pipe for the
        snippet to consume. Raw fd-level reads in the bootstrap avoid any read-ahead that
        would steal the user's stdin.
        """
        tail = stdin.encode("utf-8") if stdin is not None else b""
        return f"{len(bundle)}\n".encode("ascii") + bundle + tail

    @staticmethod
    def _bootstrap(timeout_seconds: float) -> str:
        """The in-container python that reads the bundle off stdin then runs the snippet.

        The bundle is **not** on the argv (that hit the kernel's per-arg size ceiling on
        large inputs). Instead it is streamed on stdin, length-prefixed: the first line is
        the decimal byte length, then exactly that many bundle bytes, then whatever the
        caller passed as the snippet's own stdin. We read with raw ``os.read`` on fd 0 so
        no buffered read-ahead steals the user's stdin, materialize the files onto the
        tmpfs workspace, then ``execvp`` coreutils ``timeout`` (a hard in-container
        wall-clock SIGKILL) wrapping ``python -I -B`` — the snippet inherits fd 0 and so
        receives the remaining stdin untouched.
        """
        # Kept as a single ``python -c`` string (the only argv payload now is this fixed
        # bootstrap, never the user's data). os.read loops handle short reads on the pipe.
        return (
            "import base64,json,os\n"
            "from pathlib import Path\n"
            "def _rd(n):\n"
            " b=b''\n"
            " while len(b)<n:\n"
            "  c=os.read(0,n-len(b))\n"
            "  if not c: break\n"
            "  b+=c\n"
            " return b\n"
            "h=b''\n"
            "while not h.endswith(b'\\n'):\n"
            " c=os.read(0,1)\n"
            " if not c: break\n"
            " h+=c\n"
            "d=json.loads(base64.b64decode(_rd(int(h.strip()))))\n"
            "for k,v in d.items():\n"
            " p=Path(k); p.parent.mkdir(parents=True,exist_ok=True)\n"
            " p.write_bytes(base64.b64decode(v))\n"
            "os.execvp('timeout',['timeout','-k','1','-s','KILL',"
            f"'{timeout_seconds:g}','python','-I','-B','__sandbox_main__.py'])\n"
        )

    def _argv(self, policy: SandboxPolicy, name: str) -> list[str]:
        """Build the hardened ``docker run`` argument vector from the policy.

        The bundle is no longer an argv element — it is streamed on stdin (see
        :meth:`_frame_input`) — so ``-i`` keeps the container's stdin attached for the
        bootstrap to read from.
        """
        limits = policy.limits
        mem = f"{max(6, limits.memory_mb)}m"
        argv = [self._engine, "run", "-i", "--name", name]
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
            "--memory-swap",
            mem,
            "--cpus",
            str(policy.cpus),
            "--user",
            policy.user,
        ]
        for ulimit, value in policy.ulimits.items():
            argv += ["--ulimit", f"{ulimit}={value}"]
        # The writable workspace is a tmpfs the run owns — never a host bind mount.
        # mode=1777 (sticky, world-writable) so the non-root --user can create files in
        # it; the rootfs stays read-only, so this tmpfs is the only writable surface.
        if policy.workspace_tmpfs:
            argv += [
                "--tmpfs",
                f"{_WORKDIR}:rw,nosuid,nodev,exec,mode=1777,"
                f"size={max(1, limits.file_size_mb)}m",
            ]
        argv += ["-w", _WORKDIR]
        argv += self._network_args(policy)
        if self._runtime:
            argv += ["--runtime", self._runtime]
        for key in limits.allow_env:
            passthrough = os.environ.get(key)
            if passthrough is not None:
                argv += ["-e", f"{key}={passthrough}"]
        argv.append(self._image)
        argv += ["python", "-c", self._bootstrap(limits.timeout_seconds)]
        return argv

    @staticmethod
    def _network_args(policy: SandboxPolicy) -> list[str]:
        """Network flags: default DENY, or mediated egress via a proxy (never bridge)."""
        if policy.network is NetworkMode.NONE:
            return ["--network", "none"]
        # EGRESS_PROXY: attach to the operator-provisioned isolated network and route
        # all egress through the proxy. The proxy enforces the host allow-list; the
        # container has no other route off-host (the network is otherwise internal).
        if not policy.egress_network or not policy.egress_proxy_url:
            raise HimmyError(
                "egress_proxy network mode requires both egress_network and "
                "egress_proxy_url on the SandboxPolicy"
            )
        no_proxy = ",".join(policy.egress_no_proxy)
        return [
            "--network",
            policy.egress_network,
            "-e",
            f"HTTP_PROXY={policy.egress_proxy_url}",
            "-e",
            f"HTTPS_PROXY={policy.egress_proxy_url}",
            "-e",
            f"http_proxy={policy.egress_proxy_url}",
            "-e",
            f"https_proxy={policy.egress_proxy_url}",
            "-e",
            f"NO_PROXY={no_proxy}",
            "-e",
            f"no_proxy={no_proxy}",
        ]

    async def _run(
        self,
        argv: list[str],
        name: str,
        policy: SandboxPolicy,
        limits: SandboxLimits,
        in_bytes: bytes,
    ) -> SandboxResult:
        """Launch the container, enforce the watchdog, and normalize the result.

        ``in_bytes`` is the already-framed stdin (length-prefixed bundle + the caller's
        own stdin); the bootstrap consumes the bundle and hands the remainder to the
        snippet.
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
                f"sandbox failed to start the {self._engine} container: {exc}"
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
        stdout, trunc_out = self._truncate(out, limits.max_output_bytes)
        stderr, trunc_err = self._truncate(err, limits.max_output_bytes)
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
        """Best-effort force-remove a container that overran its watchdog."""
        with suppress(Exception):
            rm = await asyncio.create_subprocess_exec(
                self._engine,
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
        """Record that a sandboxed execution is starting (if auditing is on)."""
        if not policy.audit or self._event_sink is None:
            return
        await self._emit(
            RunEvent(
                event_type=EventType.TOOL_CALLED,
                payload={
                    "tool_name": "sandbox.run_code",
                    "backend": self._backend_label(),
                    "engine": self._engine,
                    "image": self._image,
                    "container": name,
                    "runtime": self._runtime or "default",
                    "network": policy.network.value,
                    "limits": limits.model_dump(),
                },
            )
        )

    async def _audit_finish(self, name: str, result: SandboxResult) -> None:
        """Record the outcome of a sandboxed execution (if auditing is on)."""
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
                    "backend": self._backend_label(),
                    "container": name,
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

    @staticmethod
    def _reject_traversal(name: str) -> None:
        """Reject a supplied filename that would escape the workspace root."""
        from pathlib import PurePosixPath

        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise HimmyError(
                f"sandbox file path escapes the working directory: {name!r}"
            )

    @staticmethod
    def _truncate(data: bytes, limit: int) -> tuple[str, bool]:
        """Decode + cap captured output to ``limit`` bytes."""
        truncated = len(data) > limit
        return data[:limit].decode("utf-8", "replace"), truncated


__all__ = ["ContainerSandbox", "DEFAULT_IMAGE"]
