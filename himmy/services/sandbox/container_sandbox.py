"""Sandbox kernel: a hardened **container** isolate for untrusted code (WS2.2).

Where :class:`~himmy.services.sandbox.subprocess_sandbox.SubprocessSandbox` isolates
*resources and faults*, :class:`ContainerSandbox` adds an OS-level security boundary
suitable for **hostile, model-authored code** — by running each snippet in a throwaway
container with, by default:

* **no network** (``--network none``) — egress denied, the #1 exfiltration path;
* **read-only root filesystem** (``--read-only``) + a small writable ``/tmp`` tmpfs;
* **all Linux capabilities dropped** (``--cap-drop ALL``) and ``no-new-privileges``;
* a **non-root** user (``--user``), so a container escape isn't immediately root;
* **pids / memory / cpu** limits (fork-bomb + OOM + CPU containment);
* a **hard in-container wall-clock kill** (coreutils ``timeout``), backed by an outer
  watchdog that force-removes the container.

It implements the :class:`~himmy.services.sandbox.base.Sandbox` protocol, so it is a
drop-in for the subprocess backend. It shells out to the Docker/Podman CLI (no SDK); the
container image (default ``python:3.12-slim``) must already be present.

For an even stronger boundary on hostile multi-tenant workloads, run the same image under
gVisor (``runtime=runsc``) or a microVM (Kata/Firecracker) — pass ``runtime=`` / extra
args; the threat model and trade-offs are documented in ``docs/enterprise/``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from himmy.core.errors import HimmyError
from himmy.services.sandbox.models import SandboxLimits, SandboxResult

#: Default container image (must provide a ``python`` interpreter + coreutils ``timeout``).
DEFAULT_IMAGE = "python:3.12-slim"
#: Exit codes coreutils ``timeout`` uses when it had to kill the command.
_TIMEOUT_EXIT_CODES = {124, 137}


class ContainerSandbox:
    """Run Python in a hardened, throwaway container (egress-denied by default)."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        engine: str = "docker",
        limits: SandboxLimits | None = None,
        cpus: float = 1.0,
        pids_limit: int = 128,
        user: str = "65534:65534",
        runtime: str | None = None,
        startup_grace: float = 30.0,
        workdir_base: str | None = None,
    ) -> None:
        """Configure the image, engine, resource envelope, and isolation knobs."""
        self._image = image
        self._engine = engine
        self._limits = limits or SandboxLimits()
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._user = user
        self._runtime = runtime
        self._startup_grace = startup_grace
        # Where the throwaway bind-mounted workdir is created. Some engines (Docker
        # Desktop on macOS) only share specific host paths; point this at one of them.
        self._workdir_base = workdir_base or os.environ.get("HIMMY_SANDBOX_TMP")

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
        limits = self._limits
        with tempfile.TemporaryDirectory(
            prefix="himmy-csbx-", dir=self._workdir_base
        ) as workdir:
            work = Path(workdir)
            self._materialize(work, code, files)
            return await self._run(workdir, limits, stdin)

    # ------------------------------------------------------------------ internals
    def _materialize(self, work: Path, code: str, files: dict[str, str] | None) -> None:
        """Write the script + input files, world-readable for the container user."""
        for fname, content in (files or {}).items():
            target = self._safe_path(work, fname)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            with suppress(OSError):
                target.chmod(0o644)
        script = work / "__sandbox_main__.py"
        script.write_text(code, encoding="utf-8")
        with suppress(OSError):
            script.chmod(0o644)
            # The temp dir is 0700 by default; the non-root container user needs to
            # traverse + read it (mounted read-only), so widen to 0755.
            work.chmod(0o755)

    def _argv(self, workdir: str, limits: SandboxLimits, name: str) -> list[str]:
        """Build the hardened ``docker run`` argument vector."""
        mem = f"{max(6, limits.memory_mb)}m"
        argv = [
            self._engine,
            "run",
            "--rm",
            "--name",
            name,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self._pids_limit),
            "--memory",
            mem,
            "--memory-swap",
            mem,
            "--cpus",
            str(self._cpus),
            "--user",
            self._user,
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={max(1, limits.file_size_mb)}m",
            "--network",
            "bridge" if limits.network else "none",
            "-v",
            f"{workdir}:/work:ro",
            "-w",
            "/work",
        ]
        if self._runtime:
            argv += ["--runtime", self._runtime]
        for key in limits.allow_env:
            value = os.environ.get(key)
            if value is not None:
                argv += ["-e", f"{key}={value}"]
        argv.append(self._image)
        # Hard in-container wall-clock kill via coreutils ``timeout`` (SIGKILL).
        argv += [
            "timeout",
            "-k",
            "1",
            "-s",
            "KILL",
            str(limits.timeout_seconds),
            "python",
            "-I",
            "-B",
            "/work/__sandbox_main__.py",
        ]
        return argv

    async def _run(
        self, workdir: str, limits: SandboxLimits, stdin: str | None
    ) -> SandboxResult:
        """Launch the container, enforce the watchdog, and normalize the result."""
        import uuid

        name = f"himmy-csbx-{uuid.uuid4().hex[:16]}"
        argv = self._argv(workdir, limits, name)
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

        in_bytes = stdin.encode("utf-8") if stdin is not None else None
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

    @staticmethod
    def _safe_path(workdir: Path, name: str) -> Path:
        """Resolve ``name`` under ``workdir``, rejecting any path traversal."""
        base = workdir.resolve()
        target = (base / name).resolve()
        if target != base and base not in target.parents:
            raise HimmyError(
                f"sandbox file path escapes the working directory: {name!r}"
            )
        return target

    @staticmethod
    def _truncate(data: bytes, limit: int) -> tuple[str, bool]:
        """Decode + cap captured output to ``limit`` bytes."""
        truncated = len(data) > limit
        return data[:limit].decode("utf-8", "replace"), truncated


__all__ = ["ContainerSandbox", "DEFAULT_IMAGE"]
