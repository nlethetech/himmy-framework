"""Sandbox kernel: the portable, defense-in-depth subprocess sandbox (default).

THREAT MODEL — read before relying on this.
==========================================
:class:`SubprocessSandbox` isolates **resources and faults**, not a determined
adversary. Each run gets: a separate child process, POSIX ``setrlimit`` caps on
CPU time / address space / file size / core dumps, a hard wall-clock timeout that
kills the whole process group, a throwaway temporary working directory, a stripped
environment (an explicit allow-list only), bounded captured output, and Python's
``-I`` isolated mode. That makes it safe against *buggy* or *runaway* code — infinite
loops, fork bombs of CPU/RAM, accidental huge writes, environment/secret leakage.

It is NOT, on its own, a security boundary against *hostile* code: a plain
subprocess can still attempt network I/O and read world-readable files, and
``RLIMIT_AS`` is silently not enforced on some platforms (e.g. macOS). For
UNTRUSTED code, run Himmy inside an OS-level isolate (container / gVisor /
seccomp / microVM) or supply a :class:`~himmy.services.sandbox.base.Sandbox`
implementation that wraps one — the protocol exists for exactly that. The
``network`` limit is advisory here and documented as not enforced.
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from himmy.core.errors import HimmyError
from himmy.services.sandbox.models import SandboxLimits, SandboxResult


class SubprocessSandbox:
    """Run Python code in a resource-limited, fault-isolated child process."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        limits: SandboxLimits | None = None,
    ) -> None:
        """Configure the interpreter and the resource envelope for every run."""
        self._python = python_executable or sys.executable
        self._limits = limits or SandboxLimits()

    async def run_code(
        self,
        code: str,
        *,
        stdin: str | None = None,
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute ``code`` in a throwaway working dir and capture its result."""
        limits = self._limits
        with tempfile.TemporaryDirectory(prefix="himmy-sbx-") as workdir:
            work = Path(workdir)
            for fname, content in (files or {}).items():
                target = self._safe_path(work, fname)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            script = work / "__sandbox_main__.py"
            script.write_text(code, encoding="utf-8")
            return await self._run(script, workdir, limits, stdin)

    async def _run(
        self,
        script: Path,
        workdir: str,
        limits: SandboxLimits,
        stdin: str | None,
    ) -> SandboxResult:
        # ``-I`` isolated mode ignores PYTHON* env + user site; ``-B`` writes no pyc.
        argv = [self._python, "-I", "-B", str(script)]
        preexec = self._preexec(limits) if os.name == "posix" else None
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workdir,
                env=self._build_env(limits.allow_env),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec,
                start_new_session=True,
            )
        except Exception as exc:  # pragma: no cover - infra failure, not code failure
            raise HimmyError(f"sandbox failed to start a subprocess: {exc}") from exc

        in_bytes = stdin.encode("utf-8") if stdin is not None else None
        timed_out = False
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(in_bytes), timeout=limits.timeout_seconds
            )
        except TimeoutError:
            timed_out = True
            self._kill(proc)
            with suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            out = b""
            err = b"[sandbox] killed: wall-clock timeout exceeded"

        duration_ms = (time.monotonic() - started) * 1000.0
        stdout, trunc_out = self._truncate(out, limits.max_output_bytes)
        stderr, trunc_err = self._truncate(err, limits.max_output_bytes)
        exit_code = proc.returncode
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

    # ------------------------------------------------------------------ helpers
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
    def _build_env(allow_env: list[str]) -> dict[str, str]:
        """A minimal env: a safe PATH/locale plus only allow-listed passthroughs."""
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        for key in allow_env:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        return env

    @staticmethod
    def _preexec(limits: SandboxLimits) -> Callable[[], None]:  # noqa: ANN205 - returns a child callback
        """Return a POSIX child-process callback that applies the resource limits."""

        def _apply() -> None:  # pragma: no cover - runs in the forked child
            import resource

            def _set(res: int, value: int) -> None:
                try:
                    resource.setrlimit(res, (value, value))
                except (ValueError, OSError):
                    # A platform may reject a given limit (e.g. RLIMIT_AS on macOS);
                    # degrade gracefully rather than failing the whole run.
                    pass

            if limits.cpu_seconds:
                _set(resource.RLIMIT_CPU, int(math.ceil(limits.cpu_seconds)))
            if limits.memory_mb:
                _set(resource.RLIMIT_AS, limits.memory_mb * 1024 * 1024)
            if limits.file_size_mb:
                _set(resource.RLIMIT_FSIZE, limits.file_size_mb * 1024 * 1024)
            _set(resource.RLIMIT_CORE, 0)

        return _apply

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        """Kill the child's whole process group (so children die with it)."""
        with suppress(ProcessLookupError, PermissionError, OSError):
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:  # pragma: no cover - non-posix fallback
                proc.kill()

    @staticmethod
    def _truncate(data: bytes, max_bytes: int) -> tuple[str, bool]:
        """Decode ``data`` to text, clipped to ``max_bytes`` (flagging truncation)."""
        truncated = len(data) > max_bytes
        return data[:max_bytes].decode("utf-8", errors="replace"), truncated


__all__ = ["SubprocessSandbox"]
