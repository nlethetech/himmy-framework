"""Sandbox kernel: the limits requested and the result returned by a run."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SandboxLimits(BaseModel):
    """The resource envelope enforced around a sandboxed execution.

    ``cpu_seconds`` and ``memory_mb`` are enforced via POSIX ``setrlimit`` in the
    child (best-effort on platforms that reject a given limit); ``timeout_seconds``
    is a hard wall-clock kill; ``max_output_bytes`` bounds captured stdout/stderr;
    ``file_size_mb`` caps any single file the code writes. ``network`` is
    **advisory** — a bare subprocess cannot block sockets on every OS, so a
    Sandbox backend that can (container/namespace) should honor it and the default
    :class:`SubprocessSandbox` documents it as not enforced. ``allow_env`` is an
    allow-list of environment variables passed through to the child (everything
    else is stripped).
    """

    cpu_seconds: float = 5.0
    timeout_seconds: float = 10.0
    memory_mb: int = 256
    file_size_mb: int = 16
    max_output_bytes: int = 64 * 1024
    network: bool = False
    allow_env: list[str] = Field(default_factory=list)


class SandboxResult(BaseModel):
    """The outcome of one sandboxed execution."""

    ok: bool
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: float = 0.0
    truncated: bool = False
    limits: SandboxLimits = Field(default_factory=SandboxLimits)


__all__ = ["SandboxLimits", "SandboxResult"]
