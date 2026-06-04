"""Sandbox kernel: the backend-agnostic execution contract.

The default :class:`~himmy.services.sandbox.subprocess_sandbox.SubprocessSandbox`
is portable defense-in-depth; this protocol is the seam where a stronger isolate
(container / gVisor / microVM) plugs in without touching the callers, exactly like
``ClientManager`` does for inference.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from himmy.services.sandbox.models import SandboxResult


@runtime_checkable
class Sandbox(Protocol):
    """Runs a snippet of code under a resource/fault isolation envelope.

    ``run_code`` must NOT raise for code-level failures (a crash, a non-zero exit,
    a timeout): those are reported in the returned :class:`SandboxResult`. It may
    raise only for *infrastructure* failures (the isolate could not be created).
    """

    async def run_code(
        self,
        code: str,
        *,
        stdin: str | None = None,
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute ``code`` and return its captured result.

        ``files`` are written into the working directory before the run (data the
        code can ``open()`` by relative name); ``stdin`` is fed to the process.
        """
        ...


__all__ = ["Sandbox"]
