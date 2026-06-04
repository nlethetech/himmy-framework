"""Sandbox kernel: isolated, resource-limited code execution.

The default :class:`SubprocessSandbox` is portable defense-in-depth (process +
rlimits + wall-clock timeout + fs/env isolation). The :class:`Sandbox` protocol is
the seam for a stronger OS-level isolate. :func:`register_sandbox_tool` exposes a
sandbox as a policy-gated, audited agent tool.
"""

from __future__ import annotations

from opensims.services.sandbox.base import Sandbox
from opensims.services.sandbox.models import SandboxLimits, SandboxResult
from opensims.services.sandbox.subprocess_sandbox import SubprocessSandbox
from opensims.services.sandbox.tools import (
    SANDBOX_TOOL_ARGS_SCHEMA,
    register_sandbox_tool,
)

__all__ = [
    "Sandbox",
    "SandboxLimits",
    "SandboxResult",
    "SubprocessSandbox",
    "register_sandbox_tool",
    "SANDBOX_TOOL_ARGS_SCHEMA",
]
