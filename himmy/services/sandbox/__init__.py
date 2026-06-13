"""Sandbox kernel: isolated, resource-limited code execution.

The default :class:`SubprocessSandbox` is portable defense-in-depth (process +
rlimits + wall-clock timeout + fs/env isolation). Stronger OS-level isolates plug in
behind the same :class:`Sandbox` protocol, forming a boundary ladder:
:class:`ContainerSandbox` (hardened container, shared host kernel) →
:class:`GVisorSandbox` (runsc user-space kernel) →
:class:`FirecrackerSandbox` (KVM microVM, separate guest kernel per run). The last two
require Linux (+KVM for Firecracker) and are not executable on macOS — see
``docs/enterprise/sandbox.md``. :func:`register_sandbox_tool` exposes any of them as a
policy-gated, audited agent tool.
"""

from __future__ import annotations

from himmy.services.sandbox.base import Sandbox
from himmy.services.sandbox.container_sandbox import ContainerSandbox
from himmy.services.sandbox.factory import (
    CODE_EXEC_MODES,
    DisabledSandbox,
    build_sandbox,
)
from himmy.services.sandbox.firecracker_sandbox import FirecrackerSandbox
from himmy.services.sandbox.gvisor_sandbox import GVisorSandbox
from himmy.services.sandbox.models import (
    NetworkMode,
    SandboxLimits,
    SandboxPolicy,
    SandboxResult,
)
from himmy.services.sandbox.subprocess_sandbox import SubprocessSandbox
from himmy.services.sandbox.tools import (
    SANDBOX_TOOL_ARGS_SCHEMA,
    register_sandbox_tool,
)

__all__ = [
    "Sandbox",
    "SandboxLimits",
    "SandboxPolicy",
    "NetworkMode",
    "SandboxResult",
    "SubprocessSandbox",
    "ContainerSandbox",
    "GVisorSandbox",
    "FirecrackerSandbox",
    "build_sandbox",
    "DisabledSandbox",
    "CODE_EXEC_MODES",
    "register_sandbox_tool",
    "SANDBOX_TOOL_ARGS_SCHEMA",
]
