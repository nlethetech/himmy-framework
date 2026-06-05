"""Code pack: ``run_python`` — execute Python in a resource-limited sandbox.

This is a thin adapter over the existing sandbox kernel: it builds a
:class:`~himmy.services.sandbox.subprocess_sandbox.SubprocessSandbox` from the toolkit
config's limits and registers it through the kernel's own
:func:`~himmy.services.sandbox.tools.register_sandbox_tool`, so it inherits the
approval gate, arg validation, and structured ``{ok, exit_code, stdout, stderr, ...}``
result. Approval stays on by default — running code is a human-in-the-loop decision.
"""

from __future__ import annotations

from himmy.services.sandbox.subprocess_sandbox import SubprocessSandbox
from himmy.services.sandbox.tools import register_sandbox_tool
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig


def register_code_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``run_python`` backed by a SubprocessSandbox using config limits."""
    sandbox = SubprocessSandbox(limits=config.sandbox_limits)
    register_sandbox_tool(registry, sandbox, name="run_python", requires_approval=True)


__all__ = ["register_code_pack"]
