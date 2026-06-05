"""Code pack: ``run_python`` — execute Python in a sandbox (backend by policy).

A thin adapter over the sandbox kernel: it builds the configured backend via
:func:`~himmy.services.sandbox.factory.build_sandbox` (``HIMMY_CODE_EXEC`` →
``subprocess`` default, ``container`` hardened, or ``off`` = refuse) and registers it
through :func:`~himmy.services.sandbox.tools.register_sandbox_tool`, inheriting the
approval gate, arg validation, and structured result. Approval stays on by default —
running code is a human-in-the-loop decision — and served/multi-tenant deployments
should set ``HIMMY_CODE_EXEC=container`` (or ``off``) rather than run untrusted code on
the bare subprocess backend.
"""

from __future__ import annotations

from himmy.services.sandbox.factory import build_sandbox
from himmy.services.sandbox.tools import register_sandbox_tool
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig


def register_code_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``run_python`` backed by the policy-selected sandbox backend."""
    sandbox = build_sandbox(
        config.code_exec,
        limits=config.sandbox_limits,
        image=config.sandbox_image,
        engine=config.sandbox_engine,
    )
    register_sandbox_tool(registry, sandbox, name="run_python", requires_approval=True)


__all__ = ["register_code_pack"]
