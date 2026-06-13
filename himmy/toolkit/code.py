"""Code pack: ``run_python`` — execute Python in a sandbox (backend by policy).

A thin adapter over the sandbox kernel: it builds the configured backend via
:func:`~himmy.services.sandbox.factory.build_sandbox` (``HIMMY_CODE_EXEC`` →
``subprocess`` default, ``container`` hardened, or ``off`` = refuse) and registers it
through :func:`~himmy.services.sandbox.tools.register_sandbox_tool`, inheriting the
approval gate, arg validation, and structured result. Approval stays on by default —
running code is a human-in-the-loop decision.

Fail-closed posture: in a **server context** (``config.server_context``) the unsandboxed
``subprocess`` backend is refused — a served/multi-tenant deployment must run untrusted
code under ``HIMMY_CODE_EXEC=container`` (hardened isolate) or ``off``. The dev/CLI
default is unchanged. Every container-backed execution is also audited through the same
tool-event spine that already records every ``run_python`` call.
"""

from __future__ import annotations

from himmy.services.sandbox.factory import build_sandbox
from himmy.services.sandbox.tools import register_sandbox_tool
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig


def register_code_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``run_python`` backed by the policy-selected sandbox backend.

    The selected backend respects the fail-closed server posture: ``build_sandbox``
    raises if a server context asks for the unsandboxed subprocess backend, so a
    misconfigured deployment fails to start rather than silently running untrusted code
    unsandboxed.
    """
    sandbox = build_sandbox(
        config.code_exec,
        limits=config.sandbox_limits,
        policy=config.build_sandbox_policy(),
        image=config.sandbox_image,
        engine=config.sandbox_engine,
        server_context=config.server_context,
    )
    register_sandbox_tool(registry, sandbox, name="run_python", requires_approval=True)


__all__ = ["register_code_pack"]
