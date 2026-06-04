"""Sandbox kernel: expose a Sandbox as a policy-gated, audited agent tool.

Registering the sandbox as a LOCAL tool routes every execution through the
``ToolService`` pipeline — so it inherits approval gating, arg validation, event
emission, and lineage, exactly like any other tool. It defaults to
``requires_approval=True``: an agent asking to run code is a human-in-the-loop
decision, not a silent capability.
"""

from __future__ import annotations

from typing import Any

from opensims.services.sandbox.base import Sandbox
from opensims.services.tools.models import ToolDefinition
from opensims.services.tools.registry import ToolRegistry, register_local_tool

#: Argument schema for the sandbox tool (validated by the ToolService).
SANDBOX_TOOL_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python source to execute in the sandbox.",
        },
        "stdin": {
            "type": "string",
            "description": "Optional text fed to the process's stdin.",
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


def register_sandbox_tool(
    registry: ToolRegistry,
    sandbox: Sandbox,
    *,
    name: str = "run_python",
    requires_approval: bool = True,
    description: str | None = None,
) -> ToolDefinition:
    """Register a sandboxed Python-execution tool backed by ``sandbox``.

    The tool returns a structured :class:`SandboxResult` dict
    (``{ok, exit_code, stdout, stderr, timed_out, duration_ms, truncated}``).
    Approval is on by default; pass ``requires_approval=False`` only when the
    sandbox is a hardened OS-level isolate and the caller is trusted.
    """

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        code = str(args.get("code", ""))
        raw_stdin = args.get("stdin")
        stdin = str(raw_stdin) if raw_stdin is not None else None
        result = await sandbox.run_code(code, stdin=stdin)
        return result.model_dump(mode="json")

    return register_local_tool(
        registry,
        name=name,
        handler=_handler,
        description=(
            description
            or "Execute Python in an isolated, resource-limited sandbox; returns "
            "{ok, exit_code, stdout, stderr, timed_out}."
        ),
        args_json_schema=SANDBOX_TOOL_ARGS_SCHEMA,
        requires_approval=requires_approval,
        metadata={"backend": "sandbox"},
    )


__all__ = ["register_sandbox_tool", "SANDBOX_TOOL_ARGS_SCHEMA"]
