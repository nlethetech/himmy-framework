"""Sandbox kernel: expose a Sandbox as a policy-gated, audited agent tool.

Registering the sandbox as a LOCAL tool routes every execution through the
``ToolService`` pipeline — so it inherits approval gating, arg validation, event
emission, and lineage, exactly like any other tool. It defaults to
``requires_approval=True``: an agent asking to run code is a human-in-the-loop
decision, not a silent capability.
"""

from __future__ import annotations

import ast
from typing import Any

from himmy.services.sandbox.base import Sandbox
from himmy.services.tools.models import ToolDefinition
from himmy.services.tools.registry import ToolRegistry, register_local_tool

_REPL_TMP = "__himmy_repl__"


def echo_last_expression(code: str) -> str:
    """Auto-print the value of a trailing bare expression (REPL/Jupyter semantics).

    Models routinely end a snippet with a bare expression (``result`` or ``4869*17``)
    instead of ``print(...)``, IPython-style — which otherwise produces *no* stdout, so
    the agent sees an empty result and falls back to guessing. This rewrites a trailing
    expression statement to print its ``repr`` when the value is not ``None``, leaving
    everything else (including a trailing ``print(...)``, which evaluates to ``None``)
    untouched. Unparseable code is returned verbatim — the sandbox reports the error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    value = tree.body[-1].value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return code  # a trailing string/docstring — not a result to echo
    assign = ast.Assign(targets=[ast.Name(id=_REPL_TMP, ctx=ast.Store())], value=value)
    guard = ast.If(
        test=ast.Compare(
            left=ast.Name(id=_REPL_TMP, ctx=ast.Load()),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ),
        body=[
            ast.Expr(
                ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id="repr", ctx=ast.Load()),
                            args=[ast.Name(id=_REPL_TMP, ctx=ast.Load())],
                            keywords=[],
                        )
                    ],
                    keywords=[],
                )
            )
        ],
        orelse=[],
    )
    tree.body[-1:] = [assign, guard]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


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
        code = echo_last_expression(str(args.get("code", "")))
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
        read_only=False,  # executes code → an action, not a look-up
        metadata={"backend": "sandbox"},
    )


__all__ = [
    "register_sandbox_tool",
    "SANDBOX_TOOL_ARGS_SCHEMA",
    "echo_last_expression",
]
