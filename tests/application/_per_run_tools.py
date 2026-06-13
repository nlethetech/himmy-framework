"""A test-only ``tools_module`` for the per-run runtime seam (UNIT 5 / T0.2).

``register(registry)`` binds one deterministic, offline tool — ``ping`` — that
appends to :data:`CALLS` every time it runs. A spec naming this module via
``tools_module: tests.application._per_run_tools`` therefore lets a test assert that
the agent's tool ACTUALLY fired on the per-run runtime (impossible before this unit).
Because ``tools_module`` is an operator-only field, the spec must be
operator-provisioned for the field to survive sanitization (T0.3) — which is exactly
what the test verifies.
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool

#: Every ``ping`` invocation appends its args here so a test can prove the tool ran.
CALLS: list[dict[str, Any]] = []
#: Every ``wire_money`` execution appends here — used by the /v1 HITL tests (T2f) to prove
#: the approval-gated tool fired EXACTLY ONCE after approve (and NEVER after reject).
WIRE_CALLS: list[dict[str, Any]] = []


def register(registry: ToolRegistry) -> None:
    """Register the recording tools onto ``registry`` (the from_spec entrypoint).

    ``ping`` is a read-only probe (no approval). ``wire_money`` is approval-gated
    (``requires_approval=True``), so a ``hitl=True`` run that calls it PAUSES into a
    checkpoint — the seam the /v1 HITL approve/reject tests (T2f) drive.
    """

    async def _ping(args: dict[str, Any]) -> dict[str, Any]:
        CALLS.append(dict(args))
        return {"pong": True, "echo": args.get("message", "")}

    register_local_tool(
        registry,
        name="ping",
        handler=_ping,
        description="A deterministic offline probe that records each call.",
        args_json_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
        read_only=True,
    )

    async def _wire_money(args: dict[str, Any]) -> dict[str, Any]:
        WIRE_CALLS.append(dict(args))
        return {"sent": True, "amount": args.get("amount", 0)}

    register_local_tool(
        registry,
        name="wire_money",
        handler=_wire_money,
        description="An approval-gated money transfer (records each execution).",
        args_json_schema={
            "type": "object",
            "properties": {"amount": {"type": "number"}},
        },
        requires_approval=True,
    )
