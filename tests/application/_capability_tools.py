"""A test-only ``tools_module`` for the orchestration tool-capability gate (red-team r3).

Binds two deterministic offline tools so a test can prove the per-run tool-capability
authorizer is threaded into a TEAM/WORKFLOW member runtime (not just the single-agent
path):

* ``read_lookup`` — an EXPLICIT read-only tool (``read_only=True``), so the gate needs
  only ``tool:read_lookup:invoke`` to allow it; and
* ``write_action`` — a side-effecting tool that is NOT approval-gated (so a denial here is
  the CAPABILITY gate, never the HITL gate), needing ``tool:write_action:write``.

Each call records into a list so the test can assert whether the handler actually ran.
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool

#: Every ``read_lookup`` invocation appends here.
READ_CALLS: list[dict[str, Any]] = []
#: Every ``write_action`` invocation appends here (must stay empty when denied).
WRITE_CALLS: list[dict[str, Any]] = []


def register(registry: ToolRegistry) -> None:
    """Register the recording read/write tools onto ``registry``."""

    async def _read(args: dict[str, Any]) -> dict[str, Any]:
        READ_CALLS.append(dict(args))
        return {"ok": True}

    async def _write(args: dict[str, Any]) -> dict[str, Any]:
        WRITE_CALLS.append(dict(args))
        return {"sent": True}

    register_local_tool(
        registry,
        name="read_lookup",
        handler=_read,
        description="An explicit read-only lookup.",
        args_json_schema={"type": "object", "properties": {}},
        read_only=True,
    )
    register_local_tool(
        registry,
        name="write_action",
        handler=_write,
        description="A side-effecting action (not approval-gated).",
        args_json_schema={"type": "object", "properties": {}},
        read_only=False,
    )
