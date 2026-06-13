"""A test-only ``tools_module`` exposing one approval-gated tool with a SECRET arg.

``register(registry)`` binds ``push_secret`` (``requires_approval=True``) whose arg key
``api_token`` LOOKS secret, so a HITL pending-approvals view proves the (unchanged)
redaction masks the value before a human sees it. Kept separate from
``_per_run_tools`` so the shared fixture's bound-tool set stays unchanged.
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool

#: Every ``push_secret`` execution appends here so a test can assert it ran / did not run.
PUSH_CALLS: list[dict[str, Any]] = []


def register(registry: ToolRegistry) -> None:
    """Register the secret-bearing approval-gated tool onto ``registry``."""

    async def _push_secret(args: dict[str, Any]) -> dict[str, Any]:
        PUSH_CALLS.append(dict(args))
        return {"pushed": True}

    register_local_tool(
        registry,
        name="push_secret",
        handler=_push_secret,
        description="An approval-gated push carrying a secret token (records each call).",
        args_json_schema={
            "type": "object",
            "properties": {"api_token": {"type": "string"}},
        },
        requires_approval=True,
    )
