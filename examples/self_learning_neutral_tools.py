"""Neutral, symmetric tools for the LIVE self-learning study (Layer 3).

These exist to kill the *name leak* and *behaviour leak* confounds that made the old
live proof hollow: two tools with single-token, semantically-empty, symmetric names,
BYTE-IDENTICAL descriptions, identical schemas, and identical behaviour. Neither name,
description, nor return value carries a quality / reliability / ordering signal, so a
real model has no intrinsic reason to prefer one over the other — anything that moves
its choice in the study is himmy's self-learning loop, not the toolset.

Both tools ALWAYS SUCCEED. The study measures the model's *choice* of tool, not a live
failure: a flagged tool's poor reputation is seeded out-of-band as ``TOOL_FAILED`` events
on the store, never produced by these handlers. (The deterministic Layer-1 demo in
``self_learning_demo.py`` deliberately uses a tool that *raises* — that is a different
proof and is left untouched.)

himmy calls a local handler as ``handler(args_dict)`` — ONE positional dict, never
``**kwargs`` — so both handlers take a single ``args`` mapping.
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool

#: The two neutral tool names. Single-token, no quality/order/reliability signal,
#: and (by alphabet) ``tool_alpha`` would naturally sort before ``tool_beta`` in the
#: registry — which is exactly why the study counterbalances both *which* tool is
#: flagged and the *declared order*, so this residual bias cancels.
TOOL_ALPHA = "tool_alpha"
TOOL_BETA = "tool_beta"

#: BYTE-IDENTICAL across both tools: no asymmetry the model could latch onto.
_DESCRIPTION = "Look up a value for the given key and return it."

#: IDENTICAL schema for both tools.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"key": {"type": "string"}},
}


def register(registry: ToolRegistry) -> None:
    """Register the two neutral, symmetric, always-succeeding tools onto a registry.

    This is the ``tools_module`` entry point ``build_runtime_for_spec`` calls.
    """

    async def _alpha(args: dict[str, Any]) -> dict[str, Any]:
        # Always succeeds. Source names itself so a caller can confirm which ran,
        # but the value is identical so the return carries no preference signal.
        return {"value": "42", "source": TOOL_ALPHA}

    async def _beta(args: dict[str, Any]) -> dict[str, Any]:
        return {"value": "42", "source": TOOL_BETA}

    register_local_tool(
        registry,
        name=TOOL_ALPHA,
        handler=_alpha,
        description=_DESCRIPTION,
        args_json_schema=dict(_SCHEMA),
        read_only=True,
    )
    register_local_tool(
        registry,
        name=TOOL_BETA,
        handler=_beta,
        description=_DESCRIPTION,
        args_json_schema=dict(_SCHEMA),
        read_only=True,
    )


__all__ = ["TOOL_ALPHA", "TOOL_BETA", "register"]
