"""Spawn tool: let an agent run a focused *sub-agent* and use its answer.

``spawn_agent`` turns any agent into an orchestrator — it hands a sub-task to a fresh
single-agent (its own persona, optionally its own built-in tool packs) that runs to
completion and returns its answer. Unlike team delegation (which needs a declared
``team.yaml``), this is ad-hoc: the parent decides at run time who to spawn and for what.

The sub-agent shares the parent's :class:`InferenceService` (same provider/model), but
runs in a *fresh* runtime whose registry does **not** contain ``spawn_agent`` — so a
spawned worker cannot spawn again. That one-level cap is the recursion guard.

This is wired where the inference backend is known (the CLI's ``allow_spawn: true``), not
as an ordinary pack, because a pack registrar only receives a :class:`ToolkitConfig`.
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool

_SPAWN_SCHEMA = {
    "type": "object",
    "properties": {
        "instructions": {
            "type": "string",
            "description": "The sub-agent's role / system instructions.",
        },
        "prompt": {
            "type": "string",
            "description": "The concrete task to hand the sub-agent.",
        },
        "name": {"type": "string", "description": "Optional name for the sub-agent."},
        "tool_packs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional built-in tool packs to give the sub-agent.",
        },
    },
    "required": ["instructions", "prompt"],
    "additionalProperties": False,
}


def register_spawn_tool(
    registry: ToolRegistry,
    *,
    inference: Any,
    requires_approval: bool = False,
    tool_authorizer: Any = None,
) -> None:
    """Register ``spawn_agent`` bound to ``inference`` (the parent's backend).

    ``tool_authorizer`` (P0) is the PARENT run's tool-capability gate. A spawned
    sub-agent inherits it VERBATIM (via :meth:`ToolCapabilityAuthorizer.attenuate`,
    which returns the same frozen authorizer) so the sub-agent's capability set is always
    a subset of the parent's — capability can only ATTENUATE down a spawn chain, never
    amplify. ``None`` / a NON-enforcing authorizer is a no-op (the offline default), so
    the sub-runtime's tool dispatch is byte-identical to before.
    """

    async def spawn_agent(args: dict[str, Any]) -> dict[str, Any]:
        from himmy.agents.base_agent.task import Task
        from himmy.agents.personas.persona import Persona
        from himmy.runtime.builder import build_runtime
        from himmy.toolkit import BUILTIN_PACKS

        sub_overrides: dict[str, Any] = {"inference": inference}
        requested = [str(p) for p in (args.get("tool_packs") or [])]
        # Tolerate a model naming a pack that doesn't exist: use the known ones and
        # report the rest, rather than failing the whole spawn.
        packs = [p for p in requested if p in BUILTIN_PACKS]
        dropped = [p for p in requested if p not in BUILTIN_PACKS]
        if packs:
            from himmy.toolkit import ToolkitConfig, register_packs

            sub_registry = ToolRegistry()
            register_packs(sub_registry, packs, ToolkitConfig.from_env())
            # Propagate the parent's capability gate so the sub-agent can only invoke tools
            # the parent could (attenuate-never-amplify). Build the sub tool service here so
            # the authorizer reaches the sub-runtime's dispatch (a bare ``tool_registry``
            # override would otherwise get an un-gated default ToolService).
            if tool_authorizer is not None:
                from himmy.services.tools.service import ToolService

                sub_overrides["tool_service"] = ToolService(
                    sub_registry, tool_authorizer=tool_authorizer.attenuate()
                )
            else:
                sub_overrides["tool_registry"] = sub_registry

        sub_runtime, _inf, _tools = build_runtime(**sub_overrides)
        persona = Persona(
            name=str(args.get("name") or "sub-agent"),
            description="A focused sub-agent spawned to handle one task.",
            instructions=[str(args["instructions"])],
        )
        result = await sub_runtime.run_task_detailed(
            persona, Task(title="subtask", prompt=str(args["prompt"]))
        )
        out: dict[str, Any] = {
            "answer": result.output_text or "",
            "status": result.status,
            "succeeded": result.succeeded,
        }
        if dropped:
            out["unknown_tool_packs"] = dropped
        return out

    register_local_tool(
        registry,
        name="spawn_agent",
        handler=spawn_agent,
        description=(
            "Run a focused sub-agent on a sub-task and return its answer. Give it clear "
            "`instructions` (its role) and a `prompt` (the task); optionally `tool_packs`."
        ),
        args_json_schema=_SPAWN_SCHEMA,
        requires_approval=requires_approval,
        metadata={"pack": "spawn"},
    )


__all__ = ["register_spawn_tool"]
