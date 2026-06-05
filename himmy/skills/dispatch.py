"""Dispatch a single skill as an isolated, capability-scoped sub-agent.

Where ``spawn_agent`` hands a sub-task to a free-form worker, ``dispatch_skill`` runs a
*named capability*: a fresh sub-agent bound to exactly that skill's tools (its packs) and
primed with its know-how + examples — nothing else. That makes it a focused, auditable
unit ("answer this with the ``data_analysis`` skill") and a one-level recursion cap (the
sub-runtime has no ``dispatch_skill`` tool, so a dispatched skill can't dispatch again).
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool


class SkillDispatcher:
    """Run a skill by name as an isolated sub-agent scoped to its tools + know-how."""

    def __init__(
        self,
        *,
        inference: Any,
        skill_registry: Any | None = None,
        toolkit_config: Any | None = None,
        max_turns: int = 6,
    ) -> None:
        self._inference = inference
        self._skill_registry = skill_registry
        self._toolkit_config = toolkit_config
        self._max_turns = max_turns

    def _skills(self) -> Any:
        if self._skill_registry is None:
            from himmy.skills import build_skill_registry

            self._skill_registry = build_skill_registry()
        return self._skill_registry

    async def run(self, skill_name: str, prompt: str) -> dict[str, Any]:
        """Resolve ``skill_name``, run it on ``prompt``, return ``{answer, ...}``."""
        from himmy.agents.base_agent.task import Task
        from himmy.agents.personas.persona import Persona
        from himmy.runtime.builder import build_runtime
        from himmy.skills import resolve_skills
        from himmy.skills.resolve import format_examples

        bundle = resolve_skills([skill_name], self._skills())

        overrides: dict[str, Any] = {"inference": self._inference}
        has_tools = bool(bundle.tool_packs)
        if has_tools:
            from himmy.toolkit import ToolkitConfig, register_packs

            registry = ToolRegistry()
            register_packs(
                registry,
                list(bundle.tool_packs),
                self._toolkit_config or ToolkitConfig.from_env(),
            )
            overrides["tool_registry"] = registry
        runtime, _inf, _tools = build_runtime(**overrides)

        sections = list(bundle.instruction_blocks)
        if bundle.examples:
            sections.append(format_examples(bundle.examples))
        persona = Persona(
            name=f"skill:{skill_name}",
            description="\n\n".join(sections),
            metadata={"skills": list(bundle.skills)},
        )
        ctx: dict[str, Any] = {}
        if bundle.tools:
            ctx["tool_names"] = list(bundle.tools)
        task = Task(title=skill_name, prompt=prompt, context=ctx)

        if has_tools:
            loop = await runtime.run_agent_loop(
                persona, task, max_turns=self._max_turns
            )
            answer = loop.final.output_text if loop.final else ""
            status = loop.stopped_reason
        else:
            result = await runtime.run_task_detailed(persona, task)
            answer = result.output_text or ""
            status = result.status
        return {
            "answer": answer,
            "skill": skill_name,
            "applied_skills": list(bundle.skills),
            "tool_packs": list(bundle.tool_packs),
            "status": status,
        }


_DISPATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "skill": {"type": "string", "description": "The skill (capability) to run."},
        "prompt": {"type": "string", "description": "The concrete task for the skill."},
    },
    "required": ["skill", "prompt"],
    "additionalProperties": False,
}


def register_skill_dispatch_tool(
    registry: ToolRegistry,
    *,
    inference: Any,
    skill_registry: Any | None = None,
    toolkit_config: Any | None = None,
    requires_approval: bool = False,
) -> None:
    """Register ``dispatch_skill`` bound to ``inference`` and the skill catalog."""
    dispatcher = SkillDispatcher(
        inference=inference,
        skill_registry=skill_registry,
        toolkit_config=toolkit_config,
    )

    async def dispatch_skill(args: dict[str, Any]) -> dict[str, Any]:
        from himmy.skills.errors import UnknownSkillError

        try:
            return await dispatcher.run(str(args["skill"]), str(args["prompt"]))
        except UnknownSkillError as exc:
            # Tolerate a bad skill name: report it rather than failing the parent run.
            return {"answer": "", "error": str(exc), "skill": str(args.get("skill"))}

    names = ", ".join(s.name for s in (skill_registry.list() if skill_registry else []))
    register_local_tool(
        registry,
        name="dispatch_skill",
        handler=dispatch_skill,
        description=(
            "Run a named capability (skill) as a focused sub-agent scoped to just that "
            "skill's tools and know-how; returns its {answer}. "
            + (
                f"Available skills: {names}."
                if names
                else "Run `himmy skills` to list."
            )
        ),
        args_json_schema=_DISPATCH_SCHEMA,
        requires_approval=requires_approval,
        metadata={"pack": "skill_dispatch"},
    )


__all__ = ["SkillDispatcher", "register_skill_dispatch_tool"]
