"""Tier 3: requires_skills composition, few-shot examples, and routing hints."""

from __future__ import annotations

from himmy.config.agent_spec import AgentSpec, apply_skills
from himmy.skills import Skill, SkillExample, SkillRegistry, resolve_skills


def _registry() -> SkillRegistry:
    return SkillRegistry.with_builtins()


def test_composite_skill_pulls_prerequisite_tools() -> None:
    # research_writer requires [web_research, summarize] — both should apply.
    out = apply_skills(AgentSpec(name="a", skills=["research_writer"]), _registry())
    assert out.tool_packs == ["web"]  # from web_research
    assert out.metadata["skills"] == ["web_research", "summarize", "research_writer"]
    # The dependent's own block + instructions render too (after its prerequisites).
    assert "Skill — research_writer" in out.description
    assert "gather facts with web_research" in out.description


def test_examples_render_into_description() -> None:
    out = apply_skills(AgentSpec(name="a", skills=["data_analysis"]), _registry())
    assert "Worked examples:" in out.description
    assert "How many chickens" in out.description


def test_when_to_use_renders_and_routes() -> None:
    out = apply_skills(AgentSpec(name="a", skills=["data_analysis"]), _registry())
    # Rendered as guidance in the background...
    assert "Use this when" in out.description
    # ...and stashed as routing hints that make_task carries into the task context.
    assert out.metadata["skill_routing_hints"]
    task = out.make_task("how many ducks?")
    assert task.context["skill_routing_hints"]


def test_custom_composition_dedups_shared_tools() -> None:
    reg = _registry()
    reg.register(Skill(name="x", tool_packs=["utils"], tools=["calculator"]))
    reg.register(Skill(name="y", requires_skills=["x"], tool_packs=["utils"]))
    bundle = resolve_skills(["y"], reg)
    assert bundle.tool_packs == ("utils",)  # deduped across the dependency
    assert bundle.skills == ("x", "y")


def test_examples_carry_optional_note() -> None:
    reg = _registry()
    reg.register(
        Skill(
            name="noted",
            examples=[SkillExample(input="q", action="do z", note="careful")],
        )
    )
    out = apply_skills(AgentSpec(name="a", skills=["noted"]), reg)
    assert "(careful)" in out.description
