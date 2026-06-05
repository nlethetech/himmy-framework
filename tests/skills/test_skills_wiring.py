"""Tier 1: AgentSpec.skills expansion (apply_skills) and runtime tool validation."""

from __future__ import annotations

import pytest

from himmy.config.agent_spec import AgentSpec, apply_skills
from himmy.skills import Skill, SkillRegistry, UnknownSkillError


def _registry() -> SkillRegistry:
    return SkillRegistry.with_builtins()


def test_apply_skills_noop_without_skills() -> None:
    spec = AgentSpec(name="a", tool_packs=["web"])
    assert apply_skills(spec, _registry()) is spec


def test_apply_skills_unions_packs_and_injects_know_how() -> None:
    spec = AgentSpec(name="a", description="Base.", skills=["web_research"])
    out = apply_skills(spec, _registry())
    # skills consumed, pack unioned, know-how folded into the (rendered) description.
    assert out.skills == []
    assert out.tool_packs == ["web"]
    assert "Base." in out.description
    assert "Skill — web_research" in out.description
    # Skill names recorded where the prompt renderer reads them.
    assert out.metadata["skills"] == ["web_research"]
    # And they survive into the persona that drives the prompt.
    assert out.to_persona().metadata["skills"] == ["web_research"]


def test_apply_skills_dedups_against_existing_packs_and_guardrails() -> None:
    reg = _registry()
    reg.register(Skill(name="emailer", tool_packs=["comms"], guardrails=["pii"]))
    spec = AgentSpec(
        name="a", tool_packs=["comms"], guardrails=["pii"], skills=["emailer"]
    )
    out = apply_skills(spec, reg)
    assert out.tool_packs == ["comms"]  # not duplicated
    assert out.guardrails == ["pii"]


def test_apply_skills_records_explicit_tools_for_validation() -> None:
    reg = _registry()
    reg.register(Skill(name="custom", tools=["my_tool"]))
    out = apply_skills(AgentSpec(name="a", skills=["custom"]), reg)
    assert out.tools == ["my_tool"]
    assert out.metadata["resolved_skill_tools"] == ["my_tool"]


def test_apply_skills_propagates_unknown_skill() -> None:
    with pytest.raises(UnknownSkillError):
        apply_skills(AgentSpec(name="a", skills=["does_not_exist"]), _registry())


def test_spec_yaml_accepts_skills_field(tmp_path) -> None:
    from himmy.config.agent_spec import load_agent_spec

    p = tmp_path / "agent.yaml"
    p.write_text("name: a\nskills: [web_research]\n")
    spec = load_agent_spec(str(p))
    assert spec.skills == ["web_research"]
