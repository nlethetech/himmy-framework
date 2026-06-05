"""Tier 0: the Skill model, registry, resolution, and entity projection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from himmy.entities.registry import EntityRegistry
from himmy.skills import (
    BUILTIN_SKILLS,
    CyclicSkillError,
    Skill,
    SkillCollisionError,
    SkillRegistry,
    UnknownSkillError,
    resolve_skills,
)

# --- model ---------------------------------------------------------------------------


def test_skill_rejects_unknown_field() -> None:
    # extra="forbid" — a typo'd field fails loudly instead of being ignored.
    with pytest.raises(ValidationError):
        Skill.model_validate({"name": "x", "instrutions": ["oops"]})


def test_skill_requires_nonempty_name() -> None:
    with pytest.raises(ValidationError):
        Skill(name="")


def test_skill_to_record_is_deterministic_kind_skill() -> None:
    skill = Skill(name="web_research", tool_packs=["web"])
    rec = skill.to_record()
    assert rec.kind == "skill"
    assert rec.payload["tool_packs"] == ["web"]
    # Deterministic record id from (kind, stable_id, version).
    assert skill.to_record().record_id == rec.record_id


# --- registry ------------------------------------------------------------------------


def test_registry_register_get_list() -> None:
    reg = SkillRegistry()
    reg.register(Skill(name="alpha"))
    assert reg.get("alpha") is not None
    assert "alpha" in reg
    assert reg.names() == ["alpha"]


def test_registry_collision_requires_overwrite() -> None:
    reg = SkillRegistry()
    reg.register(Skill(name="alpha", description="first"))
    with pytest.raises(SkillCollisionError):
        reg.register(Skill(name="alpha", description="second"))
    reg.register(Skill(name="alpha", description="second"), overwrite=True)
    assert reg.get("alpha").description == "second"


def test_with_builtins_loads_catalog() -> None:
    reg = SkillRegistry.with_builtins()
    assert set(BUILTIN_SKILLS).issubset(set(reg.names()))
    assert reg.get("web_research").tool_packs == ["web"]


def test_registry_projects_to_entity_registry() -> None:
    entities = EntityRegistry()
    reg = SkillRegistry(entity_registry=entities)
    reg.register(Skill(name="alpha", tool_packs=["web"]))
    records = entities.list_by_kind("skill")
    assert len(records) == 1
    assert records[0].payload["name"] == "alpha"


# --- resolution ----------------------------------------------------------------------


def test_resolve_aggregates_packs_and_blocks() -> None:
    reg = SkillRegistry.with_builtins()
    bundle = resolve_skills(["web_research"], reg)
    assert bundle.tool_packs == ("web",)
    assert bundle.skills == ("web_research",)
    assert any("Skill — web_research" in b for b in bundle.instruction_blocks)
    assert bundle.when_to_use  # carries the routing hint


def test_resolve_dedups_overlapping_contributions() -> None:
    reg = SkillRegistry()
    reg.register(Skill(name="a", tools=["t1", "t2"], guardrails=["g"]))
    reg.register(Skill(name="b", tools=["t2", "t3"], guardrails=["g"]))
    bundle = resolve_skills(["a", "b"], reg)
    assert bundle.tools == ("t1", "t2", "t3")  # order-stable, deduped
    assert bundle.guardrails == ("g",)


def test_resolve_expands_requires_before_dependent() -> None:
    reg = SkillRegistry()
    reg.register(Skill(name="base", tools=["t_base"]))
    reg.register(Skill(name="top", tools=["t_top"], requires_skills=["base"]))
    bundle = resolve_skills(["top"], reg)
    # Prerequisite applied first; both tools present.
    assert bundle.skills == ("base", "top")
    assert bundle.tools == ("t_base", "t_top")


def test_resolve_detects_cycles() -> None:
    reg = SkillRegistry()
    reg.register(Skill(name="a", requires_skills=["b"]))
    reg.register(Skill(name="b", requires_skills=["a"]))
    with pytest.raises(CyclicSkillError) as exc:
        resolve_skills(["a"], reg)
    assert "a -> b -> a" in str(exc.value)


def test_resolve_unknown_skill_suggests_closest() -> None:
    reg = SkillRegistry.with_builtins()
    with pytest.raises(UnknownSkillError) as exc:
        resolve_skills(["web_reserch"], reg)  # typo
    assert "did you mean 'web_research'" in str(exc.value)
