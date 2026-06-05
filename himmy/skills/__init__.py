"""Himmy skills: first-class, typed agent capabilities (know-how + tools + guardrails).

A :class:`Skill` bundles a prompt fragment with the tools it needs; resolving skills into
an agent binds those tools and injects the guidance. Skills are validated, versioned
(``EntityRecord(kind="skill")``), and authored as small YAML files or registered in code::

    from himmy.skills import SkillRegistry, resolve_skills

    registry = SkillRegistry.with_builtins()
    bundle = resolve_skills(["web_research"], registry)
    # bundle.tool_packs -> ("web",); bundle.instruction_blocks -> the know-how
"""

from __future__ import annotations

from himmy.skills.builtin import BUILTIN_SKILLS
from himmy.skills.errors import (
    CyclicSkillError,
    SkillCollisionError,
    SkillToolError,
    UnknownSkillError,
)
from himmy.skills.models import Skill, SkillExample
from himmy.skills.registry import SkillRegistry
from himmy.skills.resolve import ResolvedSkills, resolve_skills

__all__ = [
    "Skill",
    "SkillExample",
    "SkillRegistry",
    "BUILTIN_SKILLS",
    "ResolvedSkills",
    "resolve_skills",
    "UnknownSkillError",
    "CyclicSkillError",
    "SkillToolError",
    "SkillCollisionError",
]
