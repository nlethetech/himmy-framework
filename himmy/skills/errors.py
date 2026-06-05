"""Errors raised while registering or resolving skills."""

from __future__ import annotations

from himmy.core import HimmyError


class UnknownSkillError(HimmyError):
    """A requested skill name is not in the registry (carries a did-you-mean hint)."""


class CyclicSkillError(HimmyError):
    """``requires_skills`` forms a cycle (the message names the offending path)."""


class SkillToolError(HimmyError):
    """A skill references a tool that is not available in the bound tool registry."""


class SkillCollisionError(HimmyError):
    """Registering a skill whose name already exists without ``overwrite=True``."""


__all__ = [
    "UnknownSkillError",
    "CyclicSkillError",
    "SkillToolError",
    "SkillCollisionError",
]
