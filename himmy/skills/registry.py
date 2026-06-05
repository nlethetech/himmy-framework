"""The :class:`SkillRegistry` — an in-memory catalog of skills, name → :class:`Skill`.

Mirrors :class:`~himmy.services.tools.registry.ToolRegistry`: an explicit, composable
container (not a global singleton). Pass an optional ``entity_registry`` to auto-project
every registered skill into the entity spine as a ``kind="skill"`` record.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from himmy.skills.errors import SkillCollisionError
from himmy.skills.models import Skill

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.registry import EntityRegistry


class SkillRegistry:
    """A catalog of :class:`Skill`s keyed by name."""

    def __init__(self, *, entity_registry: EntityRegistry | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._entity_registry = entity_registry

    def register(self, skill: Skill, *, overwrite: bool = False) -> Skill:
        """Add ``skill`` to the catalog.

        Raises :class:`SkillCollisionError` on a duplicate name unless ``overwrite`` is
        set (the loader uses ``overwrite=True`` for project-local skills that
        intentionally shadow built-ins). Projects to the entity registry when wired.
        """
        if not overwrite and skill.name in self._skills:
            raise SkillCollisionError(
                f"skill {skill.name!r} is already registered "
                f"(pass overwrite=True to replace it)"
            )
        self._skills[skill.name] = skill
        if self._entity_registry is not None:
            self._entity_registry.register(skill.to_record())
        return skill

    def get(self, name: str) -> Skill | None:
        """Return the skill named ``name``, or ``None`` if absent."""
        return self._skills.get(name)

    def list(self) -> builtins.list[Skill]:
        """All registered skills, ordered by name."""
        return [self._skills[n] for n in sorted(self._skills)]

    def names(self) -> builtins.list[str]:
        """All registered skill names (sorted)."""
        return sorted(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    @classmethod
    def with_builtins(
        cls, *, entity_registry: EntityRegistry | None = None
    ) -> SkillRegistry:
        """A registry pre-loaded with the built-in skill catalog."""
        from himmy.skills.builtin import BUILTIN_SKILLS

        registry = cls(entity_registry=entity_registry)
        for skill in BUILTIN_SKILLS.values():
            registry.register(skill)
        return registry


__all__ = ["SkillRegistry"]
