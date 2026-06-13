"""Author skills as YAML and discover them from the project.

A skill file is just the :class:`Skill` model serialized to YAML::

    # skills/web_research.yaml
    name: web_research
    description: Find and synthesize facts from the open web, with citations.
    tool_packs: [web]
    instructions:
      - Search before answering; never guess a fact you can look up.
      - Cite the URL(s) you used.

:func:`build_skill_registry` returns the built-in catalog with any project-local skills
(``skills/*.yaml`` under the working dir, plus ``HIMMY_SKILLS_PATH``) overlaid on top — a
project file named like a built-in intentionally **shadows** it (last-writer-wins, logged).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from himmy.core import HimmyError
from himmy.skills.models import Skill
from himmy.skills.registry import SkillRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.protocol import EntityRegistryProtocol

_LOG = logging.getLogger("himmy.skills")
_SKILL_SUFFIXES = (".yaml", ".yml")


class SkillFileError(HimmyError):
    """A skill YAML file is malformed or fails validation."""


def load_skill_file(path: str | Path) -> Skill:
    """Load and validate a single skill from a YAML file.

    The ``name`` defaults to the file stem when the file omits it, so a tidily-named
    ``web_research.yaml`` need not repeat ``name: web_research``.
    """
    p = Path(path).expanduser()
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SkillFileError(f"cannot read skill file {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SkillFileError(f"skill file {p} must be a YAML mapping")
    raw.setdefault("name", p.stem)
    try:
        return Skill.from_dict(raw)
    except Exception as exc:  # pydantic ValidationError → a clear, located error
        raise SkillFileError(f"invalid skill in {p}: {exc}") from exc


def load_skill_dir(path: str | Path) -> list[Skill]:
    """Load every ``*.yaml`` / ``*.yml`` skill in a directory (sorted, non-recursive)."""
    d = Path(path).expanduser()
    if not d.is_dir():
        return []
    files = sorted(f for f in d.iterdir() if f.suffix in _SKILL_SUFFIXES)
    return [load_skill_file(f) for f in files]


def discover_skill_dirs(extra: list[str] | None = None) -> list[Path]:
    """Directories to scan for project skills: ``./skills`` + ``HIMMY_SKILLS_PATH``.

    ``HIMMY_SKILLS_PATH`` is an ``os.pathsep``-separated list (like ``PATH``). Order is
    later-wins, so an explicitly-pointed directory overrides the conventional one.
    """
    dirs: list[Path] = [Path.cwd() / "skills"]
    env = os.environ.get("HIMMY_SKILLS_PATH", "")
    dirs.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p)
    dirs.extend(Path(p).expanduser() for p in (extra or []))
    # De-dup while preserving order.
    seen: dict[Path, None] = {}
    for d in dirs:
        seen.setdefault(d.resolve() if d.exists() else d, None)
    return list(seen)


def build_skill_registry(
    *,
    entity_registry: EntityRegistryProtocol | None = None,
    extra_dirs: list[str] | None = None,
    include_builtins: bool = True,
) -> SkillRegistry:
    """Built-in catalog with project-local skills overlaid (project shadows built-in)."""
    registry = SkillRegistry(entity_registry=entity_registry)
    if include_builtins:
        from himmy.skills.builtin import BUILTIN_SKILLS

        for skill in BUILTIN_SKILLS.values():
            registry.register(skill)
    for d in discover_skill_dirs(extra_dirs):
        for skill in load_skill_dir(d):
            if skill.name in registry:
                _LOG.info("project skill %r shadows a built-in", skill.name)
            registry.register(skill, overwrite=True)
    return registry


__all__ = [
    "SkillFileError",
    "load_skill_file",
    "load_skill_dir",
    "discover_skill_dirs",
    "build_skill_registry",
]
