"""Resolve skill names into a single, order-stable bundle for an agent.

:func:`resolve_skills` walks the requested skills (depth-first so a skill's
``requires_skills`` prerequisites are applied *before* it), detects cycles, dedups every
contribution preserving first-seen order, and returns a :class:`ResolvedSkills`. The
result is pure data: Tier 1 turns it into appended persona instructions + bound tools +
merged guardrails. Tool *existence* is validated separately, where a live tool registry
is available (a skill can name a tool that only exists once its pack is registered).
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from himmy.skills.errors import CyclicSkillError, UnknownSkillError
from himmy.skills.models import Skill, SkillExample
from himmy.skills.registry import SkillRegistry


@dataclass(frozen=True)
class ResolvedSkills:
    """The aggregate effect of a set of skills, ready to apply to an agent."""

    skills: tuple[str, ...]  # contributing skill names, in application order
    instruction_blocks: tuple[str, ...]  # one labeled know-how block per skill
    tools: tuple[str, ...]  # tools to bind (deduped, order-stable)
    tool_packs: tuple[str, ...]  # packs to register (deduped, order-stable)
    guardrails: tuple[str, ...]  # guardrails to merge (deduped, order-stable)
    examples: tuple[SkillExample, ...]  # few-shot examples (Tier 3)
    when_to_use: tuple[str, ...]  # routing hints (Tier 3)

    @property
    def is_empty(self) -> bool:
        return not self.skills


def _format_block(skill: Skill) -> str:
    """A single labeled know-how block for one skill (header + when + instructions)."""
    header = f"Skill — {skill.name}"
    if skill.description:
        header += f": {skill.description}"
    lines = [header]
    if skill.when_to_use:
        lines.append(f"Use this when {skill.when_to_use}")
    lines.extend(f"- {line}" for line in skill.instructions)
    return "\n".join(lines)


def format_examples(examples: Sequence[SkillExample]) -> str:
    """Render skill examples as a compact few-shot block for the system prompt."""
    lines = ["Worked examples:"]
    for ex in examples:
        note = f" ({ex.note})" if ex.note else ""
        lines.append(f'- When asked: "{ex.input}" → {ex.action}{note}')
    return "\n".join(lines)


def _dedup(seq: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication."""
    return tuple(dict.fromkeys(s for s in seq if s))


def _ordered_skills(names: Sequence[str], registry: SkillRegistry) -> list[Skill]:
    """Depth-first expansion of ``names`` + their prerequisites, cycle-guarded."""
    ordered: list[Skill] = []
    done: set[str] = set()

    def visit(name: str, path: tuple[str, ...]) -> None:
        if name in path:
            raise CyclicSkillError(
                "cyclic skill dependency: " + " -> ".join((*path, name))
            )
        if name in done:
            return
        skill = registry.get(name)
        if skill is None:
            known = registry.names()
            close = difflib.get_close_matches(name, known, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise UnknownSkillError(
                f"unknown skill {name!r}{hint} "
                f"(known: {', '.join(known) if known else '(none)'})"
            )
        for dep in skill.requires_skills:
            visit(dep, (*path, name))
        done.add(name)
        ordered.append(skill)

    for n in names:
        visit(n, ())
    return ordered


def resolve_skills(names: Sequence[str], registry: SkillRegistry) -> ResolvedSkills:
    """Resolve ``names`` against ``registry`` into one applied bundle."""
    ordered = _ordered_skills(names, registry)
    blocks: list[str] = []
    tools: list[str] = []
    packs: list[str] = []
    guardrails: list[str] = []
    examples: list[SkillExample] = []
    when: list[str] = []
    for skill in ordered:
        if skill.instructions or skill.description:
            blocks.append(_format_block(skill))
        tools.extend(skill.tools)
        packs.extend(skill.tool_packs)
        guardrails.extend(skill.guardrails)
        examples.extend(skill.examples)
        if skill.when_to_use:
            when.append(f"{skill.name}: {skill.when_to_use}")
    return ResolvedSkills(
        skills=tuple(s.name for s in ordered),
        instruction_blocks=tuple(blocks),
        tools=_dedup(tools),
        tool_packs=_dedup(packs),
        guardrails=_dedup(guardrails),
        examples=tuple(examples),
        when_to_use=tuple(when),
    )


__all__ = ["ResolvedSkills", "resolve_skills"]
