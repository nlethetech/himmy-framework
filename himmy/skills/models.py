"""The :class:`Skill` model — a first-class, typed agent capability.

A skill is the capability layer that sits *between* tools and agents. A tool is a
callable; a skill is the *know-how to wield a set of tools toward a job*: a prompt
fragment (``instructions``) bundled with the ``tools`` / ``tool_packs`` it needs and any
``guardrails`` it implies. Skills are declared in YAML, resolved into an agent (which
binds their tools and injects their guidance), and projected to a versioned
``EntityRecord(kind="skill")`` like every other domain artefact.

The model uses ``extra="forbid"`` so a typo'd field in a hand-written skill YAML fails
loudly at load time rather than being silently ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.records import EntityRecord


class SkillExample(BaseModel):
    """A single few-shot example for a skill (rendered into the prompt in Tier 3)."""

    model_config = ConfigDict(extra="forbid")

    input: str
    action: str
    note: str = ""


class Skill(BaseModel):
    """A named, versioned capability: instructions (know-how) + the tools it needs.

    Resolving a skill into an agent appends its ``instructions`` to the system prompt,
    unions its ``tools``/``tool_packs`` into the bound tool set, and merges its
    ``guardrails`` — so declaring ``skills: [web_research]`` grants both the capability's
    tools *and* the guidance for using them well.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = 1
    description: str = ""
    instructions: list[str] = []
    tools: list[str] = []
    tool_packs: list[str] = []
    guardrails: list[str] = []
    when_to_use: str = ""  # routing hint (feeds the tool router in Tier 3)
    examples: list[SkillExample] = []
    requires_skills: list[str] = []  # composition; cycle-guarded at resolve time
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Skill:
        """Validate a plain mapping (e.g. parsed YAML) into a :class:`Skill`."""
        return cls.model_validate(raw)

    def to_record(
        self, version: int | None = None, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this skill into its canonical ``EntityRecord`` (``kind="skill"``)."""
        from himmy.entities.projection import project

        return project(
            self,
            stable_value=self.name,
            namespace="skill",
            kind="skill",
            version=version if version is not None else self.version,
            metadata=metadata,
        )


__all__ = ["Skill", "SkillExample"]
