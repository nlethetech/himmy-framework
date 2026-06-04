"""Agents kernel: the optional Agent wrapper binding a persona to a config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from opensims.agents.personas.persona import Persona
from opensims.core.ids import new_uuid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opensims.entities.records import EntityRecord


class Agent(BaseModel):
    """A convenience wrapper binding a persona to instructions, tools, and skills.

    The runtime never requires an ``Agent`` (it takes a ``Persona`` + ``Task``);
    this exists for composing and handing around a bound configuration, e.g. in a
    multi-agent team. Projected as an entity of ``kind="agent"``.
    """

    agent_id: str = Field(default_factory=new_uuid)
    name: str
    persona: Persona
    instructions: list[str] = []
    tools: list[str] = []
    skills: list[str] = []
    metadata: dict[str, Any] = {}

    @classmethod
    def from_persona(
        cls,
        persona: Persona,
        *,
        name: str,
        tools: list[str] | None = None,
        skills: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> Agent:
        """Build an Agent from a persona, defaulting instructions to the persona's."""
        return cls(
            name=name,
            persona=persona,
            instructions=instructions
            if instructions is not None
            else list(persona.instructions),
            tools=tools or [],
            skills=skills or [],
        )

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this agent into its canonical ``EntityRecord`` (kind ``agent``)."""
        from opensims.entities.records import EntityRecord, stable_id_for

        stable_id = stable_id_for(self.agent_id, namespace="agent")
        return EntityRecord.create(
            stable_id=stable_id,
            version=version,
            kind="agent",
            payload=self.model_dump(mode="json"),
            metadata=metadata or {},
        )
