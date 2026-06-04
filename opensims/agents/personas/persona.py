"""Agents kernel: Persona and RolePersona primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from opensims.core.ids import new_uuid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opensims.entities.records import EntityRecord


class Persona(BaseModel):
    """An agent's identity: name, description, instructions, tags, role metadata.

    Personas are versioned entities (``kind="persona"``).
    """

    agent_id: str = Field(default_factory=new_uuid)
    name: str
    description: str = ""
    instructions: list[str] = []
    tags: list[str] = []
    metadata: dict[str, Any] = {}

    @property
    def role(self) -> str:
        """The role label: ``metadata['role']`` when set, otherwise the name."""
        return self.metadata.get("role") or self.name

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this persona into its canonical ``EntityRecord``."""
        from opensims.entities.records import EntityRecord, stable_id_for

        stable_id = stable_id_for(
            self.agent_id, namespace="persona", fallback_key=self.name
        )
        return EntityRecord.create(
            stable_id=stable_id,
            version=version,
            kind="persona",
            payload=self.model_dump(mode="json"),
            metadata=metadata or {},
        )


class RolePersona(Persona):
    """A Persona enriched with role/policy modeling fields (used by pack resolvers)."""

    objectives: list[str] = []
    required_tools: list[str] = []
    required_skills: list[str] = []
