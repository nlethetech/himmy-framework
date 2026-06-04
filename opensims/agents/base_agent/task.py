"""Agents kernel: the Task primitive (one unit of work)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from opensims.core.ids import new_uuid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opensims.entities.records import EntityRecord


class Task(BaseModel):
    """A single unit of work: a prompt plus structured context and constraints.

    ``context`` is the runtime's lever — recognized keys (``model_key``,
    ``tool_names``, ``output_schema``, ``context_build_spec``, etc.) steer the run.
    Tasks are projected as entities of ``kind="prompt"``.
    """

    task_id: str = Field(default_factory=new_uuid)
    title: str
    prompt: str
    context: dict[str, Any] = {}
    constraints: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this task into its canonical ``EntityRecord`` (kind ``prompt``)."""
        from opensims.entities.records import EntityRecord, stable_id_for

        stable_id = stable_id_for(self.task_id, namespace="prompt")
        return EntityRecord.create(
            stable_id=stable_id,
            version=version,
            kind="prompt",
            payload=self.model_dump(mode="json"),
            metadata=metadata or {},
        )
