"""Context kernel: snapshot/field/evidence types and the build-spec recipe."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from opensims.core.ids import new_uuid, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from opensims.entities.records import EntityRecord


class ContextSourcePreference(str, Enum):
    """Where a key's value should be sourced from when building a snapshot."""

    STORAGE_FIRST = "storage_first"
    TOOL_FIRST = "tool_first"
    TOOL_ONLY = "tool_only"


class EvidenceRef(BaseModel):
    """A pointer to where a context value came from — a row, tool result, or chunk."""

    evidence_id: str = Field(default_factory=new_uuid)
    source_type: str
    source_id: str | None = None
    row_id: str | None = None
    account_scope: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class ContextField(BaseModel):
    """One typed key-value with confidence, freshness, source, and evidence."""

    key: str
    value: Any = None
    confidence: float = 1.0
    freshness_seconds: float | None = None
    source: str = "storage"
    evidence_refs: list[EvidenceRef] = []
    metadata: dict[str, Any] = {}


class ContextSpecKey(BaseModel):
    """One entry in a build spec: key, required-ness, source preference, adapter."""

    key: str
    required: bool = False
    source_preference: ContextSourcePreference = ContextSourcePreference.STORAGE_FIRST
    adapter_name: str | None = None
    metadata: dict[str, Any] = {}


class ContextBuildSpec(BaseModel):
    """A declarative recipe: which keys to fetch, from where, with what preference."""

    spec_id: str = Field(default_factory=new_uuid)
    keys: list[ContextSpecKey] = []
    metadata: dict[str, Any] = {}


class ContextSnapshot(BaseModel):
    """A frozen, evidenced context result. First-class entity (``kind="context_snapshot"``).

    Two runs with the same snapshot produce comparable answers — the basis for
    reproducibility, evaluation, and audit.
    """

    snapshot_id: str = Field(default_factory=new_uuid)
    subject_id: str
    task_id: str | None = None
    fields: dict[str, ContextField] = {}
    missing_required_keys: list[str] = []
    metadata: dict[str, Any] = {}
    created_at: str = Field(default_factory=utc_now_iso)

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this snapshot into its canonical ``EntityRecord``."""
        from opensims.entities.records import EntityRecord, stable_id_for

        stable_id = stable_id_for(self.snapshot_id, namespace="context_snapshot")
        return EntityRecord.create(
            stable_id=stable_id,
            version=version,
            kind="context_snapshot",
            payload=self.model_dump(mode="json"),
            metadata=metadata or {},
        )


__all__ = [
    "ContextSourcePreference",
    "EvidenceRef",
    "ContextField",
    "ContextSpecKey",
    "ContextBuildSpec",
    "ContextSnapshot",
]
