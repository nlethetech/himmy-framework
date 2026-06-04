"""Storage kernel: persisted record types (runs, recommendations, memory, orchestration)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from opensims.core.ids import new_uuid, utc_now_iso


class RunStatus(str, Enum):
    """Lifecycle state of an async agent run."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class RecommendationStatus(str, Enum):
    """Lifecycle state of an advisory recommendation extracted from a run."""

    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    DISMISSED = "DISMISSED"
    SCHEDULED = "SCHEDULED"


class RunRecord(BaseModel):
    """The operational unit of work: an async run's lifecycle, output, and lineage."""

    run_id: str = Field(default_factory=new_uuid)
    workspace_id: str
    subject_id: str
    task_id: str | None = None
    thread_id: str | None = None
    snapshot_id: str | None = None
    persona_name: str | None = None
    model_key: str | None = None
    idempotency_key: str | None = None
    status: RunStatus = RunStatus.QUEUED
    output_text: str | None = None
    output_structured: Any = None
    error: str | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class RecommendationItem(BaseModel):
    """A dashboard-facing advisory output, extracted from a run's structured output."""

    recommendation_id: str = Field(default_factory=new_uuid)
    run_id: str
    workspace_id: str
    subject_id: str
    kind: str
    title: str
    summary: str = ""
    rationale: str = ""
    confidence: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)
    status: RecommendationStatus = RecommendationStatus.PROPOSED
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class MemoryObject(BaseModel):
    """A cognitive (long-lived) memory item scoped to a subject/agent."""

    memory_id: str = Field(default_factory=new_uuid)
    subject_id: str | None = None
    agent_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class EpisodicMemoryObject(BaseModel):
    """An episodic memory item — a recalled event or interaction trace."""

    episode_id: str = Field(default_factory=new_uuid)
    subject_id: str | None = None
    agent_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class AgentStateRecord(BaseModel):
    """A snapshot of an agent's internal state at a point in an orchestration."""

    state_id: str = Field(default_factory=new_uuid)
    agent_id: str | None = None
    environment_name: str | None = None
    round: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class ActionRecord(BaseModel):
    """A single action taken by an agent within an environment/round."""

    action_id: str = Field(default_factory=new_uuid)
    agent_id: str | None = None
    environment_name: str | None = None
    round: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class EnvironmentStateRecord(BaseModel):
    """A snapshot of a shared environment's state at a given round."""

    environment_state_id: str = Field(default_factory=new_uuid)
    environment_name: str | None = None
    round: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class ContextEvidenceRecord(BaseModel):
    """A persisted pointer to where a context value originated (an EvidenceRef projection)."""

    evidence_id: str = Field(default_factory=new_uuid)
    subject_id: str | None = None
    snapshot_id: str | None = None
    key: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


__all__ = [
    "RunStatus",
    "RecommendationStatus",
    "RunRecord",
    "RecommendationItem",
    "MemoryObject",
    "EpisodicMemoryObject",
    "AgentStateRecord",
    "ActionRecord",
    "EnvironmentStateRecord",
    "ContextEvidenceRecord",
]
