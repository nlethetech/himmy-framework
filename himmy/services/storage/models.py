"""Storage kernel: persisted record types (runs, recommendations, memory, orchestration)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.records import EntityRecord


#: The reserved workspace/subject for CLI + single-user-local (Studio) runs (T2.2).
#: A run authored by a one-shot CLI invocation or by the local single-user Studio GUI
#: belongs to no tenant; it is stamped ``__local__`` so it can still land in the ONE
#: canonical run store every surface reads, yet be EXCLUDED from cross-tenant
#: all-workspaces list views (an authenticated admin listing every tenant must not see
#: a local developer's runs leak in). A concrete query for ``__local__`` still returns
#: them (that is how the local Studio/CLI browse their own history).
LOCAL_WORKSPACE = "__local__"
LOCAL_SUBJECT = "__local__"

#: ``metadata`` key under which a run carries its rich Studio presentation payload
#: (transcript, timeline, cognition steps, per-model usage, tools, agent path). The
#: canonical :class:`RunRecord` is the single authoritative record; ``.himmy/studio.db``
#: is demoted to a run_id-keyed presentation CACHE rebuildable from this field, so a run
#: created by any surface (``/v1``, CLI, Studio) is fully renderable from the one store.
STUDIO_METADATA_KEY = "studio"


class RunStatus(str, Enum):
    """Lifecycle state of an async agent run.

    ``RESOLVING`` is the transient claim state for a HITL approve/reject: the resume
    that wins the atomic compare-and-set flips ``AWAITING_APPROVAL`` -> ``RESOLVING``
    BEFORE launching any resume work, so a second concurrent approve (a double-clicked
    Approve, two tabs, two workers on the shared run store) loses the claim and is
    refused at the inbox — never launching a second resume that could double-advance an
    orchestration graph. It mirrors the member-checkpoint ``resolving`` state and is an
    in-progress (non-terminal) state: a resume that crashes mid-flight leaves the run at
    ``RESOLVING`` so startup recovery can re-drive it exactly-once (via the member
    checkpoint claim + idempotency ledger), rather than being reaped to FAILED.
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    RESOLVING = "RESOLVING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    #: A run the leased queue gave up on after exhausting its retry budget (Q1/Q3): a
    #: terminal-but-recoverable parking state, distinct from ``FAILED`` so an operator
    #: can ``redrive`` it (reset attempt/next_attempt_at -> QUEUED) instead of the run
    #: being indistinguishable from a clean application failure. Additive: nothing emits
    #: it yet (the dispatcher in Q3 does), and the existing terminal-state checks treat it
    #: like any non-active status.
    PARKED = "PARKED"


class RecommendationStatus(str, Enum):
    """Lifecycle state of an advisory recommendation extracted from a run."""

    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    DISMISSED = "DISMISSED"
    SCHEDULED = "SCHEDULED"


class RunRecord(BaseModel):
    """The operational unit of work: an async run's lifecycle, output, and lineage.

    The trailing block of fields (``owner_id`` … ``lane_key`` + ``input_blob``) is the
    Q0/Q1 *leased work-queue* surface. It is fully additive and dormant on the zero-config
    offline path: a run created the old way leaves them at their defaults (``attempt=0``,
    ``next_attempt_at`` stamped to ``created_at`` on first save, everything else ``None``),
    so nothing changes until the leased dispatcher (Q3) starts claiming. They exist on the
    record now so the Q1 migration that adds the matching DB columns has somewhere to map.
    """

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

    # ----------------------------------------------------- leased work-queue (Q0/Q1)
    #: The worker/process that currently holds this run's lease (``claim_next_queued_run``
    #: stamps it). ``None`` when unclaimed. NOT an ownership/authorization field — purely
    #: the at-most-once dispatch lock holder.
    owner_id: str | None = None
    #: ISO-8601 instant after which an unrenewed RUNNING lease is considered expired and the
    #: run is re-queuable by the reaper. ``None`` for a never-claimed / terminal run.
    lease_expires_at: str | None = None
    #: ISO-8601 instant of the last lease heartbeat from the owning worker (liveness telemetry
    #: distinct from the hard ``lease_expires_at`` deadline).
    heartbeat_at: str | None = None
    #: How many times this run has been dispatched (incremented each claim). Drives the
    #: retry/backoff budget; ``0`` for a freshly enqueued run.
    attempt: int = 0
    #: The retry ceiling: once ``attempt`` reaches it a transient-failed run is PARKED rather
    #: than re-queued. The dispatcher (Q3) owns the policy; storage only persists the value.
    max_attempts: int = 1
    #: ISO-8601 earliest instant this run may be claimed (the backoff gate). Backfilled to
    #: ``created_at`` so legacy/never-delayed runs are immediately claimable.
    next_attempt_at: str | None = None
    #: The most recent transient error that caused a re-queue (operator-facing; the terminal
    #: error stays in ``error``).
    last_error: str | None = None
    #: The provider/probe LANE this run is claimed under (see
    #: :func:`himmy.services.storage.run_lane.lane_for_model_key`). The claim predicate filters
    #: on it so a stalled LOCAL-model probe cannot block claiming CLOUD-provider runs (Q2).
    lane_key: str | None = None
    #: The encrypted, recoverable run-input envelope (Q0): the serialized persona/task/
    #: llm_config/agent_spec/flags a fresh process needs to RE-EXECUTE a crashed QUEUED run,
    #: not merely re-record it. Built by :func:`himmy.services.storage.run_input.encode_run_input`
    #: and routed through the StorePayloadCipher at-rest seam (prompts are sensitive). ``None``
    #: for runs created without the recoverable-input path (back-compat / in-process-only
    #: recovery).
    input_blob: str | None = None


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

    def to_record(self) -> EntityRecord:
        """Project this recommendation into its canonical ``EntityRecord``.

        A first-class lineage node (``kind="recommendation"``) so a recommendation
        can be traced back to the run/persona/evidence that produced it. The payload
        captures only the immutable advisory CONTENT — ``status``/``notes`` are
        deliberately excluded so the content-addressed ``record_id`` stays stable
        across status transitions (re-registration remains idempotent).
        """
        from himmy.entities.projection import project

        return project(
            self,
            stable_value=self.recommendation_id,
            namespace="recommendation",
            kind="recommendation",
            version=1,
            payload={
                "recommendation_id": self.recommendation_id,
                "run_id": self.run_id,
                "workspace_id": self.workspace_id,
                "subject_id": self.subject_id,
                "kind": self.kind,
                "title": self.title,
                "summary": self.summary,
                "rationale": self.rationale,
                "confidence": self.confidence,
                "evidence_refs": list(self.evidence_refs),
            },
            metadata={"run_id": self.run_id, "workspace_id": self.workspace_id},
        )


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


class AgentDefRecord(BaseModel):
    """A reusable, workspace-scoped stored agent definition (T2e).

    The durable ``/v1/agents`` resource: a tenant stores a declarative
    :class:`~himmy.config.agent_spec.AgentSpec` once, then launches runs by
    ``agent_id`` instead of re-sending an inline persona each time. The spec is the
    SAME declarative shape the CLI writes to ``agent.yaml``, so the two surfaces
    interoperate (a stored ``/v1`` spec round-trips to a CLI file and back).

    ``spec`` is stored as the plain JSON projection of an ``AgentSpec`` (not the model
    instance) so the record stays a pure-data Pydantic type with no import-time
    dependency on the heavier agent stack; :meth:`agent_spec` rehydrates it on read.
    A tenant-submitted ``spec`` MUST be run through the T0.3 sanitizer at the write
    boundary BEFORE it reaches this record (the operator-only ``tools_module`` /
    ``http_tools`` / ``mcp_servers`` fields are an RCE/SSRF surface) — the storage
    layer is content-agnostic and does not sanitize.
    """

    agent_id: str = Field(default_factory=new_uuid)
    workspace_id: str
    name: str
    description: str = ""
    spec: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        *,
        workspace_id: str,
        agent_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentDefRecord:
        """Build a record from an :class:`AgentSpec` (stores its JSON projection)."""
        payload = spec.model_dump(mode="json")
        fields: dict[str, Any] = {
            "workspace_id": workspace_id,
            "name": spec.name,
            "description": spec.description,
            "spec": payload,
            "idempotency_key": idempotency_key,
            "metadata": metadata or {},
        }
        if agent_id is not None:
            fields["agent_id"] = agent_id
        return cls(**fields)

    def agent_spec(self) -> Any:
        """Rehydrate the stored JSON projection back into an :class:`AgentSpec`."""
        from himmy.config.agent_spec import AgentSpec

        return AgentSpec.model_validate(self.spec)

    def to_record(self) -> EntityRecord:
        """Project this stored agent into its canonical ``agent`` ``EntityRecord``.

        The ``stable_id`` is the workspace-scoped ``agent_id`` so a run launched by
        this agent can link to the SAME node across re-versions, and so two
        workspaces' agents with the same ``name`` never collide. The content-addressed
        record id is derived from the (sanitized) spec payload, so editing the spec
        creates a new version while the stable id stays put.
        """
        from himmy.entities.projection import project

        return project(
            self,
            stable_value=f"{self.workspace_id}:{self.agent_id}",
            namespace="agent",
            kind="agent",
            metadata={"workspace_id": self.workspace_id, "name": self.name},
        )


__all__ = [
    "LOCAL_SUBJECT",
    "LOCAL_WORKSPACE",
    "STUDIO_METADATA_KEY",
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
    "AgentDefRecord",
]
