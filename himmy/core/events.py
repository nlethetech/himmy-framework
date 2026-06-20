"""Core kernel: run-lifecycle event types and the event sink protocol."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.entities.records import EntityRecord


class EventType(str, Enum):
    """The closed set of run-lifecycle events emitted across the framework."""

    AGENT_RUN_STARTED = "AGENT_RUN_STARTED"
    AGENT_RUN_FINISHED = "AGENT_RUN_FINISHED"
    AGENT_TURN_STARTED = "AGENT_TURN_STARTED"
    AGENT_TURN_COMPLETED = "AGENT_TURN_COMPLETED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVAL_GRANTED = "APPROVAL_GRANTED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    INFERENCE_REQUESTED = "INFERENCE_REQUESTED"
    INFERENCE_SUCCEEDED = "INFERENCE_SUCCEEDED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_FAILED = "TOOL_FAILED"
    CONTEXT_SNAPSHOT_BUILT = "CONTEXT_SNAPSHOT_BUILT"
    CONTEXT_COMPACTED = "CONTEXT_COMPACTED"
    GUARDRAIL_APPLIED = "GUARDRAIL_APPLIED"
    WORKFLOW_STARTED = "WORKFLOW_STARTED"
    WORKFLOW_STEP_COMPLETED = "WORKFLOW_STEP_COMPLETED"
    WORKFLOW_FINISHED = "WORKFLOW_FINISHED"
    AGENT_HANDOFF = "AGENT_HANDOFF"
    AGENT_DELEGATED = "AGENT_DELEGATED"
    GROUP_CHAT_STARTED = "GROUP_CHAT_STARTED"
    GROUP_SPEAKER_SELECTED = "GROUP_SPEAKER_SELECTED"
    GROUP_SPEAKER_SPOKE = "GROUP_SPEAKER_SPOKE"
    GROUP_CHAT_FINISHED = "GROUP_CHAT_FINISHED"
    FANOUT_STARTED = "FANOUT_STARTED"
    FANOUT_WORKER_COMPLETED = "FANOUT_WORKER_COMPLETED"
    FANOUT_JOINED = "FANOUT_JOINED"
    MEMORY_REMEMBERED = "MEMORY_REMEMBERED"
    MEMORY_RECALLED = "MEMORY_RECALLED"
    MEMORY_CONSOLIDATED = "MEMORY_CONSOLIDATED"
    GRAPH_STARTED = "GRAPH_STARTED"
    GRAPH_NODE_STARTED = "GRAPH_NODE_STARTED"
    GRAPH_NODE_COMPLETED = "GRAPH_NODE_COMPLETED"
    GRAPH_EDGE_TAKEN = "GRAPH_EDGE_TAKEN"
    GRAPH_CHECKPOINTED = "GRAPH_CHECKPOINTED"
    GRAPH_FINISHED = "GRAPH_FINISHED"
    TYPED_OUTPUT_VALIDATED = "TYPED_OUTPUT_VALIDATED"
    TYPED_OUTPUT_REPAIRED = "TYPED_OUTPUT_REPAIRED"
    # Self-learning (P1): a learned signal changed behaviour — tool reputation reordered
    # the bound tools (emitted once at snapshot-refresh time, when the runtime is built)
    # and/or a learned reliability hint was injected (emitted per-run, with the run's
    # trace context). Emitted best-effort so the behavioural change is auditable.
    LEARNING_APPLIED = "LEARNING_APPLIED"
    # Self-learning (P2, outcome quality): an OUTCOME score in [0, 1] was attached to a
    # run (and, when a single tool dominated the run, to that tool). Unlike TOOL_COMPLETED
    # / TOOL_FAILED — which only say a tool RAN — this says whether the produced answer was
    # GOOD, sourced from the (same-model-guarded) LLM judge, a deterministic grader hook,
    # explicit user feedback, or a trajectory grade. Carries ``payload['tool_name']`` (the
    # same denormalised column the reputation miner already filters on) so an outcome stat
    # blends into ToolReputation with no schema change. Emitted best-effort, opt-in.
    OUTCOME_SCORED = "OUTCOME_SCORED"


class RunEvent(BaseModel):
    """A single observable event in a run's lifecycle.

    Events are first-class entities: ``to_record`` projects them into the
    registry so a run can be replayed and audited after the fact.
    """

    event_id: str = Field(default_factory=new_uuid)
    event_type: EventType
    trace_id: str | None = None
    thread_id: str | None = None
    agent_id: str | None = None
    #: The owning tenant of the run that emitted this event. Stamped only when a run is
    #: wired with a workspace scope (the per-run server path); ``None`` on the one-shot
    #: CLI / offline path, which uses an isolated in-memory store per process and needs no
    #: tenant dimension. It is denormalised into an indexed ``workspace_id`` column on the
    #: durable backends (mirroring ``tool_name``) so the self-learning reputation miner can
    #: scope its read to ONE tenant on a shared event store instead of aggregating every
    #: tenant's tool failures. ``None`` keeps the unscoped historical read byte-identical.
    workspace_id: str | None = None
    request_id: str | None = None
    tool_call_id: str | None = None
    latency_ms: float | None = None
    cost: float | None = None
    payload: dict[str, Any] = {}
    error: str | None = None
    timestamp: str = Field(default_factory=utc_now_iso)

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this event into its canonical ``EntityRecord`` (kind ``run_event``)."""
        # Imported lazily to avoid a core <-> entities import cycle.
        from himmy.entities.projection import project

        return project(
            self,
            stable_value=self.event_id,
            namespace="run_event",
            kind="run_event",
            version=version,
            metadata=metadata,
        )


@runtime_checkable
class EventSink(Protocol):
    """Anything able to durably accept ``RunEvent``s."""

    async def append_event(self, event: RunEvent) -> None:
        """Append a single event to the sink."""
        ...
