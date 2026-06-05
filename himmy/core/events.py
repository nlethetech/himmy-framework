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
    INFERENCE_REQUESTED = "INFERENCE_REQUESTED"
    INFERENCE_SUCCEEDED = "INFERENCE_SUCCEEDED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_FAILED = "TOOL_FAILED"
    CONTEXT_SNAPSHOT_BUILT = "CONTEXT_SNAPSHOT_BUILT"
    WORKFLOW_STARTED = "WORKFLOW_STARTED"
    WORKFLOW_STEP_COMPLETED = "WORKFLOW_STEP_COMPLETED"
    WORKFLOW_FINISHED = "WORKFLOW_FINISHED"


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
        from himmy.entities.records import EntityRecord, stable_id_for

        stable_id = stable_id_for(self.event_id, namespace="run_event")
        return EntityRecord.create(
            stable_id=stable_id,
            version=version,
            kind="run_event",
            payload=self.model_dump(mode="json"),
            metadata=metadata or {},
        )


@runtime_checkable
class EventSink(Protocol):
    """Anything able to durably accept ``RunEvent``s."""

    async def append_event(self, event: RunEvent) -> None:
        """Append a single event to the sink."""
        ...
