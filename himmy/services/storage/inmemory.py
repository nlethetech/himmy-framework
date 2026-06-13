"""Storage kernel: focused in-memory store implementations.

Each class owns exactly one storage concern with process-local dicts and an async
API. :class:`~himmy.services.storage.service.StorageService` composes these behind
a single backward-compatible facade. Splitting them out keeps each concern small,
independently testable, and swappable (e.g. a Redis-backed ``InMemoryRunStore``
replacement) without touching the others.

These mirror the per-concern Postgres methods 1:1 and satisfy the matching
protocols in :mod:`himmy.services.storage.protocols`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from himmy.core.events import RunEvent
from himmy.core.ids import utc_now_iso
from himmy.services.storage.models import (
    ActionRecord,
    AgentStateRecord,
    EnvironmentStateRecord,
    EpisodicMemoryObject,
    MemoryObject,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids storage <-> context cycle
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.services.context.models import ContextField, ContextSnapshot
    from himmy.services.evaluation.models import EvaluationRun


class InMemoryThreadStore:
    """Process-local chat-thread persistence keyed by ``thread_id``."""

    def __init__(self) -> None:
        self._threads: dict[str, Any] = {}

    async def save_thread(self, thread: ChatThread) -> ChatThread:
        """Upsert a chat thread keyed by ``thread_id``."""
        self._threads[thread.thread_id] = thread
        return thread

    async def load_thread(self, thread_id: str) -> ChatThread | None:
        """Return a stored chat thread by id, or None."""
        return self._threads.get(thread_id)


class InMemoryEventLog:
    """Process-local append-only run-event stream (EventSink surface)."""

    def __init__(self) -> None:
        self._events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        """Append a run event to the canonical audit stream (EventSink)."""
        self._events.append(event)

    async def list_events(
        self, thread_id: str | None = None, trace_id: str | None = None
    ) -> list[RunEvent]:
        """List events, optionally filtered by ``thread_id`` and/or ``trace_id``."""
        return [
            e
            for e in self._events
            if (thread_id is None or e.thread_id == thread_id)
            and (trace_id is None or e.trace_id == trace_id)
        ]


class InMemoryContextStore:
    """Process-local context fields, snapshots, and evidence."""

    def __init__(self) -> None:
        # context_fields keyed by (subject_id, key)
        self._context_fields: dict[tuple[str, str], Any] = {}
        self._snapshots: dict[str, Any] = {}
        self._evidence: list[Any] = []

    async def save_context_field(self, field: ContextField) -> ContextField:
        """Upsert a context field keyed by ``(subject_id, key)``.

        ``subject_id`` is read from the field metadata when present, falling back
        to a blank scope so storage-only fields still round-trip.
        """
        subject_id = str(getattr(field, "metadata", {}).get("subject_id", ""))
        self._context_fields[(subject_id, field.key)] = field
        return field

    async def get_context_field(self, subject_id: str, key: str) -> ContextField | None:
        """Return the context field for ``(subject_id, key)``, or None."""
        return self._context_fields.get((subject_id, key))

    async def list_context_fields(self, subject_id: str) -> list[ContextField]:
        """Return all context fields for a subject."""
        return [
            field
            for (sid, _), field in self._context_fields.items()
            if sid == subject_id
        ]

    async def save_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        """Upsert a context snapshot keyed by ``snapshot_id``."""
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    async def load_snapshot(self, snapshot_id: str) -> ContextSnapshot | None:
        """Return a stored snapshot by id, or None."""
        return self._snapshots.get(snapshot_id)

    async def save_context_evidence(self, record: Any) -> Any:
        """Append a context evidence record to the evidence stream."""
        self._evidence.append(record)
        return record


class InMemoryRunStore:
    """Process-local run records with an idempotency index."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        # Idempotency index: (workspace_id, idempotency_key) -> run_id. The source
        # of truth for the in-memory unique constraint (mirrors the Postgres
        # ``runs_idempotency_idx`` partial UNIQUE index).
        self._runs_by_idempotency: dict[tuple[str, str], str] = {}

    async def save_run(self, run: RunRecord) -> RunRecord:
        """Upsert a run record keyed by ``run_id``; storage stamps ``updated_at``.

        Storage owns ``updated_at`` so it can never drift: every write refreshes it.
        The idempotency index is maintained so ``load_run_by_idempotency`` is an
        O(1) lookup and the in-memory unique constraint stays consistent.
        """
        run.updated_at = utc_now_iso()
        self._index_idempotency(run)
        self._runs[run.run_id] = run
        return run

    async def save_run_if_absent_by_idempotency(
        self, run: RunRecord
    ) -> tuple[RunRecord, bool]:
        """Atomically create a run unless its idempotency key already exists.

        Returns ``(run, created)``: ``created`` is True when this call wrote a new
        run, False when an existing run for ``(workspace_id, idempotency_key)`` is
        returned instead. There is NO ``await`` between the read and the write, so
        two concurrent callers with the same key cannot both create a run (closing
        the TOCTOU race in the application layer). Runs without an idempotency key
        are always created. The Postgres backend mirrors this via
        ``INSERT ... ON CONFLICT (workspace_id, idempotency_key) DO NOTHING``.
        """
        key = run.idempotency_key
        if key is not None:
            existing_id = self._runs_by_idempotency.get((run.workspace_id, key))
            if existing_id is not None:
                existing = self._runs.get(existing_id)
                if existing is not None:
                    return existing, False
        run.updated_at = utc_now_iso()
        self._index_idempotency(run)
        self._runs[run.run_id] = run
        return run, True

    def _index_idempotency(self, run: RunRecord) -> None:
        """Maintain the (workspace_id, idempotency_key) -> run_id index."""
        if run.idempotency_key is not None:
            self._runs_by_idempotency.setdefault(
                (run.workspace_id, run.idempotency_key), run.run_id
            )

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Return a run record by id, or None."""
        return self._runs.get(run_id)

    async def list_runs(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[RunRecord]:
        """List runs filtered by workspace, subject, and/or status."""
        return [
            r
            for r in self._runs.values()
            if (workspace_id is None or r.workspace_id == workspace_id)
            and (subject_id is None or r.subject_id == subject_id)
            and (status is None or r.status == status)
        ]

    async def load_run_by_idempotency(
        self, workspace_id: str, idempotency_key: str
    ) -> RunRecord | None:
        """Return the existing run for an idempotency key, or None (O(1) lookup)."""
        run_id = self._runs_by_idempotency.get((workspace_id, idempotency_key))
        if run_id is None:
            return None
        return self._runs.get(run_id)


class InMemoryRecommendationStore:
    """Process-local recommendation items keyed by ``recommendation_id``."""

    def __init__(self) -> None:
        self._recommendations: dict[str, RecommendationItem] = {}

    async def save_recommendation(self, item: RecommendationItem) -> RecommendationItem:
        """Upsert a recommendation item keyed by ``recommendation_id``."""
        self._recommendations[item.recommendation_id] = item
        return item

    async def get_recommendation(
        self, recommendation_id: str
    ) -> RecommendationItem | None:
        """Return a recommendation by id, or None."""
        return self._recommendations.get(recommendation_id)

    async def list_recommendations(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        status: RecommendationStatus | None = None,
    ) -> list[RecommendationItem]:
        """List recommendations filtered by the given dimensions."""
        return [
            r
            for r in self._recommendations.values()
            if (workspace_id is None or r.workspace_id == workspace_id)
            and (subject_id is None or r.subject_id == subject_id)
            and (run_id is None or r.run_id == run_id)
            and (kind is None or r.kind == kind)
            and (status is None or r.status == status)
        ]

    async def update_recommendation(
        self,
        recommendation_id: str,
        *,
        status: RecommendationStatus | None = None,
        notes: str | None = None,
    ) -> RecommendationItem | None:
        """Update a recommendation's status/notes in place; return it or None."""
        item = self._recommendations.get(recommendation_id)
        if item is None:
            return None
        if status is not None:
            item.status = status
        if notes is not None:
            item.notes = notes
        return item


class InMemoryEvaluationStore:
    """Process-local evaluation runs keyed by ``run_id``."""

    def __init__(self) -> None:
        self._evaluation_runs: dict[str, Any] = {}

    async def save_evaluation_run(self, run: EvaluationRun) -> EvaluationRun:
        """Upsert an evaluation run keyed by ``run_id``."""
        self._evaluation_runs[run.run_id] = run
        return run

    async def get_evaluation_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> EvaluationRun | None:
        """Return an evaluation run by id, tenant-scoped (AAEO-4).

        When ``workspace_id`` is supplied, a run belonging to another workspace is
        treated as not found (returns None).
        """
        run = self._evaluation_runs.get(run_id)
        if run is None:
            return None
        if (
            workspace_id is not None
            and getattr(run, "workspace_id", None) != workspace_id
        ):
            return None
        return cast("EvaluationRun", run)

    async def list_evaluation_runs(
        self, suite_id: str | None = None, *, workspace_id: str | None = None
    ) -> list[EvaluationRun]:
        """List evaluation runs, optionally filtered by suite id and workspace (AAEO-4)."""
        return [
            r
            for r in self._evaluation_runs.values()
            if (suite_id is None or getattr(r, "suite_id", None) == suite_id)
            and (
                workspace_id is None
                or getattr(r, "workspace_id", None) == workspace_id
            )
        ]


class InMemoryOrchestrationStore:
    """Process-local multi-agent + world-model records.

    Cognitive memory objects, episodic memories, agent states, actions, and
    environment states used by the multi-agent orchestrators and simulations.
    """

    def __init__(self) -> None:
        self._memory: dict[str, MemoryObject] = {}
        self._episodic: dict[str, EpisodicMemoryObject] = {}
        self._agent_states: dict[str, AgentStateRecord] = {}
        self._actions: dict[str, ActionRecord] = {}
        self._environment_states: dict[str, EnvironmentStateRecord] = {}

    async def save_memory(self, obj: MemoryObject) -> MemoryObject:
        """Upsert a cognitive memory object."""
        self._memory[obj.memory_id] = obj
        return obj

    async def get_memory(self, memory_id: str) -> MemoryObject | None:
        """Return a memory object by id, or None."""
        return self._memory.get(memory_id)

    async def list_memory(self, subject_id: str | None = None) -> list[MemoryObject]:
        """List memory objects, optionally filtered by subject."""
        return [
            m
            for m in self._memory.values()
            if subject_id is None or m.subject_id == subject_id
        ]

    async def save_episodic_memory(
        self, obj: EpisodicMemoryObject
    ) -> EpisodicMemoryObject:
        """Upsert an episodic memory object."""
        self._episodic[obj.episode_id] = obj
        return obj

    async def get_episodic_memory(self, episode_id: str) -> EpisodicMemoryObject | None:
        """Return an episodic memory object by id, or None."""
        return self._episodic.get(episode_id)

    async def list_episodic_memory(
        self, subject_id: str | None = None
    ) -> list[EpisodicMemoryObject]:
        """List episodic memory objects, optionally filtered by subject."""
        return [
            m
            for m in self._episodic.values()
            if subject_id is None or m.subject_id == subject_id
        ]

    async def save_agent_state(self, record: AgentStateRecord) -> AgentStateRecord:
        """Upsert an agent state record."""
        self._agent_states[record.state_id] = record
        return record

    async def get_agent_state(self, state_id: str) -> AgentStateRecord | None:
        """Return an agent state record by id, or None."""
        return self._agent_states.get(state_id)

    async def list_agent_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[AgentStateRecord]:
        """List agent state records filtered by environment and/or round."""
        return [
            s
            for s in self._agent_states.values()
            if (environment_name is None or s.environment_name == environment_name)
            and (round is None or s.round == round)
        ]

    async def save_action(self, record: ActionRecord) -> ActionRecord:
        """Upsert an action record."""
        self._actions[record.action_id] = record
        return record

    async def get_action(self, action_id: str) -> ActionRecord | None:
        """Return an action record by id, or None."""
        return self._actions.get(action_id)

    async def list_actions(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[ActionRecord]:
        """List action records filtered by environment and/or round."""
        return [
            a
            for a in self._actions.values()
            if (environment_name is None or a.environment_name == environment_name)
            and (round is None or a.round == round)
        ]

    async def save_environment_state(
        self, record: EnvironmentStateRecord
    ) -> EnvironmentStateRecord:
        """Upsert an environment state record."""
        self._environment_states[record.environment_state_id] = record
        return record

    async def get_environment_state(
        self, environment_state_id: str
    ) -> EnvironmentStateRecord | None:
        """Return an environment state record by id, or None."""
        return self._environment_states.get(environment_state_id)

    async def list_environment_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[EnvironmentStateRecord]:
        """List environment state records filtered by environment and/or round."""
        return [
            e
            for e in self._environment_states.values()
            if (environment_name is None or e.environment_name == environment_name)
            and (round is None or e.round == round)
        ]


__all__ = [
    "InMemoryContextStore",
    "InMemoryEvaluationStore",
    "InMemoryEventLog",
    "InMemoryOrchestrationStore",
    "InMemoryRecommendationStore",
    "InMemoryRunStore",
    "InMemoryThreadStore",
]
