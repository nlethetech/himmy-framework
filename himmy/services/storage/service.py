"""Storage kernel: the in-memory async persistence facade.

``StorageService`` is the default backend: a thin facade that composes the focused,
single-responsibility stores in :mod:`himmy.services.storage.inmemory` (threads,
events, context, runs, recommendations, evaluations, orchestration) and delegates
to them. The decomposition keeps each concern small and independently testable
while preserving a single backward-compatible API for the ~15 call sites that
inject storage. The facade satisfies both :class:`~himmy.core.events.EventSink`
(``append_event``) and the
:class:`~himmy.services.storage.protocols.ThreadEventStore` protocol.

``ThreadEventStore`` (async threads + events, used by the runtime) is distinct from
:class:`himmy.services.memory.store.MemoryStore` (sync, long-term cognitive
``MemoryRecord`` persistence). The two were both named ``MemoryStore`` historically;
this one was renamed to reflect what it actually stores.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from himmy.core.events import RunEvent
from himmy.services.storage.inmemory import (
    InMemoryAgentDefStore,
    InMemoryContextStore,
    InMemoryEvaluationStore,
    InMemoryEventLog,
    InMemoryOrchestrationStore,
    InMemoryRecommendationStore,
    InMemoryRunStore,
    InMemoryThreadStore,
    InMemoryTriggerDedupStore,
)
from himmy.services.storage.models import (
    ActionRecord,
    AgentDefRecord,
    AgentStateRecord,
    EnvironmentStateRecord,
    EpisodicMemoryObject,
    MemoryObject,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)
from himmy.services.storage.protocols import ThreadEventStore

logger = logging.getLogger("himmy.services.storage")

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids storage <-> context cycle
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.services.context.models import ContextField, ContextSnapshot
    from himmy.services.evaluation.models import EvaluationRun
    from himmy.services.storage.trigger_dedup import DedupClaim


class StorageService:
    """In-memory async storage backend (default for tests, examples, local dev).

    A facade that composes focused per-concern stores and delegates to them, so
    each domain (threads, events, context, runs, recommendations, evaluations,
    orchestration) is implemented and testable in isolation. Satisfies
    :class:`~himmy.core.events.EventSink` (``append_event``) and the
    :class:`~himmy.services.storage.protocols.ThreadEventStore` protocol. All state
    is process-local and lost on exit.
    """

    def __init__(self) -> None:
        """Compose the focused in-memory stores behind the facade."""
        self._thread_store = InMemoryThreadStore()
        self._event_log = InMemoryEventLog()
        self._context_store = InMemoryContextStore()
        self._run_store = InMemoryRunStore()
        self._agent_def_store = InMemoryAgentDefStore()
        self._recommendation_store = InMemoryRecommendationStore()
        self._evaluation_store = InMemoryEvaluationStore()
        self._orchestration_store = InMemoryOrchestrationStore()
        self._trigger_dedup_store = InMemoryTriggerDedupStore()

    # ------------------------------------------------------------------ threads
    async def save_thread(self, thread: ChatThread) -> ChatThread:
        """Upsert a chat thread keyed by ``thread_id``."""
        return await self._thread_store.save_thread(thread)

    async def load_thread(self, thread_id: str) -> ChatThread | None:
        """Return a stored chat thread by id, or None."""
        return await self._thread_store.load_thread(thread_id)

    # ------------------------------------------------------------------- events
    async def append_event(self, event: RunEvent) -> None:
        """Append a run event to the canonical audit stream (EventSink)."""
        await self._event_log.append_event(event)

    async def list_events(
        self,
        thread_id: str | None = None,
        trace_id: str | None = None,
        *,
        event_type: Any = None,
        tool_name: str | None = None,
        workspace_id: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[RunEvent]:
        """List events filtered by thread/trace/event_type/tool_name (insertion order)."""
        return await self._event_log.list_events(
            thread_id,
            trace_id,
            event_type=event_type,
            tool_name=tool_name,
            workspace_id=workspace_id,
            limit=limit,
            newest_first=newest_first,
        )

    def delete_by_subject(self, subject_id: str) -> int:
        """Hard-DELETE a subject's chat_threads + run_events (S4 right-to-erasure).

        Mirrors the durable backends' :meth:`SqliteStorageService.delete_by_subject`: a
        subject's runs name the threads + event streams the runtime persisted for it, so
        we resolve those ids and drop the matching threads + events. Synchronous so the
        sync ``SubjectReachMap.erase`` can drive it directly. Returns rows removed.
        """
        thread_ids, trace_ids = self._run_store.thread_and_trace_ids_for_subject(
            subject_id
        )
        removed = self._thread_store.delete_threads(thread_ids)
        removed += self._event_log.delete_events(thread_ids, trace_ids)
        return removed

    # ------------------------------------------------------------------ context
    async def save_context_field(self, field: ContextField) -> ContextField:
        """Upsert a context field keyed by ``(subject_id, key)``."""
        return await self._context_store.save_context_field(field)

    async def get_context_field(self, subject_id: str, key: str) -> ContextField | None:
        """Return the context field for ``(subject_id, key)``, or None."""
        return await self._context_store.get_context_field(subject_id, key)

    async def list_context_fields(self, subject_id: str) -> list[ContextField]:
        """Return all context fields for a subject."""
        return await self._context_store.list_context_fields(subject_id)

    async def save_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        """Upsert a context snapshot keyed by ``snapshot_id``."""
        return await self._context_store.save_snapshot(snapshot)

    async def load_snapshot(self, snapshot_id: str) -> ContextSnapshot | None:
        """Return a stored snapshot by id, or None."""
        return await self._context_store.load_snapshot(snapshot_id)

    async def save_context_evidence(self, record: object) -> object:
        """Append a context evidence record to the evidence stream."""
        return await self._context_store.save_context_evidence(record)

    # --------------------------------------------------------------------- runs
    async def save_run(self, run: RunRecord) -> RunRecord:
        """Upsert a run record keyed by ``run_id``; storage stamps ``updated_at``."""
        return await self._run_store.save_run(run)

    async def save_run_if_absent_by_idempotency(
        self, run: RunRecord
    ) -> tuple[RunRecord, bool]:
        """Atomically create a run unless its idempotency key already exists."""
        return await self._run_store.save_run_if_absent_by_idempotency(run)

    async def claim_run_for_resume(
        self, run_id: str, *, workspace_id: str
    ) -> bool:
        """Atomically claim an AWAITING_APPROVAL run for resume (True iff we won)."""
        return await self._run_store.claim_run_for_resume(
            run_id, workspace_id=workspace_id
        )

    async def claim_next_queued_run(
        self,
        owner_id: str,
        lease_seconds: float,
        *,
        lanes: list[str] | None = None,
        now: str | None = None,
    ) -> RunRecord | None:
        """Atomically claim the oldest ready QUEUED run for ``owner_id`` (Q2; or None)."""
        return await self._run_store.claim_next_queued_run(
            owner_id, lease_seconds, lanes=lanes, now=now
        )

    async def renew_lease(
        self,
        run_id: str,
        owner_id: str,
        lease_seconds: float,
        *,
        now: str | None = None,
    ) -> bool:
        """Extend a RUNNING run's lease iff ``owner_id`` still holds it (Q2)."""
        return await self._run_store.renew_lease(
            run_id, owner_id, lease_seconds, now=now
        )

    async def requeue_expired_leases(
        self, *, now: str | None = None, lanes: list[str] | None = None
    ) -> list[str]:
        """Re-queue RUNNING runs whose lease expired; return the re-queued ids (Q2)."""
        return await self._run_store.requeue_expired_leases(now=now, lanes=lanes)

    async def redrive_run(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        now: str | None = None,
    ) -> bool:
        """Reset a PARKED/FAILED run back to QUEUED for another attempt (Q2)."""
        return await self._run_store.redrive_run(
            run_id, workspace_id=workspace_id, now=now
        )

    # ------------------------------------------------------ inbound dedup (Q4)
    async def dedup_try_claim(
        self,
        scope: str,
        key: str,
        *,
        lease_seconds: float,
        now: str | None = None,
    ) -> DedupClaim:
        """Atomically claim ``(scope, key)`` for execution, or report a duplicate (Q4)."""
        return await self._trigger_dedup_store.dedup_try_claim(
            scope, key, lease_seconds=lease_seconds, now=now
        )

    async def dedup_complete(
        self,
        scope: str,
        key: str,
        *,
        result: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> None:
        """Upgrade a won in-flight dedup claim to COMPLETED with ``result`` + TTL (Q4)."""
        await self._trigger_dedup_store.dedup_complete(
            scope, key, result=result, ttl_seconds=ttl_seconds, now=now
        )

    async def dedup_release(
        self, scope: str, key: str, *, now: str | None = None
    ) -> None:
        """Drop a won-but-failed in-flight dedup claim so a redelivery re-runs (Q4)."""
        await self._trigger_dedup_store.dedup_release(scope, key, now=now)

    async def dedup_sweep(self, *, now: str | None = None) -> int:
        """Delete expired dedup rows (lazy GC); return the count removed (Q4)."""
        return await self._trigger_dedup_store.dedup_sweep(now=now)

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Return a run record by id, or None."""
        return await self._run_store.get_run(run_id)

    async def list_runs(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[RunRecord]:
        """List runs filtered by workspace, subject, and/or status."""
        return await self._run_store.list_runs(workspace_id, subject_id, status)

    async def load_run_by_idempotency(
        self, workspace_id: str, idempotency_key: str
    ) -> RunRecord | None:
        """Return the existing run for an idempotency key, or None (O(1) lookup)."""
        return await self._run_store.load_run_by_idempotency(
            workspace_id, idempotency_key
        )

    # ------------------------------------------------------------- agent defs (T2e)
    async def save_agent_def(self, record: AgentDefRecord) -> AgentDefRecord:
        """Upsert a stored agent definition keyed by ``agent_id``."""
        return await self._agent_def_store.save_agent_def(record)

    async def save_agent_def_if_absent(
        self, record: AgentDefRecord
    ) -> tuple[AgentDefRecord, bool]:
        """Atomically create an agent def unless its idempotency key already exists."""
        return await self._agent_def_store.save_agent_def_if_absent(record)

    async def get_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> AgentDefRecord | None:
        """Return a stored agent def by id, tenant-scoped (out-of-workspace → None)."""
        return await self._agent_def_store.get_agent_def(
            agent_id, workspace_id=workspace_id
        )

    async def list_agent_defs(
        self, *, workspace_id: str | None = None
    ) -> list[AgentDefRecord]:
        """List stored agent defs for a workspace."""
        return await self._agent_def_store.list_agent_defs(workspace_id=workspace_id)

    async def delete_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> bool:
        """Delete a stored agent def, tenant-scoped. Returns True iff removed."""
        return await self._agent_def_store.delete_agent_def(
            agent_id, workspace_id=workspace_id
        )

    # ---------------------------------------------------------- recommendations
    async def save_recommendation(self, item: RecommendationItem) -> RecommendationItem:
        """Upsert a recommendation item keyed by ``recommendation_id``."""
        return await self._recommendation_store.save_recommendation(item)

    async def get_recommendation(
        self, recommendation_id: str
    ) -> RecommendationItem | None:
        """Return a recommendation by id, or None."""
        return await self._recommendation_store.get_recommendation(recommendation_id)

    async def list_recommendations(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        status: RecommendationStatus | None = None,
    ) -> list[RecommendationItem]:
        """List recommendations filtered by the given dimensions."""
        return await self._recommendation_store.list_recommendations(
            workspace_id, subject_id, run_id, kind, status
        )

    async def update_recommendation(
        self,
        recommendation_id: str,
        *,
        status: RecommendationStatus | None = None,
        notes: str | None = None,
    ) -> RecommendationItem | None:
        """Update a recommendation's status/notes in place; return it or None."""
        return await self._recommendation_store.update_recommendation(
            recommendation_id, status=status, notes=notes
        )

    # --------------------------------------------------------------- evaluation
    async def save_evaluation_run(self, run: EvaluationRun) -> EvaluationRun:
        """Upsert an evaluation run keyed by ``run_id``."""
        return await self._evaluation_store.save_evaluation_run(run)

    async def get_evaluation_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> EvaluationRun | None:
        """Return an evaluation run by id, tenant-scoped (AAEO-4)."""
        return await self._evaluation_store.get_evaluation_run(
            run_id, workspace_id=workspace_id
        )

    async def list_evaluation_runs(
        self, suite_id: str | None = None, *, workspace_id: str | None = None
    ) -> list[EvaluationRun]:
        """List evaluation runs, optionally filtered by suite id and workspace (AAEO-4)."""
        return await self._evaluation_store.list_evaluation_runs(
            suite_id, workspace_id=workspace_id
        )

    # --------------------------------------------- memory + orchestration records
    async def save_memory(self, obj: MemoryObject) -> MemoryObject:
        """Upsert a cognitive memory object."""
        return await self._orchestration_store.save_memory(obj)

    async def get_memory(self, memory_id: str) -> MemoryObject | None:
        """Return a memory object by id, or None."""
        return await self._orchestration_store.get_memory(memory_id)

    async def list_memory(self, subject_id: str | None = None) -> list[MemoryObject]:
        """List memory objects, optionally filtered by subject."""
        return await self._orchestration_store.list_memory(subject_id)

    async def save_episodic_memory(
        self, obj: EpisodicMemoryObject
    ) -> EpisodicMemoryObject:
        """Upsert an episodic memory object."""
        return await self._orchestration_store.save_episodic_memory(obj)

    async def get_episodic_memory(self, episode_id: str) -> EpisodicMemoryObject | None:
        """Return an episodic memory object by id, or None."""
        return await self._orchestration_store.get_episodic_memory(episode_id)

    async def list_episodic_memory(
        self, subject_id: str | None = None
    ) -> list[EpisodicMemoryObject]:
        """List episodic memory objects, optionally filtered by subject."""
        return await self._orchestration_store.list_episodic_memory(subject_id)

    async def save_agent_state(self, record: AgentStateRecord) -> AgentStateRecord:
        """Upsert an agent state record."""
        return await self._orchestration_store.save_agent_state(record)

    async def get_agent_state(self, state_id: str) -> AgentStateRecord | None:
        """Return an agent state record by id, or None."""
        return await self._orchestration_store.get_agent_state(state_id)

    async def list_agent_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[AgentStateRecord]:
        """List agent state records filtered by environment and/or round."""
        return await self._orchestration_store.list_agent_states(
            environment_name, round
        )

    async def save_action(self, record: ActionRecord) -> ActionRecord:
        """Upsert an action record."""
        return await self._orchestration_store.save_action(record)

    async def get_action(self, action_id: str) -> ActionRecord | None:
        """Return an action record by id, or None."""
        return await self._orchestration_store.get_action(action_id)

    async def list_actions(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[ActionRecord]:
        """List action records filtered by environment and/or round."""
        return await self._orchestration_store.list_actions(environment_name, round)

    async def save_environment_state(
        self, record: EnvironmentStateRecord
    ) -> EnvironmentStateRecord:
        """Upsert an environment state record."""
        return await self._orchestration_store.save_environment_state(record)

    async def get_environment_state(
        self, environment_state_id: str
    ) -> EnvironmentStateRecord | None:
        """Return an environment state record by id, or None."""
        return await self._orchestration_store.get_environment_state(
            environment_state_id
        )

    async def list_environment_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[EnvironmentStateRecord]:
        """List environment state records filtered by environment and/or round."""
        return await self._orchestration_store.list_environment_states(
            environment_name, round
        )


__all__ = ["StorageService", "ThreadEventStore"]
