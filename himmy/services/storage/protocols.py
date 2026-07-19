"""Storage kernel: focused, single-responsibility persistence protocols.

The storage surface is split into focused contracts so each concern — threads,
events, context, runs, recommendations, evaluations, and orchestration records —
can be implemented, tested, and swapped independently rather than funnelled
through one god object. Both the in-memory stores
(:mod:`himmy.services.storage.inmemory`) and the Postgres backend
(:class:`~himmy.services.storage.postgres.PostgresStorageService`) satisfy these
protocols; :class:`~himmy.services.storage.service.StorageService` composes the
in-memory stores behind a single backward-compatible facade.

``ThreadEventStore`` is the runtime-facing union of ``ThreadStore`` + ``EventLog``
(the conversation + event subset the single-agent runtime depends on). It is
distinct from :class:`himmy.services.memory.store.MemoryStore`, which persists
long-term cognitive ``MemoryRecord`` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from himmy.core.events import RunEvent
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

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids storage <-> context cycle
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.services.context.models import ContextField, ContextSnapshot
    from himmy.services.evaluation.models import EvaluationRun
    from himmy.services.storage.conversations import (
        ConversationSummary,
        FlatMessage,
    )


def normalize_event_type(event_type: Any) -> str | None:
    """Coerce an ``event_type`` filter to its stored string value (or ``None``).

    The ``list_events`` filters accept an
    :class:`~himmy.core.events.EventType` or its raw string value; both backends store
    the enum's ``.value`` in the denormalised ``event_type`` column. This mirrors the
    ``getattr(et, "value", str(et))`` coercion ``append_event`` uses on write so a read
    filter matches what was written, regardless of which form the caller passes.
    """
    if event_type is None:
        return None
    return str(getattr(event_type, "value", event_type))


@runtime_checkable
class ThreadStore(Protocol):
    """Persistence for chat threads, keyed by ``thread_id``."""

    async def save_thread(self, thread: ChatThread) -> ChatThread: ...

    async def load_thread(self, thread_id: str) -> ChatThread | None: ...


@runtime_checkable
class EventLog(Protocol):
    """Append-only run-event audit stream (also the EventSink surface).

    ``list_events`` carries keyword-only ``event_type``/``tool_name``/``workspace_id``/
    ``limit``/``newest_first`` filters on top of the historical positional ``thread_id``/
    ``trace_id`` scope so the self-learning miners can answer "which tools fail most"
    with a bounded (``limit``) index seek over the P0-B indexed columns rather than
    scanning the whole audit stream. ``event_type`` accepts an
    :class:`~himmy.core.events.EventType` or its string value (normalised exactly as
    ``append_event`` denormalises it). ``workspace_id`` scopes the read to one tenant's
    events on a shared store (``None`` = unscoped, the historical behaviour) so a tenant's
    tool-reputation mining never aggregates another tenant's failures.
    """

    async def append_event(self, event: RunEvent) -> None: ...

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
    ) -> list[RunEvent]: ...


@runtime_checkable
class ThreadEventStore(Protocol):
    """Runtime-facing union of threads + events used by the single-agent runtime.

    Distinct from :class:`himmy.services.memory.store.MemoryStore` (long-term
    cognitive memory). This is the thread/event persistence backbone.
    """

    async def save_thread(self, thread: Any) -> Any: ...

    async def load_thread(self, thread_id: str) -> Any | None: ...

    async def append_event(self, event: RunEvent) -> None: ...

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
    ) -> list[RunEvent]: ...


@runtime_checkable
class ContextStore(Protocol):
    """Persistence for context fields, snapshots, and evidence records."""

    async def save_context_field(self, field: ContextField) -> ContextField: ...

    async def get_context_field(
        self, subject_id: str, key: str, *, workspace_id: str | None = None
    ) -> ContextField | None: ...

    async def list_context_fields(self, subject_id: str) -> list[ContextField]: ...

    async def save_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot: ...

    async def load_snapshot(self, snapshot_id: str) -> ContextSnapshot | None: ...

    async def save_context_evidence(self, record: Any) -> Any: ...


@runtime_checkable
class RunStore(Protocol):
    """Persistence for run records, including atomic idempotent creation."""

    async def save_run(self, run: RunRecord) -> RunRecord: ...

    async def save_run_if_owner(
        self, run: RunRecord, owner_id: str
    ) -> bool: ...

    async def save_run_if_absent_by_idempotency(
        self, run: RunRecord
    ) -> tuple[RunRecord, bool]: ...

    async def save_run_if_under_quota(
        self, run: RunRecord, *, cap: int
    ) -> tuple[RunRecord, bool]: ...

    async def claim_run_for_resume(
        self, run_id: str, *, workspace_id: str
    ) -> bool: ...

    async def claim_next_queued_run(
        self,
        owner_id: str,
        lease_seconds: float,
        *,
        lanes: list[str] | None = None,
        now: str | None = None,
        fairness: bool = False,
        workspace_concurrency: int = 0,
    ) -> RunRecord | None: ...

    async def renew_lease(
        self,
        run_id: str,
        owner_id: str,
        lease_seconds: float,
        *,
        now: str | None = None,
    ) -> bool: ...

    async def requeue_expired_leases(
        self, *, now: str | None = None, lanes: list[str] | None = None
    ) -> list[str]: ...

    async def redrive_run(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        now: str | None = None,
    ) -> bool: ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...

    async def list_runs(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[RunRecord]: ...

    async def count_runs(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
        *,
        exclude_local_workspace: bool = False,
    ) -> int: ...

    async def count_active_runs_for_workspace(self, workspace_id: str) -> int: ...

    async def load_run_by_idempotency(
        self, workspace_id: str, idempotency_key: str
    ) -> RunRecord | None: ...

    async def prune_runs(
        self, *, older_than_days: float | None = None, keep_last: int | None = None
    ) -> int: ...


@runtime_checkable
class AgentDefStore(Protocol):
    """Persistence for workspace-scoped stored agent definitions (T2e).

    The durable ``/v1/agents`` resource: reusable :class:`AgentDefRecord` rows keyed
    by ``agent_id`` and always read tenant-scoped on ``workspace_id`` so one tenant's
    stored agent is invisible (404) to another. ``save_agent_def_if_absent`` is the
    non-run idempotent-create analog of ``save_run_if_absent_by_idempotency``.
    """

    async def save_agent_def(self, record: AgentDefRecord) -> AgentDefRecord: ...

    async def save_agent_def_if_absent(
        self, record: AgentDefRecord
    ) -> tuple[AgentDefRecord, bool]: ...

    async def get_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> AgentDefRecord | None: ...

    async def list_agent_defs(
        self, *, workspace_id: str | None = None
    ) -> list[AgentDefRecord]: ...

    async def delete_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> bool: ...


@runtime_checkable
class RecommendationStore(Protocol):
    """Persistence for recommendation items."""

    async def save_recommendation(
        self, item: RecommendationItem
    ) -> RecommendationItem: ...

    async def get_recommendation(
        self, recommendation_id: str
    ) -> RecommendationItem | None: ...

    async def list_recommendations(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        status: RecommendationStatus | None = None,
    ) -> list[RecommendationItem]: ...

    async def update_recommendation(
        self,
        recommendation_id: str,
        *,
        status: RecommendationStatus | None = None,
        notes: str | None = None,
    ) -> RecommendationItem | None: ...

    async def prune_recommendations(
        self, *, older_than_days: float | None = None, keep_last: int | None = None
    ) -> int: ...


@runtime_checkable
class EvaluationStore(Protocol):
    """Persistence for evaluation runs."""

    async def save_evaluation_run(self, run: EvaluationRun) -> EvaluationRun: ...

    async def get_evaluation_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> EvaluationRun | None: ...

    async def list_evaluation_runs(
        self, suite_id: str | None = None, *, workspace_id: str | None = None
    ) -> list[EvaluationRun]: ...


@runtime_checkable
class OrchestrationStore(Protocol):
    """Persistence for multi-agent + world-model records.

    Cognitive memory objects, episodic memories, agent states, actions, and
    environment states used by the multi-agent orchestrators and simulations.
    """

    async def save_memory(self, obj: MemoryObject) -> MemoryObject: ...

    async def get_memory(self, memory_id: str) -> MemoryObject | None: ...

    async def list_memory(
        self, subject_id: str | None = None
    ) -> list[MemoryObject]: ...

    async def save_episodic_memory(
        self, obj: EpisodicMemoryObject
    ) -> EpisodicMemoryObject: ...

    async def get_episodic_memory(
        self, episode_id: str
    ) -> EpisodicMemoryObject | None: ...

    async def list_episodic_memory(
        self, subject_id: str | None = None
    ) -> list[EpisodicMemoryObject]: ...

    async def prune_memory(
        self, *, older_than_days: float | None = None, keep_last: int | None = None
    ) -> int: ...

    async def save_agent_state(self, record: AgentStateRecord) -> AgentStateRecord: ...

    async def get_agent_state(self, state_id: str) -> AgentStateRecord | None: ...

    async def list_agent_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[AgentStateRecord]: ...

    async def save_action(self, record: ActionRecord) -> ActionRecord: ...

    async def get_action(self, action_id: str) -> ActionRecord | None: ...

    async def list_actions(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[ActionRecord]: ...

    async def save_environment_state(
        self, record: EnvironmentStateRecord
    ) -> EnvironmentStateRecord: ...

    async def get_environment_state(
        self, environment_state_id: str
    ) -> EnvironmentStateRecord | None: ...

    async def list_environment_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[EnvironmentStateRecord]: ...


@runtime_checkable
class ConversationStoreProtocol(Protocol):
    """Shared contract for the durable conversation store (SQLite ↔ Postgres twins).

    The auxiliary conversation store is hand-mirrored across two backends —
    :class:`himmy.services.storage.conversations.ConversationStore` (durable SQLite) and
    :class:`himmy.services.storage.postgres_aux.PostgresConversationStore` (the K4 Postgres
    mirror) — that :func:`~himmy.services.storage.conversations.get_conversation_store`
    swaps behind one handle by ``HIMMY_DATABASE_URL``. Because callers hold that handle
    polymorphically, the two twins must expose the SAME method surface or a Postgres
    deployment silently loses a method the SQLite path has. This ``runtime_checkable``
    Protocol pins that shared surface so the conformance test fails loudly on drift, exactly
    like the focused core-store protocols above.

    Method signatures use ``Any`` for the record/thread payloads (mirroring
    ``ThreadEventStore``) so the contract stays decoupled from the concrete model modules;
    ``runtime_checkable`` verifies method PRESENCE, which is the drift that actually bites.
    ``prune`` is part of the surface for parity with the other pruneable stores
    (``RunStore.prune_runs``, ``RecommendationStore.prune_recommendations``,
    ``OrchestrationStore.prune_memory``); the SQLite-only ``import_legacy`` boot-path helper
    (which folds local ``sessions.db``/``chats.db`` sidecars a Postgres deployment never has)
    is deliberately NOT part of the abstract contract.
    """

    def save_thread(
        self, conversation_id: str, thread: Any, **kwargs: Any
    ) -> ConversationSummary: ...

    def save_flat(self, **kwargs: Any) -> ConversationSummary: ...

    def load_thread(self, conversation_id: str) -> ChatThread | None: ...

    def get_summary(
        self, conversation_id: str, **kwargs: Any
    ) -> ConversationSummary | None: ...

    def flat_messages(self, conversation_id: str) -> list[FlatMessage]: ...

    def list_summaries(self, **kwargs: Any) -> list[ConversationSummary]: ...

    def rename(self, conversation_id: str, title: str, **kwargs: Any) -> bool: ...

    def delete(self, conversation_id: str, **kwargs: Any) -> bool: ...

    def subject_of(self, conversation_id: str) -> str | None: ...

    def conversation_ids_for_subject(self, subject_id: str) -> list[str]: ...

    def delete_by_subject(self, subject_id: str) -> int: ...

    def set_project(self, conversation_id: str, project_id: str | None) -> bool: ...

    def prune(
        self, *, older_than_days: float | None = None, keep_last: int | None = None
    ) -> int: ...

    def create_project(self, *, name: str, **kwargs: Any) -> Any: ...

    def get_project_row(self, project_id: str, **kwargs: Any) -> Any | None: ...

    def project_chat_count(self, project_id: str, **kwargs: Any) -> int: ...

    def list_project_rows(self, **kwargs: Any) -> list[Any]: ...

    def update_project(
        self, project_id: str, assignments: dict[str, str | None], **kwargs: Any
    ) -> bool: ...

    def delete_project(self, project_id: str, **kwargs: Any) -> bool: ...

    def project_conversation_summaries(
        self, project_id: str, **kwargs: Any
    ) -> list[ConversationSummary]: ...

    def assign_conversation(
        self, project_id: str, conversation_id: str, **kwargs: Any
    ) -> bool: ...

    def unassign_conversation(self, conversation_id: str, **kwargs: Any) -> bool: ...

    def close(self) -> None: ...


__all__ = [
    "AgentDefStore",
    "ContextStore",
    "ConversationStoreProtocol",
    "EvaluationStore",
    "EventLog",
    "OrchestrationStore",
    "RecommendationStore",
    "RunStore",
    "ThreadEventStore",
    "ThreadStore",
    "normalize_event_type",
]
