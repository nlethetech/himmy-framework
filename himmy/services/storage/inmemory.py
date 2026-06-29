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
from himmy.core.ids import iso_plus_seconds as _iso_plus_seconds
from himmy.core.ids import utc_now_iso
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
from himmy.services.storage.protocols import normalize_event_type

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

    def delete_threads(self, thread_ids: set[str]) -> int:
        """Drop the named threads; return the count removed (S4 erasure)."""
        removed = 0
        for tid in thread_ids:
            if self._threads.pop(tid, None) is not None:
                removed += 1
        return removed


class InMemoryEventLog:
    """Process-local append-only run-event stream (EventSink surface)."""

    def __init__(self) -> None:
        self._events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        """Append a run event to the canonical audit stream (EventSink)."""
        self._events.append(event)

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
        """List events filtered by thread/trace/event_type/tool_name (insertion order).

        ``event_type`` matches an :class:`EventType` or its string value; ``tool_name``
        matches the denormalised ``payload['tool_name']`` (the in-memory analog of the
        durable backends' indexed columns). ``workspace_id`` scopes the read to one
        tenant's events (``None`` = unscoped). ``limit`` bounds the result and
        ``newest_first`` flips the order — the learning miners take the most recent N.
        """
        want_type = normalize_event_type(event_type)
        matches = [
            e
            for e in self._events
            if (thread_id is None or e.thread_id == thread_id)
            and (trace_id is None or e.trace_id == trace_id)
            and (
                want_type is None
                or getattr(e.event_type, "value", str(e.event_type)) == want_type
            )
            and (tool_name is None or (e.payload or {}).get("tool_name") == tool_name)
            and (workspace_id is None or e.workspace_id == workspace_id)
        ]
        if newest_first:
            matches = list(reversed(matches))
        if limit is not None:
            matches = matches[: max(0, limit)]
        return matches

    def delete_events(self, thread_ids: set[str], trace_ids: set[str]) -> int:
        """Drop events whose thread_id OR trace_id is named; return count (S4 erasure)."""
        before = len(self._events)
        self._events = [
            e
            for e in self._events
            if e.thread_id not in thread_ids and e.trace_id not in trace_ids
        ]
        return before - len(self._events)


class InMemoryContextStore:
    """Process-local context fields, snapshots, and evidence."""

    def __init__(self) -> None:
        # context_fields keyed by (workspace_id, subject_id, key) — the workspace
        # component (blank offline / single-tenant) partitions the store by tenant so a
        # tenant's upsert can never clobber another tenant's row under a shared subject_id.
        self._context_fields: dict[tuple[str, str, str], ContextField] = {}
        self._snapshots: dict[str, Any] = {}
        self._evidence: list[Any] = []

    async def save_context_field(self, field: ContextField) -> ContextField:
        """Upsert a context field keyed by ``(workspace_id, subject_id, key)``.

        ``subject_id`` and the owning ``workspace_id`` are read from the field metadata
        when present, falling back to a blank scope so storage-only / single-tenant fields
        still round-trip (the blank workspace is the offline partition). Including
        ``workspace_id`` in the key tenant-PARTITIONS the store (red-team reattack-r9): two
        tenants sharing a ``subject_id`` write to DISTINCT rows, so a cross-tenant upsert
        can never overwrite/poison another tenant's field.
        """
        meta = getattr(field, "metadata", {}) or {}
        subject_id = str(meta.get("subject_id", ""))
        workspace_id = str(meta.get("workspace_id", "") or "")
        self._context_fields[(workspace_id, subject_id, field.key)] = field
        return field

    async def get_context_field(
        self, subject_id: str, key: str, *, workspace_id: str | None = None
    ) -> ContextField | None:
        """Return the context field for ``(subject_id, key)``, tenant-scoped on workspace.

        When ``workspace_id`` is supplied, ONLY that tenant's partition is consulted (so
        the snapshot-resolution path cannot pick up a sibling tenant's row). The offline
        path (``None``) applies NO workspace filter — byte-identical to the historical
        ``(subject_id, key)`` lookup (a single-tenant store holds one such row).
        """
        if workspace_id is None:
            for (_, sid, k), field in self._context_fields.items():
                if sid == subject_id and k == key:
                    return field
            return None
        return self._context_fields.get((str(workspace_id), subject_id, key))

    async def list_context_fields(self, subject_id: str) -> list[ContextField]:
        """Return all context fields for a subject (across every workspace partition)."""
        return [
            field
            for (_, sid, _), field in self._context_fields.items()
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

    async def save_run_if_under_quota(
        self, run: RunRecord, *, cap: int
    ) -> tuple[RunRecord, bool]:
        """Atomically create ``run`` IFF the tenant is under its outstanding-run cap (T3).

        Fuses the idempotency check, the active-run COUNT, and the INSERT — with NO ``await``
        between them, so on the single event loop two concurrent same-workspace creates cannot
        interleave: each sees the other's already-inserted run in the count. The result is
        EXACTLY ``cap`` admits, never the unlocked count-then-insert overshoot.

        Returns ``(run, admitted)``. ``admitted=False`` ⇒ the tenant was at/over ``cap`` and
        NOTHING was inserted. An idempotent re-submit of an existing key returns the prior run
        with ``admitted=True`` and does NOT consume a slot (it already holds one). ``cap <= 0``
        admits unconditionally (cap disabled).
        """
        # Idempotent re-submit returns the prior run WITHOUT counting against the cap.
        key = run.idempotency_key
        if key is not None:
            existing_id = self._runs_by_idempotency.get((run.workspace_id, key))
            if existing_id is not None:
                existing = self._runs.get(existing_id)
                if existing is not None:
                    return existing, True
        if cap > 0:
            from himmy.services.storage.models import ACTIVE_RUN_STATUSES

            active = set(ACTIVE_RUN_STATUSES)
            current = sum(
                1
                for r in self._runs.values()
                if r.workspace_id == run.workspace_id and r.status in active
            )
            # Count BEFORE insert ⇒ reject at/over the cap so a cap of N admits exactly N.
            if current >= cap:
                return run, False
        run.updated_at = utc_now_iso()
        self._index_idempotency(run)
        self._runs[run.run_id] = run
        return run, True

    async def claim_run_for_resume(
        self, run_id: str, *, workspace_id: str
    ) -> bool:
        """Atomically claim an AWAITING_APPROVAL run for resume (True iff we won).

        Compare-and-set ``AWAITING_APPROVAL`` -> ``RESOLVING`` for the run, scoped to
        ``workspace_id``. There is NO ``await`` between the read and the write, so two
        concurrent resumes on one event loop cannot both win — exactly one flips the
        status and returns True; the loser sees a non-AWAITING status and returns False.
        This is the in-memory analog of the SQLite/Postgres conditional UPDATE, and the
        run-level mirror of the member checkpoint ``claim()``: only AWAITING_APPROVAL is
        claimable here (a fresh approve), so a SECOND concurrent click loses and is
        refused, while a stale RESOLVING run (a crashed resume) is re-driven through the
        recovery path, not this method.
        """
        run = self._runs.get(run_id)
        if (
            run is None
            or run.workspace_id != workspace_id
            or run.status != RunStatus.AWAITING_APPROVAL
        ):
            return False
        run.status = RunStatus.RESOLVING
        run.updated_at = utc_now_iso()
        return True

    async def claim_next_queued_run(
        self,
        owner_id: str,
        lease_seconds: float,
        *,
        lanes: list[str] | None = None,
        now: str | None = None,
        fairness: bool = False,
        workspace_concurrency: int = 0,
    ) -> RunRecord | None:
        """Atomically claim the next ready QUEUED run for ``owner_id`` (or None).

        The in-memory analog of the SQLite/Postgres claim CAS: with NO ``await`` between the
        scan and the mutation, two concurrent claims on one event loop cannot both win a run.
        A READY (``QUEUED`` and ``next_attempt_at <= now``) run is selected by ``created_at``
        order; ``lanes`` restricts it to the given ``lane_key`` set (Q2 lane keying). The
        winner flips to ``RUNNING`` with owner/lease/heartbeat stamped and ``attempt``
        incremented.

        T3-fairness (opt-in via ``fairness=True``): candidates are ordered by their workspace's
        live-leased RUNNING count first (least-loaded round-robin), and a positive
        ``workspace_concurrency`` excludes a workspace already at that cap — the same per-tenant
        fairness + concurrency cap the durable backends apply, so the in-memory path stays
        behaviourally faithful for tests. Both default OFF (FIFO unchanged).
        """
        now_iso = now or utc_now_iso()
        if lanes is not None and not lanes:
            return None
        lane_set = set(lanes) if lanes is not None else None
        ready = [
            r
            for r in self._runs.values()
            if r.status == RunStatus.QUEUED
            and (r.next_attempt_at is None or r.next_attempt_at <= now_iso)
            and (lane_set is None or r.lane_key in lane_set)
        ]
        if not ready:
            return None
        if fairness:
            # Per-workspace live-leased RUNNING load (an expired lease will be reaped and
            # does not count toward the live concurrency).
            load: dict[str, int] = {}
            for r in self._runs.values():
                if r.status == RunStatus.RUNNING and (
                    r.lease_expires_at is None or r.lease_expires_at > now_iso
                ):
                    load[r.workspace_id] = load.get(r.workspace_id, 0) + 1
            if workspace_concurrency > 0:
                ready = [
                    r
                    for r in ready
                    if load.get(r.workspace_id, 0) < workspace_concurrency
                ]
                if not ready:
                    return None
            ready.sort(
                key=lambda r: (
                    load.get(r.workspace_id, 0),
                    r.next_attempt_at or r.created_at,
                    r.created_at,
                )
            )
            run = ready[0]
            run.status = RunStatus.RUNNING
            run.owner_id = owner_id
            run.lease_expires_at = _iso_plus_seconds(now_iso, lease_seconds)
            run.heartbeat_at = now_iso
            run.attempt = int(run.attempt or 0) + 1
            run.updated_at = now_iso
            return run
        # Oldest by next_attempt_at then created_at (insertion-order proxy), stable.
        ready.sort(key=lambda r: (r.next_attempt_at or r.created_at, r.created_at))
        run = ready[0]
        run.status = RunStatus.RUNNING
        run.owner_id = owner_id
        run.lease_expires_at = _iso_plus_seconds(now_iso, lease_seconds)
        run.heartbeat_at = now_iso
        run.attempt = int(run.attempt or 0) + 1
        run.updated_at = now_iso
        return run

    async def renew_lease(
        self,
        run_id: str,
        owner_id: str,
        lease_seconds: float,
        *,
        now: str | None = None,
    ) -> bool:
        """Extend a RUNNING run's lease iff ``owner_id`` still holds it (True iff renewed)."""
        now_iso = now or utc_now_iso()
        run = self._runs.get(run_id)
        if run is None or run.owner_id != owner_id or run.status != RunStatus.RUNNING:
            return False
        run.lease_expires_at = _iso_plus_seconds(now_iso, lease_seconds)
        run.heartbeat_at = now_iso
        run.updated_at = now_iso
        return True

    async def requeue_expired_leases(
        self, *, now: str | None = None, lanes: list[str] | None = None
    ) -> list[str]:
        """Re-queue RUNNING runs whose lease expired; return the re-queued run_ids.

        Only ``RUNNING`` runs with ``lease_expires_at <= now`` are touched (a renewing peer
        has a future deadline and is never reaped; HITL/terminal/QUEUED states are excluded by
        the status predicate). Each resets to ``QUEUED`` with owner/lease cleared and
        ``next_attempt_at`` set to ``now``.
        """
        now_iso = now or utc_now_iso()
        if lanes is not None and not lanes:
            return []
        lane_set = set(lanes) if lanes is not None else None
        requeued: list[str] = []
        for run in self._runs.values():
            if (
                run.status == RunStatus.RUNNING
                and run.lease_expires_at is not None
                and run.lease_expires_at <= now_iso
                and (lane_set is None or run.lane_key in lane_set)
            ):
                run.status = RunStatus.QUEUED
                run.owner_id = None
                run.lease_expires_at = None
                run.next_attempt_at = now_iso
                run.updated_at = now_iso
                requeued.append(run.run_id)
        return requeued

    async def redrive_run(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        now: str | None = None,
    ) -> bool:
        """Reset a PARKED/FAILED run back to QUEUED for another attempt (True iff redriven)."""
        now_iso = now or utc_now_iso()
        run = self._runs.get(run_id)
        if (
            run is None
            or (workspace_id is not None and run.workspace_id != workspace_id)
            or run.status not in (RunStatus.PARKED, RunStatus.FAILED)
        ):
            return False
        run.status = RunStatus.QUEUED
        run.attempt = 0
        run.owner_id = None
        run.lease_expires_at = None
        run.last_error = None
        run.error = None
        run.next_attempt_at = now_iso
        run.updated_at = now_iso
        return True

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

    async def count_active_runs_for_workspace(self, workspace_id: str) -> int:
        """Count a workspace's non-terminal (in-flight) runs (T3 quota).

        QUEUED/RUNNING/AWAITING_APPROVAL/RESOLVING occupy the tenant's outstanding budget;
        terminal SUCCEEDED/FAILED/PARKED do not.
        """
        from himmy.services.storage.models import ACTIVE_RUN_STATUSES

        active = set(ACTIVE_RUN_STATUSES)
        return sum(
            1
            for r in self._runs.values()
            if r.workspace_id == workspace_id and r.status in active
        )

    async def load_run_by_idempotency(
        self, workspace_id: str, idempotency_key: str
    ) -> RunRecord | None:
        """Return the existing run for an idempotency key, or None (O(1) lookup)."""
        run_id = self._runs_by_idempotency.get((workspace_id, idempotency_key))
        if run_id is None:
            return None
        return self._runs.get(run_id)

    def thread_and_trace_ids_for_subject(
        self, subject_id: str
    ) -> tuple[set[str], set[str]]:
        """The thread_ids + trace_ids of a subject's runs (the S4 erasure linkage)."""
        thread_ids: set[str] = set()
        trace_ids: set[str] = set()
        for run in self._runs.values():
            if run.subject_id != subject_id:
                continue
            if run.thread_id:
                thread_ids.add(run.thread_id)
            if run.trace_id:
                trace_ids.add(run.trace_id)
        return thread_ids, trace_ids


class InMemoryAgentDefStore:
    """Process-local workspace-scoped stored agent definitions (T2e).

    Keyed by ``agent_id`` with a ``(workspace_id, idempotency_key)`` index mirroring
    the run store's idempotency primitive, so an idempotent re-POST returns the prior
    record. All reads are tenant-scoped: a record from another workspace reads as None.
    """

    def __init__(self) -> None:
        self._defs: dict[str, AgentDefRecord] = {}
        # (workspace_id, idempotency_key) -> agent_id
        self._by_idempotency: dict[tuple[str, str], str] = {}

    async def save_agent_def(self, record: AgentDefRecord) -> AgentDefRecord:
        """Upsert an agent definition keyed by ``agent_id`` (refreshes ``updated_at``)."""
        record.updated_at = utc_now_iso()
        self._index_idempotency(record)
        self._defs[record.agent_id] = record
        return record

    async def save_agent_def_if_absent(
        self, record: AgentDefRecord
    ) -> tuple[AgentDefRecord, bool]:
        """Create an agent def unless its ``(workspace, idempotency_key)`` already exists.

        Returns ``(record, created)``. No ``await`` between the read and the write, so
        two concurrent callers with the same key cannot both create (the in-memory
        analog of the run store's atomic idempotent insert).
        """
        key = record.idempotency_key
        if key is not None:
            existing_id = self._by_idempotency.get((record.workspace_id, key))
            if existing_id is not None:
                existing = self._defs.get(existing_id)
                if existing is not None:
                    return existing, False
        record.updated_at = utc_now_iso()
        self._index_idempotency(record)
        self._defs[record.agent_id] = record
        return record, True

    def _index_idempotency(self, record: AgentDefRecord) -> None:
        """Maintain the (workspace_id, idempotency_key) -> agent_id index."""
        if record.idempotency_key is not None:
            self._by_idempotency.setdefault(
                (record.workspace_id, record.idempotency_key), record.agent_id
            )

    async def get_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> AgentDefRecord | None:
        """Return an agent def by id, tenant-scoped (out-of-workspace reads as None)."""
        rec = self._defs.get(agent_id)
        if rec is None:
            return None
        if workspace_id is not None and rec.workspace_id != workspace_id:
            return None
        return rec

    async def list_agent_defs(
        self, *, workspace_id: str | None = None
    ) -> list[AgentDefRecord]:
        """List agent defs for a workspace (created_at asc)."""
        items = [
            r
            for r in self._defs.values()
            if workspace_id is None or r.workspace_id == workspace_id
        ]
        return sorted(items, key=lambda r: r.created_at)

    async def delete_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> bool:
        """Delete an agent def, tenant-scoped. Returns True iff a row was removed."""
        rec = self._defs.get(agent_id)
        if rec is None:
            return False
        if workspace_id is not None and rec.workspace_id != workspace_id:
            return False
        self._defs.pop(agent_id, None)
        if rec.idempotency_key is not None:
            self._by_idempotency.pop((rec.workspace_id, rec.idempotency_key), None)
        return True


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


class InMemoryTriggerDedupStore:
    """Process-local TTL-CAS dedup ledger (the in-memory analog of ``trigger_dedup``, Q4).

    Mirrors the SQLite/Postgres durable dedup exactly so the
    :class:`~himmy.services.storage.trigger_dedup.DurableIdempotencyStore` behaves the same
    on the zero-config default — minus durability across a process exit (the dict is gone on
    restart, which is precisely the in-memory contract). A row is keyed by ``(scope, key)``;
    ``result is None`` marks an in-flight lease, a string marks a COMPLETED entry. The
    mutation methods have NO ``await`` between read and write, so two concurrent callers on
    one loop cannot both win a claim.
    """

    def __init__(self) -> None:
        # (scope, key) -> {"result": str | None, "expires_at": str | None}
        self._rows: dict[tuple[str, str], dict[str, str | None]] = {}

    async def dedup_try_claim(
        self,
        scope: str,
        key: str,
        *,
        lease_seconds: float,
        now: str | None = None,
    ) -> DedupClaim:
        """Claim ``(scope, key)`` for execution, or report a COMPLETED/in-flight duplicate."""
        from himmy.services.storage.trigger_dedup import (
            CLAIM_DONE,
            CLAIM_IN_FLIGHT,
            CLAIM_WON,
            DedupClaim,
        )

        now_iso = now or utc_now_iso()
        row = self._rows.get((scope, key))
        if row is not None:
            expires = row.get("expires_at")
            live = expires is None or expires > now_iso
            if live:
                if row.get("result") is not None:
                    return DedupClaim(CLAIM_DONE, result=row.get("result"))
                return DedupClaim(CLAIM_IN_FLIGHT)
            # Expired (in-flight lease from a crashed worker OR a lapsed COMPLETED row):
            # fall through and reclaim it as a fresh in-flight lease.
        self._rows[(scope, key)] = {
            "result": None,
            "expires_at": _iso_plus_seconds(now_iso, lease_seconds),
        }
        return DedupClaim(CLAIM_WON)

    async def dedup_complete(
        self,
        scope: str,
        key: str,
        *,
        result: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> None:
        """Upgrade the in-flight row to COMPLETED with ``result`` + the real TTL."""
        now_iso = now or utc_now_iso()
        self._rows[(scope, key)] = {
            "result": result,
            "expires_at": _iso_plus_seconds(now_iso, ttl_seconds),
        }

    async def dedup_release(
        self, scope: str, key: str, *, now: str | None = None
    ) -> None:
        """Drop a won-but-failed in-flight row so a redelivery re-runs (only if in-flight)."""
        row = self._rows.get((scope, key))
        if row is not None and row.get("result") is None:
            self._rows.pop((scope, key), None)

    async def dedup_sweep(self, *, now: str | None = None) -> int:
        """Delete expired rows; return the count removed."""
        now_iso = now or utc_now_iso()
        doomed = [
            k
            for k, row in self._rows.items()
            if row.get("expires_at") is not None
            and str(row.get("expires_at")) <= now_iso
        ]
        for k in doomed:
            self._rows.pop(k, None)
        return len(doomed)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.storage.trigger_dedup import DedupClaim


__all__ = [
    "InMemoryAgentDefStore",
    "InMemoryContextStore",
    "InMemoryEvaluationStore",
    "InMemoryEventLog",
    "InMemoryOrchestrationStore",
    "InMemoryRecommendationStore",
    "InMemoryRunStore",
    "InMemoryThreadStore",
    "InMemoryTriggerDedupStore",
]
