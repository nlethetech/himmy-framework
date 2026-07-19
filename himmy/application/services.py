"""Application kernel: the async run lifecycle, context, recommendations, dashboard.

These services sit above the runtime and storage kernels and own the production
concerns the FastAPI BFF surfaces: idempotent run creation with background
execution, recommendation extraction, status transitions, and a dashboard
summary. They are storage-backed and degrade cleanly without a registry.

Production hardening (see IMPROVEMENTS AAEO-1/3/4/6/8/16):

- Background runs are durable-ish for one process: tracked, cancellable, drained
  on shutdown, bounded by a per-run wall-clock timeout, and swept on startup so
  stuck non-terminal runs reach a terminal state.
- The FAILED-inference path is honoured: ``_execute_run`` reads the terminal
  :class:`~himmy.runtime.single_agent.RunResult` status and sets the run
  FAILED + ``run.error`` instead of recording a garbage SUCCEEDED run.
- Structured output is validated against the requested schema before extraction;
  validation failures are recorded in ``run.metadata['extraction_error']``.
- Read paths are tenant-scoped on ``workspace_id`` (404/None on mismatch).
- List endpoints paginate (limit/offset) with a deterministic created_at-desc
  order and a hard cap.
- ``model_key`` resolves with explicit precedence (caller-set llm_config wins,
  else task.context, else None).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncIterator, Callable, Collection
from typing import TYPE_CHECKING, Any, cast

from himmy.application.models import RecommendationEnvelope
from himmy.application.run_context import _RunContext
from himmy.application.workspace_quota import (
    WorkspaceQuota,
    WorkspaceRunQuotaExceeded,
)
from himmy.config.spec_sanitizer import sanitize_tenant_spec
from himmy.core.errors import HimmyError
from himmy.core.ids import iso_plus_seconds, utc_now_iso
from himmy.entities.lineage import DEFAULT_TRACE_DEPTH
from himmy.entities.records import record_id_for, stable_id_for
from himmy.services.storage.conversations import ORIGIN_CLI
from himmy.services.storage.models import (
    LOCAL_WORKSPACE,
    AgentDefRecord,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)
from himmy.services.tools.validation import validate_against_schema

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.config.agent_spec import AgentSpec
    from himmy.entities.lineage import LineageGraph
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.context.models import ContextField
    from himmy.services.context.service import ContextService
    from himmy.services.inference.models import LLMConfig
    from himmy.services.storage.service import StorageService

logger = logging.getLogger("himmy.application")


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` when it is awaitable, else return it unchanged.

    Lets the run service drive both the synchronous in-memory ``EntityRegistry``
    and an async ``PostgresEntityRegistry`` through one code path.
    """
    if inspect.isawaitable(value):
        return await value
    return value


#: Default and maximum page sizes for list endpoints (AAEO-8).
DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 1000

#: Default per-run wall-clock execution budget (AAEO-1). A run that exceeds this
#: transitions to FAILED with a timeout error rather than hanging forever.
DEFAULT_RUN_TIMEOUT_SECONDS = 300.0

#: Default per-workspace concurrency cap (T0.4): at most this many of one
#: workspace's background runs execute at once; the rest queue on a per-workspace
#: semaphore. Keeps one tenant's fan-out (compare/team/routine) from starving the
#: shared event loop + provider quota for every other tenant.
DEFAULT_WORKSPACE_CONCURRENCY = 8

#: Default cap on a single workspace's OUTSTANDING (created-but-not-finished)
#: background runs (T0.4). A burst beyond this is rejected at create time with a
#: :class:`WorkspaceRunQuotaExceeded` rather than being silently queued forever, so a
#: runaway fan-out cannot pin unbounded memory/tasks. ``0`` disables the cap.
DEFAULT_WORKSPACE_MAX_OUTSTANDING = 64

#: Sentinel default for ``LLMConfig.model_key`` (AAEO-16): a config carrying this
#: value is treated as "caller did not pick a model" so task.context can win.
_DEFAULT_MODEL_KEY = "default"

#: Default retry ceiling for a leased-dispatch run (Q3): the number of CLAIM attempts a
#: transient-failed run gets before it is PARKED. ``1`` would mean no retry; ``3`` gives two
#: re-queues with exponential backoff, which absorbs the common laptop transient (provider
#: blip / model still loading) without churning on a permanently-broken run.
DEFAULT_QUEUE_MAX_ATTEMPTS = 3

#: Margin (seconds) added to ``run_timeout_seconds`` to form the lease TTL (Q3). A run holds
#: its lease for slightly longer than its own execution budget so the terminal-state write
#: lands before the reaper could consider the lease expired.
_LEASE_MARGIN_SECONDS = 30.0

#: The leased-dispatch retry/backoff/PARK policy — the ``_QUEUE_BACKOFF_*`` /
#: ``DEFAULT_QUEUE_MAX_AGE_SECONDS`` constants, the ``_TRANSIENT_ERROR_MARKERS`` list, and the
#: ``_is_transient_run_error`` classifier — moved to :mod:`himmy.application.run_retry` as the
#: ``RetryPolicyEngine`` collaborator (LANE runapp step 5). They are re-imported below so
#: ``himmy.application.services._is_transient_run_error`` / ``DEFAULT_QUEUE_MAX_AGE_SECONDS`` /
#: ``_QUEUE_BACKOFF_BASE_SECONDS`` / ``_QUEUE_BACKOFF_MAX_SECONDS`` stay importable unchanged.


# ``WorkspaceRunQuotaExceeded`` is defined in and re-exported from
# :mod:`himmy.application.workspace_quota` (imported at module top). It stays importable
# from this module — ``from himmy.application.services import WorkspaceRunQuotaExceeded``
# — so the API error handlers and tests that import it here are unchanged.


class HitlNotSupportedError(Exception):
    """A ``hitl=True`` run was requested but no checkpoint store is wired (T2f → 400).

    Raised by :meth:`RunAppService.create_run` when the deployment has no HITL inbox to
    pause into (e.g. a programmatic offline service built without a checkpoint store), so
    a caller gets a clear error rather than a silently non-pausable "HITL" run.
    """


class HitlRequiresAgentError(Exception):
    """A ``hitl=True`` run was requested without a stored agent (T2f → 422).

    HITL needs a tool-bearing per-run runtime (the shared inline runtime has no tools to
    gate), so the run MUST resolve a stored agent (``agent_id``). Raised by
    :meth:`RunAppService.create_run` when ``hitl=True`` is paired with an inline persona.
    """


class RunNotApprovableError(Exception):
    """An approve/reject targeted a run that is not AWAITING_APPROVAL (T2f → 409).

    Raised by :meth:`RunAppService.resume_run` when the run is missing, already terminal,
    or otherwise not paused on a human decision. Carries the run id + observed status so
    the router can return an informative 409 (and a double-approve of an already-resumed
    run is a clean no-op-with-409, never a re-execution of the gated tool).
    """

    def __init__(self, run_id: str, *, status: str) -> None:
        """Record the run and its observed (non-approvable) status."""
        self.run_id = run_id
        self.status = status
        super().__init__(
            f"run {run_id!r} is {status}, not AWAITING_APPROVAL; cannot approve/reject."
        )


def _now() -> str:
    """ISO timestamp helper."""
    return utc_now_iso()


def _iso_plus_seconds(base_iso: str, seconds: float) -> str:
    """ISO instant advanced by ``seconds``."""
    return iso_plus_seconds(base_iso, seconds)


# ``_TRANSIENT_ERROR_MARKERS`` + ``_is_transient_run_error`` moved to
# :mod:`himmy.application.run_retry` (RetryPolicyEngine, LANE runapp step 5); re-imported below
# so ``himmy.application.services._is_transient_run_error`` stays importable unchanged.


def _is_resume_claim_loss(exc: BaseException) -> bool:
    """True when ``exc`` is the exactly-once claim loss from a member resume.

    ``resume_agent_loop`` raises ``HimmyError('checkpoint ... already resolved ...')``
    when the atomic claim is lost (a double/concurrent approve, or a crash-retry the
    winner already finished). For an ORCHESTRATION resume this propagates up through
    ``run_orchestration``, so it must be recognized as a clean NO-OP rather than a
    failure — exactly as the single-agent resume path treats it.
    """
    return isinstance(exc, HimmyError) and "already resolved" in str(exc)


def _paginate(
    items: list[Any],
    *,
    limit: int | None,
    offset: int,
    sort_key: Callable[[Any], Any],
    reverse: bool = True,
) -> list[Any]:
    """Deterministically sort + window a list (AAEO-8).

    ``limit`` is clamped to ``[0, MAX_PAGE_LIMIT]`` (``None`` -> default cap);
    ``offset`` is floored at 0. The sort is stable and total (the ``sort_key``
    must break ties), so paging is repeatable across calls and backends.
    """
    ordered = sorted(items, key=sort_key, reverse=reverse)
    start = max(0, offset)
    if limit is None:
        limit = DEFAULT_PAGE_LIMIT
    capped = max(0, min(limit, MAX_PAGE_LIMIT))
    return ordered[start : start + capped]


class ContextAppService:
    """Operator-facing wrapper over :class:`ContextService` + :class:`StorageService`."""

    def __init__(
        self,
        *,
        context_service: ContextService,
        storage: StorageService,
    ) -> None:
        """Wire the context engine and its backing store."""
        self._context = context_service
        self._storage = storage

    async def upsert_fields(
        self,
        workspace_id: str,
        subject_id: str,
        fields: list[ContextField],
    ) -> list[ContextField]:
        """Bulk-upsert context fields for a subject (stamping subject + workspace scope)."""
        saved: list[Any] = []
        for field in fields:
            meta = getattr(field, "metadata", {}) or {}
            if (
                meta.get("subject_id") != subject_id
                or meta.get("workspace_id") != workspace_id
            ):
                field.metadata = {
                    **meta,
                    "subject_id": subject_id,
                    "workspace_id": workspace_id,
                }
            saved.append(await self._storage.save_context_field(field))
        return saved

    async def list_fields(
        self, subject_id: str, *, workspace_id: str | None = None
    ) -> list[ContextField]:
        """List stored context fields for a subject, scoped to a workspace (AAEO-4).

        When ``workspace_id`` is supplied, only fields stamped with that workspace
        are returned, so two workspaces sharing a ``subject_id`` cannot see each
        other's fields.
        """
        fields = await self._storage.list_context_fields(subject_id)
        if workspace_id is None:
            return fields
        return [
            f
            for f in fields
            if (getattr(f, "metadata", {}) or {}).get("workspace_id") == workspace_id
        ]

    async def build_snapshot(
        self,
        *,
        subject_id: str,
        task_id: str | None = None,
        build_spec: Any,
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        """Build and persist an evidenced context snapshot.

        ``workspace_id`` (AAEO-4) tenant-scopes the field-RESOLUTION path so a snapshot
        built in one workspace cannot surface another workspace's stored ``context_fields``
        value under a shared ``subject_id`` (the store is keyed globally by
        ``(subject_id, key)``). ``None`` keeps resolution unscoped (offline/all-tenants).
        """
        return await self._context.build_snapshot(
            subject_id=subject_id,
            task_id=task_id,
            build_spec=build_spec,
            metadata=metadata,
            workspace_id=workspace_id,
        )

    async def get_snapshot(
        self, snapshot_id: str, *, workspace_id: str | None = None
    ) -> Any:
        """Load a stored snapshot by id, or None.

        When ``workspace_id`` is supplied (AAEO-4), a snapshot whose subject's
        fields/metadata do not belong to that workspace is treated as not found.
        Snapshots carry their workspace under ``metadata['workspace_id']`` when
        built through this service.
        """
        snapshot = await self._storage.load_snapshot(snapshot_id)
        if snapshot is None:
            return None
        if workspace_id is not None and not _snapshot_in_workspace(
            snapshot, workspace_id
        ):
            return None
        return snapshot


def _snapshot_in_workspace(snapshot: Any, workspace_id: str) -> bool:
    """Whether a snapshot belongs to ``workspace_id`` (FAILS CLOSED on unstamped).

    The only tenant boundary for a snapshot read: ``load_snapshot`` is keyed by
    ``snapshot_id`` alone (the ``context_snapshots`` table has no ``workspace_id`` column),
    so a lenient verdict here is the sole gate. This is reached ONLY when a concrete
    ``workspace_id`` was supplied — i.e. a tenant-bound caller (an unrestricted /
    ``all_tenants`` caller resolves to ``workspace_id is None`` and never calls this; see
    :meth:`ContextAppService.get_snapshot`). For such a tenant-bound caller it FAILS CLOSED:
    the snapshot must carry a workspace stamp (its own ``metadata['workspace_id']`` or a
    field-level stamp) that EQUALS the caller's workspace. An UNSTAMPED snapshot — built by
    the offline/CLI/``all_tenants``/routine path that does not stamp a workspace — is treated
    as NOT in any tenant's workspace and is refused, closing the cross-tenant IDOR where any
    tenant-bound principal could read an unstamped snapshot by guessing its id. The offline /
    ``all_tenants`` path is byte-unchanged because it never supplies a workspace and so never
    reaches this gate.
    """
    meta = getattr(snapshot, "metadata", {}) or {}
    stamped = meta.get("workspace_id")
    if stamped is None:
        # Also check field-level scope as a fallback.
        for field in (getattr(snapshot, "fields", {}) or {}).values():
            fmeta = getattr(field, "metadata", {}) or {}
            ws = fmeta.get("workspace_id")
            if ws is not None:
                stamped = ws
                break
    # Fail CLOSED: a tenant-bound caller (workspace_id is non-None to even reach here) may
    # read a snapshot ONLY if it carries a matching workspace stamp. Unstamped → refused.
    return stamped == workspace_id


class RecommendationAppService:
    """Extract, list, and transition advisory recommendations from runs."""

    def __init__(
        self,
        *,
        storage: StorageService,
        entity_registry: EntityRegistryProtocol | None = None,
    ) -> None:
        """Wire the backing store and (optionally) the lineage registry.

        When ``entity_registry`` is present, extracted recommendations are projected
        as first-class ``recommendation`` entities and linked back to the run they
        came from (``derived_from``) and the evidence they cite (``cites``), so a
        recommendation is queryable in the provenance graph.
        """
        self._storage = storage
        self._registry = entity_registry

    async def extract_from_run(self, run: RunRecord) -> list[RecommendationItem]:
        """Parse a run's structured output as a RecommendationEnvelope and persist items.

        Returns the created items. Non-matching output yields an empty list; this
        is the no-hand-wiring path the API relies on. A dict that *looks* like an
        envelope (has a ``recommendations`` key) but fails coercion records an
        ``extraction_error`` on ``run.metadata`` (AAEO-6) and persists the run so
        the schema-failure is visible instead of silently swallowed.
        """
        envelope, error = self._coerce_envelope(run.output_structured)
        if error is not None:
            run.metadata = {**(run.metadata or {}), "extraction_error": error}
            try:
                await self._storage.save_run(run)
            except Exception:  # pragma: no cover - best-effort persistence
                logger.warning(
                    "failed to persist extraction_error for run %s", run.run_id
                )
        if envelope is None:
            return []
        items: list[RecommendationItem] = []
        for rec in envelope.recommendations:
            item = RecommendationItem(
                run_id=run.run_id,
                workspace_id=run.workspace_id,
                subject_id=run.subject_id,
                kind=rec.kind,
                title=rec.title,
                summary=rec.summary,
                rationale=rec.rationale,
                confidence=rec.confidence,
                evidence_refs=list(rec.evidence_refs),
                metadata=dict(rec.metadata),
            )
            await self._storage.save_recommendation(item)
            await self._project_lineage(item, run)
            items.append(item)
        return items

    async def _project_lineage(self, item: RecommendationItem, run: RunRecord) -> None:
        """Register the recommendation as a lineage node and link its provenance.

        Best-effort and idempotent: a registry failure never loses the (already
        persisted) recommendation, and re-extracting the same run neither
        duplicates the record (content-addressed) nor its links (deduped). Links:
        ``derived_from`` -> the run's chat_thread hub (so the recommendation joins
        the run's existing graph), and ``cites`` -> each cited evidence record that
        actually exists in the registry (dangling citations stay in the payload,
        not the graph). Works against the sync in-memory and async Postgres
        registries alike via :func:`_maybe_await`.
        """
        if self._registry is None:
            return
        try:
            rec_record = await _maybe_await(self._registry.register(item.to_record()))

            if run.thread_id:
                thread_sid = stable_id_for(run.thread_id, namespace="chat_thread")
                thread_record = await _maybe_await(
                    self._registry.get_latest(thread_sid)
                )
                if thread_record is not None:
                    await self._link_once(
                        rec_record.record_id,
                        thread_record.record_id,
                        "derived_from",
                    )

            for ref in item.evidence_refs:
                ev_sid = stable_id_for(ref, namespace="context_evidence")
                ev_rid = record_id_for(
                    stable_id=ev_sid, version=1, kind="context_evidence"
                )
                # Only graph a citation whose evidence is a real registered node.
                if await _maybe_await(self._registry.get(ev_rid)) is not None:
                    await self._link_once(
                        rec_record.record_id,
                        ev_rid,
                        "cites",
                        metadata={"evidence_ref": ref},
                    )
        except Exception:  # pragma: no cover - lineage projection is best-effort
            logger.warning(
                "failed to project recommendation lineage for %s",
                item.recommendation_id,
            )

    async def _link_once(
        self,
        from_record_id: str,
        to_record_id: str,
        relation: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a link only if an identical (from, to, relation) edge is absent."""
        assert self._registry is not None  # guarded by caller
        existing = await _maybe_await(self._registry.links_from(from_record_id))
        for link in existing:
            if link.to_record_id == to_record_id and link.relation == relation:
                return
        await _maybe_await(
            self._registry.link(
                from_record_id=from_record_id,
                to_record_id=to_record_id,
                relation=relation,
                metadata=metadata or {},
            )
        )

    @staticmethod
    def _coerce_envelope(
        structured: Any,
    ) -> tuple[RecommendationEnvelope | None, str | None]:
        """Coerce structured output into an envelope, surfacing the failure reason.

        Returns ``(envelope, error)``: ``error`` is non-None only when the output
        looked like an envelope (a dict with ``recommendations``) but could not be
        validated into one — that is the schema-failure signal AAEO-6 records.
        Output that is simply not envelope-shaped returns ``(None, None)``.
        """
        if structured is None:
            return None, None
        if isinstance(structured, RecommendationEnvelope):
            return structured, None
        if isinstance(structured, dict) and "recommendations" in structured:
            try:
                return RecommendationEnvelope.model_validate(structured), None
            except Exception as exc:  # noqa: BLE001 - record, don't swallow
                return None, f"recommendation envelope coercion failed: {exc}"
        return None, None

    async def list(
        self,
        *,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        status: RecommendationStatus | None = None,
        limit: int | None = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> list[RecommendationItem]:
        """List recommendation items filtered by dimensions, paginated (AAEO-8).

        Results are ordered ``created_at`` desc with ``recommendation_id`` as the
        tiebreak, then windowed by ``offset``/``limit`` (capped at
        :data:`MAX_PAGE_LIMIT`).
        """
        items = await self._storage.list_recommendations(
            workspace_id=workspace_id,
            subject_id=subject_id,
            run_id=run_id,
            kind=kind,
            status=status,
        )
        return _paginate(
            items,
            limit=limit,
            offset=offset,
            sort_key=lambda r: (r.created_at, r.recommendation_id),
        )

    async def count(
        self,
        *,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        status: RecommendationStatus | None = None,
    ) -> int:
        """Total recommendation count for the filter (for pagination envelopes)."""
        items = await self._storage.list_recommendations(
            workspace_id=workspace_id,
            subject_id=subject_id,
            run_id=run_id,
            kind=kind,
            status=status,
        )
        return len(items)

    async def get(
        self,
        recommendation_id: str,
        *,
        workspace_id: str | None = None,
    ) -> RecommendationItem | None:
        """Return one recommendation by id, tenant-scoped (AAEO-4).

        The app-service getter the read surfaces (``himmy recommendations show`` and
        the /v1 GET-by-id path) need: a recommendation belonging to another workspace
        is treated as not found (returns None) instead of being read cross-tenant.
        Until now the service exposed only ``list``/``count``/``update_status`` plus
        ``get_recommendation_lineage`` — a single-item read had no app-layer entry, so
        a caller had to reach into ``storage.get_recommendation`` and re-implement the
        scope check. This closes that gap (reviewer must_fix for T2h).
        """
        item = await self._storage.get_recommendation(recommendation_id)
        if item is None:
            return None
        if workspace_id is not None and item.workspace_id != workspace_id:
            return None
        return item

    async def update_status(
        self,
        recommendation_id: str,
        *,
        status: RecommendationStatus,
        notes: str | None = None,
        workspace_id: str | None = None,
    ) -> RecommendationItem | None:
        """Transition a recommendation's status and optionally attach notes.

        When ``workspace_id`` is supplied (AAEO-4), a recommendation belonging to
        a different workspace is treated as not found (returns None) instead of
        being mutated cross-tenant.
        """
        if workspace_id is not None:
            existing = await self._storage.get_recommendation(recommendation_id)
            if existing is None or existing.workspace_id != workspace_id:
                return None
        return await self._storage.update_recommendation(
            recommendation_id, status=status, notes=notes
        )

    async def get_recommendation_lineage(
        self,
        recommendation_id: str,
        *,
        workspace_id: str | None = None,
        max_depth: int = DEFAULT_TRACE_DEPTH,
        relations: Collection[str] | None = None,
    ) -> LineageGraph | None:
        """Return the provenance subgraph for a recommendation, or None.

        Traces from the recommendation node outward: ``derived_from`` reaches the
        run's thread hub (and through it the persona, prompt, and context
        snapshot), while ``cites`` reaches the evidence the recommendation stands
        on. This is the literal "trace THIS recommendation back to its persona +
        evidence" demo the README promises.

        Returns None when the recommendation is unknown / out-of-workspace, no
        registry is wired, or it was never projected into the graph.
        """
        item = await self._storage.get_recommendation(recommendation_id)
        if item is None or self._registry is None:
            return None
        if workspace_id is not None and item.workspace_id != workspace_id:
            return None
        stable_id = stable_id_for(recommendation_id, namespace="recommendation")
        root = await _maybe_await(self._registry.get_latest(stable_id))
        if root is None:
            return None
        return cast(
            "LineageGraph | None",
            await _maybe_await(
                self._registry.trace(
                    root.record_id, max_depth=max_depth, relations=relations
                )
            ),
        )


# Imported here (not at module top) to break the import cycle: ``run_reads`` reads the
# ``_paginate`` / ``_maybe_await`` / ``DEFAULT_PAGE_LIMIT`` helpers defined above from this
# module, so its import must run only after those names are bound.
from himmy.application.run_enqueue import RunEnqueuer  # noqa: E402
from himmy.application.run_reads import RunReadService  # noqa: E402
from himmy.application.run_recovery import RunRecovery  # noqa: E402
from himmy.application.run_retry import (  # noqa: E402
    RetryPolicyEngine,
)
from himmy.application.run_side_effects import RunSideEffects  # noqa: E402


class RunAppService:
    """Owns the async run lifecycle: idempotent create + background execution."""

    def __init__(
        self,
        *,
        runtime: SingleAgentRuntime,
        storage: StorageService,
        entity_registry: EntityRegistryProtocol | None = None,
        recommendation_app: RecommendationAppService | None = None,
        run_timeout_seconds: float = DEFAULT_RUN_TIMEOUT_SECONDS,
        workspace_concurrency: int = DEFAULT_WORKSPACE_CONCURRENCY,
        workspace_max_outstanding: int = DEFAULT_WORKSPACE_MAX_OUTSTANDING,
        checkpoint_store: Any = None,
        agent_resolver: Callable[..., Any] | None = None,
        conversation_sink: Callable[..., Any] | None = None,
        graph_checkpoint_store_provider: Callable[[], Any] | None = None,
        access_policy: Any = None,
    ) -> None:
        """Wire the runtime, store, optional registry, and recommendation service.

        ``run_timeout_seconds`` (AAEO-1) bounds each background run's wall clock;
        a run that exceeds it transitions to FAILED with a timeout error.

        ``workspace_concurrency`` / ``workspace_max_outstanding`` (T0.4) bound the
        per-workspace fan-out so one tenant cannot exhaust the shared event loop +
        provider quota: at most ``workspace_concurrency`` of a workspace's runs
        execute the runtime at once (the rest wait on a per-workspace semaphore), and
        a workspace may hold at most ``workspace_max_outstanding`` created-but-not-
        terminal runs before :meth:`create_run` rejects with
        :class:`WorkspaceRunQuotaExceeded`. Both default to safe values and apply to
        EVERY background run path (inline persona and per-run-spec alike), so the
        offline single-workspace default behaves identically while a multi-tenant
        deployment is protected.

        ``checkpoint_store`` (T2f) is the SURFACE-OWNED durable HITL inbox: when a
        ``hitl=True`` run hits an approval-gated tool the per-run runtime pauses into a
        checkpoint here, and :meth:`resume_run` re-claims it on approve/reject. This is
        /v1's OWN store, distinct from Studio's — the two surfaces rebuild a paused run
        from different spec sources (Studio from a filesystem ``agent_path``, /v1 from
        the stored DB ``AgentSpec`` resolved via ``agent_resolver``), so a shared inbox
        is a deferred item. ``None`` (the default) leaves the offline/inline path
        byte-unchanged and disables HITL (a ``hitl=True`` create is then rejected).

        ``agent_resolver`` is an async ``(agent_id, workspace_id) -> AgentDefRecord |
        None`` used to rebuild the per-run tool-bearing runtime FROM THE STORED DB SPEC
        on resume (``resume_agent_loop`` HARD-requires a ``tool_service``, so /v1 must
        rebuild from the spec, not a filesystem path it never had).

        ``conversation_sink`` (T2g) is an optional async ``(conversation_id, run) ->
        None`` invoked whenever a THREAD-LINKED run (one carrying
        ``metadata['conversation_id']``) reaches a terminal / AWAITING_APPROVAL state, so
        the :class:`~himmy.services.storage.conversations.ConversationStore` projection
        stays in sync after a continuation OR an approval-resume (which completes on a
        background task the thread router cannot await). Best-effort: a sink failure never
        fails the run. ``None`` (the default) is a no-op — runs not started via the thread
        router carry no ``conversation_id`` and never reach it.
        """
        # -- Shared run-lifecycle context (foundation collaborator). Holds the immutable
        # handles, the SINGLE background-task set, and the RUNTIME-mutable dispatch tunables
        # in one object the service — and later collaborators — read THROUGH. The former
        # inline attributes (``_runtime`` / ``_storage`` / ``_tasks`` / ``_dispatch_enabled`` …)
        # are preserved as delegating properties below, so the read/poke surface is
        # byte-identical.
        #
        # Tunable defaults seeded here match the former inline assignment order:
        #   ``_dispatch_enabled`` False = the inline fire-and-forget behaviour (today's
        #   ``asyncio.create_task`` per run), preserving the offline single-box +
        #   bare-TestClient path byte-for-byte; the Q3 :class:`RunDispatcher` flips it on via
        #   :meth:`enable_dispatch`. ``_default_max_attempts`` is the enqueue retry ceiling;
        #   ``_dispatch_fairness`` False = global FIFO (byte-identical single-tenant claim).
        #
        # ``access_policy`` (P0 tool authz) rebuilds a run's tool-capability gate from the
        # persisted ``actor`` at build time; ``None`` (offline default) builds no per-run
        # authorizer (byte-identical dispatch). ``graph_checkpoint_store_provider`` supplies
        # the durable graph store an orchestration HITL resume rejoins; ``None`` falls back
        # to an in-memory store. The ``_tasks`` set keeps strong refs so background tasks are
        # not GC'd mid-flight and can be drained/cancelled on shutdown (AAEO-1).
        self._ctx = _RunContext(
            runtime=runtime,
            storage=storage,
            registry=entity_registry,
            recommendations=recommendation_app
            or RecommendationAppService(
                storage=storage, entity_registry=entity_registry
            ),
            run_timeout_seconds=run_timeout_seconds,
            checkpoint_store=checkpoint_store,
            agent_resolver=agent_resolver,
            conversation_sink=conversation_sink,
            access_policy=access_policy,
            graph_checkpoint_store_provider=graph_checkpoint_store_provider,
            dispatch_enabled=False,
            default_max_attempts=DEFAULT_QUEUE_MAX_ATTEMPTS,
            dispatch_fairness=False,
        )
        # T0.4 per-workspace quotas, owned by a :class:`WorkspaceQuota` collaborator: it
        # gates concurrent EXECUTION (a per-workspace semaphore, created lazily on the
        # running loop) and counts created-but-not-terminal runs for the admission cap.
        # The former inline attributes are preserved as delegating properties/methods
        # (``_workspace_max_outstanding`` / ``_ws_outstanding`` / ``_admit_workspace_run`` …)
        # so behaviour and the introspection surface are byte-identical.
        self._ws_quota = WorkspaceQuota(
            concurrency=workspace_concurrency,
            max_outstanding=workspace_max_outstanding,
        )
        # Tenant-scoped NO-MUTATION reads (get_run/list_runs/lineage/events/thread/…) live on
        # a :class:`RunReadService` collaborator reading its handles LIVE through the shared
        # context; the public read methods below are thin delegating shims (behaviour and the
        # internal ``self.get_run`` callers are byte-identical).
        self._reads = RunReadService(context=self._ctx)
        # Cross-cutting run side effects (tool-authz gate, subject scope, and the two
        # best-effort lineage/conversation projections) live on a :class:`RunSideEffects`
        # collaborator reading its handles LIVE through the shared context; the former
        # private methods below are thin delegating shims (behaviour, the ambient
        # contextvar set/reset lifetime at the call sites, and every internal caller/test
        # poke are byte-identical).
        self._side_effects = RunSideEffects(context=self._ctx)
        # The leased-dispatch retry/backoff/PARK policy (Q3) lives on a
        # :class:`RetryPolicyEngine` collaborator reading its storage handle LIVE through the
        # shared context; ``_apply_retry_policy`` below is a thin delegating shim (behaviour and
        # the test poke of ``run_app._apply_retry_policy`` are byte-identical).
        self._retry = RetryPolicyEngine(context=self._ctx)
        # The startup/shutdown recovery lifecycle (shutdown ``drain`` + startup
        # ``sweep_stuck_runs`` / ``reconcile_resolving_runs``) lives on a :class:`RunRecovery`
        # collaborator reading its ``storage`` / the SINGLE shared ``_tasks`` set /
        # ``dispatch_enabled`` LIVE through the context; ``reconcile_resolving_runs`` re-drives a
        # stranded RESOLVING run back through ``self.resume_run`` (passed as a callback). The
        # public methods below are thin delegating shims (behaviour and every lifespan/test poke
        # are byte-identical).
        self._recovery = RunRecovery(context=self._ctx, resume_run=self.resume_run)
        # Idempotent create + Q3 queue-field stamping (LANE runapp step 7 — enqueue): the
        # :class:`RunEnqueuer` collaborator owns ``create_run`` / ``continue_thread`` /
        # ``_stamp_queue_fields`` / ``_launch_or_enqueue``. It reads its store/checkpoint/
        # dispatch/attempts/tasks handles LIVE through the shared context and the outstanding
        # cap LIVE from the shared :class:`WorkspaceQuota`, and delegates the durable count-cap
        # + the drive path back to the service (``_admit_workspace_run_durable`` /
        # ``_execute_run``). The public/private methods below are thin delegating shims so every
        # router, CLI path, and test caller stays byte-identical.
        self._enqueue = RunEnqueuer(
            context=self._ctx,
            ws_quota=self._ws_quota,
            admit_workspace_run_durable=self._admit_workspace_run_durable,
            execute_run=self._execute_run,
        )
        # The coupled execution/HITL/orchestration CORE (LANE runapp step 8 — the
        # riskiest, last slice): the :class:`RunDriveEngine` collaborator owns the run-drive
        # primitives (``_execute_run`` / ``_execute_on_runtime`` / ``_finalize_succeeded_run`` /
        # ``_resolve_runtime`` / ``dispatch_claimed_run``), the HITL approve/reject resume
        # coordinator (``resume_run`` / ``pending_approvals`` / ``_resume_in_background`` /
        # ``_apply_loop_outcome`` / the orchestration-resume band), and the team/workflow
        # executor (``create_orchestration_run`` / ``_execute_orchestration_run``). They share
        # the drive primitives, finalizers, and the ambient tool-authorizer scope, so they are
        # extracted TOGETHER into one collaborator. It reads every store/runtime/dispatch/
        # timeout handle LIVE through the shared context, the outstanding cap + semaphores LIVE
        # from the shared :class:`WorkspaceQuota`, and delegates the cross-cutting side effects
        # (tool-authz gate, subject scope, lineage/conversation projection), the retry policy,
        # and the tenant-scoped reads to the SAME collaborators the service uses — so dispatch
        # order, lease/idempotency/HITL semantics, and the error taxonomy are byte-identical.
        # ``himmy.application.orchestration_runner`` stays FUNCTION-LOCAL inside it (re-entering
        # ``application/__init__`` at import time would partial-import). The public/test-poked
        # methods below are thin delegating shims so every router, dispatcher, CLI path, and
        # test caller stays byte-identical.
        self._drive = RunDriveEngine(
            context=self._ctx,
            ws_quota=self._ws_quota,
            side_effects=self._side_effects,
            retry=self._retry,
            reads=self._reads,
        )

    # -- Backward-compatible views over the shared :class:`_RunContext`. These preserve the
    # exact private attributes the former inline implementation exposed, so callers/tests
    # that read or poke them behave identically, while the single source of truth (and the
    # live-read tunables) lives on the context.
    @property
    def _runtime(self) -> SingleAgentRuntime:
        return self._ctx.runtime

    @_runtime.setter
    def _runtime(self, value: SingleAgentRuntime) -> None:
        self._ctx.runtime = value

    @property
    def _storage(self) -> StorageService:
        return self._ctx.storage

    @_storage.setter
    def _storage(self, value: StorageService) -> None:
        self._ctx.storage = value

    @property
    def _registry(self) -> EntityRegistryProtocol | None:
        return self._ctx.registry

    @_registry.setter
    def _registry(self, value: EntityRegistryProtocol | None) -> None:
        self._ctx.registry = value

    @property
    def _recommendations(self) -> RecommendationAppService:
        return self._ctx.recommendations

    @_recommendations.setter
    def _recommendations(self, value: RecommendationAppService) -> None:
        self._ctx.recommendations = value

    @property
    def _checkpoint_store(self) -> Any:
        return self._ctx.checkpoint_store

    @_checkpoint_store.setter
    def _checkpoint_store(self, value: Any) -> None:
        self._ctx.checkpoint_store = value

    @property
    def _agent_resolver(self) -> Callable[..., Any] | None:
        return self._ctx.agent_resolver

    @_agent_resolver.setter
    def _agent_resolver(self, value: Callable[..., Any] | None) -> None:
        self._ctx.agent_resolver = value

    @property
    def _conversation_sink(self) -> Callable[..., Any] | None:
        return self._ctx.conversation_sink

    @_conversation_sink.setter
    def _conversation_sink(self, value: Callable[..., Any] | None) -> None:
        self._ctx.conversation_sink = value

    @property
    def _access_policy(self) -> Any:
        return self._ctx.access_policy

    @_access_policy.setter
    def _access_policy(self, value: Any) -> None:
        self._ctx.access_policy = value

    @property
    def _graph_checkpoint_store_provider(self) -> Callable[[], Any] | None:
        return self._ctx.graph_checkpoint_store_provider

    @_graph_checkpoint_store_provider.setter
    def _graph_checkpoint_store_provider(
        self, value: Callable[[], Any] | None
    ) -> None:
        self._ctx.graph_checkpoint_store_provider = value

    @property
    def _tasks(self) -> set[asyncio.Task[Any]]:
        return self._ctx.tasks

    @_tasks.setter
    def _tasks(self, value: set[asyncio.Task[Any]]) -> None:
        self._ctx.tasks = value

    @property
    def _run_timeout_seconds(self) -> float:
        return self._ctx.run_timeout_seconds

    @_run_timeout_seconds.setter
    def _run_timeout_seconds(self, value: float) -> None:
        self._ctx.run_timeout_seconds = value

    @property
    def _dispatch_enabled(self) -> bool:
        return self._ctx.dispatch_enabled

    @_dispatch_enabled.setter
    def _dispatch_enabled(self, value: bool) -> None:
        self._ctx.dispatch_enabled = value

    @property
    def _default_max_attempts(self) -> int:
        return self._ctx.default_max_attempts

    @_default_max_attempts.setter
    def _default_max_attempts(self, value: int) -> None:
        self._ctx.default_max_attempts = value

    @property
    def _dispatch_fairness(self) -> bool:
        return self._ctx.dispatch_fairness

    @_dispatch_fairness.setter
    def _dispatch_fairness(self, value: bool) -> None:
        self._ctx.dispatch_fairness = value

    def enable_dispatch(
        self,
        *,
        max_attempts: int | None = None,
        run_timeout_seconds: float | None = None,
        fairness: bool | None = None,
    ) -> None:
        """Switch this service into leased-dispatch mode (the Q3 dispatcher owns execution).

        Called by the :class:`~himmy.application.dispatcher.RunDispatcher` at server startup
        once it has confirmed the durable run store is active. From this point new runs are
        ENQUEUED (persisted QUEUED with recoverable input) instead of fire-and-forgotten, and
        the dispatcher claims them. Idempotent.

        ``max_attempts`` (the enqueue retry ceiling) and ``run_timeout_seconds`` (each run's
        wall clock, which also bases the lease TTL) are optional operator overrides; ``None``
        leaves the construction-time value untouched. A non-positive value is clamped up.
        """
        self._dispatch_enabled = True
        if max_attempts is not None:
            self._default_max_attempts = max(1, int(max_attempts))
        if run_timeout_seconds is not None:
            self._run_timeout_seconds = max(1.0, float(run_timeout_seconds))
        if fairness is not None:
            self._dispatch_fairness = bool(fairness)

    @property
    def dispatch_enabled(self) -> bool:
        """Whether the leased dispatcher owns execution (vs. inline fire-and-forget)."""
        return self._dispatch_enabled

    @property
    def dispatch_fairness(self) -> bool:
        """Whether the dispatcher claims runs with per-tenant fairness + the cross-node cap (T3)."""
        return self._dispatch_fairness

    @property
    def workspace_concurrency(self) -> int:
        """The per-workspace execution-concurrency cap (T0.4 in-process / T3 cross-node)."""
        return self._ws_quota.concurrency

    @property
    def workspace_max_outstanding(self) -> int:
        """The per-workspace outstanding-run cap enforced at enqueue (T0.4 / T3)."""
        return self._ws_quota.max_outstanding

    # -- Backward-compatible views over the WorkspaceQuota collaborator's state. These
    # preserve the exact private attributes the former inline implementation exposed, so
    # existing callers/tests that read or poke them behave identically.
    @property
    def _workspace_max_outstanding(self) -> int:
        return self._ws_quota.max_outstanding

    @_workspace_max_outstanding.setter
    def _workspace_max_outstanding(self, value: int) -> None:
        self._ws_quota.max_outstanding = value

    @property
    def _ws_outstanding(self) -> dict[str, int]:
        return self._ws_quota.outstanding_counts

    @property
    def run_timeout_seconds(self) -> float:
        """Each background run's wall-clock timeout (AAEO-1); also the lease TTL basis."""
        return self._run_timeout_seconds

    @property
    def default_max_attempts(self) -> int:
        """The retry ceiling stamped on every enqueued run (the dispatcher's backoff budget)."""
        return self._default_max_attempts

    @property
    def lease_seconds(self) -> float:
        """The lease TTL a claim should hold, derived from the run timeout (Q3).

        A run can't legitimately run longer than its own wall-clock timeout, so the lease is
        the timeout plus a margin for the terminal-state write — long enough that a live
        worker's heartbeat keeps it, short enough that a crashed worker's run is re-queued
        soon after it would have finished.
        """
        return self._run_timeout_seconds + _LEASE_MARGIN_SECONDS

    @property
    def storage(self) -> StorageService:
        """The backing run/thread/event store (the dispatcher claims runs through it, Q3)."""
        return self._storage

    async def create_run(
        self,
        *,
        workspace_id: str,
        subject_id: str,
        persona: Persona,
        task: Task,
        idempotency_key: str | None = None,
        llm_config: LLMConfig | None = None,
        actor: dict[str, Any] | None = None,
        agent_spec: AgentSpec | None = None,
        agent_def: AgentDefRecord | None = None,
        operator_provisioned: bool = False,
        hitl: bool = False,
        plan: bool = False,
    ) -> RunRecord:
        """Create (or return the existing) run and launch background execution.

        Thin delegating shim over the :class:`RunEnqueuer` collaborator (LANE runapp
        step 7 — enqueue), which owns the idempotent create + Q3 queue-field stamping +
        admission-then-drive fork. Signature, idempotency, admission ordering, and the
        :class:`WorkspaceRunQuotaExceeded` / :class:`HitlNotSupportedError` /
        :class:`HitlRequiresAgentError` error surface are byte-identical.
        """
        return await self._enqueue.create_run(
            workspace_id=workspace_id,
            subject_id=subject_id,
            persona=persona,
            task=task,
            idempotency_key=idempotency_key,
            llm_config=llm_config,
            actor=actor,
            agent_spec=agent_spec,
            agent_def=agent_def,
            operator_provisioned=operator_provisioned,
            hitl=hitl,
            plan=plan,
        )

    async def continue_thread(
        self,
        *,
        workspace_id: str,
        subject_id: str,
        conversation_id: str,
        thread: ChatThread,
        prompt: str,
        agent_spec: AgentSpec,
        agent_def: AgentDefRecord,
        llm_config: LLMConfig | None = None,
        idempotency_key: str | None = None,
        actor: dict[str, Any] | None = None,
        operator_provisioned: bool = False,
        hitl: bool = False,
        plan: bool = False,
    ) -> RunRecord:
        """Continue a stored conversation with a new user turn on the per-run runtime (T2g).

        Thin delegating shim over the :class:`RunEnqueuer` collaborator; the admission +
        sanitizer + quota path and the run<->conversation linkage are byte-identical.
        """
        return await self._enqueue.continue_thread(
            workspace_id=workspace_id,
            subject_id=subject_id,
            conversation_id=conversation_id,
            thread=thread,
            prompt=prompt,
            agent_spec=agent_spec,
            agent_def=agent_def,
            llm_config=llm_config,
            idempotency_key=idempotency_key,
            actor=actor,
            operator_provisioned=operator_provisioned,
            hitl=hitl,
            plan=plan,
        )

    # --------------------------------------------------------- Q3 enqueue/dispatch
    def _stamp_queue_fields(
        self,
        run: RunRecord,
        *,
        model_key: str | None,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None,
        hitl: bool,
        plan: bool,
    ) -> None:
        """Populate the leased-queue fields on a single-agent run when dispatch is on (Q3).

        Thin delegating shim over the :class:`RunEnqueuer` collaborator (kept so any caller
        or test poke of ``run_app._stamp_queue_fields`` stays byte-identical).
        """
        self._enqueue._stamp_queue_fields(
            run,
            model_key=model_key,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            hitl=hitl,
            plan=plan,
        )

    async def _launch_or_enqueue(
        self,
        stored: RunRecord,
        *,
        workspace_id: str,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None,
        agent_def: AgentDefRecord | None,
        hitl: bool,
        plan: bool,
        thread: ChatThread | None = None,
        quota_already_admitted: bool = False,
    ) -> RunRecord:
        """Admit + (inline) launch OR (dispatch) leave QUEUED for the dispatcher (Q3).

        Thin delegating shim over the :class:`RunEnqueuer` collaborator (kept so any caller
        or test poke of ``run_app._launch_or_enqueue`` stays byte-identical).
        """
        return await self._enqueue._launch_or_enqueue(
            stored,
            workspace_id=workspace_id,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            agent_def=agent_def,
            hitl=hitl,
            plan=plan,
            thread=thread,
            quota_already_admitted=quota_already_admitted,
        )

    # ----------------------------------------------------------- T0.4 quotas
    def _admit_workspace_run(self, workspace_id: str) -> None:
        """Reserve one outstanding-run slot for ``workspace_id`` or reject (T0.4).

        Delegates to :class:`WorkspaceQuota`. Raises :class:`WorkspaceRunQuotaExceeded`
        when the workspace already holds its outstanding-run cap. ``0`` disables the cap.
        """
        self._ws_quota.admit(workspace_id)

    async def _admit_workspace_run_durable(
        self, workspace_id: str, stored: RunRecord
    ) -> None:
        """Reject a QUEUED enqueue when the tenant is at its outstanding cap (T3, dispatch).

        The cross-node analog of :meth:`_admit_workspace_run`: instead of an in-RAM counter
        (which a sibling worker process cannot see), it COUNTS the workspace's non-terminal
        runs in the shared store. A workspace already at :attr:`_workspace_max_outstanding`
        in-flight runs has THIS run marked FAILED (record preserved, not orphaned QUEUED) and
        :class:`WorkspaceRunQuotaExceeded` raised (HTTP 429). The reserved ``__local__``
        single-user workspace is EXEMPT so the offline/single-tenant dispatch path is unchanged
        (it had no cap before). ``0`` disables the cap. The stored run already exists, so the
        count includes it; the cap is treated as a ceiling ON TOP of this run (``> cap``).
        """
        if self._workspace_max_outstanding <= 0:
            return
        if workspace_id == LOCAL_WORKSPACE:
            return
        try:
            active = await self._storage.count_active_runs_for_workspace(workspace_id)
        except Exception:  # noqa: BLE001 - fail OPEN: a count error must not block enqueue
            logger.warning(
                "durable outstanding-quota count failed for workspace %s; admitting",
                workspace_id,
            )
            return
        # ``active`` includes the just-stored QUEUED run; reject only when it pushes the tenant
        # ABOVE the cap (so a cap of N admits exactly N concurrent in-flight runs).
        if active > self._workspace_max_outstanding:
            stored.status = RunStatus.FAILED
            stored.error = "rejected: workspace run-concurrency quota exceeded"
            stored.updated_at = _now()
            try:
                await self._storage.save_run(stored)
            except Exception:  # pragma: no cover - best-effort terminal mark
                logger.warning("failed to mark quota-rejected run %s", stored.run_id)
            raise WorkspaceRunQuotaExceeded(
                workspace_id,
                cap=self._workspace_max_outstanding,
                outstanding=active - 1,
            )

    def _release_workspace_run(self, workspace_id: str) -> None:
        """Release one outstanding-run slot for ``workspace_id`` (floors at 0)."""
        self._ws_quota.release(workspace_id)

    def _workspace_semaphore(self, workspace_id: str) -> asyncio.Semaphore:
        """Lazily get/create the per-workspace execution semaphore (T0.4)."""
        return self._ws_quota.semaphore(workspace_id)

    def workspace_outstanding(self, workspace_id: str) -> int:
        """Return the current count of in-flight runs for a workspace (introspection)."""
        return self._ws_quota.outstanding(workspace_id)

    @contextlib.asynccontextmanager
    async def workspace_run_slot(
        self, workspace_id: str
    ) -> AsyncIterator[None]:
        """Reserve one workspace run-slot for a non-``RunRecord`` fan-out (T0.4).

        The same per-workspace admission + concurrency gate that guards full runs, exposed
        for fan-out work that is NOT a stored run — today the model-compare endpoint
        (``POST /v1/models/compare``), which spawns N concurrent model calls per request.
        One compare request counts as ONE outstanding slot (admission caps how many
        concurrent compares a tenant may pin) and is executed under the per-workspace
        execution semaphore, so a tenant's fan-outs cannot starve the shared loop/provider
        quota of other tenants.

        Raises :class:`WorkspaceRunQuotaExceeded` (mapped to HTTP 429) when the workspace is
        already at its outstanding cap; releases the slot in ``finally`` even on error.
        """
        self._admit_workspace_run(workspace_id)
        try:
            async with self._workspace_semaphore(workspace_id):
                yield
        finally:
            self._release_workspace_run(workspace_id)

    async def _execute_run(
        self,
        run_id: str,
        *,
        workspace_id: str,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None = None,
        agent_def: AgentDefRecord | None = None,
        hitl: bool = False,
        plan: bool = False,
        thread: ChatThread | None = None,
    ) -> None:
        """Background worker: RUNNING -> run_task -> SUCCEEDED/FAILED + extraction.

        Reads the terminal :class:`RunResult` status (invariant #4 / AAEO-3): a
        FAILED inference response is recorded as a FAILED run with ``run.error``
        populated and recommendation extraction skipped, instead of being marked
        SUCCEEDED with garbage output. The whole run is bounded by
        ``run_timeout_seconds`` (AAEO-1).

        Runtime selection (T0.2): with ``agent_spec`` set, the run executes on a
        PER-RUN tool-bearing runtime built from the spec (so the agent's tools fire);
        otherwise it stays on the shared tool-less runtime (inline-persona
        back-compat). Execution holds the per-workspace concurrency semaphore (T0.4)
        and the outstanding-run reservation taken in :meth:`create_run` is released
        in ``finally`` so a failed/cancelled run frees its slot. ``workspace_id`` is
        passed in (not re-read from the record) so the slot is always released for the
        right workspace even on the defensive ``run is None`` path.
        """
        return await self._drive._execute_run(
            run_id,
            workspace_id=workspace_id,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            agent_def=agent_def,
            hitl=hitl,
            plan=plan,
            thread=thread,
        )
    async def dispatch_claimed_run(self, run: RunRecord) -> None:
        """Execute a leased-queue run the dispatcher just CLAIMED, with retry/backoff (Q3).

        The dispatcher hands this a run already flipped to RUNNING with a fresh lease (the Q2
        ``claim_next_queued_run`` CAS) and ``attempt`` incremented. This:

        1. REHYDRATES the recoverable launch input from ``run.input_blob`` (the Q0 blob the
           enqueue persisted). A run with no blob (legacy / non-recoverable) cannot be
           re-executed from a fresh process, so it is failed with a clear reason rather than
           silently dropped.
        2. RE-RESOLVES the stored ``agent_def`` from ``metadata['agent_id']`` (for the
           run<->agent lineage edge) via the same resolver the resume path uses.
        3. Drives :meth:`_execute_on_runtime` under the per-workspace concurrency semaphore.
        4. On a TRANSIENT failure (provider blip, timeout, model-not-loaded) RE-QUEUES the run
           with exponential backoff while attempts + age remain; once the budget is exhausted
           (or the failure is PERMANENT) it leaves the terminal FAILED set by the runtime,
           or PARKS it so an operator can ``redrive``. A SUCCEEDED / AWAITING_APPROVAL /
           RESOLVING outcome is left as-is (a paused HITL run is NOT a dispatcher failure).

        The lease-renewal heartbeat is run by the dispatcher as a sibling sub-task, not here.
        """
        return await self._drive.dispatch_claimed_run(run)
    async def _apply_retry_policy(self, run_id: str) -> None:
        """Re-queue a transient-failed run with backoff, else PARK it (Q3).

        Reads the run's terminal state after :meth:`_execute_on_runtime`. Only a FAILED run is
        considered — SUCCEEDED is done; AWAITING_APPROVAL/RESOLVING are paused (NOT failures).
        A FAILED run is classified transient (provider/timeout/connection blip) vs permanent
        (validation, build, unknown-tool). A transient failure with attempts AND age remaining
        is RE-QUEUED with exponential backoff (``next_attempt_at`` in the future, so the claim
        CAS leaves it until then — the backoff survives a restart); otherwise it is PARKED
        (terminal-but-redrivable) so an operator can intervene, distinct from a clean FAILED.
        A permanent failure is left FAILED untouched.
        """
        await self._retry.apply_retry_policy(run_id)

    async def _notify_conversation_sink(self, run: RunRecord) -> None:
        """Re-project a thread-linked run's updated ChatThread (T2g, best-effort).

        Invoked when a run that the thread router started (carrying
        ``metadata['conversation_id']``) reaches a terminal / AWAITING_APPROVAL state, so
        the ConversationStore projection reflects the latest turns even when the run
        finished on a background task the router never awaited (an approval-resume). A
        no-op when no sink is wired or the run is not thread-linked; any sink error is
        swallowed (provenance, not the work).
        """
        await self._side_effects.notify_conversation_sink(run)

    def _extract_plan_from_checkpoint(
        self, checkpoint_id: str | None
    ) -> list[dict[str, str]]:
        """Read the bounded plan steps out of a PLAN-READY checkpoint (T2g).

        The plan-first run pauses on its gated ``update_plan`` call; that call's args
        carry the proposed steps. Returns the normalized, bounded steps (an empty list
        when no checkpoint / no plan call is pending), so a caller can read the plan
        before approving it.
        """
        if not checkpoint_id or self._checkpoint_store is None:
            return []
        checkpoint = self._checkpoint_store.load(checkpoint_id)
        if checkpoint is None:  # pragma: no cover - the pause just wrote it
            return []
        from himmy.runtime.plan_mode import PLAN_TOOL, normalize_plan_steps

        for pending in checkpoint.pending_tool_calls:
            if pending.tool_name == PLAN_TOOL:
                return normalize_plan_steps((pending.args or {}).get("steps"))
        return []

    def _build_tool_authorizer(self, actor: dict[str, Any] | None) -> Any:
        """Rebuild this run's tool-capability gate from its persisted actor (P0).

        Returns ``None`` when no RBAC policy is wired (the offline / zero-config default
        and every programmatic caller) so ``build_runtime_for_spec`` builds a per-run
        runtime with NO authorizer — tool dispatch byte-identical to before. When a policy
        IS wired the gate is rebuilt from the actor descriptor: an actor carrying
        ``tool_authz_enforce`` yields an ENFORCING authorizer over its recorded roles; the
        ANONYMOUS / all_tenants offline actor (no flag) yields a NON-enforcing pass-through.
        Building from the persisted actor (not a live Principal) is what lets the gate
        survive the leased-dispatch recovery path, where a fresh process re-executes the
        run with no in-memory principal.
        """
        return self._side_effects.build_tool_authorizer(actor)

    @staticmethod
    def _subject_scope_from_actor(actor: dict[str, Any] | None) -> str | None:
        """The within-tenant subject axis for a run's tool stores, from its persisted actor.

        Mirrors the single-agent path: under a ``subject_scoped`` per-user actor the run's
        memory/KB/tasks/notes packs are namespaced by the user so two users of ONE tenant never
        read each other's facts/docs. The flag is persisted by ``Principal.actor_metadata`` (set
        only when actually ``subject_scoped`` + not a ``tenant_admin``), so a non-subject-scoped /
        offline run returns ``None`` — byte-for-byte unchanged.
        """
        return RunSideEffects.subject_scope_from_actor(actor)

    async def _resolve_runtime(
        self,
        agent_spec: AgentSpec | None,
        *,
        checkpoint_store: Any = None,
        plan_mode: bool = False,
        workspace_id: str | None = None,
        actor: dict[str, Any] | None = None,
    ) -> SingleAgentRuntime:
        """Pick the runtime for a run: shared tool-less, or a per-run tool-bearing one.

        With ``agent_spec is None`` the existing shared (tool-less) runtime is
        returned — the inline-persona fast path stays byte-identical (back-compat).

        With a spec present (T0.2) a PER-RUN runtime is built via
        :func:`himmy.runtime.from_spec.build_runtime_for_spec`, which wires the spec's
        tool packs / tools / guardrails / knowledge / connectors / MCP + a tool
        service, so the run can finally CALL the agent's tools (impossible on the
        shared runtime, which carries no tool_service). It is built off-loop in a
        worker thread because ``build_runtime_for_spec`` may run an inner
        ``asyncio.run`` (knowledge ingest) that cannot nest in the running loop.

        Wiring choices that preserve the zero-config offline default: the per-run
        runtime SHARES this service's storage (so its thread/events/memory land in the
        one store the app layer reads) and REUSES the shared runtime's inference
        service when the spec pins no provider — so an offline deployment keeps the
        stub and a configured deployment keeps its gateway, with no surprise provider
        switch. When the spec names a provider explicitly, ``build_runtime_for_spec``
        honors it. ``checkpoint_store`` is threaded so a HITL run can pause (T2f).

        ``plan_mode`` (T2g) registers the APPROVAL-GATED ``update_plan`` tool into the
        per-run registry so a plan-first run pauses at PLAN-READY through the SAME
        approval machinery (it MUST be registered on the resume runtime too, hence this
        flag is threaded both on the initial drive and on resume).

        ``workspace_id`` (P1 tenancy) is the run's owning tenant, threaded into
        ``build_runtime_for_spec(subject=...)`` so a ``self_learning`` agent's tool-
        reputation mining is scoped to this tenant on the SHARED ``/v1`` event store
        instead of aggregating every tenant's tool failures.
        """
        return await self._drive._resolve_runtime(
            agent_spec,
            checkpoint_store=checkpoint_store,
            plan_mode=plan_mode,
            workspace_id=workspace_id,
            actor=actor,
        )
    async def _project_run_agent_link(
        self, run: RunRecord, agent_def: AgentDefRecord
    ) -> None:
        """Register the stored-agent node + link the run's thread -> agent (T2e).

        Ensures the ``agent`` entity exists in the shared spine (idempotent,
        content-addressed) and draws a ``run_of_agent`` edge from the run's
        ``chat_thread`` hub to that agent node, so a run launched by ``agent_id`` joins
        the agent's lineage graph. Best-effort: a projection failure never fails the run
        (it is provenance, not the work).
        """
        await self._side_effects.project_run_agent_link(run, agent_def)

    # ----------------------------------------------------------- lifecycle (AAEO-1)
    async def drain(self, *, timeout: float = 30.0) -> None:
        """Cancel + await all in-flight background runs (FastAPI shutdown hook).

        Each task is cancelled and awaited; the per-task cancel handler records
        the run FAILED('run cancelled'). Bounded by ``timeout`` so shutdown cannot
        hang forever.
        """
        await self._recovery.drain(timeout=timeout)

    async def sweep_stuck_runs(self, *, ttl_seconds: float = 0.0) -> list[str]:
        """Recover runs left non-terminal by a dead process (AAEO-1; Q3 rewrite).

        Intended for startup. The behaviour now depends on whether the leased dispatcher owns
        execution:

        * DISPATCH mode (the durable store + lifespan dispatcher): a crashed worker's RUNNING
          run holds an EXPIRED lease, so it is RE-QUEUED (not failed) via the Q2 reaper
          ``requeue_expired_leases`` — the dispatcher then re-claims and re-executes it from
          its recoverable input. A QUEUED run is NOT touched (it is already recoverable — the
          dispatcher will claim it); a LIVE peer's RUNNING run (lease not yet expired) is NEVER
          re-queued; AWAITING_APPROVAL / RESOLVING are preserved (HITL pauses + re-drivable
          resumes). This is the fix for the old "crash -> mass-FAIL" and "reap a live peer"
          behaviours. Returns the re-queued run ids.

        * INLINE mode (no dispatcher — a bare ``create_app`` / CLI / TestClient): there is
          nothing to re-claim a QUEUED/RUNNING run, so the old fail-loud sweep is preserved
          byte-for-byte — those runs are transitioned to FAILED so they do not hang forever.
          ``ttl_seconds=0`` sweeps all; AWAITING_APPROVAL / RESOLVING are still excluded.
        """
        return await self._recovery.sweep_stuck_runs(ttl_seconds=ttl_seconds)

    async def reconcile_resolving_runs(self) -> list[str]:
        """Re-drive HITL resumes left at RESOLVING by a crash, exactly-once (UNIT 1d).

        A resume that wins the run-level claim flips the run to RESOLVING and persists the
        approve/reject decision BEFORE launching its background task. If the process dies
        between the claim and the run reaching a terminal/AWAITING-again state, the run is
        stranded at RESOLVING — but it is RE-DRIVABLE: the member checkpoint's own
        ``claim()`` accepts a RESOLVING checkpoint on retry, and the per-tool idempotency
        ledger replays an already-executed gated tool, so re-driving the recorded decision
        executes the side effect at most once total (across the crashed attempt + this
        recovery). Intended for startup, after :meth:`sweep_stuck_runs` (which never reaps
        a RESOLVING run). Returns the re-driven run ids.

        This is DISTINCT from a concurrent second approve: that loses the run-level CAS at
        the inbox and 409s without launching anything (no RESOLVING strand to recover);
        only an in-flight crash leaves a RESOLVING run for this method to pick up.
        """
        return await self._recovery.reconcile_resolving_runs()

    # ----------------------------------------------------------------- HITL (T2f)
    async def pending_approvals(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> list[dict[str, Any]] | None:
        """The redacted pending tool call(s) a HITL-paused run awaits (T2f).

        Tenant-scoped: a run outside ``workspace_id`` reads as None (404). Returns the
        list of ``{tool_name, args}`` for the checkpoint the run paused on, with secret-
        looking arg values masked (the same redaction Studio's approvals inbox uses), so
        a reviewer can see WHAT will run before approving without leaking a credential.
        None when the run is unknown/out-of-workspace; an empty list when the run carries
        no checkpoint (e.g. not actually paused) or the checkpoint has been resolved.
        """
        return await self._drive.pending_approvals(
            run_id, workspace_id=workspace_id
        )
    async def resume_run(
        self,
        run_id: str,
        *,
        approved: bool,
        workspace_id: str | None = None,
        actor: str = "human",
    ) -> RunRecord:
        """Approve/reject a HITL-paused run; resume it on a tracked bg task (T2f).

        Loads the run tenant-scoped (a 404 for unknown/out-of-workspace, a 409 for a
        terminal/non-paused run), then ATOMICALLY claims ``AWAITING_APPROVAL`` ->
        ``RESOLVING`` via :meth:`StorageService.claim_run_for_resume` — the run-level
        compare-and-set that mirrors the member checkpoint ``claim()``. A SECOND concurrent
        approve (a double-clicked Approve, two tabs, two workers) loses this CAS and is
        refused with :class:`RunNotApprovableError` (409) BEFORE launching any resume, so
        for an ORCHESTRATION run the graph advance — which has no claim of its own and
        could otherwise double-fire DOWNSTREAM members' tools — only ever happens once.

        The winner then REBUILDS its OWN per-run tool-bearing runtime FROM THE STORED DB
        ``AgentSpec`` (resolved by ``agent_id`` — /v1 has no filesystem ``agent_path`` to
        rebuild from, and ``resume_agent_loop`` HARD-requires a ``tool_service``) and
        launches :meth:`SingleAgentRuntime.resume_agent_loop` (or the orchestration graph
        resume) on a fresh tracked background task. The background task drives the run to
        SUCCEEDED / FAILED / AWAITING_APPROVAL-again. A resume that crashes mid-flight
        leaves the run at ``RESOLVING`` so startup recovery can re-drive it exactly-once
        (the member checkpoint ``claim()`` + idempotency ledger), distinct from this
        rejected "concurrent second click". Returns the in-progress record (fire-and-
        forget, mirroring :meth:`create_run`).
        """
        return await self._drive.resume_run(
            run_id, approved=approved, workspace_id=workspace_id, actor=actor
        )
    async def _resume_in_background(
        self,
        run_id: str,
        *,
        checkpoint_id: str,
        approved: bool,
        actor: str,
        agent_def: AgentDefRecord,
        workspace_id: str,
    ) -> None:
        """Background worker: rebuild the runtime from the DB spec + resume the loop (T2f).

        Holds the per-workspace concurrency semaphore (T0.4) for the duration, rebuilds a
        tool-bearing runtime FROM THE STORED SPEC carrying /v1's checkpoint store, and
        drives :meth:`SingleAgentRuntime.resume_agent_loop`. A loser of the exactly-once
        ``claim()`` race (``HimmyError('already resolved')``) is a NO-OP — the run is left
        at whatever the winner set it to (never re-failed, never re-run).
        """
        return await self._drive._resume_in_background(
            run_id,
            checkpoint_id=checkpoint_id,
            approved=approved,
            actor=actor,
            agent_def=agent_def,
            workspace_id=workspace_id,
        )
    # ``self.get_run`` callers hit ``get_run`` here, which forwards to the same collaborator.
    async def get_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> RunRecord | None:
        """Return a run record by id, scoped to a workspace (AAEO-4).

        When ``workspace_id`` is supplied, a run belonging to another workspace is
        treated as not found (returns None).
        """
        return await self._reads.get_run(run_id, workspace_id=workspace_id)

    async def get_run_lineage(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        max_depth: int = DEFAULT_TRACE_DEPTH,
        relations: set[str] | list[str] | None = None,
    ) -> LineageGraph | None:
        """Return the provenance subgraph for a run, or None.

        Resolves the run (tenant-scoped), finds its ``chat_thread`` entity — the
        lineage hub — and traces the connected records: the persona it used, the
        prompt, and the context snapshot it was built from. This is the read side
        of the captured lineage that fulfils the documented "trace any run back to
        its persona + evidence" promise.

        Returns None when the run is unknown / out-of-workspace, no entity registry
        is wired, or the thread was never projected into the registry.
        """
        return await self._reads.get_run_lineage(
            run_id,
            workspace_id=workspace_id,
            max_depth=max_depth,
            relations=relations,
        )

    async def list_runs(
        self,
        *,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> list[RunRecord]:
        """List runs filtered by workspace, subject, and/or status, paginated.

        Ordered ``created_at`` desc with ``run_id`` tiebreak, then windowed by
        ``offset``/``limit`` (capped at :data:`MAX_PAGE_LIMIT`) — AAEO-8.

        Cross-tenant safety (T2.2): the all-workspaces view (``workspace_id is None``,
        an authenticated admin listing every tenant) EXCLUDES the reserved
        ``__local__`` workspace so CLI/single-user-local runs never leak into a
        multi-tenant admin list. An explicit query for ``__local__`` still returns
        them (the local Studio/CLI browse their own history).
        """
        return await self._reads.list_runs(
            workspace_id=workspace_id,
            subject_id=subject_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    async def count_runs(
        self,
        *,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
    ) -> int:
        """Total run count for the filter (for pagination envelopes).

        Mirrors :meth:`list_runs`' cross-tenant rule so the count matches the page:
        the all-workspaces view excludes the reserved ``__local__`` runs (T2.2).

        T2-runs-pagination: when the backing store exposes a native ``count_runs``
        (the durable SQLite/Postgres services), the total is computed with a single
        ``SELECT COUNT(*)`` — the ``__local__`` exclusion folded into the same WHERE
        clause — instead of loading + JSON-deserializing the entire runs table just
        to ``len()`` it. Stores without a native counter (the in-memory facade) fall
        back to the load+filter path, which is cheap for an in-RAM dict and preserves
        byte-for-byte the same count semantics.
        """
        return await self._reads.count_runs(
            workspace_id=workspace_id, subject_id=subject_id, status=status
        )

    async def get_run_events(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> list[Any]:
        """Replay the canonical event stream for one run (by its trace id).

        Tenant-scoped (AAEO-4): a run outside ``workspace_id`` yields ``[]``.
        """
        return await self._reads.get_run_events(run_id, workspace_id=workspace_id)

    async def get_run_thread(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> Any:
        """Replay the full conversation thread for one run (tenant-scoped, AAEO-4)."""
        return await self._reads.get_run_thread(run_id, workspace_id=workspace_id)

    async def get_run_thread_by_thread_id(self, thread_id: str) -> Any:
        """Load the authoritative ChatThread the runtime saved under ``thread_id`` (T2g).

        The thread router pins a run's ``thread_id`` to the conversation id, so this reads
        the latest in-flight thread state from the canonical run store directly (the run
        record may not yet carry the thread_id when a continuation has only just started).
        Returns None when nothing is stored under that id.
        """
        return await self._reads.get_run_thread_by_thread_id(thread_id)

    async def await_run(self, run_id: str, timeout: float = 5.0) -> RunRecord | None:
        """Poll until the run reaches a terminal state (test/example helper)."""
        return await self._reads.await_run(run_id, timeout=timeout)

    # ----------------------------------------------- T3b team/workflow runs
    async def create_orchestration_run(
        self,
        *,
        workspace_id: str,
        subject_id: str,
        kind: str,
        members: list[AgentDefRecord],
        prompt: str,
        resource_kind: str,
        resource_id: str,
        idempotency_key: str | None = None,
        actor: dict[str, Any] | None = None,
        operator_provisioned: bool = False,
        graph_checkpoint_store: Any = None,
        graph_resume_id: str | None = None,
    ) -> RunRecord:
        """Launch a team/workflow orchestration on the EXISTING run machinery (T3b).

        A team/workflow run is NOT a second executor: it creates a canonical
        :class:`RunRecord`, admits it against the SAME T0.4 per-workspace quota, and
        executes on a tracked background task under the per-workspace concurrency semaphore
        — exactly like :meth:`create_run`. The difference is the body: instead of one
        per-run agent runtime it builds a TEAM runtime from the ordered member
        :class:`AgentDefRecord`s (resolved + sanitized) and drives the matching orchestrator
        (``multi_agent`` | ``group_chat`` for a team; an ordered pipeline for a workflow).

        ``kind`` selects the orchestrator. ``members`` is the ordered, pre-resolved member
        list (the router validated each exists in the workspace + drew the same-workspace
        membership check). ``resource_kind``/``resource_id`` (``team``/``workflow`` + its id)
        are stamped into the run metadata so a run is traceable to the team/workflow that
        launched it. ``graph_checkpoint_store``/``graph_resume_id`` (the ``graph`` kind)
        thread the durable :class:`SqliteGraphCheckpointStore` so a long graph run resumes
        after a restart.

        Returns the QUEUED :class:`RunRecord` immediately; poll ``get_run`` for the outcome.
        Raises :class:`WorkspaceRunQuotaExceeded` (429) when the workspace is at its cap.
        """
        return await self._drive.create_orchestration_run(
            workspace_id=workspace_id,
            subject_id=subject_id,
            kind=kind,
            members=members,
            prompt=prompt,
            resource_kind=resource_kind,
            resource_id=resource_id,
            idempotency_key=idempotency_key,
            actor=actor,
            operator_provisioned=operator_provisioned,
            graph_checkpoint_store=graph_checkpoint_store,
            graph_resume_id=graph_resume_id,
        )
class AgentDefReferencedError(Exception):
    """A stored agent could not be deleted because it is referenced (T2e must_fix).

    Raised by :meth:`AgentDefAppService.delete_agent_def` when a referential check
    finds the agent is still used by a team/routine (and ``cascade`` is not set), so a
    DELETE returns HTTP 409 rather than silently orphaning the reference.
    """

    def __init__(self, agent_id: str, references: list[str]) -> None:
        """Record the agent and the references that block its deletion."""
        self.agent_id = agent_id
        self.references = list(references)
        joined = ", ".join(self.references) or "another resource"
        super().__init__(
            f"agent {agent_id} cannot be deleted: still referenced by {joined}"
        )


class AgentDefAppService:
    """Owns the stored-agent (``/v1/agents``) resource lifecycle (T2e).

    A workspace-scoped CRUD service over :class:`AgentDefRecord` that, on every
    tenant write, runs the spec through the T0.3 sanitizer (the operator-only
    ``tools_module``/``http_tools``/``mcp_servers`` RCE/SSRF surface is rejected/
    stripped) BEFORE it is persisted, projects each stored agent as an ``agent``
    entity into the shared spine (so run->agent lineage links resolve), enforces
    tenant scoping on every read, supports idempotent create, and refuses to delete an
    agent still referenced by a team/routine (referential integrity).
    """

    def __init__(
        self,
        *,
        storage: StorageService,
        entity_registry: EntityRegistryProtocol | None = None,
        reference_finder: Any = None,
    ) -> None:
        """Wire the store, the optional spine, and an optional reference finder.

        ``reference_finder`` is an optional callable
        ``(agent_id, workspace_id) -> awaitable[list[str]]`` returning the human
        references (team/routine ids) that block a delete. It is a forward-looking seam
        for T3b/T3c (which introduce /v1 teams + routines that reference an agent_id);
        when ``None`` the referential check is a no-op (nothing references agents yet),
        so the contract is in place without coupling to not-yet-built resources.
        """
        self._storage = storage
        self._registry = entity_registry
        self._reference_finder = reference_finder

    async def save_agent_def(
        self,
        spec: AgentSpec,
        *,
        workspace_id: str,
        agent_id: str | None = None,
        idempotency_key: str | None = None,
        actor: dict[str, Any] | None = None,
        operator_provisioned: bool = False,
    ) -> tuple[AgentDefRecord, bool]:
        """Sanitize, store (idempotent), and project a stored agent. Returns (rec, created).

        The spec is fail-closed sanitized against the tenant attack surface (T0.3)
        BEFORE persistence — a tenant spec carrying ``tools_module``/``http_tools``/
        ``mcp_servers`` is rejected (or stripped when the deployment opts into strip
        mode), unless operator-provisioned AND the operator opted in. ``idempotency_key``
        makes a re-submit return the prior record (``created=False``) without creating a
        duplicate. The stored agent is projected as an ``agent`` entity into the spine.
        """
        clean = sanitize_tenant_spec(
            spec, operator_provisioned=operator_provisioned
        ).spec
        metadata: dict[str, Any] = {}
        if actor:
            metadata["actor"] = actor
        record = AgentDefRecord.from_spec(
            clean,
            workspace_id=workspace_id,
            agent_id=agent_id,
            idempotency_key=idempotency_key,
            metadata=metadata,
        )
        stored, created = await self._storage.save_agent_def_if_absent(record)
        # Re-project on every accepted write (create OR explicit update). An idempotent
        # no-op re-submit (created=False, same record) still re-registers — the spine is
        # content-addressed so it dedupes; an edit creates a new version.
        await self._project_agent(stored)
        return stored, created

    async def update_agent_def(
        self,
        agent_id: str,
        spec: AgentSpec,
        *,
        workspace_id: str,
        actor: dict[str, Any] | None = None,
        operator_provisioned: bool = False,
    ) -> AgentDefRecord | None:
        """Replace a stored agent's spec in place (tenant-scoped). None when absent.

        The existing record's ``agent_id``/``created_at``/``idempotency_key`` are
        preserved; only the (re-sanitized) spec + name/description + ``updated_at``
        change. Returns ``None`` when the agent does not exist in the workspace (404).
        """
        existing = await self._storage.get_agent_def(
            agent_id, workspace_id=workspace_id
        )
        if existing is None:
            return None
        clean = sanitize_tenant_spec(
            spec, operator_provisioned=operator_provisioned
        ).spec
        metadata = dict(existing.metadata)
        if actor:
            metadata["actor"] = actor
        updated = AgentDefRecord.from_spec(
            clean,
            workspace_id=workspace_id,
            agent_id=existing.agent_id,
            idempotency_key=existing.idempotency_key,
            metadata=metadata,
        )
        updated.created_at = existing.created_at
        stored = await self._storage.save_agent_def(updated)
        await self._project_agent(stored)
        return stored

    async def get_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> AgentDefRecord | None:
        """Return a stored agent def by id, tenant-scoped (out-of-workspace → None)."""
        return await self._storage.get_agent_def(
            agent_id, workspace_id=workspace_id
        )

    async def list_agent_defs(
        self, *, workspace_id: str | None = None
    ) -> list[AgentDefRecord]:
        """List stored agent defs for a workspace."""
        return await self._storage.list_agent_defs(workspace_id=workspace_id)

    async def delete_agent_def(
        self,
        agent_id: str,
        *,
        workspace_id: str | None = None,
        cascade: bool = False,
    ) -> bool:
        """Delete a stored agent, refusing if referenced (unless ``cascade``).

        Returns ``True`` when a row was removed, ``False`` when the agent did not exist
        in the workspace (404). Raises :class:`AgentDefReferencedError` (HTTP 409) when
        a referential check finds the agent is still used by a team/routine and
        ``cascade`` is not set — the reviewer must_fix against silently orphaning a
        reference. ``cascade`` is accepted for forward-compat; the caller owns removing
        the referencing resources first.
        """
        existing = await self._storage.get_agent_def(
            agent_id, workspace_id=workspace_id
        )
        if existing is None:
            return False
        if not cascade and self._reference_finder is not None:
            references = await _maybe_await(
                self._reference_finder(agent_id, existing.workspace_id)
            )
            if references:
                raise AgentDefReferencedError(agent_id, list(references))
        return await self._storage.delete_agent_def(
            agent_id, workspace_id=workspace_id
        )

    async def _project_agent(self, record: AgentDefRecord) -> None:
        """Register the stored agent as an ``agent`` entity in the spine (best-effort)."""
        if self._registry is None:
            return
        try:
            await _maybe_await(self._registry.register(record.to_record()))
        except Exception:  # pragma: no cover - projection is best-effort
            logger.warning("failed to project agent entity for %s", record.agent_id)


class ThreadNotFoundError(Exception):
    """A /v1 thread operation targeted an unknown / out-of-workspace conversation (T2g → 404).

    Workspace ownership is a property of the stored ChatThread (its
    ``metadata['workspace_id']``); a thread owned by another workspace is treated as
    absent so a cross-tenant ``thread_id`` is a clean 404, never a leak.
    """


class ThreadAgentRequiredError(Exception):
    """A /v1 thread continuation was sent without resolving an agent (T2g → 422).

    A continuation runs on the per-run tool-bearing runtime, which needs a stored
    ``AgentSpec`` (an inline persona carries no tools), so the thread must have been
    created with an ``agent_id`` (or one must be supplied on the message).
    """


#: Where a /v1 thread stamps its owning workspace in the authoritative ChatThread JSON.
#: Threads have no native workspace column (ConversationStore is single-user-local by
#: origin), so workspace ownership rides in the thread metadata and is enforced by
#: :class:`ThreadAppService` on every read (404 cross-tenant).
THREAD_WORKSPACE_KEY = "workspace_id"
#: The stored ``agent_id`` a thread continues with, recorded on the thread metadata at
#: create so every message turn resolves the same agent without re-supplying it.
THREAD_AGENT_KEY = "agent_id"


class ThreadAppService:
    """Owns the /v1 thread (conversation) resource over the ONE ConversationStore (T2g).

    A thin orchestration layer that makes a ``/v1`` conversation a first-class,
    workspace-scoped, agentic resource: it CREATES a workspace-stamped thread, CONTINUES
    it (loading the authoritative :class:`ChatThread`, appending the user turn, and running
    it on :class:`RunAppService`'s PER-RUN tool-bearing runtime so the conversation has
    tools + HITL/plan pauses — reviewer must_fix), and PERSISTS the updated thread back to
    the same durable ``.himmy/conversations.db`` the CLI and Studio read (so a /v1
    conversation is browsable everywhere). Workspace ownership is enforced on every read
    via the thread's ``metadata['workspace_id']`` (cross-tenant → 404).

    It does NOT own a second run executor — every turn lands a canonical
    :class:`RunRecord` through :class:`RunAppService` (one writer), linked to the
    conversation by ``metadata['conversation_id']``.
    """

    def __init__(
        self,
        *,
        run_app: RunAppService,
        agent_app: AgentDefAppService,
        conversation_store: Any,
    ) -> None:
        """Wire the run service, the stored-agent service, and the conversation store."""
        self._run_app = run_app
        self._agent_app = agent_app
        self._store = conversation_store

    # -- the conversation projection sink (wired into RunAppService) ---------

    async def project_run_thread(self, conversation_id: str, run: RunRecord) -> None:
        """Re-project a run's updated ChatThread into the ConversationStore (T2g sink).

        Wired as :class:`RunAppService`'s ``conversation_sink`` so a continuation that
        completes on a background task (notably an approval-resume the message handler
        never awaited) still lands its latest turns in the durable conversation projection.
        Loads the authoritative thread from the canonical run store (where the runtime
        saved it) and re-saves it under the conversation id, preserving the workspace /
        agent ownership metadata. Best-effort and idempotent.
        """
        thread = await self._run_app.get_run_thread(
            run.run_id, workspace_id=run.workspace_id
        )
        if thread is None:
            return
        self._persist(
            conversation_id, thread, run.workspace_id, subject_id=run.subject_id
        )

    def _persist(
        self,
        conversation_id: str,
        thread: Any,
        workspace_id: str,
        *,
        subject_id: str | None = None,
    ) -> None:
        """Stamp ownership onto the thread + upsert it into the ConversationStore.

        ``subject_id`` (the run's data subject) is threaded through so a governed deployment's
        :class:`ConsentGatedConversationStore` can gate + per-subject-encrypt the transcript
        and record the subject linkage the S4 reach map needs; the bare store stores it as the
        linkage column and is otherwise unchanged.
        """
        thread.thread_id = conversation_id
        agent_id = (thread.metadata or {}).get(THREAD_AGENT_KEY)
        thread.metadata = {
            **(thread.metadata or {}),
            THREAD_WORKSPACE_KEY: workspace_id,
        }
        # ``origin`` defaults to the CLI/lossless origin (this is an authoritative,
        # full-fidelity ChatThread, not a flat Studio transcript) so list views show it
        # as a first-class conversation across surfaces.
        self._store.save_thread(
            conversation_id,
            thread,
            origin=ORIGIN_CLI,
            agent_path=agent_id,
            subject_id=subject_id,
        )

    # -- create -------------------------------------------------------------

    def create_thread(
        self,
        *,
        workspace_id: str,
        agent_id: str | None = None,
        title: str | None = None,
    ) -> Any:
        """Create an empty, workspace-stamped thread (T2g). Returns the ChatThread.

        The workspace (and the optional default ``agent_id`` the thread continues with)
        are stamped into the thread metadata so every later message turn is scoped + runs
        the right agent. The thread is persisted immediately so it is listable.
        """
        from himmy.agents.base_agent.thread import ChatThread

        metadata: dict[str, Any] = {THREAD_WORKSPACE_KEY: workspace_id}
        if agent_id:
            metadata[THREAD_AGENT_KEY] = agent_id
        thread = ChatThread(agent_id=agent_id, metadata=metadata)
        self._persist(thread.thread_id, thread, workspace_id)
        if title:
            self._store.rename(thread.thread_id, title)
        return thread

    # -- read ---------------------------------------------------------------

    def load_owned_thread(self, conversation_id: str, *, workspace_id: str) -> Any:
        """Load a thread, enforcing workspace ownership (T2g). Raises on mismatch.

        Returns the authoritative :class:`ChatThread`. Raises
        :class:`ThreadNotFoundError` when the conversation is unknown OR owned by another
        workspace (cross-tenant → 404), so ownership is enforced at one chokepoint.
        """
        thread = self._store.load_thread(conversation_id)
        if thread is None:
            raise ThreadNotFoundError(conversation_id)
        owner = (thread.metadata or {}).get(THREAD_WORKSPACE_KEY)
        if owner != workspace_id:
            raise ThreadNotFoundError(conversation_id)
        return thread

    def flat_messages(self, conversation_id: str, *, workspace_id: str) -> list[Any]:
        """The flat (user/agent) projection of an owned thread (404 cross-tenant)."""
        # Ownership check first (raises 404 on mismatch / unknown).
        self.load_owned_thread(conversation_id, workspace_id=workspace_id)
        return cast("list[Any]", self._store.flat_messages(conversation_id))

    def owned_subject_id(
        self, conversation_id: str, *, workspace_id: str
    ) -> str | None:
        """The data subject linked to an OWNED thread, for object-level (BOLA) gating.

        Resolves the thread under its workspace owner FIRST (so a cross-tenant id is a
        clean 404 before any subject is revealed), then returns the conversation's stored
        ``subject_id`` (the S3 erasure-linkage column) via the store's ``subject_of`` —
        or ``None`` when the store does not expose it or the thread is un-attributed. The
        router folds this into :func:`~himmy.api.auth.authorize_object`; ``None`` means
        "no subject to narrow on", so an un-attributed thread (and the offline path) is
        unaffected.
        """
        self.load_owned_thread(conversation_id, workspace_id=workspace_id)
        subject_of = getattr(self._store, "subject_of", None)
        if subject_of is None:
            return None
        return cast("str | None", subject_of(conversation_id))

    # -- continue -----------------------------------------------------------

    async def append_message(
        self,
        *,
        conversation_id: str,
        workspace_id: str,
        subject_id: str,
        prompt: str,
        agent_id: str | None = None,
        idempotency_key: str | None = None,
        actor: dict[str, Any] | None = None,
        operator_provisioned: bool = False,
        hitl: bool = False,
        plan: bool = False,
    ) -> RunRecord:
        """Append a user turn to an owned thread and run it agentically (T2g).

        Loads the authoritative thread (404 cross-tenant), resolves the agent to continue
        with (the message's ``agent_id`` overrides the thread's default), and drives
        :meth:`RunAppService.continue_thread` — so the turn runs WITH TOOLS + can pause for
        approval (``hitl``) or at PLAN-READY (``plan``). The updated thread is persisted
        back to the ConversationStore (and re-persisted by the sink when the run later
        completes on a background task). Returns the linked :class:`RunRecord`.
        """
        thread = self.load_owned_thread(conversation_id, workspace_id=workspace_id)
        resolved_agent_id = agent_id or (thread.metadata or {}).get(THREAD_AGENT_KEY)
        if not resolved_agent_id:
            raise ThreadAgentRequiredError(conversation_id)
        agent_def = await self._agent_app.get_agent_def(
            resolved_agent_id, workspace_id=workspace_id
        )
        if agent_def is None:
            # The thread references an agent that does not exist in this workspace.
            raise ThreadAgentRequiredError(resolved_agent_id)
        agent_spec = agent_def.agent_spec()

        # Carry the resolved agent forward on the thread so a later turn defaults to it
        # and the persisted thread records which agent it continues with.
        thread.metadata = {
            **(thread.metadata or {}),
            THREAD_AGENT_KEY: resolved_agent_id,
            THREAD_WORKSPACE_KEY: workspace_id,
        }

        run = await self._run_app.continue_thread(
            workspace_id=workspace_id,
            subject_id=subject_id,
            conversation_id=conversation_id,
            thread=thread,
            prompt=prompt,
            agent_spec=agent_spec,
            agent_def=agent_def,
            idempotency_key=idempotency_key,
            actor=actor,
            operator_provisioned=operator_provisioned,
            hitl=hitl,
            plan=plan,
        )
        # Persist the latest thread state synchronously too (the run also re-projects via
        # the sink when it reaches a terminal/paused state, but persisting here makes the
        # new turn immediately visible even before the background run lands).
        await self._reproject_after_turn(conversation_id, workspace_id)
        return run

    async def _reproject_after_turn(
        self, conversation_id: str, workspace_id: str
    ) -> None:
        """Best-effort sync of the conversation projection right after a turn starts."""
        try:
            thread = await self._run_app.get_run_thread_by_thread_id(conversation_id)
            if thread is not None:
                self._persist(conversation_id, thread, workspace_id)
        except Exception:  # noqa: BLE001 - projection is best-effort
            logger.warning("failed to reproject conversation %s", conversation_id)


def _resolve_model_key(llm_config: LLMConfig | None, task: Task) -> str | None:
    """Resolve a run's model_key with explicit precedence (AAEO-16).

    A caller-set ``llm_config.model_key`` wins — but ``LLMConfig.model_key``
    defaults to ``"default"`` (non-None), so a config carrying *only* that default
    must NOT shadow a ``model_key`` the task supplied in its context. Precedence:
    caller-set llm_config.model_key, else task.context['model_key'], else None.
    """
    ctx_key = cast("str | None", (task.context or {}).get("model_key"))
    if llm_config is not None and llm_config.model_key != _DEFAULT_MODEL_KEY:
        return llm_config.model_key
    if ctx_key is not None:
        return ctx_key
    if llm_config is not None:
        return llm_config.model_key
    return None


def _requested_schema(
    llm_config: LLMConfig | None, task: Task
) -> dict[str, Any] | None:
    """Resolve the structured-output JSON schema requested for a run (AAEO-6)."""
    if llm_config is not None and llm_config.output_json_schema is not None:
        return llm_config.output_json_schema
    ctx = task.context or {}
    schema = ctx.get("output_schema")
    return schema if isinstance(schema, dict) else None


def _validate_structured(value: Any, schema: dict[str, Any]) -> str | None:
    """Validate structured output against a schema; return an error or None.

    Delegates to the tools kernel's shared validator (jsonschema when available,
    a stdlib subset otherwise).
    """
    return validate_against_schema(value, schema)


def _parse_iso_epoch(value: str | None) -> float:
    """Best-effort parse of an ISO-8601 timestamp into epoch seconds (0.0 on fail)."""
    if not value:
        return 0.0
    from datetime import datetime

    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):  # pragma: no cover - tolerant of odd formats
        return 0.0


class DashboardQueryService:
    """Aggregates the operator overview tile: context, runs, recommendations."""

    def __init__(
        self,
        *,
        storage: StorageService,
    ) -> None:
        """Wire the backing store."""
        self._storage = storage

    async def summary(
        self,
        *,
        subject_id: str,
        workspace_id: str,
        all_tenants: bool = False,
    ) -> dict[str, Any]:
        """Return one summary object: context stats + run/recommendation/eval counts.

        Context fields are scoped to ``workspace_id`` (AAEO-4) and the latest
        evaluation aggregate is folded in (AAEO-15) so the documented "scorecards
        on a dashboard" story is real.

        ``all_tenants`` is the caller principal's cross-tenant flag (True for the
        offline / admin all-tenants path). It is threaded into BOTH the context-tile and
        eval-tile scoping so a None-stamped (unstamped) field/run an ``all_tenants`` caller
        explicitly asks for stays visible — byte-unchanged zero-config — while a tenant-bound
        caller gets strict ``== workspace_id`` and never folds another tenant's / an admin
        run (scope-r3) or a None-stamped context field for a shared subject (scope-r4).
        """
        all_fields = await self._storage.list_context_fields(subject_id)
        # Scope the context tile exactly like the dedicated reader
        # ContextAppService.list_fields() and the eval tile (scope-r3): a
        # TENANT-BOUND caller gets STRICT ``== workspace_id`` so a None-stamped
        # (unstamped) context field written for this subject by an offline / admin /
        # other-tenant unstamped path never inflates its aggregate count. Only the
        # ``all_tenants`` (offline / admin) principal keeps the lenient
        # ``in (None, workspace_id)`` branch — byte-unchanged zero-config (scope-r4).
        if all_tenants:
            fields = [
                f
                for f in all_fields
                if (getattr(f, "metadata", {}) or {}).get("workspace_id")
                in (
                    None,
                    workspace_id,
                )
            ]
        else:
            fields = [
                f
                for f in all_fields
                if (getattr(f, "metadata", {}) or {}).get("workspace_id") == workspace_id
            ]
        confidences = [getattr(f, "confidence", 0.0) for f in fields]
        freshness = [
            getattr(f, "freshness_seconds", None)
            for f in fields
            if getattr(f, "freshness_seconds", None) is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        # freshness is filtered non-None above; getattr's static type stays Any | None.
        avg_freshness = (
            sum(freshness) / len(freshness) if freshness else None  # type: ignore[arg-type]
        )

        runs = await self._storage.list_runs(
            workspace_id=workspace_id, subject_id=subject_id
        )
        run_counts: dict[str, int] = {s.value: 0 for s in RunStatus}
        for run in runs:
            run_counts[run.status.value] = run_counts.get(run.status.value, 0) + 1

        recs = await self._storage.list_recommendations(
            workspace_id=workspace_id, subject_id=subject_id
        )
        rec_counts: dict[str, int] = {s.value: 0 for s in RecommendationStatus}
        for rec in recs:
            rec_counts[rec.status.value] = rec_counts.get(rec.status.value, 0) + 1

        evaluation = await self._evaluation_summary(
            workspace_id=workspace_id, all_tenants=all_tenants
        )

        return {
            "subject_id": subject_id,
            "workspace_id": workspace_id,
            "context": {
                "field_count": len(fields),
                "avg_confidence": avg_confidence,
                "avg_freshness_seconds": avg_freshness,
            },
            "runs": {
                "total": len(runs),
                "by_status": run_counts,
            },
            "recommendations": {
                "total": len(recs),
                "by_status": rec_counts,
            },
            "evaluation": evaluation,
        }

    async def _evaluation_summary(
        self, *, workspace_id: str | None = None, all_tenants: bool = False
    ) -> dict[str, Any]:
        """Fold the latest evaluation run's aggregate into the dashboard (AAEO-15).

        Scoped to ``workspace_id`` (AAEO-4) so the tile never leaks ANOTHER tenant's
        evaluation runs. A TENANT-BOUND caller sees ONLY runs stamped with its own
        workspace — STRICT ``== workspace_id`` (matching the GET-by-id IDOR path): a
        None-stamped run (produced by an offline / admin all-tenants run) was previously
        folded into a tenant tile via the lenient ``in (None, workspace_id)`` filter,
        surfacing another tenant's / an admin run's aggregate_score, suite_name and
        run_id — a cross-tenant metadata leak (scope-r3). It is now EXCLUDED for a
        tenant-bound caller.

        For the ``all_tenants`` principal (offline / admin) the filter is SKIPPED entirely
        when ``workspace_id`` is None, and otherwise stays LENIENT (a None-stamped run the
        caller explicitly scoped to a workspace stays visible) — so the zero-config /
        admin path is byte-unchanged.
        """
        lister = getattr(self._storage, "list_evaluation_runs", None)
        if lister is None:
            return {"total": 0, "latest_aggregate": None}
        try:
            runs = list(await lister())
        except Exception:  # pragma: no cover - eval persistence optional
            return {"total": 0, "latest_aggregate": None}
        if workspace_id is not None:
            if all_tenants:
                # Byte-unchanged offline/admin behaviour: an unstamped run scoped to a
                # workspace stays visible alongside that workspace's own runs.
                runs = [
                    r
                    for r in runs
                    if getattr(r, "workspace_id", None) in (None, workspace_id)
                ]
            else:
                # Tenant-bound: strict — a None-stamped admin/offline run never leaks.
                runs = [
                    r for r in runs if getattr(r, "workspace_id", None) == workspace_id
                ]
        if not runs:
            return {"total": 0, "latest_aggregate": None}
        latest = max(runs, key=lambda r: getattr(r, "created_at", "") or "")
        return {
            "total": len(runs),
            "latest_aggregate": getattr(latest, "aggregate_score", None),
            "latest_suite_name": getattr(latest, "suite_name", None),
            "latest_run_id": getattr(latest, "run_id", None),
        }


# Imported at the BOTTOM (not the collaborator block above the class) to break the import
# cycle the other way: ``run_drive`` reads the schema-validation helpers
# (``_requested_schema`` / ``_validate_structured``) + ``_resolve_model_key`` defined LOWER
# in this module, so its import must run only after those names are bound. ``RunDriveEngine``
# is resolved at ``RunAppService`` INSTANTIATION time (runtime), by which point this module
# is fully loaded, so a bottom import is safe.
from himmy.application.run_drive import RunDriveEngine  # noqa: E402

__all__ = [
    "ContextAppService",
    "RunAppService",
    "RecommendationAppService",
    "DashboardQueryService",
    "AgentDefAppService",
    "AgentDefReferencedError",
    "ThreadAppService",
    "ThreadNotFoundError",
    "ThreadAgentRequiredError",
    "THREAD_AGENT_KEY",
    "THREAD_WORKSPACE_KEY",
    "WorkspaceRunQuotaExceeded",
    "HitlNotSupportedError",
    "HitlRequiresAgentError",
    "RunNotApprovableError",
    "DEFAULT_PAGE_LIMIT",
    "MAX_PAGE_LIMIT",
    "DEFAULT_WORKSPACE_CONCURRENCY",
    "DEFAULT_WORKSPACE_MAX_OUTSTANDING",
]
