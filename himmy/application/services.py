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
import time
from collections.abc import AsyncIterator, Callable, Collection
from typing import TYPE_CHECKING, Any, cast

from himmy.application.models import RecommendationEnvelope
from himmy.entities.lineage import DEFAULT_TRACE_DEPTH
from himmy.services.storage.models import (
    LOCAL_WORKSPACE,
    AgentDefRecord,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.config.agent_spec import AgentSpec
    from himmy.core.ids import utc_now_iso  # noqa: F401
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

#: Backoff schedule for a transient-failed run's re-queue (Q3): ``base * 2**(attempt-1)``
#: seconds, capped at ``max``. Mirrors the :class:`~himmy.connectors.sdk.RetryPolicy`
#: exponential discipline, but expressed as a DB ``next_attempt_at`` delay so the backoff
#: survives a process restart (it is the gate the claim CAS already honours).
_QUEUE_BACKOFF_BASE_SECONDS = 2.0
_QUEUE_BACKOFF_MAX_SECONDS = 300.0

#: Hard age ceiling (seconds) for a run in the leased queue (Q3): once a run has been in
#: flight (created_at -> now) longer than this it is PARKED even if attempts remain, so a
#: run can't be re-queued forever on a persistently-flapping backend.
DEFAULT_QUEUE_MAX_AGE_SECONDS = 3600.0


class WorkspaceRunQuotaExceeded(Exception):
    """A workspace exceeded its outstanding-run cap (T0.4 — surfaces as HTTP 429).

    Raised by :meth:`RunAppService.create_run` when a workspace already has its
    maximum number of in-flight (created-but-not-terminal) background runs. Carries
    the workspace + the cap so the router can return an informative 429.
    """

    def __init__(self, workspace_id: str, *, cap: int, outstanding: int) -> None:
        """Record the workspace and the cap it hit for the API error surface."""
        self.workspace_id = workspace_id
        self.cap = cap
        self.outstanding = outstanding
        super().__init__(
            f"workspace {workspace_id!r} has {outstanding} outstanding run(s); "
            f"the per-workspace cap is {cap}. Retry once in-flight runs complete."
        )


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
    """ISO timestamp helper (kept local to avoid a top-level core import cycle)."""
    from himmy.core.ids import utc_now_iso

    return utc_now_iso()


def _iso_plus_seconds(base_iso: str, seconds: float) -> str:
    """ISO instant advanced by ``seconds`` (kept local to avoid a top-level import cycle)."""
    from himmy.core.ids import iso_plus_seconds

    return iso_plus_seconds(base_iso, seconds)


#: Substrings that mark a run failure as TRANSIENT (worth a re-queue), matched case-
#: insensitively against the recorded ``error`` (Q3). These are the recoverable conditions a
#: laptop/offline deployment hits — the provider was briefly unreachable, a request timed out,
#: the local model was still loading. Everything NOT matching is treated as PERMANENT (a
#: validation/build/tool error that re-running cannot fix), so the dispatcher does not churn on
#: a genuinely-broken run. Conservative on purpose: it is safe to leave an ambiguous failure
#: PERMANENT (the operator can redrive) — re-queuing a truly-permanent one wastes attempts.
_TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "connect error",
    "temporarily unavailable",
    "unreachable",
    "rate limit",
    "429",
    "503",
    "502",
    "504",
    "overloaded",
    "provider",
    "econnreset",
    "broken pipe",
    "model not found",  # ollama: the model is not (yet) pulled/loaded
    "no route to host",
)


def _is_transient_run_error(error: str) -> bool:
    """Whether a recorded run ``error`` looks TRANSIENT (re-queuable) vs PERMANENT (Q3)."""
    if not error:
        return False
    low = error.lower()
    return any(marker in low for marker in _TRANSIENT_ERROR_MARKERS)


def _is_resume_claim_loss(exc: BaseException) -> bool:
    """True when ``exc`` is the exactly-once claim loss from a member resume.

    ``resume_agent_loop`` raises ``HimmyError('checkpoint ... already resolved ...')``
    when the atomic claim is lost (a double/concurrent approve, or a crash-retry the
    winner already finished). For an ORCHESTRATION resume this propagates up through
    ``run_orchestration``, so it must be recognized as a clean NO-OP rather than a
    failure — exactly as the single-agent resume path treats it.
    """
    from himmy.core.errors import HimmyError

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
    ) -> Any:
        """Build and persist an evidenced context snapshot."""
        return await self._context.build_snapshot(
            subject_id=subject_id,
            task_id=task_id,
            build_spec=build_spec,
            metadata=metadata,
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
    """Whether a snapshot belongs to ``workspace_id`` (lenient when unstamped).

    A snapshot stamped with a different workspace is rejected; an unstamped
    snapshot (legacy/no workspace metadata) is allowed so existing callers don't
    break. New snapshots built via the app layer carry the workspace.
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
    return stamped is None or stamped == workspace_id


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
            from himmy.entities.records import record_id_for, stable_id_for

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
        from himmy.entities.records import stable_id_for

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
        self._runtime = runtime
        self._storage = storage
        self._registry = entity_registry
        self._recommendations = recommendation_app or RecommendationAppService(
            storage=storage, entity_registry=entity_registry
        )
        self._run_timeout_seconds = run_timeout_seconds
        self._checkpoint_store = checkpoint_store
        self._agent_resolver = agent_resolver
        self._conversation_sink = conversation_sink
        # P0 tool authz: the RBAC policy used to rebuild a run's tool-capability gate from
        # the persisted ``actor`` metadata at runtime-build time. ``None`` (the offline /
        # zero-config default, and every programmatic caller that doesn't wire RBAC) means
        # NO per-run tool authorizer is built — tool dispatch is byte-identical to before.
        # The server container passes the active :class:`AccessPolicy` so a tenant-bound
        # run's tools are gated deny-by-default; an ANONYMOUS / all_tenants actor still
        # yields a NON-enforcing authorizer (pass-through), preserving offline behavior.
        self._access_policy = access_policy
        # HITL orchestration resume needs the SAME durable graph checkpoint store the
        # team/workflow run paused into. The surface provides it lazily (a getter, so a
        # path/env change is honoured); ``None`` means orchestration HITL resume falls
        # back to an in-memory store (offline/test default — the run paused in-process).
        self._graph_checkpoint_store_provider = graph_checkpoint_store_provider
        # Keep strong refs to background tasks so they are not GC'd mid-flight and
        # so they can be drained/cancelled on shutdown (AAEO-1).
        self._tasks: set[asyncio.Task[Any]] = set()
        # T0.4 per-workspace quotas. ``_ws_semaphores`` gates concurrent EXECUTION;
        # ``_ws_outstanding`` counts created-but-not-terminal runs for the admission
        # cap. Both are keyed on workspace_id and created lazily so an unused
        # workspace costs nothing. A semaphore binds the loop it was created on, so it
        # is created lazily inside the (async) create path, never at construction.
        self._workspace_concurrency = max(1, int(workspace_concurrency))
        self._workspace_max_outstanding = max(0, int(workspace_max_outstanding))
        self._ws_semaphores: dict[str, asyncio.Semaphore] = {}
        self._ws_outstanding: dict[str, int] = {}
        # Q3 leased-dispatch mode. DEFAULT False = the inline fire-and-forget behaviour
        # (today's ``asyncio.create_task`` per run) — preserving the offline single-box +
        # bare-TestClient path byte-for-byte. A :class:`~himmy.application.dispatcher.
        # RunDispatcher`, started in the server lifespan ONLY when the durable run store is
        # active, flips this on via :meth:`enable_dispatch`: from then on
        # ``create_run``/``continue_thread``/``create_orchestration_run`` persist a QUEUED run
        # carrying its recoverable input (Q0) and return WITHOUT a background task — the
        # dispatcher claims + executes it, so a crash between enqueue and execution leaves the
        # run QUEUED (recoverable), never FAILED. The lease TTL is derived from
        # ``run_timeout_seconds`` (a run can't outlive its own timeout), with a safety margin.
        self._dispatch_enabled = False
        # The retry ceiling stamped on every enqueued run (the dispatcher's backoff budget).
        self._default_max_attempts = DEFAULT_QUEUE_MAX_ATTEMPTS
        # T3 per-tenant FAIRNESS on the durable claim. DEFAULT off = global FIFO (byte-identical
        # single-tenant behaviour). When on, the dispatcher claims the least-loaded workspace's
        # run first (round-robin) and refuses to claim a workspace already at its CROSS-NODE
        # per-tenant concurrency cap (``_workspace_concurrency`` — the same cap the in-process
        # execution semaphore uses, now enforced globally across worker processes via the DB).
        self._dispatch_fairness = False

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
        return self._workspace_concurrency

    @property
    def workspace_max_outstanding(self) -> int:
        """The per-workspace outstanding-run cap enforced at enqueue (T0.4 / T3)."""
        return self._workspace_max_outstanding

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

        Idempotent on ``(workspace_id, idempotency_key)``: re-submitting a key
        returns the prior run rather than creating a duplicate. Returns
        immediately with a ``QUEUED`` record; execution runs in the background.

        The create+check is atomic via
        :meth:`StorageService.save_run_if_absent_by_idempotency` (no ``await``
        between read and write in-memory; ``ON CONFLICT DO NOTHING`` in Postgres),
        so two concurrent requests with the same key cannot both create a run.
        Background execution is only launched for the run this call actually
        created.

        ``agent_spec`` (T0.2) is the load-bearing seam: when a run resolves to a
        stored/declarative :class:`AgentSpec` the run executes on a PER-RUN
        tool-bearing runtime built via
        :func:`himmy.runtime.from_spec.build_runtime_for_spec` — so a ``/v1`` run can
        finally call the agent's TOOLS, which the shared (tool-less) runtime cannot.
        When ``agent_spec`` is ``None`` the existing inline-persona fast path runs on
        the shared tool-less runtime, byte-unchanged (back-compat). ``agent_spec`` is
        sanitized against the tenant RCE/SSRF surface (T0.3) unless
        ``operator_provisioned`` is set AND the operator opted in — a tenant spec
        carrying ``tools_module``/``http_tools``/``mcp_servers`` is rejected here.

        Per-workspace admission (T0.4): a workspace already holding
        ``workspace_max_outstanding`` in-flight runs is rejected with
        :class:`WorkspaceRunQuotaExceeded` BEFORE a record is created, so a runaway
        fan-out cannot pin unbounded tasks. An idempotent re-submit of an existing
        run never counts against the cap.
        """
        # HITL admission (T2f): a ``hitl=True`` run pauses on an approval-gated tool, so
        # it REQUIRES a per-run tool-bearing runtime (the shared inline runtime carries
        # no tool_service and cannot call — let alone gate — a tool) AND a checkpoint
        # store to pause into. Reject early (before any record is written) when either is
        # missing, so a caller never gets a silently tool-less "HITL" run that can never
        # pause. The store is resolved on resume from the stored ``agent_id``.
        #
        # Plan mode (T2g) pauses at PLAN-READY through the SAME approval gate (a gated
        # synthetic ``update_plan`` tool), so it has the identical preconditions: a stored
        # agent (so a per-run registry exists to register the plan tool onto) + a
        # checkpoint store. A run may set ``plan`` and ``hitl`` together — the plan gate is
        # simply the first gate the loop hits.
        if hitl or plan:
            if self._checkpoint_store is None:
                raise HitlNotSupportedError(
                    "this deployment has no checkpoint store; hitl/plan runs are "
                    "unavailable"
                )
            if agent_spec is None or agent_def is None:
                raise HitlRequiresAgentError(
                    "hitl/plan runs require a stored agent (agent_id); an inline persona "
                    "carries no tools to gate"
                )
            # Plan mode synthesizes a gated ``update_plan`` tool and registers it onto the
            # per-run tool registry to force the PLAN-READY pause. A bare persona spec
            # (e.g. {"name": "x"}) builds NO registry (from_spec returns registry=None and
            # the runtime gets no tool_service), so there is nowhere to register the plan
            # tool and the run would silently complete WITHOUT pausing. Reject up front
            # (422) — consistent with the inline-persona rejection — rather than letting a
            # legitimate-but-tool-less stored agent slip through to a silently wrong
            # "plan=true never paused" outcome.
            if plan and not agent_spec.builds_tool_registry():
                raise HitlRequiresAgentError(
                    "plan mode requires a tool-bearing stored agent (one declaring "
                    "tool_packs/tools_module/http_tools/knowledge/mcp_servers/connectors/"
                    "allow_spawn/allow_skill_dispatch); a tool-less agent builds no "
                    "registry to host the gated update_plan tool, so the run could never "
                    "pause at PLAN-READY"
                )

        # Fail-closed sanitize a per-run spec against the tenant attack surface
        # BEFORE anything is persisted or executed (T0.3). Operator-provisioned specs
        # may keep their tools when the operator opted in; everyone else is stripped
        # or (default) rejected by ``sanitize_tenant_spec`` raising.
        if agent_spec is not None:
            from himmy.config.spec_sanitizer import sanitize_tenant_spec

            agent_spec = sanitize_tenant_spec(
                agent_spec, operator_provisioned=operator_provisioned
            ).spec

        # Stamp the authenticated actor ("who launched this") into the durable
        # metadata JSONB (round-trips on both in-memory and Postgres) so every run
        # records its initiator — the operational half of "who did what" (WS1.3).
        metadata: dict[str, Any] = {}
        if actor:
            metadata["actor"] = actor
        if agent_spec is not None:
            metadata["agent_name"] = agent_spec.name
        # T2e: a run launched by a stored agent records its agent_id so the run<->agent
        # lineage is reconstructable and the Studio/CLI can show "run of agent X".
        if agent_def is not None:
            metadata["agent_id"] = agent_def.agent_id
        # T2f: mark the run HITL-driven so the resume path (which re-resolves the spec
        # from agent_id) knows this run is approvable and rebuilds the same gated runtime.
        # T2g: plan mode is also an agentic, pausable run — mark it so the resume path
        # re-registers the gated ``update_plan`` tool when rebuilding the runtime.
        if hitl or plan:
            metadata["hitl"] = True
            # ``operator_provisioned`` governs whether the stored spec's privileged tools
            # survive the run-time re-sanitize; the resume must honor the SAME status so
            # the gated tool the run paused on is still present when it is approved.
            metadata["operator_provisioned"] = bool(operator_provisioned)
        if plan:
            metadata["plan_mode"] = True
        model_key = _resolve_model_key(llm_config, task)
        run = RunRecord(
            workspace_id=workspace_id,
            subject_id=subject_id,
            task_id=task.task_id,
            persona_name=persona.name,
            model_key=model_key,
            idempotency_key=idempotency_key,
            status=RunStatus.QUEUED,
            metadata=metadata,
        )
        # Q3: in leased-dispatch mode the run carries its lane (provider health gate) + retry
        # ceiling + recoverable input so the dispatcher (possibly in a FRESH process after a
        # crash) can claim and re-execute it. A no-op on the inline path (fields stay default).
        self._stamp_queue_fields(
            run,
            model_key=model_key,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            hitl=hitl,
            plan=plan,
        )
        # T3 HARD per-tenant outstanding-run cap (dispatch / multi-node path): when the cap
        # is ON and the workspace is not the exempt ``__local__``, the create goes through the
        # ATOMIC store op that FUSES the idempotency check, the active-run COUNT, and the
        # INSERT into ONE advisory-locked transaction (Postgres) / ``BEGIN IMMEDIATE``
        # (SQLite) — so concurrent same-workspace creates serialise and land EXACTLY at the
        # cap, never the unlocked count-then-insert overshoot. ``admitted=False`` means the
        # tenant was at/over cap and NOTHING was written (cleaner than the old
        # insert-then-mark-FAILED): raise the same 429 surface. An idempotent re-submit
        # returns the prior run with ``admitted=True`` and does NOT consume a slot — it must
        # NOT relaunch, so it is distinguished via ``created``.
        if (
            self._dispatch_enabled
            and self._workspace_max_outstanding > 0
            and workspace_id != LOCAL_WORKSPACE
        ):
            stored, admitted = await self._storage.save_run_if_under_quota(
                run, cap=self._workspace_max_outstanding
            )
            if not admitted:
                raise WorkspaceRunQuotaExceeded(
                    workspace_id,
                    cap=self._workspace_max_outstanding,
                    outstanding=self._workspace_max_outstanding,
                )
            # A re-submit of an existing run (same run object identity is NOT guaranteed;
            # detect it by idempotency key resolving to a prior row) must not relaunch. The
            # atomic op returns the stored row; ``run.run_id != stored.run_id`` (or a
            # mismatched created_at) signals a re-submit. The op returns the PRIOR run on a
            # key hit, so compare ids: a fresh insert returns the same object we passed in.
            if stored.run_id != run.run_id:
                return stored
            return await self._launch_or_enqueue(
                stored,
                workspace_id=workspace_id,
                persona=persona,
                task=task,
                llm_config=llm_config,
                agent_spec=agent_spec,
                agent_def=agent_def,
                hitl=hitl,
                plan=plan,
                quota_already_admitted=True,
            )

        # Atomic idempotent insert FIRST (the race-safe primitive), so an idempotent
        # re-submit (created=False) returns the prior run without ever touching the
        # T0.4 cap — a duplicate spawns no new task. Only a NEWLY-created run is
        # admitted against the per-workspace outstanding cap below. (Used by the INLINE
        # path, the exempt ``__local__`` workspace, and the cap-disabled case.)
        stored, created = await self._storage.save_run_if_absent_by_idempotency(run)
        if not created:
            return stored

        return await self._launch_or_enqueue(
            stored,
            workspace_id=workspace_id,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            agent_def=agent_def,
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

        Unlike :meth:`create_run` (a fresh inline/agent run), this APPENDS a turn to an
        existing :class:`ChatThread` and drives the AGENTIC loop on the per-run
        tool-bearing runtime (T0.2) so a ``/v1`` conversation is as capable as a Studio one
        — the model sees prior turns AND can call the agent's tools / pause for approval
        (reviewer must_fix: NOT a tool-less single-shot ``run_task_detailed`` masquerading
        as chat). A continuation always resolves a stored ``agent_id`` (a per-run runtime
        needs a spec), runs through the SAME admission + sanitizer + quota path as a fresh
        run, and links the new :class:`RunRecord` to the conversation via
        ``metadata['conversation_id']`` (the run's ``thread_id`` IS the conversation id) so
        the thread router (and the ConversationStore projection sink) can find it.

        With ``hitl``/``plan`` the run can pause at ``AWAITING_APPROVAL`` exactly like a
        fresh agentic run; the SAME ``approve``/``reject`` machinery resumes it.
        """
        # A continuation is always agentic + needs a checkpoint store when it can pause.
        if (hitl or plan) and self._checkpoint_store is None:
            raise HitlNotSupportedError(
                "this deployment has no checkpoint store; hitl/plan runs are unavailable"
            )
        # Plan mode hosts its gated ``update_plan`` tool on the per-run registry; a
        # tool-less stored agent builds none, so the run could never pause at PLAN-READY.
        # Reject up front (422) — same gate as ``create_run`` — rather than silently
        # running plan=true to completion without a pause.
        if plan and not agent_spec.builds_tool_registry():
            raise HitlRequiresAgentError(
                "plan mode requires a tool-bearing stored agent; a tool-less agent "
                "builds no registry to host the gated update_plan tool, so the run "
                "could never pause at PLAN-READY"
            )

        # Fail-closed sanitize the spec under the request's operator status BEFORE it
        # runs (defense-in-depth: the stored spec was sanitized at write, this honors the
        # caller's operator status so a tenant continuation can never reach privileged
        # tools the stored spec might carry).
        from himmy.config.spec_sanitizer import sanitize_tenant_spec

        agent_spec = sanitize_tenant_spec(
            agent_spec, operator_provisioned=operator_provisioned
        ).spec

        # Pin the thread id to the conversation id so the run's thread_id IS the
        # conversation id (the run<->conversation join the router/sink rely on). The new
        # user turn is NOT appended here — the runtime's ``run_agent_loop`` /
        # ``run_task_detailed`` appends it from ``task.prompt`` (appending it twice would
        # duplicate the message), so the loaded thread carries only the PRIOR turns.
        thread.thread_id = conversation_id

        persona = agent_spec.to_persona()
        if llm_config is None:
            llm_config = agent_spec.to_llm_config()
        task = agent_spec.make_task(prompt)

        metadata: dict[str, Any] = {"conversation_id": conversation_id}
        if actor:
            metadata["actor"] = actor
        metadata["agent_name"] = agent_spec.name
        metadata["agent_id"] = agent_def.agent_id
        if hitl or plan:
            metadata["hitl"] = True
            metadata["operator_provisioned"] = bool(operator_provisioned)
        if plan:
            metadata["plan_mode"] = True

        model_key = _resolve_model_key(llm_config, task)
        run = RunRecord(
            workspace_id=workspace_id,
            subject_id=subject_id,
            task_id=task.task_id,
            persona_name=persona.name,
            model_key=model_key,
            idempotency_key=idempotency_key,
            status=RunStatus.QUEUED,
            thread_id=conversation_id,
            metadata=metadata,
        )
        # Q3: stamp lane/retry/recoverable-input so a continuation can be claimed + re-run by
        # the dispatcher (a no-op on the inline path). The recovered run carries thread_id =
        # conversation_id, so the rebuilt task continues the SAME conversation.
        self._stamp_queue_fields(
            run,
            model_key=model_key,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            hitl=hitl,
            plan=plan,
        )
        # T3 HARD per-tenant outstanding-run cap (dispatch / multi-node path): a continuation
        # is a real run and MUST land under the SAME atomic gate as a fresh ``create_run`` —
        # the prior soft ``save_run_if_absent`` + count-then-mark-FAILED ``_admit`` had a
        # TOCTOU window letting concurrent same-conversation/same-workspace continuations
        # overshoot the cap. Mirror create_run (count+insert fused in ONE advisory-locked
        # xact): ``admitted=False`` means the tenant was at/over cap and NOTHING was written
        # (raise the 429); an idempotent re-submit returns the prior row WITHOUT relaunching
        # and WITHOUT consuming a slot; a fresh admit launches with the soft re-check skipped.
        if (
            self._dispatch_enabled
            and self._workspace_max_outstanding > 0
            and workspace_id != LOCAL_WORKSPACE
        ):
            stored, admitted = await self._storage.save_run_if_under_quota(
                run, cap=self._workspace_max_outstanding
            )
            if not admitted:
                raise WorkspaceRunQuotaExceeded(
                    workspace_id,
                    cap=self._workspace_max_outstanding,
                    outstanding=self._workspace_max_outstanding,
                )
            if stored.run_id != run.run_id:
                # idempotent re-submit: do not relaunch, do not consume a slot.
                return stored
            return await self._launch_or_enqueue(
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
                quota_already_admitted=True,
            )

        # OFF / __local__ / cap<=0 / inline — byte-identical legacy soft path.
        stored, created = await self._storage.save_run_if_absent_by_idempotency(run)
        if not created:
            return stored

        return await self._launch_or_enqueue(
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

        A no-op on the inline path (the fields keep their RunRecord defaults). In dispatch
        mode it stamps: ``lane_key`` (the provider health-gate lane derived from the model
        key), ``max_attempts`` (the retry ceiling), and ``input_blob`` — the Q0 recoverable
        launch input serialized so a dispatcher in a FRESH process can rehydrate the exact
        persona/task/spec and re-execute. The blob is stored PLAINTEXT here; the durable run
        store encrypts it at rest (bound to run_id) on write, exactly like chat content.
        """
        if not self._dispatch_enabled:
            return
        from himmy.services.storage.run_input import encode_run_input
        from himmy.services.storage.run_lane import lane_for_model_key

        run.lane_key = lane_for_model_key(model_key)
        run.max_attempts = self._default_max_attempts
        run.input_blob = encode_run_input(
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            hitl=hitl,
            plan=plan,
            run_id=run.run_id,
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

        INLINE mode (default, offline single-box + bare TestClient): admits the run against
        the per-workspace outstanding cap and launches today's fire-and-forget background
        task — byte-identical to the pre-Q3 behaviour. DISPATCH mode (durable store + the
        lifespan dispatcher): the run is left QUEUED for the dispatcher to claim, so a crash
        between enqueue and execution leaves it recoverable (not FAILED). Admission/concurrency
        in dispatch mode is the dispatcher's job (its bounded claim loop + the per-workspace
        execution semaphore at run time), so the in-memory outstanding counter — which a
        cross-process claim could never release — is NOT taken here.

        ``quota_already_admitted`` is set by the dispatch caller when the per-tenant
        outstanding cap was already enforced ATOMICALLY (count+insert fused) via
        :meth:`StorageService.save_run_if_under_quota` — the HARD path. In that case the
        durable count-then-mark-FAILED admit below is SKIPPED (it would double-count and the
        atomic op already rejected an over-cap create before any row was written).
        """
        if self._dispatch_enabled:
            # T3 per-tenant QUOTA at enqueue (dispatch / multi-node path): the in-RAM
            # outstanding counter cannot span worker processes, so the cap is enforced by
            # COUNTING the workspace's non-terminal runs in the SHARED store before leaving the
            # run QUEUED. When the caller already enforced the cap atomically
            # (``quota_already_admitted``), this soft re-check is skipped — the atomic op is
            # the HARD enforcer and a second count here would only double-work. Otherwise (the
            # legacy soft fallback) a burst beyond the cap is rejected (the record is marked
            # FAILED, not orphaned QUEUED) and the 429 propagates. ``0`` disables the cap
            # (single-tenant default unaffected, since the LOCAL workspace is exempted).
            if not quota_already_admitted:
                await self._admit_workspace_run_durable(workspace_id, stored)
            # Recoverable QUEUED state; the dispatcher claims + executes it.
            return stored

        # T0.4 admission (inline only): a workspace already at its outstanding-run cap cannot
        # launch another background task. The record exists (atomicity preserved), so it is
        # marked FAILED rather than orphaned QUEUED, and the quota error propagates (HTTP 429).
        try:
            self._admit_workspace_run(workspace_id)
        except WorkspaceRunQuotaExceeded:
            stored.status = RunStatus.FAILED
            stored.error = "rejected: workspace run-concurrency quota exceeded"
            stored.updated_at = _now()
            try:
                await self._storage.save_run(stored)
            except Exception:  # pragma: no cover - best-effort terminal mark
                logger.warning("failed to mark quota-rejected run %s", stored.run_id)
            raise

        bg = asyncio.create_task(
            self._execute_run(
                stored.run_id,
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
        )
        self._tasks.add(bg)
        bg.add_done_callback(self._tasks.discard)
        return stored

    # ----------------------------------------------------------- T0.4 quotas
    def _admit_workspace_run(self, workspace_id: str) -> None:
        """Reserve one outstanding-run slot for ``workspace_id`` or reject (T0.4).

        Raises :class:`WorkspaceRunQuotaExceeded` when the workspace already holds
        :attr:`_workspace_max_outstanding` in-flight runs. ``0`` disables the cap.
        Runs single-threaded in the event loop, so the read-modify-write is atomic.
        """
        if self._workspace_max_outstanding <= 0:
            return
        current = self._ws_outstanding.get(workspace_id, 0)
        if current >= self._workspace_max_outstanding:
            raise WorkspaceRunQuotaExceeded(
                workspace_id,
                cap=self._workspace_max_outstanding,
                outstanding=current,
            )
        self._ws_outstanding[workspace_id] = current + 1

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
        if self._workspace_max_outstanding <= 0:
            return
        current = self._ws_outstanding.get(workspace_id, 0)
        if current <= 1:
            self._ws_outstanding.pop(workspace_id, None)
        else:
            self._ws_outstanding[workspace_id] = current - 1

    def _workspace_semaphore(self, workspace_id: str) -> asyncio.Semaphore:
        """Lazily get/create the per-workspace execution semaphore (T0.4).

        Created on first use inside the running loop so it binds the correct event
        loop (a semaphore created at construction could bind the wrong loop under
        ``asyncio.run`` test harnesses).
        """
        sem = self._ws_semaphores.get(workspace_id)
        if sem is None:
            sem = asyncio.Semaphore(self._workspace_concurrency)
            self._ws_semaphores[workspace_id] = sem
        return sem

    def workspace_outstanding(self, workspace_id: str) -> int:
        """Return the current count of in-flight runs for a workspace (introspection)."""
        return self._ws_outstanding.get(workspace_id, 0)

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
        semaphore = self._workspace_semaphore(workspace_id)
        try:
            async with semaphore:
                run = await self._storage.get_run(run_id)
                if run is None:  # pragma: no cover - defensive
                    return
                await self._execute_on_runtime(
                    run,
                    persona=persona,
                    task=task,
                    llm_config=llm_config,
                    agent_spec=agent_spec,
                    agent_def=agent_def,
                    hitl=hitl,
                    plan=plan,
                    thread=thread,
                )
        finally:
            self._release_workspace_run(workspace_id)

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
        run_id = run.run_id
        workspace_id = run.workspace_id
        # Orchestration (team/workflow/graph) runs reconstruct from member_agent_ids + the
        # persisted prompt, not a single-agent input_blob — route them to their own driver.
        if (run.metadata or {}).get("orchestration"):
            await self._dispatch_orchestration_run(run)
            await self._apply_retry_policy(run_id)
            return
        # 1. rehydrate the recoverable input.
        if not run.input_blob:
            run.status = RunStatus.FAILED
            run.error = (
                "run has no recoverable input (input_blob missing); cannot be "
                "re-executed by the dispatcher"
            )
            run.last_error = run.error
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        from himmy.services.storage.run_input import RunInputError, decode_run_input

        try:
            rinput = decode_run_input(run.input_blob, run_id=run_id)
        except RunInputError as exc:
            run.status = RunStatus.FAILED
            run.error = f"run input could not be rehydrated: {exc}"
            run.last_error = run.error
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

        # 2. re-resolve the stored agent_def (lineage edge) — best-effort; a removed agent
        # just means no run<->agent link, not a failure.
        agent_def: AgentDefRecord | None = None
        agent_id = (run.metadata or {}).get("agent_id")
        if agent_id and self._agent_resolver is not None:
            try:
                agent_def = await _maybe_await(
                    self._agent_resolver(agent_id, workspace_id=workspace_id)
                )
            except Exception:  # noqa: BLE001 - resolver failure must not crash the worker
                agent_def = None

        # 3. execute under the per-workspace concurrency semaphore (same back-pressure as the
        # inline path). A continuation carries thread_id == its conversation id, so reload the
        # prior thread to continue it; a fresh run starts a new thread.
        thread: ChatThread | None = None
        if run.thread_id:
            try:
                thread = await self._storage.load_thread(run.thread_id)
            except Exception:  # noqa: BLE001 - a missing thread just starts fresh
                thread = None

        semaphore = self._workspace_semaphore(workspace_id)
        async with semaphore:
            await self._execute_on_runtime(
                run,
                persona=rinput.persona,
                task=rinput.task,
                llm_config=rinput.llm_config,
                agent_spec=rinput.agent_spec,
                agent_def=agent_def,
                hitl=rinput.hitl,
                plan=rinput.plan,
                thread=thread,
            )

        # 4. retry/backoff/PARK on a transient failure.
        await self._apply_retry_policy(run_id)

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
        run = await self._storage.get_run(run_id)
        if run is None or run.status != RunStatus.FAILED:
            return
        error = run.error or ""
        if not _is_transient_run_error(error):
            return  # permanent failure: leave it FAILED for the operator/caller.
        age = time.time() - _parse_iso_epoch(run.created_at)
        attempts_left = run.attempt < max(1, run.max_attempts)
        if attempts_left and age < DEFAULT_QUEUE_MAX_AGE_SECONDS:
            delay = min(
                _QUEUE_BACKOFF_BASE_SECONDS * (2.0 ** max(0, run.attempt - 1)),
                _QUEUE_BACKOFF_MAX_SECONDS,
            )
            now_iso = _now()
            run.status = RunStatus.QUEUED
            run.owner_id = None
            run.lease_expires_at = None
            run.last_error = error
            run.error = None
            run.next_attempt_at = _iso_plus_seconds(now_iso, delay)
            run.updated_at = now_iso
            await self._storage.save_run(run)
            logger.info(
                "re-queued transient-failed run %s (attempt %d/%d) in %.0fs",
                run_id,
                run.attempt,
                run.max_attempts,
                delay,
            )
            return
        # Budget exhausted: PARK (terminal-but-redrivable), preserving the last error.
        run.status = RunStatus.PARKED
        run.last_error = error
        run.owner_id = None
        run.lease_expires_at = None
        run.updated_at = _now()
        await self._storage.save_run(run)
        logger.info(
            "parked run %s after %d attempt(s) of transient failure", run_id, run.attempt
        )

    async def _dispatch_orchestration_run(self, run: RunRecord) -> None:
        """Reconstruct + drive a CLAIMED orchestration run from a fresh process (Q3).

        The team/workflow/graph run carries its member ids + the persisted prompt in metadata
        (not a single-agent input_blob), so recovery re-resolves the members via the same
        resolver the resume path uses and rebuilds the graph checkpoint store from the
        provider. A member that has since been removed fails the run with a clear reason
        (the retry policy then classifies it permanent). Delegates the actual orchestration to
        :meth:`_execute_orchestration_run`, which sets the terminal/AWAITING state.
        """
        meta = run.metadata or {}
        member_agent_ids = list(meta.get("member_agent_ids") or [])
        prompt = meta.get("orchestration_prompt", "")
        kind = meta.get("orchestration_kind", "graph")
        resource_kind = meta.get("orchestration", "workflow")
        operator_provisioned = bool(meta.get("operator_provisioned", False))
        graph_resume_id = meta.get("graph_resume_id")

        if not member_agent_ids or self._agent_resolver is None:
            run.status = RunStatus.FAILED
            run.error = "orchestration run cannot be recovered: no resolvable members"
            run.last_error = run.error
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        members: list[AgentDefRecord] = []
        for agent_id in member_agent_ids:
            rec = await _maybe_await(
                self._agent_resolver(agent_id, workspace_id=run.workspace_id)
            )
            if rec is None:
                run.status = RunStatus.FAILED
                run.error = f"orchestration recovery failed: member agent {agent_id} removed"
                run.last_error = run.error
                run.updated_at = _now()
                await self._storage.save_run(run)
                return
            members.append(rec)

        await self._execute_orchestration_run(
            run.run_id,
            workspace_id=run.workspace_id,
            kind=kind,
            members=members,
            prompt=prompt,
            resource_kind=resource_kind,
            operator_provisioned=operator_provisioned,
            graph_checkpoint_store=self._resolve_graph_checkpoint_store(),
            graph_resume_id=graph_resume_id,
        )

    async def _execute_on_runtime(
        self,
        run: RunRecord,
        *,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None,
        agent_def: AgentDefRecord | None = None,
        hitl: bool = False,
        plan: bool = False,
        thread: ChatThread | None = None,
    ) -> None:
        """Drive one run to a terminal state on the resolved runtime (T0.2 core).

        With ``hitl=True`` (T2f) the run drives :meth:`SingleAgentRuntime.run_agent_loop`
        with ``hitl=True`` on a per-run runtime carrying /v1's checkpoint store; if it
        pauses on an approval-gated tool the run goes AWAITING_APPROVAL (carrying the
        checkpoint id) and STOPS — never swept, resumable via :meth:`resume_run`. The
        non-HITL path is byte-identical to before (single ``run_task_detailed`` turn).

        ``plan=True`` (T2g) is an agentic, pausable run too: the per-run runtime carries
        the gated ``update_plan`` tool so the loop pauses at PLAN-READY through the same
        approval machinery. ``thread`` (T2g) is the continuation seam — a prior
        :class:`ChatThread` the loop continues so the model sees earlier turns; ``None``
        starts a fresh thread (every pre-T2g call site).
        """
        # HITL/plan runs pause into /v1's OWN checkpoint store; the plain single-turn path
        # passes None so the per-run runtime stays exactly as the T0.2 build wired it.
        agentic = hitl or plan
        checkpoint_store = self._checkpoint_store if agentic else None
        run.status = RunStatus.RUNNING
        run.updated_at = _now()
        await self._storage.save_run(run)

        # T2g: a plan-first run prepends the plan nudge + binds ``update_plan`` so the
        # agent publishes (and pauses on) its plan first.
        if plan:
            from himmy.runtime.plan_mode import apply_plan_mode_to_task

            apply_plan_mode_to_task(task)

        try:
            runtime = await self._resolve_runtime(
                agent_spec,
                checkpoint_store=checkpoint_store,
                plan_mode=plan,
                workspace_id=run.workspace_id,
                actor=(run.metadata or {}).get("actor"),
            )
        except Exception as exc:  # noqa: BLE001 - spec wiring failure is terminal
            run.status = RunStatus.FAILED
            run.error = f"agent runtime build failed: {exc}"
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

        # T2f/T2g: the agentic path drives the loop (which can PAUSE on an approval-gated
        # tool — the HITL gate OR the plan gate); the plain path is the single-turn fast
        # path, optionally continuing a prior thread (T2g).
        if agentic:
            await self._drive_hitl_run(
                run,
                persona,
                task,
                runtime,
                llm_config=llm_config,
                agent_def=agent_def,
                thread=thread,
                plan=plan,
            )
            return

        try:
            # Pass ``thread`` ONLY when continuing a prior conversation (T2g); the
            # fresh-run path omits it so the call is byte-identical to the pre-T2g signature
            # (``run_task_detailed(persona, task, llm_config=...)``) — back-compat for every
            # existing call site and test double.
            coro = (
                runtime.run_task_detailed(persona, task, thread, llm_config=llm_config)
                if thread is not None
                else runtime.run_task_detailed(persona, task, llm_config=llm_config)
            )
            result = await asyncio.wait_for(coro, timeout=self._run_timeout_seconds)
        except TimeoutError:
            run.status = RunStatus.FAILED
            run.error = (
                f"run exceeded {self._run_timeout_seconds:.0f}s execution timeout"
            )
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        except asyncio.CancelledError:
            # Shutdown drain / explicit cancel: record FAILED then re-raise so the
            # task unwinds as a cancellation (AAEO-1).
            run.status = RunStatus.FAILED
            run.error = "run cancelled"
            run.updated_at = _now()
            try:
                await self._storage.save_run(run)
            except Exception:  # pragma: no cover - best-effort during cancel
                pass
            raise
        except Exception as exc:  # noqa: BLE001 - terminal failure transition
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

        thread = result.thread
        run.thread_id = thread.thread_id
        run.trace_id = result.trace_id

        # T2e: link the run's chat_thread hub -> the stored agent node so a run launched
        # by ``agent_id`` carries a durable run<->agent lineage edge (best-effort; the
        # run never fails because lineage projection did).
        if agent_def is not None:
            await self._project_run_agent_link(run, agent_def)

        # AAEO-3: honour the FAILED inference path. ``RunResult.succeeded`` is the
        # typed status surface; a failed run records the error and skips extraction.
        if not result.succeeded:
            run.status = RunStatus.FAILED
            run.error = result.error or (result.error_code or "inference failed")
            run.output_text = result.output_text or None
            run.updated_at = _now()
            await self._storage.save_run(run)
            await self._notify_conversation_sink(run)
            return

        await self._finalize_succeeded_run(run, result, task=task, llm_config=llm_config)
        await self._notify_conversation_sink(run)

    async def _finalize_succeeded_run(
        self,
        run: RunRecord,
        result: Any,
        *,
        task: Task,
        llm_config: LLMConfig | None,
    ) -> None:
        """Record a terminal-SUCCEEDED run + extract recommendations (shared path).

        Factored out so both the single-turn path and the HITL loop's terminal turn
        finish identically (structured-output parse, AAEO-6 schema validation, the
        SUCCEEDED transition, and recommendation extraction).
        """
        run.output_text = result.output_text or None
        # Prefer the typed structured output; fall back to parsing the text.
        structured = result.output_structured
        if structured is None:
            structured = self._parse_structured(result.output_text)
        run.output_structured = structured

        # AAEO-6: validate the structured output against the requested schema
        # before extraction, recording any failure on the run.
        schema = _requested_schema(llm_config, task)
        if structured is not None and schema is not None:
            error = _validate_structured(structured, schema)
            if error is not None:
                run.metadata = {
                    **(run.metadata or {}),
                    "extraction_error": f"schema validation failed: {error}",
                }

        run.status = RunStatus.SUCCEEDED
        run.updated_at = _now()
        await self._storage.save_run(run)

        # Auto-extract recommendations when the output matches the envelope.
        if run.output_structured is not None:
            await self._recommendations.extract_from_run(run)

    async def _drive_hitl_run(
        self,
        run: RunRecord,
        persona: Persona,
        task: Task,
        runtime: SingleAgentRuntime,
        *,
        llm_config: LLMConfig | None,
        agent_def: AgentDefRecord | None,
        thread: ChatThread | None = None,
        plan: bool = False,
    ) -> None:
        """Drive a hitl/plan run's agentic loop; pause to AWAITING_APPROVAL on a gate.

        Runs :meth:`SingleAgentRuntime.run_agent_loop` (``hitl=True``) bounded by the
        per-run timeout. The loop either:

        * pauses on an approval-gated tool — ``stopped_reason == 'awaiting_approval'`` —
          in which case the run is stamped AWAITING_APPROVAL + ``metadata['checkpoint_id']``
          and STOPS (never swept, resumable via approve/reject), or
        * completes — handled exactly like the single-turn success/failure paths via the
          shared finalizers.

        ``thread`` (T2g) continues a prior conversation; ``plan`` marks a plan-first run
        so :meth:`_apply_loop_outcome` extracts the published plan into
        ``metadata['plan']`` when the run pauses at PLAN-READY.
        """
        try:
            # Pass ``thread`` only on a continuation (T2g) so a fresh agentic run's call is
            # byte-identical to the pre-T2g signature (back-compat for test doubles).
            loop_coro = (
                runtime.run_agent_loop(
                    persona, task, thread, llm_config=llm_config, hitl=True
                )
                if thread is not None
                else runtime.run_agent_loop(
                    persona, task, llm_config=llm_config, hitl=True
                )
            )
            loop = await asyncio.wait_for(
                loop_coro, timeout=self._run_timeout_seconds
            )
        except TimeoutError:
            run.status = RunStatus.FAILED
            run.error = (
                f"run exceeded {self._run_timeout_seconds:.0f}s execution timeout"
            )
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        except asyncio.CancelledError:
            run.status = RunStatus.FAILED
            run.error = "run cancelled"
            run.updated_at = _now()
            try:
                await self._storage.save_run(run)
            except Exception:  # pragma: no cover - best-effort during cancel
                pass
            raise
        except Exception as exc:  # noqa: BLE001 - terminal failure transition
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

        await self._apply_loop_outcome(
            run, loop, task=task, llm_config=llm_config, agent_def=agent_def, plan=plan
        )

    async def _apply_loop_outcome(
        self,
        run: RunRecord,
        loop: Any,
        *,
        task: Task,
        llm_config: LLMConfig | None,
        agent_def: AgentDefRecord | None,
        plan: bool = False,
    ) -> None:
        """Project a finished/paused :class:`AgentLoopResult` onto the run record (T2f).

        Shared by the initial HITL drive and the resume path so a run pauses-again,
        succeeds, or fails identically regardless of which entry produced the loop.

        ``plan`` (T2g) makes a PLAN-READY pause stamp the published plan steps into
        ``metadata['plan']`` so a caller can read the proposed plan before approving.
        """
        thread = loop.thread
        run.thread_id = thread.thread_id
        # The loop's trace id is derived as ``{thread_id}:{task_id}``; record it so the
        # canonical event replay (``get_run_events``) finds the turn/tool events.
        run.trace_id = f"{thread.thread_id}:{task.task_id}"

        # T2e: link the run's chat_thread hub -> the stored agent node (best-effort).
        if agent_def is not None:
            await self._project_run_agent_link(run, agent_def)

        if loop.stopped_reason == "awaiting_approval":
            # The headline T2f transition: the run paused on an approval-gated tool.
            # Stamp the checkpoint id so approve/reject can re-claim it, and STOP — the
            # sweeper explicitly skips AWAITING_APPROVAL, so this run is never reaped.
            run.status = RunStatus.AWAITING_APPROVAL
            metadata = {
                **(run.metadata or {}),
                "checkpoint_id": loop.checkpoint_id,
            }
            # T2g: a plan-first run paused on its (gated) ``update_plan`` call — surface
            # the proposed plan in metadata so a caller reads it before approving.
            if plan:
                plan_steps = self._extract_plan_from_checkpoint(loop.checkpoint_id)
                if plan_steps:
                    metadata["plan"] = plan_steps
            run.metadata = metadata
            run.updated_at = _now()
            await self._storage.save_run(run)
            await self._notify_conversation_sink(run)
            return

        final = loop.final
        if not final.succeeded:
            run.status = RunStatus.FAILED
            run.error = final.error or (final.error_code or "inference failed")
            run.output_text = final.output_text or None
            run.updated_at = _now()
            await self._storage.save_run(run)
            await self._notify_conversation_sink(run)
            return

        await self._finalize_succeeded_run(run, final, task=task, llm_config=llm_config)
        await self._notify_conversation_sink(run)

    async def _notify_conversation_sink(self, run: RunRecord) -> None:
        """Re-project a thread-linked run's updated ChatThread (T2g, best-effort).

        Invoked when a run that the thread router started (carrying
        ``metadata['conversation_id']``) reaches a terminal / AWAITING_APPROVAL state, so
        the ConversationStore projection reflects the latest turns even when the run
        finished on a background task the router never awaited (an approval-resume). A
        no-op when no sink is wired or the run is not thread-linked; any sink error is
        swallowed (provenance, not the work).
        """
        if self._conversation_sink is None:
            return
        conversation_id = (run.metadata or {}).get("conversation_id")
        if not conversation_id:
            return
        try:
            await _maybe_await(self._conversation_sink(conversation_id, run))
        except Exception:  # noqa: BLE001 - conversation projection is best-effort
            logger.warning(
                "failed to project conversation %s for run %s",
                conversation_id,
                run.run_id,
            )

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
        if self._access_policy is None:
            return None
        from himmy.services.tools.capability import ToolCapabilityAuthorizer

        return ToolCapabilityAuthorizer.from_actor(actor, self._access_policy)

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
        if agent_spec is None:
            return self._runtime
        # Reuse the shared runtime's inference only when the spec does not pin its own
        # provider; otherwise let from_spec build the provider-specific service.
        shared_inference = (
            getattr(self._runtime, "inference_service", None)
            if not agent_spec.provider
            else None
        )
        from himmy.runtime.from_spec import build_runtime_for_spec

        # P0: rebuild the run principal's tool-capability gate from the persisted actor and
        # thread it into the per-run runtime (no-op offline / when no RBAC policy is wired).
        tool_authorizer = self._build_tool_authorizer(actor)

        runtime, registry = await asyncio.to_thread(
            build_runtime_for_spec,
            agent_spec,
            inference=shared_inference,
            storage=self._storage,
            checkpoint_store=checkpoint_store,
            subject=workspace_id,
            tool_authorizer=tool_authorizer,
        )
        runtime = cast("SingleAgentRuntime", runtime)
        if plan_mode:
            # The plan tool must live on a tool registry; a spec with no tools builds no
            # registry, so resolve the runtime's own tool_service registry as the target.
            target = registry
            if target is None:
                target = getattr(
                    getattr(runtime, "tool_service", None), "registry", None
                )
            if target is not None:
                from himmy.runtime.plan_mode import register_plan_tool

                register_plan_tool(target)
        return runtime

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
        if self._registry is None or not run.thread_id:
            return
        try:
            from himmy.entities.records import stable_id_for

            agent_record = await _maybe_await(
                self._registry.register(agent_def.to_record())
            )
            thread_sid = stable_id_for(run.thread_id, namespace="chat_thread")
            thread_record = await _maybe_await(self._registry.get_latest(thread_sid))
            if thread_record is None:
                return
            # Dedupe the edge so a re-run of the same thread/agent never doubles it.
            existing = await _maybe_await(
                self._registry.links_from(thread_record.record_id)
            )
            for link in existing:
                if (
                    link.to_record_id == agent_record.record_id
                    and link.relation == "run_of_agent"
                ):
                    return
            await _maybe_await(
                self._registry.link(
                    from_record_id=thread_record.record_id,
                    to_record_id=agent_record.record_id,
                    relation="run_of_agent",
                    metadata={"run_id": run.run_id, "agent_id": agent_def.agent_id},
                )
            )
        except Exception:  # pragma: no cover - lineage projection is best-effort
            logger.warning(
                "failed to project run<->agent lineage for run %s agent %s",
                run.run_id,
                agent_def.agent_id,
            )

    @staticmethod
    def _parse_structured(content: str | None) -> Any:
        """Parse JSON content into a structure, returning None on non-JSON text."""
        if not content:
            return None
        import json

        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, (dict, list)):
            return parsed
        return None

    # ----------------------------------------------------------- lifecycle (AAEO-1)
    async def drain(self, *, timeout: float = 30.0) -> None:
        """Cancel + await all in-flight background runs (FastAPI shutdown hook).

        Each task is cancelled and awaited; the per-task cancel handler records
        the run FAILED('run cancelled'). Bounded by ``timeout`` so shutdown cannot
        hang forever.
        """
        tasks = list(self._tasks)
        if not tasks:
            return
        for t in tasks:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )
        except TimeoutError:  # pragma: no cover - shutdown best-effort
            logger.warning(
                "drain timed out after %.0fs with tasks still running", timeout
            )

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
        if self._dispatch_enabled:
            # Re-queue ONLY lease-expired RUNNING runs; a live peer (future lease) and every
            # QUEUED/AWAITING/RESOLVING run is left untouched by the lease predicate.
            requeued = await self._storage.requeue_expired_leases()
            if requeued:
                logger.info(
                    "re-queued %d lease-expired run(s) on startup", len(requeued)
                )
            return requeued

        swept: list[str] = []
        runs = await self._storage.list_runs()
        now = time.time()
        for run in runs:
            # Only QUEUED/RUNNING are "abandoned"; SUCCEEDED/FAILED/PARKED are terminal,
            # AWAITING_APPROVAL is intentionally paused, and RESOLVING is a re-drivable
            # crashed resume (reconcile_resolving_runs) — never sweep any of those.
            if run.status not in (RunStatus.QUEUED, RunStatus.RUNNING):
                continue
            if (
                ttl_seconds > 0
                and (now - _parse_iso_epoch(run.updated_at)) < ttl_seconds
            ):
                continue
            run.status = RunStatus.FAILED
            run.error = "run abandoned (process restart); swept to terminal state"
            run.updated_at = _now()
            await self._storage.save_run(run)
            swept.append(run.run_id)
        return swept

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
        redriven: list[str] = []
        runs = await self._storage.list_runs(status=RunStatus.RESOLVING)
        for run in runs:
            decision = (run.metadata or {}).get("resume_decision")
            if decision not in ("approved", "rejected"):
                # No recorded decision — cannot faithfully re-drive; leave it for an
                # operator (never guess approve, which would run the gated tool).
                continue
            actor = (run.metadata or {}).get("resume_actor") or "recovery"
            # Re-open to AWAITING_APPROVAL so resume_run re-claims it through the SAME
            # proven path. Startup recovery is single-threaded (no concurrent approves
            # yet), so the brief re-open carries no double-resume risk.
            run.status = RunStatus.AWAITING_APPROVAL
            run.updated_at = _now()
            await self._storage.save_run(run)
            try:
                await self.resume_run(
                    run.run_id,
                    approved=decision == "approved",
                    workspace_id=run.workspace_id,
                    actor=actor,
                )
            except RunNotApprovableError:  # pragma: no cover - lost a concurrent re-claim
                continue
            redriven.append(run.run_id)
        return redriven

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
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None:
            return None
        if self._checkpoint_store is None:
            return []
        checkpoint_id = (run.metadata or {}).get("checkpoint_id")
        if not checkpoint_id:
            return []
        checkpoint = self._checkpoint_store.load(checkpoint_id)
        if checkpoint is None:
            return []
        from himmy.runtime.checkpoint import redact_tool_args

        return [
            {"tool_name": p.tool_name, "args": redact_tool_args(p.args)}
            for p in checkpoint.pending_tool_calls
        ]

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
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None:
            raise RunNotApprovableError(run_id, status="unknown")
        if run.status != RunStatus.AWAITING_APPROVAL:
            raise RunNotApprovableError(run_id, status=run.status.value)
        if self._checkpoint_store is None:  # pragma: no cover - guarded at create
            raise HitlNotSupportedError("no checkpoint store wired; cannot resume")

        # Validate that the resume CAN proceed BEFORE claiming, so a config error
        # (missing checkpoint / unresolvable agent) 409s without first stranding the run
        # in RESOLVING. Orchestration runs validate their member ids/graph checkpoint
        # inside ``_resume_orchestration`` (also pre-claim, below).
        is_orchestration = (run.metadata or {}).get("hitl_kind") == "orchestration"
        checkpoint_id = ""
        agent_def: AgentDefRecord | None = None
        if not is_orchestration:
            checkpoint_id = (run.metadata or {}).get("checkpoint_id") or ""
            if not checkpoint_id:  # pragma: no cover - an AWAITING run always has one
                raise RunNotApprovableError(run_id, status="no checkpoint")
            agent_id = (run.metadata or {}).get("agent_id")
            if not agent_id or self._agent_resolver is None:
                raise RunNotApprovableError(run_id, status="no resolvable agent")
            agent_def = await _maybe_await(
                self._agent_resolver(agent_id, workspace_id=run.workspace_id)
            )
            if agent_def is None:
                raise RunNotApprovableError(run_id, status="agent removed")

        # ATOMIC run-level claim: flip AWAITING_APPROVAL -> RESOLVING exactly once. The
        # non-atomic check above is only for a clean 404/409 message; THIS compare-and-set
        # is the authoritative gate. The loser of a concurrent double-approve fails the CAS
        # here and 409s without launching a resume — so an orchestration graph advance can
        # never double-fire downstream members' tools. (For a single-agent run the member
        # checkpoint claim() is a second backstop; for orchestration this is the ONLY gate
        # on the post-member graph advance.)
        claimed = await self._storage.claim_run_for_resume(
            run_id, workspace_id=run.workspace_id
        )
        if not claimed:
            raise RunNotApprovableError(run_id, status=RunStatus.RESOLVING.value)
        # Reflect the won claim on the local record (the background task flips RESOLVING ->
        # RUNNING when it actually starts; the returned record shows the in-progress state).
        # Persist the approve/reject decision + actor onto the run so a resume that CRASHES
        # mid-flight (leaving the run at RESOLVING) can be re-driven by startup recovery
        # with the SAME decision — exactly-once is preserved by the member checkpoint
        # claim() + idempotency ledger; only the decision (run the tool, or not) is needed.
        run.status = RunStatus.RESOLVING
        run.metadata = {
            **(run.metadata or {}),
            "resume_decision": "approved" if approved else "rejected",
            "resume_actor": actor,
        }
        run.updated_at = _now()
        await self._storage.save_run(run)

        # HITL ORCHESTRATION pause (a team/workflow member paused): a graph/workflow run
        # carries a member-agent-id LIST (not a single ``agent_id``) and resumes via the
        # durable graph splice, not the single-agent resume.
        if is_orchestration:
            return await self._resume_orchestration(run, approved=approved, actor=actor)

        # Flip RESOLVING -> RUNNING up front so the inbox reflects "in progress". The
        # authoritative exactly-once gates are the run-level claim above + the runtime
        # ``claim()`` (crash-retry replays the per-tool idempotency ledger).
        run.status = RunStatus.RUNNING
        run.updated_at = _now()
        await self._storage.save_run(run)

        assert agent_def is not None  # noqa: S101 - narrowed: validated pre-claim above
        bg = asyncio.create_task(
            self._resume_in_background(
                run_id,
                checkpoint_id=checkpoint_id,
                approved=approved,
                actor=actor,
                agent_def=agent_def,
                workspace_id=run.workspace_id,
            )
        )
        self._tasks.add(bg)
        bg.add_done_callback(self._tasks.discard)
        return run

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
        semaphore = self._workspace_semaphore(workspace_id)
        async with semaphore:
            run = await self._storage.get_run(run_id)
            if run is None:  # pragma: no cover - defensive
                return
            # Re-sanitize the stored spec under the SAME operator status the run was
            # created with, so the approval-gated tool the run paused on is still present
            # when it executes (a tenant spec would have had its tools stripped at create,
            # so a paused run could only exist for an operator-provisioned/clean spec). If
            # the operator has since REVOKED the opt-in (env var unset between pause and
            # resume), the re-sanitize fail-closes — the resume becomes a clean FAILED,
            # never a crashed task and never a privileged-tool execution without the opt-in.
            operator_provisioned = bool(
                (run.metadata or {}).get("operator_provisioned", False)
            )
            # T2g: a plan-first run paused on its gated ``update_plan`` tool — the rebuilt
            # resume runtime MUST re-register that synthetic tool, else the now-approved
            # plan call has no handler to execute against.
            plan_mode = bool((run.metadata or {}).get("plan_mode", False))
            from himmy.config.spec_sanitizer import sanitize_tenant_spec

            try:
                spec = sanitize_tenant_spec(
                    agent_def.agent_spec(),
                    operator_provisioned=operator_provisioned,
                ).spec
                runtime = await self._resolve_runtime(
                    spec,
                    checkpoint_store=self._checkpoint_store,
                    plan_mode=plan_mode,
                    workspace_id=run.workspace_id,
                    # Carry the PERSISTED launch actor (stamped at create into
                    # run.metadata['actor']) into the rebuilt runtime, mirroring the drive
                    # path above. Without it the tool-capability gate would rebuild
                    # NON-enforcing and the now-approved, side-effecting tool would execute
                    # with the gate DISABLED — re-opening the confused-deputy hole on the
                    # highest-value path (the approved write).
                    actor=(run.metadata or {}).get("actor"),
                )
            except Exception as exc:  # noqa: BLE001 - spec rebuild failure is terminal
                run.status = RunStatus.FAILED
                run.error = f"resume runtime build failed: {exc}"
                run.updated_at = _now()
                await self._storage.save_run(run)
                return

            from himmy.core.errors import HimmyError

            try:
                loop = await asyncio.wait_for(
                    runtime.resume_agent_loop(
                        checkpoint_id, approved=approved, actor=actor
                    ),
                    timeout=self._run_timeout_seconds,
                )
            except HimmyError as exc:
                # The exactly-once loser: the checkpoint was already resolved by a
                # concurrent/earlier resume. This is a NO-OP — do NOT touch the run (the
                # winner owns its terminal state). Leaving it as-is means a double-approve
                # neither re-runs the tool nor flips a SUCCEEDED run to FAILED.
                logger.info(
                    "resume of run %s was a no-op (%s)", run_id, exc
                )
                return
            except TimeoutError:
                run.status = RunStatus.FAILED
                run.error = (
                    f"resume exceeded {self._run_timeout_seconds:.0f}s timeout"
                )
                run.updated_at = _now()
                await self._storage.save_run(run)
                return
            except asyncio.CancelledError:
                run.status = RunStatus.FAILED
                run.error = "resume cancelled"
                run.updated_at = _now()
                try:
                    await self._storage.save_run(run)
                except Exception:  # pragma: no cover - best-effort during cancel
                    pass
                raise
            except Exception as exc:  # noqa: BLE001 - terminal failure transition
                run.status = RunStatus.FAILED
                run.error = str(exc)
                run.updated_at = _now()
                await self._storage.save_run(run)
                return

            # Reconstruct the task so the shared finalizers compute the right trace id +
            # validate structured output against the originally-requested schema.
            from himmy.agents.base_agent.task import Task as _Task

            task = _Task(title=run.persona_name or "resume", prompt="", context={})
            if run.task_id:
                task.task_id = run.task_id
            await self._apply_loop_outcome(
                run, loop, task=task, llm_config=None, agent_def=agent_def
            )

    async def _resume_orchestration(
        self, run: RunRecord, *, approved: bool, actor: str
    ) -> RunRecord:
        """Approve/reject a HITL-paused TEAM/WORKFLOW run; resume the graph (WI-6).

        By the time this runs the caller (``resume_run``) has already won the atomic
        run-level ``claim_run_for_resume`` CAS, flipping AWAITING_APPROVAL -> RESOLVING;
        that run-level claim is now the PRIMARY exactly-once gate — a concurrent second
        approve loses the CAS and 409s before ever reaching here, so the graph advance
        happens exactly once. This method flips RESOLVING -> RUNNING and launches the
        durable graph resume on a tracked background task. The MEMBER checkpoint
        ``claim()`` inside ``resume_agent_loop`` remains a defence-in-depth backstop (a
        crash re-drive that re-enters here finds the member already resolved -> no-op).
        Returns the RUNNING record.
        """
        graph_resume_id = (run.metadata or {}).get("orchestration_checkpoint_id")
        if not graph_resume_id:  # pragma: no cover - an orchestration pause always has one
            raise RunNotApprovableError(run.run_id, status="no orchestration checkpoint")
        member_agent_ids = list((run.metadata or {}).get("member_agent_ids") or [])
        if not member_agent_ids or self._agent_resolver is None:
            raise RunNotApprovableError(run.run_id, status="no resolvable members")

        run.status = RunStatus.RUNNING
        run.updated_at = _now()
        await self._storage.save_run(run)

        bg = asyncio.create_task(
            self._resume_orchestration_in_background(
                run.run_id,
                approved=approved,
                actor=actor,
                workspace_id=run.workspace_id,
                graph_resume_id=graph_resume_id,
                member_agent_ids=member_agent_ids,
            )
        )
        self._tasks.add(bg)
        bg.add_done_callback(self._tasks.discard)
        return run

    def _resolve_graph_checkpoint_store(self) -> Any:
        """The durable graph checkpoint store to resume an orchestration from.

        Uses the surface-provided getter (file-backed in a server, so the SAME db the run
        paused into is reopened) and falls back to an in-memory store offline/in tests.
        """
        if self._graph_checkpoint_store_provider is not None:
            return self._graph_checkpoint_store_provider()
        from himmy.runtime.checkpoint import InMemoryGraphCheckpointStore

        return InMemoryGraphCheckpointStore()

    async def _resume_orchestration_in_background(
        self,
        run_id: str,
        *,
        approved: bool,
        actor: str,
        workspace_id: str,
        graph_resume_id: str,
        member_agent_ids: list[str],
    ) -> None:
        """Background worker: re-resolve members + drive the durable graph resume (WI-6).

        Holds the per-workspace concurrency semaphore for the duration, re-resolves and
        (via ``run_orchestration``) re-sanitizes the members under the run's operator
        status — so a revoked opt-in fails closed — rebuilds the graph member runtime with
        BOTH the member and graph checkpoint stores wired, and resumes the graph. The
        outcome is projected exactly like the execute path, INCLUDING pausing again at a
        later member. A claim loser (the member checkpoint already resolved) is a clean
        no-op — the run is left at the winner's terminal state.
        """
        from himmy.application.orchestration_runner import run_orchestration

        semaphore = self._workspace_semaphore(workspace_id)
        async with semaphore:
            run = await self._storage.get_run(run_id)
            if run is None:  # pragma: no cover - defensive
                return
            operator_provisioned = bool(
                (run.metadata or {}).get("operator_provisioned", False)
            )
            kind = (run.metadata or {}).get("orchestration_kind", "graph")
            resource_kind = (run.metadata or {}).get("orchestration", "workflow")

            members: list[AgentDefRecord] = []
            for agent_id in member_agent_ids:
                rec = await _maybe_await(
                    self._agent_resolver(agent_id, workspace_id=workspace_id)
                    if self._agent_resolver is not None
                    else None
                )
                if rec is None:
                    run.status = RunStatus.FAILED
                    run.error = f"resume failed: member agent {agent_id} removed"
                    run.updated_at = _now()
                    await self._storage.save_run(run)
                    return
                members.append(rec)

            try:
                outcome = await asyncio.wait_for(
                    run_orchestration(
                        kind=kind,
                        members=members,
                        prompt="",
                        resource_kind=resource_kind,
                        storage=self._storage,
                        shared_inference=getattr(
                            self._runtime, "inference_service", None
                        ),
                        operator_provisioned=operator_provisioned,
                        graph_checkpoint_store=self._resolve_graph_checkpoint_store(),
                        graph_resume_id=graph_resume_id,
                        checkpoint_store=self._checkpoint_store,
                        approve_member=approved,
                        actor=actor,
                    ),
                    timeout=self._run_timeout_seconds,
                )
            except TimeoutError:
                run.status = RunStatus.FAILED
                run.error = f"resume exceeded {self._run_timeout_seconds:.0f}s timeout"
                run.updated_at = _now()
                await self._storage.save_run(run)
                return
            except asyncio.CancelledError:
                run.status = RunStatus.FAILED
                run.error = "resume cancelled"
                run.updated_at = _now()
                try:
                    await self._storage.save_run(run)
                except Exception:  # pragma: no cover - best-effort during cancel
                    pass
                raise
            except Exception as exc:  # noqa: BLE001 - terminal failure transition
                # A claim-loser raises HimmyError('already resolved') from the member
                # resume — that is a clean NO-OP (the winner owns the terminal state),
                # NOT a failure. Distinguish it so a double-approve never flips a
                # SUCCEEDED run to FAILED.
                if _is_resume_claim_loss(exc):
                    logger.info(
                        "orchestration resume of run %s was a no-op (%s)", run_id, exc
                    )
                    return
                run.status = RunStatus.FAILED
                run.error = str(exc)
                run.updated_at = _now()
                await self._storage.save_run(run)
                return

            await self._apply_orchestration_outcome(run, outcome)

    async def _apply_orchestration_outcome(
        self, run: RunRecord, outcome: Any
    ) -> None:
        """Project a resumed orchestration outcome onto the run (shared with execute).

        Pauses AGAIN at a later member (AWAITING_APPROVAL with restamped ids), or lands
        terminal SUCCEEDED/FAILED — mirroring the initial execute-path projection.
        """
        run.thread_id = outcome.thread_id
        run.output_text = outcome.output_text or None
        run.metadata = {
            **(run.metadata or {}),
            "stopped_reason": outcome.stopped_reason,
            "route": outcome.route,
        }
        if outcome.graph_checkpoint_id:
            run.metadata["graph_checkpoint_id"] = outcome.graph_checkpoint_id
        if outcome.awaiting_approval:
            run.status = RunStatus.AWAITING_APPROVAL
            run.metadata["checkpoint_id"] = outcome.member_checkpoint_id
            run.metadata["orchestration_checkpoint_id"] = (
                outcome.orchestration_checkpoint_id
            )
            run.metadata["awaiting_member"] = outcome.awaiting_member
            run.metadata["hitl_kind"] = "orchestration"
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        run.status = RunStatus.FAILED if outcome.failed else RunStatus.SUCCEEDED
        if outcome.failed:
            run.error = outcome.error or "orchestration failed"
        run.updated_at = _now()
        await self._storage.save_run(run)

    # --------------------------------------------------------------------- reads
    async def get_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> RunRecord | None:
        """Return a run record by id, scoped to a workspace (AAEO-4).

        When ``workspace_id`` is supplied, a run belonging to another workspace is
        treated as not found (returns None).
        """
        run = await self._storage.get_run(run_id)
        if run is None:
            return None
        if workspace_id is not None and run.workspace_id != workspace_id:
            return None
        return run

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
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None or self._registry is None or run.thread_id is None:
            return None
        from himmy.entities.records import stable_id_for

        stable_id = stable_id_for(run.thread_id, namespace="chat_thread")
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
        runs = await self._storage.list_runs(
            workspace_id=workspace_id, subject_id=subject_id, status=status
        )
        if workspace_id is None:
            runs = [r for r in runs if r.workspace_id != LOCAL_WORKSPACE]
        return _paginate(
            runs,
            limit=limit,
            offset=offset,
            sort_key=lambda r: (r.created_at, r.run_id),
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
        """
        runs = await self._storage.list_runs(
            workspace_id=workspace_id, subject_id=subject_id, status=status
        )
        if workspace_id is None:
            runs = [r for r in runs if r.workspace_id != LOCAL_WORKSPACE]
        return len(runs)

    async def get_run_events(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> list[Any]:
        """Replay the canonical event stream for one run (by its trace id).

        Tenant-scoped (AAEO-4): a run outside ``workspace_id`` yields ``[]``.
        """
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None:
            return []
        if run.trace_id is not None:
            events = await self._storage.list_events(trace_id=run.trace_id)
            if events:
                return events
        if run.thread_id is not None:
            return await self._storage.list_events(thread_id=run.thread_id)
        return []

    async def get_run_thread(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> Any:
        """Replay the full conversation thread for one run (tenant-scoped, AAEO-4)."""
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None or run.thread_id is None:
            return None
        return await self._storage.load_thread(run.thread_id)

    async def get_run_thread_by_thread_id(self, thread_id: str) -> Any:
        """Load the authoritative ChatThread the runtime saved under ``thread_id`` (T2g).

        The thread router pins a run's ``thread_id`` to the conversation id, so this reads
        the latest in-flight thread state from the canonical run store directly (the run
        record may not yet carry the thread_id when a continuation has only just started).
        Returns None when nothing is stored under that id.
        """
        return await self._storage.load_thread(thread_id)

    async def await_run(self, run_id: str, timeout: float = 5.0) -> RunRecord | None:
        """Poll until the run reaches a terminal state (test/example helper)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = await self._storage.get_run(run_id)
            if run is not None and run.status in (
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
            ):
                return run
            await asyncio.sleep(0.01)
        return await self._storage.get_run(run_id)

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
        metadata: dict[str, Any] = {
            "orchestration": resource_kind,
            "orchestration_kind": kind,
            f"{resource_kind}_id": resource_id,
            "member_agent_ids": [m.agent_id for m in members],
            # Persisted so a HITL resume re-sanitizes the members under the SAME operator
            # status (a revoked opt-in between pause and resume then fails closed).
            "operator_provisioned": bool(operator_provisioned),
        }
        if actor:
            metadata["actor"] = actor
        if graph_resume_id:
            metadata["graph_resume_id"] = graph_resume_id
        # Q3: persist the launch PROMPT so the dispatcher can reconstruct the orchestration
        # run from a fresh process (the members are re-resolved from ``member_agent_ids`` and
        # the graph checkpoint store is rebuilt from the provider — both already crash-safe).
        if self._dispatch_enabled:
            metadata["orchestration_prompt"] = prompt
        run = RunRecord(
            workspace_id=workspace_id,
            subject_id=subject_id,
            task_id=None,
            persona_name=members[0].name if members else resource_kind,
            model_key="default",
            idempotency_key=idempotency_key,
            status=RunStatus.QUEUED,
            metadata=metadata,
        )
        if self._dispatch_enabled:
            # The orchestration run has no single-agent input_blob; its lane is the neutral
            # default (members may target mixed providers) + its retry ceiling is stamped so
            # the dispatcher claims + drives it via the orchestration reconstruction path.
            # It MUST be the literal LANE_DEFAULT, not NULL: the claim filter is
            # ``lane_key IN (...)`` and SQL NULL never matches an IN/ANY list, so a NULL lane
            # would sit QUEUED forever whenever the local probe gates out the local lane —
            # exactly the laptop-transient the health gate is meant to drain through.
            from himmy.services.storage.run_lane import LANE_DEFAULT

            run.lane_key = LANE_DEFAULT
            run.max_attempts = self._default_max_attempts
        # T3 HARD per-tenant outstanding-run cap (dispatch / multi-node path): an N-member
        # orchestration writes EXACTLY ONE run row (this single PARENT; members execute
        # in-process under the parent's per-workspace semaphore and never persist their own
        # RunRecords), so the parent IS the single admission unit and gating it is the whole
        # logical-unit gate. In dispatch mode the old code short-circuited to ``return stored``
        # WITHOUT ever admitting against the outstanding cap — the cap was simply NOT enforced
        # at create time (deferred to the dispatcher's runtime concurrency cap). Route it
        # through the SAME atomic op as create_run so a concurrent burst lands EXACTLY at the
        # cap. ``admitted=False`` => at/over cap and NOTHING written (no parent row, no member
        # state => zero partial/orphaned orchestration): raise the same 429. An idempotent
        # re-submit returns the prior row and does NOT consume a slot. A fresh admit leaves the
        # parent QUEUED for the dispatcher (recoverable on crash) — the SAME dispatch tail as
        # before, just now atomically capped. No soft ``_admit_workspace_run_durable`` is added
        # (the dispatch orchestration path never called it; the atomic op is the enforcer).
        if (
            self._dispatch_enabled
            and self._workspace_max_outstanding > 0
            and workspace_id != LOCAL_WORKSPACE
        ):
            stored, admitted = await self._storage.save_run_if_under_quota(
                run, cap=self._workspace_max_outstanding
            )
            if not admitted:
                raise WorkspaceRunQuotaExceeded(
                    workspace_id,
                    cap=self._workspace_max_outstanding,
                    outstanding=self._workspace_max_outstanding,
                )
            if stored.run_id != run.run_id:
                # idempotent re-submit: do not relaunch, do not consume a slot.
                return stored
            # Fresh admit: leave QUEUED for the dispatcher (recoverable on crash).
            return stored

        stored, created = await self._storage.save_run_if_absent_by_idempotency(run)
        if not created:
            return stored

        # Q3 dispatch mode: leave QUEUED for the dispatcher to claim (recoverable on crash).
        if self._dispatch_enabled:
            return stored

        try:
            self._admit_workspace_run(workspace_id)
        except WorkspaceRunQuotaExceeded:
            stored.status = RunStatus.FAILED
            stored.error = "rejected: workspace run-concurrency quota exceeded"
            stored.updated_at = _now()
            try:
                await self._storage.save_run(stored)
            except Exception:  # pragma: no cover - best-effort terminal mark
                logger.warning("failed to mark quota-rejected run %s", stored.run_id)
            raise

        bg = asyncio.create_task(
            self._execute_orchestration_run(
                stored.run_id,
                workspace_id=workspace_id,
                kind=kind,
                members=members,
                prompt=prompt,
                resource_kind=resource_kind,
                operator_provisioned=operator_provisioned,
                graph_checkpoint_store=graph_checkpoint_store,
                graph_resume_id=graph_resume_id,
            )
        )
        self._tasks.add(bg)
        bg.add_done_callback(self._tasks.discard)
        return stored

    async def _execute_orchestration_run(
        self,
        run_id: str,
        *,
        workspace_id: str,
        kind: str,
        members: list[AgentDefRecord],
        prompt: str,
        resource_kind: str,
        operator_provisioned: bool,
        graph_checkpoint_store: Any,
        graph_resume_id: str | None,
    ) -> None:
        """Background worker: build the team runtime + drive the orchestrator (T3b).

        Holds the per-workspace concurrency semaphore (T0.4) and releases the outstanding
        reservation in ``finally`` (mirroring :meth:`_execute_run`), so a team/workflow run
        is bounded and accounted exactly like a single-agent run.
        """
        from himmy.application.orchestration_runner import run_orchestration

        semaphore = self._workspace_semaphore(workspace_id)
        try:
            async with semaphore:
                run = await self._storage.get_run(run_id)
                if run is None:  # pragma: no cover - defensive
                    return
                run.status = RunStatus.RUNNING
                run.updated_at = _now()
                await self._storage.save_run(run)
                try:
                    outcome = await asyncio.wait_for(
                        run_orchestration(
                            kind=kind,
                            members=members,
                            prompt=prompt,
                            resource_kind=resource_kind,
                            storage=self._storage,
                            shared_inference=getattr(
                                self._runtime, "inference_service", None
                            ),
                            operator_provisioned=operator_provisioned,
                            graph_checkpoint_store=graph_checkpoint_store,
                            graph_resume_id=graph_resume_id,
                            # HITL: thread the surface-owned AgentCheckpoint store so a
                            # graph/workflow member calling an approval-gated tool pauses
                            # to a durable member checkpoint (None disables nested HITL).
                            checkpoint_store=self._checkpoint_store,
                        ),
                        timeout=self._run_timeout_seconds,
                    )
                except TimeoutError:
                    run.status = RunStatus.FAILED
                    run.error = (
                        f"run exceeded {self._run_timeout_seconds:.0f}s execution timeout"
                    )
                    run.updated_at = _now()
                    await self._storage.save_run(run)
                    return
                except asyncio.CancelledError:
                    run.status = RunStatus.FAILED
                    run.error = "run cancelled"
                    run.updated_at = _now()
                    try:
                        await self._storage.save_run(run)
                    except Exception:  # pragma: no cover - best-effort during cancel
                        pass
                    raise
                except Exception as exc:  # noqa: BLE001 - terminal failure transition
                    run.status = RunStatus.FAILED
                    run.error = str(exc)
                    run.updated_at = _now()
                    await self._storage.save_run(run)
                    return

                run.thread_id = outcome.thread_id
                run.output_text = outcome.output_text or None
                run.metadata = {
                    **(run.metadata or {}),
                    "stopped_reason": outcome.stopped_reason,
                    "route": outcome.route,
                }
                if outcome.graph_checkpoint_id:
                    run.metadata["graph_checkpoint_id"] = outcome.graph_checkpoint_id
                # HITL pause: a member called an approval-gated tool. Stamp the MEMBER
                # checkpoint id under the ``checkpoint_id`` key so the unchanged
                # pending-approvals path reads it, plus the orchestration (graph)
                # checkpoint id + awaiting member + a hitl_kind discriminator so
                # ``resume_run`` routes to the orchestration resume. STOP here — the
                # sweeper skips AWAITING_APPROVAL, so the paused run is never reaped.
                if outcome.awaiting_approval:
                    run.status = RunStatus.AWAITING_APPROVAL
                    run.metadata["checkpoint_id"] = outcome.member_checkpoint_id
                    run.metadata["orchestration_checkpoint_id"] = (
                        outcome.orchestration_checkpoint_id
                    )
                    run.metadata["awaiting_member"] = outcome.awaiting_member
                    run.metadata["hitl_kind"] = "orchestration"
                    run.updated_at = _now()
                    await self._storage.save_run(run)
                    return
                run.status = (
                    RunStatus.FAILED if outcome.failed else RunStatus.SUCCEEDED
                )
                if outcome.failed:
                    run.error = outcome.error or "orchestration failed"
                run.updated_at = _now()
                await self._storage.save_run(run)
        finally:
            self._release_workspace_run(workspace_id)


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
        from himmy.config.spec_sanitizer import sanitize_tenant_spec

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
        from himmy.config.spec_sanitizer import sanitize_tenant_spec

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
        from himmy.services.storage.conversations import ORIGIN_CLI

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
    a stdlib subset otherwise). Import is local to avoid a top-level cross-kernel
    dependency on the tools package.
    """
    try:
        from himmy.services.tools.validation import validate_against_schema
    except Exception:  # pragma: no cover - tools kernel always present
        return None
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

    async def summary(self, *, subject_id: str, workspace_id: str) -> dict[str, Any]:
        """Return one summary object: context stats + run/recommendation/eval counts.

        Context fields are scoped to ``workspace_id`` (AAEO-4) and the latest
        evaluation aggregate is folded in (AAEO-15) so the documented "scorecards
        on a dashboard" story is real.
        """
        all_fields = await self._storage.list_context_fields(subject_id)
        fields = [
            f
            for f in all_fields
            if (getattr(f, "metadata", {}) or {}).get("workspace_id")
            in (
                None,
                workspace_id,
            )
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

        evaluation = await self._evaluation_summary(workspace_id=workspace_id)

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
        self, *, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Fold the latest evaluation run's aggregate into the dashboard (AAEO-15).

        Scoped to ``workspace_id`` (AAEO-4) so the tile never leaks ANOTHER tenant's
        evaluation runs. The same inclusive rule the context-field tile uses applies:
        a run stamped with a *different* workspace is excluded, but a legacy/offline
        run with no stamped workspace stays visible (it belongs to no tenant). The
        strict-equality filter lives on the GET-by-id IDOR path, not this overview.
        """
        lister = getattr(self._storage, "list_evaluation_runs", None)
        if lister is None:
            return {"total": 0, "latest_aggregate": None}
        try:
            runs = list(await lister())
        except Exception:  # pragma: no cover - eval persistence optional
            return {"total": 0, "latest_aggregate": None}
        if workspace_id is not None:
            runs = [
                r
                for r in runs
                if getattr(r, "workspace_id", None) in (None, workspace_id)
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
