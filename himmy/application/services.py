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
import inspect
import logging
import time
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any, cast

from himmy.application.models import RecommendationEnvelope
from himmy.entities.lineage import DEFAULT_TRACE_DEPTH
from himmy.services.storage.models import (
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona
    from himmy.core.ids import utc_now_iso  # noqa: F401
    from himmy.entities.lineage import LineageGraph
    from himmy.entities.registry import EntityRegistry
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

#: Sentinel default for ``LLMConfig.model_key`` (AAEO-16): a config carrying this
#: value is treated as "caller did not pick a model" so task.context can win.
_DEFAULT_MODEL_KEY = "default"


def _now() -> str:
    """ISO timestamp helper (kept local to avoid a top-level core import cycle)."""
    from himmy.core.ids import utc_now_iso

    return utc_now_iso()


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
        entity_registry: EntityRegistry | None = None,
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
        entity_registry: EntityRegistry | None = None,
        recommendation_app: RecommendationAppService | None = None,
        run_timeout_seconds: float = DEFAULT_RUN_TIMEOUT_SECONDS,
    ) -> None:
        """Wire the runtime, store, optional registry, and recommendation service.

        ``run_timeout_seconds`` (AAEO-1) bounds each background run's wall clock;
        a run that exceeds it transitions to FAILED with a timeout error.
        """
        self._runtime = runtime
        self._storage = storage
        self._registry = entity_registry
        self._recommendations = recommendation_app or RecommendationAppService(
            storage=storage, entity_registry=entity_registry
        )
        self._run_timeout_seconds = run_timeout_seconds
        # Keep strong refs to background tasks so they are not GC'd mid-flight and
        # so they can be drained/cancelled on shutdown (AAEO-1).
        self._tasks: set[asyncio.Task[Any]] = set()

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
        """
        # Stamp the authenticated actor ("who launched this") into the durable
        # metadata JSONB (round-trips on both in-memory and Postgres) so every run
        # records its initiator — the operational half of "who did what" (WS1.3).
        run = RunRecord(
            workspace_id=workspace_id,
            subject_id=subject_id,
            task_id=task.task_id,
            persona_name=persona.name,
            model_key=_resolve_model_key(llm_config, task),
            idempotency_key=idempotency_key,
            status=RunStatus.QUEUED,
            metadata={"actor": actor} if actor else {},
        )
        stored, created = await self._storage.save_run_if_absent_by_idempotency(run)
        if not created:
            return stored

        bg = asyncio.create_task(
            self._execute_run(
                stored.run_id,
                persona=persona,
                task=task,
                llm_config=llm_config,
            )
        )
        self._tasks.add(bg)
        bg.add_done_callback(self._tasks.discard)
        return stored

    async def _execute_run(
        self,
        run_id: str,
        *,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
    ) -> None:
        """Background worker: RUNNING -> run_task -> SUCCEEDED/FAILED + extraction.

        Reads the terminal :class:`RunResult` status (invariant #4 / AAEO-3): a
        FAILED inference response is recorded as a FAILED run with ``run.error``
        populated and recommendation extraction skipped, instead of being marked
        SUCCEEDED with garbage output. The whole run is bounded by
        ``run_timeout_seconds`` (AAEO-1).
        """
        run = await self._storage.get_run(run_id)
        if run is None:  # pragma: no cover - defensive
            return
        run.status = RunStatus.RUNNING
        run.updated_at = _now()
        await self._storage.save_run(run)

        try:
            result = await asyncio.wait_for(
                self._runtime.run_task_detailed(persona, task, llm_config=llm_config),
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

        # AAEO-3: honour the FAILED inference path. ``RunResult.succeeded`` is the
        # typed status surface; a failed run records the error and skips extraction.
        if not result.succeeded:
            run.status = RunStatus.FAILED
            run.error = result.error or (result.error_code or "inference failed")
            run.output_text = result.output_text or None
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

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
        """Mark non-terminal runs older than ``ttl_seconds`` as FAILED (AAEO-1).

        Intended for startup: runs left in QUEUED/RUNNING when the process died
        cannot complete (their background task is gone), so they are transitioned
        to FAILED with a recovery error. ``ttl_seconds=0`` sweeps all non-terminal
        runs. Returns the swept run ids.
        """
        swept: list[str] = []
        runs = await self._storage.list_runs()
        now = time.time()
        for run in runs:
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
        """
        runs = await self._storage.list_runs(
            workspace_id=workspace_id, subject_id=subject_id, status=status
        )
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
        """Total run count for the filter (for pagination envelopes)."""
        runs = await self._storage.list_runs(
            workspace_id=workspace_id, subject_id=subject_id, status=status
        )
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

        evaluation = await self._evaluation_summary()

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

    async def _evaluation_summary(self) -> dict[str, Any]:
        """Fold the latest evaluation run's aggregate into the dashboard (AAEO-15)."""
        lister = getattr(self._storage, "list_evaluation_runs", None)
        if lister is None:
            return {"total": 0, "latest_aggregate": None}
        try:
            runs = await lister()
        except Exception:  # pragma: no cover - eval persistence optional
            return {"total": 0, "latest_aggregate": None}
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
    "DEFAULT_PAGE_LIMIT",
    "MAX_PAGE_LIMIT",
]
