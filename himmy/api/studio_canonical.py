"""Bridge: project Studio runs into the ONE canonical RunRecord store (T2.2).

Before T2.2 a Studio run was an *authoritative* record living only in
``.himmy/studio.db`` — invisible to ``/v1`` and the CLI, which read the canonical
:class:`~himmy.services.storage.models.RunRecord` store (``.himmy/storage.db`` /
Postgres). That made three islands: a run created on one surface could not be read on
another.

This module collapses them. There is now exactly ONE authoritative run store (the
``RunAppService`` storage every surface reads) and ``.himmy/studio.db`` is demoted to a
run_id-keyed *presentation cache*: the rich UI payload (transcript, timeline, cognition
steps, per-model usage, the tools touched, the agent path/provider) rides in
``RunRecord.metadata['studio']`` and is serialized into the canonical ``runs.payload``
JSON column. So a run created by ANY surface — ``/v1``, the CLI, or Studio — is fully
renderable from the one store; the studio.db row is a rebuildable convenience, never the
source of truth.

Two reviewer must-fixes are baked into the write path here:

* **Main-loop write, never a nested loop.** ``_record_run`` runs in a worker thread
  (``asyncio.to_thread``) to do the synchronous ``studio.db`` write; the canonical write
  is an *async* upsert performed on the MAIN event loop AFTER the worker returns. We
  never nest ``asyncio.run`` inside ``to_thread`` (an asyncpg pool is bound to the loop
  it was created on, so a fresh loop in a worker thread would explode).
* **Last-write-wins that preserves the authoritative status.** :func:`save_canonical_run`
  upserts by ``run_id`` but, when a record already exists, keeps the *more advanced*
  status (so a late Studio cache write of an ``ok`` run cannot clobber an ``/v1`` row
  that the HITL machinery already moved to ``AWAITING_APPROVAL`` or a terminal state).

Single-user-local runs (Studio + CLI) are stamped with the reserved ``__local__``
workspace/subject so they land in the one store yet are EXCLUDED from cross-tenant
all-workspaces list views (see ``RunAppService.list_runs``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from himmy.core.ids import utc_now_iso
from himmy.services.storage.models import (
    LOCAL_SUBJECT,
    LOCAL_WORKSPACE,
    STUDIO_METADATA_KEY,
    RunRecord,
    RunStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.api.studio_runs import StudioRun, StudioRunSummary

logger = logging.getLogger("himmy.api.studio_canonical")

#: Process-wide provider of the ONE canonical run store (T2.2). The app factory sets
#: this to read ``app.state.container.storage`` so every background run path that lacks a
#: request (the routine scheduler, the mission registry) mirrors its run into the SAME
#: store the request-driven Studio/​v1/CLI surfaces read. When unset (a unit test calling
#: a streamer directly), :func:`resolve_canonical_storage` falls back to the durable
#: server-context SQLite store — the canonical store in the offline default.
_CANONICAL_STORAGE_PROVIDER: Any | None = None


def set_canonical_storage_provider(provider: Any | None) -> None:
    """Install (or clear) the process-wide canonical-run-store provider (T2.2).

    ``provider`` is a zero-arg callable returning the canonical storage (typically
    ``lambda: app.state.container.storage``), so a lifespan rebind to the durable store
    is reflected. Passing ``None`` clears it (shutdown / tests).
    """
    global _CANONICAL_STORAGE_PROVIDER
    _CANONICAL_STORAGE_PROVIDER = provider


def resolve_canonical_storage() -> Any | None:
    """Resolve the canonical run store for a request-less run path, or None (T2.2).

    Prefers the app-installed provider (the container's storage — correct for a
    Postgres deployment and consistent with the request-driven readers); otherwise
    resolves the durable server-context SQLite store (the canonical ``.himmy/storage.db``
    in the zero-config default). Returns None only when nothing resolves, in which case
    the run still records to the studio.db presentation cache.
    """
    provider = _CANONICAL_STORAGE_PROVIDER
    if provider is not None:
        try:
            return provider()
        except Exception:  # noqa: BLE001 - never let a bad provider break a run
            logger.debug("canonical storage provider failed", exc_info=True)
    try:
        from himmy.services.storage.factory import StoreFactory

        return StoreFactory.for_context(server=True)
    except Exception:  # noqa: BLE001 - resolution is best-effort
        logger.debug("server-context canonical store unavailable", exc_info=True)
        return None

#: Map a Studio run's loose status string to the canonical :class:`RunStatus`.
#: Studio uses ``ok``/``error``/``awaiting_approval``; the canonical enum is the single
#: vocabulary all surfaces filter on.
_STUDIO_STATUS_TO_RUN: dict[str, RunStatus] = {
    "ok": RunStatus.SUCCEEDED,
    "error": RunStatus.FAILED,
    "awaiting_approval": RunStatus.AWAITING_APPROVAL,
}

#: The reverse map (canonical → Studio string) so a ``/v1``-authored RunRecord with no
#: studio.db cache row still renders with a Studio-native status in the GUI.
_RUN_STATUS_TO_STUDIO: dict[RunStatus, str] = {
    RunStatus.QUEUED: "ok",
    RunStatus.RUNNING: "ok",
    RunStatus.AWAITING_APPROVAL: "awaiting_approval",
    # RESOLVING is the transient HITL-resume claim state: a non-terminal "in
    # progress" run (like RUNNING) — both for the normal resume window and for a
    # crash-stranded run awaiting recovery. It MUST live here: this map is keyed by
    # the exhaustive RunStatus enum and subscripted unguarded from the Studio list +
    # detail readers (which scan EVERY canonical run with no tenant scope), so a
    # missing key would KeyError-crash the Studio runs screen whenever any run is
    # RESOLVING.
    RunStatus.RESOLVING: "ok",
    RunStatus.SUCCEEDED: "ok",
    RunStatus.FAILED: "error",
    # PARKED is the leased-queue give-up state (Q1): terminal-but-redrivable. It renders as
    # "error" in the Studio GUI (the run did not complete) and MUST live here for the same
    # reason RESOLVING does — this map is subscripted unguarded over every canonical run, so a
    # missing key would KeyError-crash the Studio runs screen the moment any run is PARKED.
    RunStatus.PARKED: "error",
}

#: Status ordering for the last-write-wins guard. A higher rank is "more advanced": a
#: terminal/paused status must never be overwritten by a re-projected non-terminal one.
_STATUS_RANK: dict[RunStatus, int] = {
    RunStatus.QUEUED: 0,
    RunStatus.RUNNING: 1,
    # RESOLVING ranks with RUNNING: both are non-terminal "in progress" states. This
    # keeps the last-write-wins guard from letting a re-projected RESOLVING row roll
    # back an AWAITING_APPROVAL/terminal row (the .get default of 0 would rank it
    # below RUNNING, which is directionally wrong).
    RunStatus.RESOLVING: 1,
    RunStatus.AWAITING_APPROVAL: 2,
    RunStatus.SUCCEEDED: 3,
    RunStatus.FAILED: 3,
    # PARKED ranks terminal (like FAILED): the queue gave up, so a re-projected non-terminal
    # row must never roll it back.
    RunStatus.PARKED: 3,
}


def studio_run_to_record(
    run: StudioRun,
    *,
    workspace_id: str = LOCAL_WORKSPACE,
    subject_id: str = LOCAL_SUBJECT,
) -> RunRecord:
    """Project a :class:`StudioRun` into a canonical :class:`RunRecord` (T2.2).

    The heavy presentation fields (full transcript, timeline, cognition steps,
    per-model usage, tools, agent path/provider) are tucked into
    ``metadata['studio']`` so the canonical row is the single source of truth a
    studio.db cache row can always be rebuilt from. ``run_id``, ``status``,
    ``output_text``, ``thread_id`` and timing are promoted to first-class
    ``RunRecord`` columns so the ``/v1`` + CLI run readers see them natively.
    """
    status = _STUDIO_STATUS_TO_RUN.get(run.status, RunStatus.SUCCEEDED)
    studio_payload: dict[str, Any] = {
        "agent_name": run.agent_name,
        "agent_path": run.agent_path,
        "provider": run.provider,
        "model": run.model,
        "prompt": run.prompt,
        "output": run.output,
        "status": run.status,
        "duration_ms": run.duration_ms,
        "tools": list(run.tools),
        "messages": [m.model_dump() for m in run.messages],
        "timeline": [s.model_dump() for s in run.timeline],
        "steps": [s.model_dump() for s in run.steps],
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cost": run.cost,
        "usage_by_model": [u.model_dump() for u in run.usage_by_model],
        "checkpoint_id": run.checkpoint_id,
    }
    metadata: dict[str, Any] = {
        STUDIO_METADATA_KEY: studio_payload,
        "agent_name": run.agent_name,
        "source": "studio",
    }
    if run.checkpoint_id:
        metadata["checkpoint_id"] = run.checkpoint_id
    return RunRecord(
        run_id=run.id,
        workspace_id=workspace_id,
        subject_id=subject_id,
        thread_id=run.thread_id,
        persona_name=run.agent_name,
        model_key=run.model,
        status=status,
        output_text=run.output or None,
        error=(run.output or None) if run.status == "error" else None,
        metadata=metadata,
        created_at=run.created_at,
        updated_at=utc_now_iso(),
    )


def record_to_studio_run(rec: RunRecord) -> StudioRun:
    """Render a canonical :class:`RunRecord` as a :class:`StudioRun` for the GUI (T2.2).

    Used when the Studio runs screen reads a run authored by another surface
    (``/v1`` / CLI ``--persist``) that has no studio.db cache row: the rich fields
    are taken from ``metadata['studio']`` when present, else synthesized from the
    canonical columns so the GUI always renders *something* truthful (a one-line
    transcript + status) rather than 404ing on a run that demonstrably exists.
    """
    from himmy.api.studio_runs import (
        CognitionStep,
        ModelUsage,
        StudioRun,
        TimelineStep,
        TranscriptMessage,
    )

    studio = (rec.metadata or {}).get(STUDIO_METADATA_KEY) or {}
    if studio:
        output = str(studio.get("output") or rec.output_text or "")
        return StudioRun(
            id=rec.run_id,
            created_at=rec.created_at,
            agent_name=studio.get("agent_name") or rec.persona_name,
            agent_path=studio.get("agent_path"),
            provider=studio.get("provider"),
            model=studio.get("model") or rec.model_key,
            prompt=str(studio.get("prompt") or ""),
            output=output,
            output_preview=output,
            status=str(studio.get("status") or _RUN_STATUS_TO_STUDIO[rec.status]),
            duration_ms=studio.get("duration_ms"),
            thread_id=rec.thread_id,
            tools=list(studio.get("tools") or []),
            tool_count=len(studio.get("tools") or []),
            messages=[
                TranscriptMessage(**m) for m in (studio.get("messages") or [])
            ],
            timeline=[TimelineStep(**s) for s in (studio.get("timeline") or [])],
            steps=[CognitionStep(**s) for s in (studio.get("steps") or [])],
            input_tokens=int(studio.get("input_tokens") or 0),
            output_tokens=int(studio.get("output_tokens") or 0),
            cost=float(studio.get("cost") or 0.0),
            usage_by_model=[
                ModelUsage(**u) for u in (studio.get("usage_by_model") or [])
            ],
            checkpoint_id=studio.get("checkpoint_id")
            or (rec.metadata or {}).get("checkpoint_id"),
        )

    # No Studio projection (a run authored purely by /v1 or the CLI): synthesize a
    # minimal but honest view from the canonical columns.
    output = str(rec.output_text or "")
    prompt = ""
    messages: list[TranscriptMessage] = []
    if prompt or output:
        if prompt:
            messages.append(TranscriptMessage(role="user", content=prompt))
        messages.append(TranscriptMessage(role="assistant", content=output))
    return StudioRun(
        id=rec.run_id,
        created_at=rec.created_at,
        agent_name=rec.persona_name,
        agent_path=None,
        provider=None,
        model=rec.model_key,
        prompt=prompt,
        output=output,
        output_preview=output,
        status=_RUN_STATUS_TO_STUDIO[rec.status],
        duration_ms=None,
        thread_id=rec.thread_id,
        tools=[],
        tool_count=0,
        messages=messages,
        checkpoint_id=(rec.metadata or {}).get("checkpoint_id"),
    )


def _merge_status(existing: RunStatus, incoming: RunStatus) -> RunStatus:
    """Pick the status to keep on a re-upsert: the more advanced one wins.

    A re-projected Studio cache write (``ok``/``error``) must never roll back an
    ``/v1`` row the HITL machinery already advanced to ``AWAITING_APPROVAL`` or a
    terminal state. When the incoming status is at least as advanced (e.g. a real
    completion landing on a prior RUNNING row) it wins.
    """
    if _STATUS_RANK.get(incoming, 0) >= _STATUS_RANK.get(existing, 0):
        return incoming
    return existing


async def save_canonical_run(storage: Any, record: RunRecord) -> RunRecord:
    """Upsert a run into the canonical store, preserving the authoritative status.

    Last-write-wins-SAFE (reviewer must-fix): a plain ``save_run`` upsert would let a
    late, lower-status Studio cache projection overwrite a row another writer (the
    ``/v1`` HITL resume, the run lifecycle) has already advanced. So we read the
    existing row first and keep the *more advanced* status; everything else
    (output/metadata/thread) is refreshed from the incoming projection. Best-effort:
    a persistence error must never break the live stream, so the caller wraps this.
    """
    existing = await storage.get_run(record.run_id)
    if existing is not None:
        record.status = _merge_status(existing.status, record.status)
        # Preserve the original creation time + any authoritative metadata an earlier
        # writer stamped (e.g. the launching actor, a checkpoint id), letting the
        # Studio projection refresh only the presentation payload.
        record.created_at = existing.created_at
        merged_meta = dict(existing.metadata or {})
        merged_meta.update(record.metadata or {})
        record.metadata = merged_meta
    saved: RunRecord = await storage.save_run(record)
    return saved


def _summary_of(run: StudioRun) -> StudioRunSummary:
    """Project a full :class:`StudioRun` to its list :class:`StudioRunSummary`."""
    from himmy.api.studio_runs import StudioRunSummary, _preview

    return StudioRunSummary(
        id=run.id,
        created_at=run.created_at,
        agent_name=run.agent_name,
        agent_path=run.agent_path,
        provider=run.provider,
        model=run.model,
        prompt=_preview(run.prompt),
        output_preview=_preview(run.output),
        status=run.status,
        duration_ms=run.duration_ms,
        tool_count=len(run.tools),
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cost=run.cost,
        feedback_verdict=run.feedback_verdict,
        feedback_note=run.feedback_note,
        feedback_at=run.feedback_at,
    )


async def list_studio_runs_unified(
    storage: Any,
    *,
    limit: int,
    offset: int,
    accessible_workspaces: frozenset[str] | None = None,
    accessible_subject: str | None = None,
) -> tuple[list[StudioRunSummary], int]:
    """List runs for the Studio runs screen from the ONE canonical store (T2.2).

    Studio is single-user-local, so by default it shows EVERY canonical run (no tenant
    scope) — including runs created by ``/v1`` and the CLI ``--persist`` path — so "all
    interfaces, one store" is visible in the GUI. Each canonical run is rendered via
    :func:`record_to_studio_run`; when a matching studio.db presentation-cache row
    exists it supplies the richer fields + human feedback (which lives only on the
    cache). Returns ``(summaries, total)`` newest-first, windowed by offset/limit.

    ``accessible_workspaces`` is the tenant allow-list (from
    :func:`himmy.api.auth.context.studio_tenant_filter`): ``None`` (the default) means
    NO filtering — the byte-unchanged single-box behavior — while a set restricts the
    result to runs in those workspaces so a tenant-bound principal cannot read another
    tenant's runs through Studio. The filter is applied BEFORE the total/window so the
    count + pagination describe only the visible slice.

    ``accessible_subject`` is the object-level (BOLA) subject narrowing (from
    :func:`himmy.api.auth.context.studio_subject_filter`): ``None`` (the default) means NO
    subject filtering — byte-unchanged — while a value pins the result to runs attributed to
    that data subject (plus the reserved ``__local__`` single-user runs, which belong to no
    end subject), so a ``subject_scoped`` principal cannot read another subject's runs through
    Studio the way the ``/v1`` reader and missions already prevent. This closes the alternate-
    surface BOLA where a subject-scoped caller blocked on ``/v1/runs`` and missions read the
    same run via ``/api/studio/runs``.

    Falls back to the studio.db cache alone when ``storage`` is None (a bare test app
    with no container) so the screen never errors.
    """
    from himmy.api.studio_runs import get_run_store

    cache = get_run_store()
    if storage is None:
        items: list[StudioRunSummary] = cache.list(limit=limit, offset=offset)
        return items, cache.count()

    records: list[RunRecord] = await storage.list_runs()
    if accessible_workspaces is not None:
        records = [r for r in records if r.workspace_id in accessible_workspaces]
    if accessible_subject is not None:
        records = [
            r
            for r in records
            if r.subject_id in (accessible_subject, LOCAL_SUBJECT)
        ]
    # Newest first, run_id tiebreak (stable ordering across equal timestamps).
    records.sort(key=lambda r: (r.created_at, r.run_id), reverse=True)
    total = len(records)
    window = records[offset : offset + limit]
    summaries: list[StudioRunSummary] = []
    for rec in window:
        cached = cache.get(rec.run_id)
        if cached is not None:
            # The cache row is the richest projection (+ feedback); prefer it but keep
            # the canonical status authoritative so a paused/terminal run reads true.
            cached.status = _RUN_STATUS_TO_STUDIO[rec.status]
            summaries.append(_summary_of(cached))
        else:
            summaries.append(_summary_of(record_to_studio_run(rec)))
    return summaries, total


async def get_studio_run_unified(
    storage: Any,
    run_id: str,
    *,
    accessible_workspaces: frozenset[str] | None = None,
    accessible_subject: str | None = None,
) -> StudioRun | None:
    """Fetch one run in full for the Studio detail view from the canonical store.

    Prefers the studio.db presentation cache (richest fields + feedback) but reflects
    the canonical status, and falls back to projecting the canonical RunRecord when no
    cache row exists (a ``/v1``- or CLI-authored run). Returns None when the run is
    unknown in BOTH stores.

    ``accessible_workspaces`` is the tenant allow-list (from
    :func:`himmy.api.auth.context.studio_tenant_filter`): ``None`` (the default) means
    NO scoping — byte-unchanged single-box behavior — while a set makes a run whose
    canonical ``workspace_id`` is outside it read as **not found** (``None``), so a
    tenant-bound principal cannot fetch — or even confirm the existence of — another
    tenant's run by id. The studio.db presentation cache carries NO tenant stamp, so a
    cache-only row (canonical record absent/aged while the cache lingers) is
    unattributable to any tenant: it is served ONLY when scoping is off
    (``accessible_workspaces is None``) — either no canonical ``storage`` (the
    bare/single-box path) or a tenant-bound principal whose canonical lookup missed. For a
    tenant-bound principal (``accessible_workspaces`` is a set) a cache-only row is read as
    **not found** rather than leaked cross-tenant, since its owning tenant cannot be
    verified against the allow-list.

    ``accessible_subject`` is the object-level (BOLA) subject narrowing (from
    :func:`himmy.api.auth.context.studio_subject_filter`): ``None`` (the default) means NO
    subject scoping — byte-unchanged — while a value makes a run whose canonical
    ``subject_id`` is neither that subject nor the reserved ``__local__`` single-user subject
    read as **not found** (``None``), so a ``subject_scoped`` principal cannot fetch — or even
    confirm the existence of — another data subject's run by id through Studio. A cache-only
    row (no canonical record) is likewise refused to a subject-scoped reader, since the cache
    carries no subject stamp to attribute it to the caller.
    """
    from himmy.api.studio_runs import get_run_store

    cache = get_run_store()
    cached = cache.get(run_id)
    if storage is None:
        return cached
    rec = await storage.get_run(run_id)
    if rec is None:
        # No canonical record to attribute to a tenant/subject. An unscoped (single-box)
        # reader still gets the cache row; a tenant-bound OR subject-scoped reader must NOT —
        # the cache has no tenant/subject stamp, so serving it would bypass the allow-lists.
        if accessible_workspaces is not None or accessible_subject is not None:
            return None
        return cached
    if (
        accessible_workspaces is not None
        and rec.workspace_id not in accessible_workspaces
    ):
        return None  # cross-tenant → not found (never leak existence)
    if accessible_subject is not None and rec.subject_id not in (
        accessible_subject,
        LOCAL_SUBJECT,
    ):
        return None  # cross-subject → not found (never leak existence)
    if cached is not None:
        cached.status = _RUN_STATUS_TO_STUDIO[rec.status]
        return cached
    return record_to_studio_run(rec)


__all__ = [
    "get_studio_run_unified",
    "list_studio_runs_unified",
    "record_to_studio_run",
    "resolve_canonical_storage",
    "save_canonical_run",
    "set_canonical_storage_provider",
    "studio_run_to_record",
]
