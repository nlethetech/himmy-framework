"""API kernel: Studio routines router — scheduled agent runs.

Mounted under ``/api/studio/routines`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). Behavior lives in
:mod:`himmy.api.routines` (store + scheduler); this file is the transport:

* full CRUD over routines (validated schedule grammar, bounded inputs),
* ``POST /{id}/run-now`` — a manual trigger through the same safety rails,
* router startup/shutdown events start/stop the scheduler loop. FastAPI's
  ``include_router`` merges these into the app lifespan, so the loop lives
  and dies with the server process. ``HIMMY_ROUTINES_SCHEDULER=off`` disables
  the background loop (run-now still works).
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field

from himmy.api import routines as svc
from himmy.api.auth import authorize_studio_object, scoped_read
from himmy.api.routers.studio_common import build_studio_router, studio_permission

router = build_studio_router("routines", tag="studio-routines")
# Every routine read is tenant-scoped (LIST pinned to ``__local__``; by-id gated by
# ``authorize_studio_object``), so stamp the router as a scoped reader for the whole-app
# tenant-scope coverage gate. A no-op marker; the real scoping is the handlers above.
router.dependencies.append(Depends(scoped_read))

#: Creating/editing/deleting/firing a routine schedules an autonomous agent run — a
#: privileged mutation gated by ``studio.routines:write`` (admin-only by default),
#: additively on top of the router's ``studio.routines:read`` baseline so a read-only
#: role can browse routines but never mutate one.
_routines_write = Depends(studio_permission("studio.routines", "write"))


def _wake_scheduler() -> None:
    """Wake the sleep-to-next-fire loop after a CRUD so a changed schedule re-plans now.

    Fail-open: a scheduler that is not running (or in another process / another node)
    simply doesn't wake — it will pick the change up on its next sleep cap, which is still
    correct. Only the in-process scheduler can be nudged this way.
    """
    sched = svc.get_scheduler()
    if sched.active:
        sched.notify_change()


# ---- request/response models -------------------------------------------------


class RoutineCreate(BaseModel):
    """POST body: a new routine. The schedule grammar is validated server-side."""

    name: str = Field(..., min_length=1, max_length=200)
    agent_path: str = Field(..., min_length=1, max_length=500)
    prompt: str = Field(..., min_length=1, max_length=100_000)
    schedule: svc.Schedule
    provider: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    deliver: svc.DeliverKind = "none"
    enabled: bool = True


class RoutineUpdate(BaseModel):
    """PATCH body: partial update — only the provided fields change."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    agent_path: str | None = Field(default=None, min_length=1, max_length=500)
    prompt: str | None = Field(default=None, min_length=1, max_length=100_000)
    schedule: svc.Schedule | None = None
    provider: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    deliver: svc.DeliverKind | None = None
    enabled: bool | None = None


class RoutineView(BaseModel):
    """One routine, including last-run info, as the single-user-local GUI sees it.

    Studio routines are filesystem (``agent_path``) bound; the ``/v1`` workspace/``agent_id``
    fields the shared store now carries are not part of this single-user-local view.
    """

    id: str
    name: str
    agent_path: str | None
    prompt: str
    schedule: svc.Schedule
    provider: str | None
    model: str | None
    deliver: svc.DeliverKind
    enabled: bool
    created_at: str
    updated_at: str
    last_run_at: str | None
    last_status: str | None
    last_preview: str
    last_error: str | None
    last_delivery: str | None
    running: bool


def _view(routine: svc.Routine) -> RoutineView:
    return RoutineView(
        id=routine.id,
        name=routine.name,
        agent_path=routine.agent_path,
        prompt=routine.prompt,
        schedule=routine.schedule,
        provider=routine.provider,
        model=routine.model,
        deliver=routine.deliver,
        enabled=routine.enabled,
        created_at=routine.created_at,
        updated_at=routine.updated_at,
        last_run_at=routine.last_run_at,
        last_status=routine.last_status,
        last_preview=routine.last_preview,
        last_error=routine.last_error,
        last_delivery=routine.last_delivery,
        running=svc.get_scheduler().is_running(routine.id),
    )


def _load_owned_routine(request: Request, routine_id: str) -> svc.Routine:
    """Resolve a routine by id and gate it on the caller's tenant entitlement (BOLA).

    The single by-id choke point for every Studio routine reader/mutator. The Studio LIST
    is scoped to the ``__local__`` workspace, but the by-id paths share ``.himmy/routines.db``
    with the ``/v1`` surface, where tenants create rows stamped with their OWN ``workspace_id``
    (POST /v1/routines). Without this gate a tenant-bound principal could read/mutate another
    tenant's routine row by id — a cross-tenant BOLA over routine definitions (prompt, agent
    binding, provider/model, last run preview/error). This intersects the routine's owning
    ``workspace_id`` against the caller's tenants via :func:`authorize_studio_object`, folding
    a foreign / unknown row into a uniform **404** (existence is never leaked), mirroring the
    ``/v1`` ``store.get(routine_id, workspace_id=...)`` sibling.

    A NO-OP for an unrestricted principal (offline / ``all_tenants`` / ANONYMOUS): every row
    is allowed and the zero-config single-box Studio path is byte-unchanged. A ``None``
    routine ``workspace_id`` (a legacy/uncategorized row predating tenant binding) is allowed.
    """
    routine = svc.get_routines_store().get(routine_id)
    if routine is None or not authorize_studio_object(request, routine.workspace_id):
        raise HTTPException(status_code=404, detail="routine not found")
    return routine


def _validate_agent_path(rel_path: str) -> None:
    """Reject a routine pointing at a missing/escaping agent spec up front."""
    from himmy.api.studio_service import resolve_spec_path

    try:
        resolve_spec_path(rel_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---- CRUD ---------------------------------------------------------------------


@router.get("", response_model=list[RoutineView])
async def list_routines() -> list[RoutineView]:
    """Local routines, newest first, with their last-run info.

    Scoped to the ``__local__`` workspace so the single-user Studio GUI shows only its
    own ``agent_path`` routines and never an ``agent_id``-bound ``/v1`` tenant row (whose
    ``agent_path`` is null) when both surfaces share ``.himmy/routines.db``.
    """
    rows = svc.get_routines_store().list(workspace_id=svc.LOCAL_WORKSPACE)
    return [_view(r) for r in rows]


@router.post("", response_model=RoutineView, dependencies=[_routines_write])
async def create_routine(body: RoutineCreate) -> RoutineView:
    """Create a routine. The agent path must resolve inside the project root."""
    _validate_agent_path(body.agent_path)
    routine = svc.Routine(
        name=body.name,
        agent_path=body.agent_path,
        prompt=body.prompt,
        schedule=body.schedule,
        provider=body.provider,
        model=body.model,
        deliver=body.deliver,
        enabled=body.enabled,
    )
    stored = svc.get_routines_store().upsert(routine)
    _wake_scheduler()
    return _view(stored)


@router.get("/{routine_id}", response_model=RoutineView)
async def get_routine(routine_id: str, request: Request) -> RoutineView:
    """Read one routine by id (404 when unknown / out-of-tenant, BOLA-gated)."""
    routine = _load_owned_routine(request, routine_id)
    return _view(routine)


@router.patch(
    "/{routine_id}", response_model=RoutineView, dependencies=[_routines_write]
)
async def update_routine(
    routine_id: str, body: RoutineUpdate, request: Request
) -> RoutineView:
    """Partial update; the schedule (when given) is re-validated as a whole.

    BOLA-gated: a tenant-bound principal may only mutate a routine in a workspace it is
    entitled to (:func:`_load_owned_routine`); a foreign row is a 404, never mutated.
    """
    store = svc.get_routines_store()
    routine = _load_owned_routine(request, routine_id)
    patch = body.model_dump(exclude_unset=True)
    # Non-nullable fields: an explicit null means "leave unchanged", never None.
    for key in ("name", "agent_path", "prompt", "schedule", "deliver", "enabled"):
        if key in patch and patch[key] is None:
            patch.pop(key)
    if "schedule" in patch and body.schedule is not None:
        patch["schedule"] = body.schedule  # the validated model, not its dict dump
    if "agent_path" in patch:
        _validate_agent_path(patch["agent_path"])
    updated = routine.model_copy(update=patch)
    stored = store.upsert(updated)
    _wake_scheduler()
    return _view(stored)


@router.delete("/{routine_id}", dependencies=[_routines_write])
async def delete_routine(routine_id: str, request: Request) -> dict[str, bool]:
    """Delete a routine (404 when unknown / out-of-tenant, BOLA-gated)."""
    _load_owned_routine(request, routine_id)
    if not svc.get_routines_store().delete(routine_id):
        raise HTTPException(status_code=404, detail="routine not found")
    _wake_scheduler()
    return {"ok": True}


# ---- manual trigger -------------------------------------------------------------


@router.post(
    "/{routine_id}/run-now", response_model=RoutineView, dependencies=[_routines_write]
)
async def run_now(routine_id: str, request: Request) -> RoutineView:
    """Run a routine immediately through the same unattended rails.

    Same pipeline, same timeout, same approval pause — the response carries the
    refreshed routine once the run finishes (or pauses/fails). A routine that is
    already executing is refused with a 409, never run twice concurrently.

    BOLA-gated: a tenant-bound principal may only fire a routine in a workspace it is
    entitled to (:func:`_load_owned_routine`); a foreign row is a 404, never executed.
    """
    _load_owned_routine(request, routine_id)
    try:
        routine = await svc.get_scheduler().run_now(routine_id)
    except svc.RoutineBusyError as exc:
        raise HTTPException(
            status_code=409, detail="routine is already running"
        ) from exc
    if routine is None:  # deleted mid-run
        raise HTTPException(status_code=404, detail="routine not found")
    # last_run_at advanced -> re-plan the next boundary off the new anchor.
    _wake_scheduler()
    return _view(routine)


# ---- scheduler lifecycle (merged into the app lifespan by include_router) -------


def _scheduler_enabled() -> bool:
    from himmy.config.flags import env_falsy

    # Default-ON off-switch: enabled unless an explicit falsy token disables it. Shares
    # the canonical falsy vocabulary (env_falsy) with HIMMY_STUDIO_AUTH so a typo can
    # never silently disable it.
    return not env_falsy("HIMMY_ROUTINES_SCHEDULER")


#: Module-scoped leadership lease + failover watchdog for the FastAPI-hosted scheduler.
#: In a multi-replica deployment EVERY replica runs this startup event, so without leader
#: election every replica would tick. The lease (Postgres) / topology guard (SQLite) ensures
#: at most one replica actually ticks; the watchdog promotes a follower on failover.
_LEADERSHIP: object | None = None
_WATCHDOG: object | None = None


async def _start_scheduler() -> None:
    """Bid for scheduler leadership, then start the tick loop iff this replica wins.

    Fail-open: if leadership wiring is unavailable (e.g. no container in a bare app), fall back
    to the legacy unconditional start so a single-process server keeps working — the per-fire
    dedup-CAS stays the single-fire backstop either way.
    """
    global _LEADERSHIP, _WATCHDOG
    if not _scheduler_enabled():
        return
    container = svc.resolve_routine_container()
    if container is None:
        # No container wired (a bare/in-memory app): nothing to coordinate — start directly.
        svc.get_scheduler().start()
        return
    try:
        import asyncio

        from himmy.api.scheduler_leader import acquire_scheduler_leadership
        from himmy.config.flags import env_truthy

        require_ack = env_truthy("HIMMY_SCHEDULER_REQUIRE_ACK")
        leadership = await acquire_scheduler_leadership(
            container, require_single_scheduler_ack=require_ack
        )
        _LEADERSHIP = leadership
        if leadership.is_leader:
            sched = svc.get_scheduler()
            try:
                await sched.catch_up_on_launch()
            except Exception:  # noqa: BLE001 - catch-up must not block startup
                pass
            sched.start()
        if leadership.mode in ("postgres-lease", "follower"):
            _WATCHDOG = asyncio.create_task(
                _failover_loop(leadership), name="himmy-studio-scheduler-failover"
            )
    except Exception:  # noqa: BLE001 - fail open to the legacy direct start
        svc.get_scheduler().start()


async def _failover_loop(leadership: object) -> None:
    """Promote this replica's scheduler on the leader's failover (Studio-hosted variant)."""
    import asyncio

    raw = os.environ.get("HIMMY_SCHEDULER_FAILOVER_INTERVAL", "").strip()
    try:
        interval = float(raw) if raw else 5.0
    except ValueError:
        interval = 5.0
    if interval <= 0:
        interval = 5.0
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        try:
            was = leadership.is_leader  # type: ignore[attr-defined]
            now = await leadership.refresh()  # type: ignore[attr-defined]
            sched = svc.get_scheduler()
            if now and not was:
                try:
                    await sched.catch_up_on_launch()
                except Exception:  # noqa: BLE001
                    pass
                sched.start()
            elif not now and was and sched.active:
                await sched.stop()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - watchdog must not crash the server
            pass


async def _stop_scheduler() -> None:
    global _LEADERSHIP, _WATCHDOG
    if _WATCHDOG is not None:
        try:
            _WATCHDOG.cancel()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        _WATCHDOG = None
    await svc.get_scheduler().stop()
    if _LEADERSHIP is not None:
        try:
            await _LEADERSHIP.release()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        _LEADERSHIP = None


router.add_event_handler("startup", _start_scheduler)
router.add_event_handler("shutdown", _stop_scheduler)


__all__ = ["router"]
