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

from fastapi import HTTPException
from pydantic import BaseModel, Field

from himmy.api import routines as svc
from himmy.api.routers.studio_common import build_studio_router

router = build_studio_router("routines", tag="studio-routines")


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
    """One routine, including last-run info, as the GUI sees it."""

    id: str
    name: str
    agent_path: str
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
        **routine.model_dump(),
        running=svc.get_scheduler().is_running(routine.id),
    )


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
    """All routines, newest first, with their last-run info."""
    return [_view(r) for r in svc.get_routines_store().list()]


@router.post("", response_model=RoutineView)
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
    return _view(svc.get_routines_store().upsert(routine))


@router.get("/{routine_id}", response_model=RoutineView)
async def get_routine(routine_id: str) -> RoutineView:
    routine = svc.get_routines_store().get(routine_id)
    if routine is None:
        raise HTTPException(status_code=404, detail="routine not found")
    return _view(routine)


@router.patch("/{routine_id}", response_model=RoutineView)
async def update_routine(routine_id: str, body: RoutineUpdate) -> RoutineView:
    """Partial update; the schedule (when given) is re-validated as a whole."""
    store = svc.get_routines_store()
    routine = store.get(routine_id)
    if routine is None:
        raise HTTPException(status_code=404, detail="routine not found")
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
    return _view(store.upsert(updated))


@router.delete("/{routine_id}")
async def delete_routine(routine_id: str) -> dict[str, bool]:
    if not svc.get_routines_store().delete(routine_id):
        raise HTTPException(status_code=404, detail="routine not found")
    return {"ok": True}


# ---- manual trigger -------------------------------------------------------------


@router.post("/{routine_id}/run-now", response_model=RoutineView)
async def run_now(routine_id: str) -> RoutineView:
    """Run a routine immediately through the same unattended rails.

    Same pipeline, same timeout, same approval pause — the response carries the
    refreshed routine once the run finishes (or pauses/fails). A routine that is
    already executing is refused with a 409, never run twice concurrently.
    """
    if svc.get_routines_store().get(routine_id) is None:
        raise HTTPException(status_code=404, detail="routine not found")
    try:
        routine = await svc.get_scheduler().run_now(routine_id)
    except svc.RoutineBusyError as exc:
        raise HTTPException(
            status_code=409, detail="routine is already running"
        ) from exc
    if routine is None:  # deleted mid-run
        raise HTTPException(status_code=404, detail="routine not found")
    return _view(routine)


# ---- scheduler lifecycle (merged into the app lifespan by include_router) -------


def _scheduler_enabled() -> bool:
    raw = os.environ.get("HIMMY_ROUTINES_SCHEDULER", "on").lower()
    return raw not in ("off", "0", "false", "no")


async def _start_scheduler() -> None:
    if _scheduler_enabled():
        svc.get_scheduler().start()


async def _stop_scheduler() -> None:
    await svc.get_scheduler().stop()


router.add_event_handler("startup", _start_scheduler)
router.add_event_handler("shutdown", _stop_scheduler)


__all__ = ["router"]
