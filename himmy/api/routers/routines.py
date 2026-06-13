"""API kernel: the ``/v1/routines`` router — workspace-scoped scheduled agent runs (T3c).

A tenant schedules a saved prompt against one of its STORED agents (``/v1/agents``, T2e)
to run unattended on a simple schedule. The routine references the agent ONLY by
``agent_id`` — a filesystem ``agent_path`` is REJECTED (a server path would leak the
host filesystem to a tenant, exactly as ``/v1/knowledge`` refuses a tenant server path).

This router shares the ONE routines store (``.himmy/routines.db``) the Studio routines
screen and ``himmy routines`` drive, but every read/write is tenant-scoped on
``workspace_id`` via :func:`resolve_workspace` / :func:`require_workspace`, so one
tenant's routine is invisible (404) to another in an authenticated deployment.

Two reviewer must-fixes are load-bearing here:

* **Runs land in the CANONICAL store, in the TENANT's workspace.** ``POST /{id}/run-now``
  dispatches through :func:`himmy.api.routines.execute_routine`, which for an
  ``agent_id`` routine launches via ``RunAppService.create_run`` under the routine's
  ``workspace_id`` (T0.4 quota + HITL rails) — so a scheduled run shows in
  ``GET /v1/runs`` and ``himmy runs`` and Studio, not just a private cache.
* **Cross-process single-flight.** ``execute_routine`` holds a host-level ``flock`` keyed
  ``routine-<id>`` for the whole run, so a Studio scheduler tick and a CLI / ``/v1``
  run-now can never double-execute one routine; a concurrent attempt is a clean 409.

Unattended safety: the run executes with ``hitl=True`` when the stored agent bears tools,
so an approval-gated tool PAUSES the run at ``AWAITING_APPROVAL`` (the routine records
``awaiting_approval``) instead of firing — the scheduler never auto-approves.

Offline note: with no authenticator (the zero-config default) the caller is the
unrestricted operator, so RBAC + workspace scoping are inert — the standalone posture.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from himmy.api import routines as svc
from himmy.api.auth import (
    require_permission,
    require_workspace,
    resolve_workspace,
)
from himmy.api.models import NOT_FOUND_RESPONSE
from himmy.api.security_audit import audit_event

router = APIRouter(prefix="/v1/routines", tags=["routines"])

_READ = [Depends(require_permission("routine", "read"))]
_WRITE = [Depends(require_permission("routine", "write"))]


# ---- request/response models -------------------------------------------------


class CreateRoutineRequest(BaseModel):
    """The POST /v1/routines body: a workspace-scoped routine bound to a stored agent.

    ``agent_id`` references a stored ``/v1/agents`` record in the same workspace. There is
    deliberately NO ``agent_path`` field — a tenant may never name a server filesystem
    path (FS-leak surface). ``idempotency_key`` makes a re-POST return the prior routine.
    """

    workspace_id: str
    agent_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=100_000)
    schedule: svc.Schedule
    provider: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    deliver: svc.DeliverKind = "none"
    enabled: bool = True
    idempotency_key: str | None = None


class UpdateRoutineRequest(BaseModel):
    """The PATCH /v1/routines/{id} body: partial update (only provided fields change).

    ``agent_id`` may be re-pointed (re-validated to exist in the workspace); there is no
    ``agent_path`` field, by design.
    """

    workspace_id: str | None = None
    agent_id: str | None = Field(default=None, min_length=1, max_length=200)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    prompt: str | None = Field(default=None, min_length=1, max_length=100_000)
    schedule: svc.Schedule | None = None
    provider: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    deliver: svc.DeliverKind | None = None
    enabled: bool | None = None


class RoutineView(BaseModel):
    """The API view of a stored routine (no ``agent_path`` — that field is local-only)."""

    id: str
    workspace_id: str
    agent_id: str | None
    name: str
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

    @classmethod
    def of(cls, routine: svc.Routine) -> RoutineView:
        """Project a stored :class:`Routine` into the API view (drops ``agent_path``)."""
        return cls(
            id=routine.id,
            workspace_id=routine.workspace_id,
            agent_id=routine.agent_id,
            name=routine.name,
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


class RoutineListResponse(BaseModel):
    """The GET /v1/routines envelope: the workspace's routines."""

    items: list[RoutineView]
    total: int


def _agent_app(request: Request) -> Any:
    """Pull the wired :class:`AgentDefAppService` off the app container."""
    return request.app.state.container.agent_app


async def _require_stored_agent(
    request: Request, agent_id: str, workspace_id: str
) -> None:
    """404 unless ``agent_id`` is a stored agent in ``workspace_id`` (no FS path leak)."""
    record = await _agent_app(request).get_agent_def(
        agent_id, workspace_id=workspace_id
    )
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"stored agent {agent_id!r} not found in this workspace",
        )


# ---- CRUD ---------------------------------------------------------------------


@router.post("", response_model=RoutineView, dependencies=_WRITE, status_code=201)
async def create_routine(
    body: CreateRoutineRequest, request: Request
) -> RoutineView:
    """Create a workspace-scoped routine bound to a stored agent (idempotent).

    The ``agent_id`` must resolve to a stored agent in the same workspace (404 otherwise).
    A re-POST of an existing ``idempotency_key`` returns the prior routine unchanged.
    """
    workspace_id = require_workspace(request, body.workspace_id)
    await _require_stored_agent(request, body.agent_id, workspace_id)
    store = svc.get_routines_store()
    if body.idempotency_key is not None:
        for existing in store.list(workspace_id=workspace_id):
            if existing.idempotency_key == body.idempotency_key:
                return RoutineView.of(existing)
    routine = svc.Routine(
        workspace_id=workspace_id,
        agent_id=body.agent_id,
        name=body.name,
        prompt=body.prompt,
        schedule=body.schedule,
        provider=body.provider,
        model=body.model,
        deliver=body.deliver,
        enabled=body.enabled,
        idempotency_key=body.idempotency_key,
    )
    stored = store.upsert(routine)
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="routine",
        action="create",
        workspace_id=workspace_id,
        detail=f"routine {stored.id} created",
    )
    return RoutineView.of(stored)


@router.get("", response_model=RoutineListResponse, dependencies=_READ)
async def list_routines(
    request: Request, workspace_id: str | None = None
) -> RoutineListResponse:
    """List the routines in a workspace (tenant-scoped), newest first."""
    workspace_id = resolve_workspace(request, workspace_id)
    routines = svc.get_routines_store().list(workspace_id=workspace_id)
    items = [RoutineView.of(r) for r in routines]
    return RoutineListResponse(items=items, total=len(items))


@router.get(
    "/{routine_id}",
    response_model=RoutineView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_routine(
    routine_id: str, request: Request, workspace_id: str | None = None
) -> RoutineView:
    """Read one routine by id (404 when unknown / out-of-workspace)."""
    workspace_id = resolve_workspace(request, workspace_id)
    routine = svc.get_routines_store().get(routine_id, workspace_id=workspace_id)
    if routine is None:
        raise HTTPException(status_code=404, detail="routine not found")
    return RoutineView.of(routine)


@router.patch(
    "/{routine_id}",
    response_model=RoutineView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def update_routine(
    routine_id: str, body: UpdateRoutineRequest, request: Request
) -> RoutineView:
    """Partial update of a routine (tenant-scoped). 404 when absent.

    A re-pointed ``agent_id`` is re-validated to exist in the workspace; the schedule, when
    given, is re-validated as a whole. ``agent_path`` cannot be set (no field).
    """
    workspace_id = require_workspace(request, body.workspace_id or "")
    store = svc.get_routines_store()
    routine = store.get(routine_id, workspace_id=workspace_id)
    if routine is None:
        raise HTTPException(status_code=404, detail="routine not found")
    patch = body.model_dump(exclude_unset=True)
    patch.pop("workspace_id", None)
    # Non-nullable fields: an explicit null means "leave unchanged".
    for key in ("agent_id", "name", "prompt", "schedule", "deliver", "enabled"):
        if key in patch and patch[key] is None:
            patch.pop(key)
    if "schedule" in patch and body.schedule is not None:
        patch["schedule"] = body.schedule  # the validated model, not its dict dump
    if "agent_id" in patch:
        await _require_stored_agent(request, patch["agent_id"], workspace_id)
    updated = routine.model_copy(update=patch)
    stored = store.upsert(updated)
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="routine",
        action="update",
        workspace_id=workspace_id,
        detail=f"routine {routine_id} updated",
    )
    return RoutineView.of(stored)


@router.delete(
    "/{routine_id}", responses=NOT_FOUND_RESPONSE, dependencies=_WRITE
)
async def delete_routine(
    routine_id: str, request: Request, workspace_id: str | None = None
) -> dict[str, str]:
    """Delete a routine (tenant-scoped). 404 when absent / out-of-workspace."""
    workspace_id = resolve_workspace(request, workspace_id)
    if not svc.get_routines_store().delete(routine_id, workspace_id=workspace_id):
        raise HTTPException(status_code=404, detail="routine not found")
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="routine",
        action="delete",
        workspace_id=workspace_id,
        detail=f"routine {routine_id} deleted",
    )
    return {"routine_id": routine_id, "status": "deleted"}


# ---- manual trigger -----------------------------------------------------------


@router.post(
    "/{routine_id}/run-now",
    response_model=RoutineView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def run_now(
    routine_id: str, request: Request, workspace_id: str | None = None
) -> RoutineView:
    """Run a routine immediately through the unattended rails (tenant-scoped).

    Dispatches via ``RunAppService.create_run`` under the routine's workspace (canonical
    store + T0.4 quota + HITL pause). A routine already executing — in THIS process or in
    the Studio scheduler process (the cross-process flock) — is refused with 409, never
    run twice. The response carries the refreshed routine once the run settles/pauses.
    """
    workspace_id = resolve_workspace(request, workspace_id)
    store = svc.get_routines_store()
    if store.get(routine_id, workspace_id=workspace_id) is None:
        raise HTTPException(status_code=404, detail="routine not found")
    try:
        routine = await svc.get_scheduler().run_now(routine_id)
    except svc.RoutineBusyError as exc:
        raise HTTPException(
            status_code=409, detail="routine is already running"
        ) from exc
    if routine is None:  # deleted mid-run
        raise HTTPException(status_code=404, detail="routine not found")
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="routine",
        action="run_now",
        workspace_id=workspace_id,
        detail=f"routine {routine_id} run ({routine.last_status})",
    )
    return RoutineView.of(routine)


__all__ = ["router"]
