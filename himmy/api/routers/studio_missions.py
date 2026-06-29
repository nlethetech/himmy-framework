"""API kernel: Studio missions router — background runs, steering, plan-first.

Mounted under ``/api/studio/missions`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). Thin HTTP shell over
:mod:`himmy.api.missions` (the registry on ``app.state``):

* ``POST   /api/studio/missions``                 — start a background run
* ``GET    /api/studio/missions``                 — list (newest first)
* ``GET    /api/studio/missions/{id}``            — one mission's detail
* ``GET    /api/studio/missions/{id}/stream``     — SSE: replay buffer, follow live
* ``POST   /api/studio/missions/{id}/steer``      — queue between-turns guidance
* ``POST   /api/studio/missions/{id}/interrupt``  — checkpoint-aware stop

The registry is process-local: a restart loses RUNNING missions; finished ones
persist in the runs store via the same ``_record_run`` path foreground runs use.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from himmy.api.auth import scoped_read
from himmy.api.auth.context import (
    authorize_object,
    authorize_studio_object,
    get_principal,
    resolve_workspace,
)
from himmy.api.auth.service_principal import require_no_capability_amplification
from himmy.api.missions import (
    STEER_TEXT_MAX,
    Mission,
    MissionLimitError,
    MissionNotFoundError,
    MissionNotRunningError,
    MissionRegistry,
    get_registry,
)
from himmy.api.routers.studio_common import build_studio_router, studio_permission

router = build_studio_router("missions", tag="studio-missions")
#: This surface derives its read scope from the verified principal (the tenant/subject the
#: mission was stamped with at start time) and refuses out-of-scope missions — so it carries
#: the tenant-scope coverage gate's ``scoped_read`` marker and is no longer on that gate's
#: time-boxed ``_STUDIO_TENANT_PENDING`` allow-list. A pure no-op offline / ``all_tenants``.
router.dependencies.append(Depends(scoped_read))

#: Launching/steering/interrupting an autonomous mission is a privileged mutation gated
#: by ``studio.missions:write`` (admin-only by default), additively on top of the
#: router's ``studio.missions:read`` baseline so a read-only role can watch a mission's
#: progress but never start, redirect, or interrupt one.
_missions_write = Depends(studio_permission("studio.missions", "write"))


# ---- request/response models ---------------------------------------------


class MissionTurn(BaseModel):
    """One prior conversation turn replayed to preserve multi-turn context."""

    role: str = Field(..., max_length=20)  # "user" | "assistant"
    content: str = Field(..., max_length=100_000)


class MissionStartRequest(BaseModel):
    """POST /api/studio/missions body: which agent, the goal, optional overrides."""

    agent_path: str = Field(
        ..., max_length=500, description="agent spec path, relative to the root"
    )
    prompt: str = Field(..., min_length=1, max_length=100_000)
    provider: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    history: list[MissionTurn] = Field(default=[], max_length=200)
    plan_mode: bool = False


class MissionView(BaseModel):
    """The mission row/detail projection (never exposes queue/task internals)."""

    id: str
    agent: str
    agent_path: str
    prompt: str
    status: str  # running | paused | done | error
    created_at: str
    finished_at: str | None = None
    result_preview: str = ""
    error: str | None = None
    run_id: str | None = None
    checkpoint_id: str | None = None
    plan_mode: bool = False
    frame_count: int = 0


class MissionListResponse(BaseModel):
    items: list[MissionView]
    running: int
    limit: int


class SteerRequest(BaseModel):
    """POST /missions/{id}/steer body: between-turns guidance for the agent."""

    text: str = Field(..., min_length=1, max_length=STEER_TEXT_MAX)


class InterruptResponse(BaseModel):
    """What actually happened: a checkpoint pause or a cooperative cancel."""

    status: str
    mode: str  # "checkpoint" | "cancelled"
    detail: str


def _registry(request: Request) -> MissionRegistry:
    return get_registry(request.app)


def _view(mission: Any) -> MissionView:
    return MissionView(**mission.summary())


def _may_read(request: Request, mission: Mission) -> bool:
    """Whether this request's principal may see ``mission`` (tenant AND subject axes).

    The centralized data-scoping gate for the process-local mission registry: a mission is
    visible iff its OWNING ``workspace_id`` passes :func:`authorize_studio_object` (the
    tenant/IDOR axis) AND its ``subject_id`` passes :func:`authorize_object` (the subject/
    BOLA axis). Both are strict NO-OPs for an unrestricted (offline / ``all_tenants``)
    principal — and an unowned mission (``None`` workspace/subject, the single-box default)
    is allowed by both — so the zero-config path is byte-unchanged. A tenant-bound or
    ``subject_scoped`` caller never sees another tenant's / subject's mission.
    """
    return authorize_studio_object(
        request, mission.workspace_id
    ) and authorize_object(request, mission.subject_id)


def _owned_or_404(request: Request, mission_id: str) -> Mission:
    """Fetch a mission, folding a cross-tenant/cross-subject one to a uniform 404.

    Returning 404 (not 403) means a tenant-bound / subject-scoped principal cannot even
    probe the existence of another tenant's / subject's mission by id — mirroring the
    not-found-on-cross-scope convention of the ``/v1`` and Studio run readers.
    """
    try:
        mission = _registry(request).get(mission_id)
    except MissionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not _may_read(request, mission):
        raise HTTPException(status_code=404, detail=f"unknown mission {mission_id!r}")
    return mission


# ---- endpoints -------------------------------------------------------------


@router.post("", response_model=dict[str, str], dependencies=[_missions_write])
async def start_mission(body: MissionStartRequest, request: Request) -> dict[str, str]:
    """Start a background agent run; returns immediately with the mission id.

    The mission is STAMPED with the owning tenant + data subject derived from the verified
    principal (``resolve_workspace`` / ``get_principal``), never from the request body, so a
    later read can be scoped to it. Both are ``None`` for an unrestricted offline /
    ``all_tenants`` caller, leaving the mission unowned and visible to all — byte-unchanged.

    Capability-amplification gate (scope-r4): a mission's tools run under the operator
    SERVICE identity (``tool:*``), so a caller granted ``studio.missions:write`` but NOT the
    broad tool reach the mission would exercise is refused (403) — mirroring the routines
    arm-and-fire gate. NO-OP offline (unrestricted principal), byte-unchanged.
    """
    require_no_capability_amplification(
        request, resource="mission", action="start"
    )
    registry = _registry(request)
    principal = get_principal(request)
    owner_workspace = None if principal.all_tenants else resolve_workspace(request, None)
    owner_subject = None if principal.all_tenants else principal.subject
    try:
        mission = registry.start_mission(
            agent_path=body.agent_path,
            prompt=body.prompt,
            provider=body.provider,
            model=body.model,
            history=[t.model_dump() for t in body.history],
            plan_mode=body.plan_mode,
            workspace_id=owner_workspace,
            subject_id=owner_subject,
        )
    except MissionLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"mission_id": mission.id}


@router.get("", response_model=MissionListResponse)
async def list_missions(request: Request) -> MissionListResponse:
    """Every mission this process knows about, newest first.

    Process-local by design: finished missions are also in the runs store, but
    RUNNING ones do not survive a server restart.
    """
    registry = _registry(request)
    visible = [m for m in registry.list() if _may_read(request, m)]
    return MissionListResponse(
        items=[_view(m) for m in visible],
        running=sum(1 for m in visible if m.status == "running"),
        limit=registry.max_running,
    )


@router.get("/{mission_id}", response_model=MissionView)
async def get_mission(mission_id: str, request: Request) -> MissionView:
    return _view(_owned_or_404(request, mission_id))


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.get("/{mission_id}/stream")
async def stream_mission(mission_id: str, request: Request) -> StreamingResponse:
    """SSE: replay every buffered frame, then follow live until the mission ends.

    Reconnect-safe — each connection starts from the oldest buffered frame, so a
    client that dropped mid-run simply reattaches and catches up.
    """
    registry = _registry(request)
    # 404 before the stream starts (cross-tenant/subject folds to a uniform not-found).
    _owned_or_404(request, mission_id)

    async def _gen() -> AsyncIterator[str]:
        try:
            frames = await registry.stream(mission_id)
            async for frame in frames:
                yield _sse(frame)
        except Exception as exc:  # noqa: BLE001 - end the stream cleanly
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/{mission_id}/steer", response_model=MissionView, dependencies=[_missions_write]
)
async def steer_mission(
    mission_id: str, body: SteerRequest, request: Request
) -> MissionView:
    """Queue guidance the loop injects as a USER message before its next turn."""
    # Steering re-drives the mission's agent loop (which executes tools under the operator
    # service identity), so it carries the same capability-amplification gate as start
    # (scope-r4): a least-privilege ``studio.missions:write`` caller cannot redirect a
    # mission into broader tool authority than it holds. NO-OP offline.
    require_no_capability_amplification(
        request, resource="mission", action="steer"
    )
    _owned_or_404(request, mission_id)  # cross-scope mission is an opaque 404
    try:
        mission = await _registry(request).steer(mission_id, body.text)
    except MissionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MissionNotRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _view(mission)


@router.post(
    "/{mission_id}/interrupt",
    response_model=InterruptResponse,
    dependencies=[_missions_write],
)
async def interrupt_mission(mission_id: str, request: Request) -> InterruptResponse:
    """Stop a mission — honestly reporting checkpoint-pause vs cooperative cancel."""
    registry = _registry(request)
    _owned_or_404(request, mission_id)  # cross-scope mission is an opaque 404
    try:
        outcome = await registry.interrupt(mission_id)
    except MissionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MissionNotRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Re-read the (possibly just-flipped) status for an accurate response.
    status = registry.get(mission_id).status
    return InterruptResponse(
        status=status, mode=str(outcome["mode"]), detail=str(outcome["detail"])
    )


__all__ = ["router"]
