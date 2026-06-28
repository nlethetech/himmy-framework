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

from himmy.api.missions import (
    STEER_TEXT_MAX,
    MissionLimitError,
    MissionNotFoundError,
    MissionNotRunningError,
    MissionRegistry,
    get_registry,
)
from himmy.api.routers.studio_common import build_studio_router, studio_permission

router = build_studio_router("missions", tag="studio-missions")

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


# ---- endpoints -------------------------------------------------------------


@router.post("", response_model=dict[str, str], dependencies=[_missions_write])
async def start_mission(body: MissionStartRequest, request: Request) -> dict[str, str]:
    """Start a background agent run; returns immediately with the mission id."""
    registry = _registry(request)
    try:
        mission = registry.start_mission(
            agent_path=body.agent_path,
            prompt=body.prompt,
            provider=body.provider,
            model=body.model,
            history=[t.model_dump() for t in body.history],
            plan_mode=body.plan_mode,
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
    return MissionListResponse(
        items=[_view(m) for m in registry.list()],
        running=registry.running_count(),
        limit=registry.max_running,
    )


@router.get("/{mission_id}", response_model=MissionView)
async def get_mission(mission_id: str, request: Request) -> MissionView:
    try:
        return _view(_registry(request).get(mission_id))
    except MissionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.get("/{mission_id}/stream")
async def stream_mission(mission_id: str, request: Request) -> StreamingResponse:
    """SSE: replay every buffered frame, then follow live until the mission ends.

    Reconnect-safe — each connection starts from the oldest buffered frame, so a
    client that dropped mid-run simply reattaches and catches up.
    """
    registry = _registry(request)
    try:
        registry.get(mission_id)  # 404 before the stream starts, not inside it
    except MissionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
