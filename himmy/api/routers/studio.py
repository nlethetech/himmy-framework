"""API kernel: the Himmy Studio router (the local web GUI's backend).

Studio is the no-code front door to himmy: chat with an agent, build/edit an
``agent.yaml``, and browse past runs/traces — all over the same FastAPI BFF. This
router is intentionally GUI-shaped (not the tenant-scoped ``/v1`` surface): it is
meant to be served on loopback by ``himmy studio`` for a single local user.

Endpoints:
  * ``GET  /api/studio/doctor``  — environment diagnostics as JSON.
  * ``GET  /api/studio/agents``  — discover agent.yaml files in the project.
  * ``POST /api/studio/run``     — run an agent for one turn (SSE token stream).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from himmy.api import studio_agents, studio_service
from himmy.api.studio_runs import (
    StudioRun,
    StudioRunListResponse,
    get_run_store,
)

router = APIRouter(prefix="/api/studio", tags=["studio"])


@router.get("/doctor")
async def doctor() -> dict:
    """Environment diagnostics: extras, providers, keys, and the next step.

    The JSON twin of ``himmy doctor`` — same
    :func:`himmy.runtime.diagnostics.collect_doctor_report` snapshot the CLI prints.
    """
    from himmy.runtime.diagnostics import collect_doctor_report

    return collect_doctor_report().to_dict()


@router.get("/benchmarks")
async def benchmarks() -> dict:
    """Cached per-model reliability scorecards (from `himmy bench` / the probe)."""
    from himmy.api import studio_bench

    return {"entries": studio_bench.list_cached()}


@router.post("/benchmarks/probe")
async def benchmarks_probe() -> dict:
    """Run a quick tool-focused reliability check against detected local models.

    Slow (real model calls) but bounded — a small suite against ≤3 local models, one
    trial each. Caches the result so Doctor shows it without re-running.
    """
    from himmy.api import studio_bench

    return await studio_bench.run_probe()


@router.get("/agents", response_model=list[studio_service.AgentSummary])
async def list_agents() -> list[studio_service.AgentSummary]:
    """Discover single-agent specs under the project root."""
    return studio_service.list_agents()


@router.get("/teams", response_model=list[studio_service.TeamSummary])
async def list_teams() -> list[studio_service.TeamSummary]:
    """Discover multi-agent team specs (manager + workers) under the project root."""
    return studio_service.list_teams()


class TurnInput(BaseModel):
    """One prior conversation turn replayed to preserve multi-turn context."""

    role: str  # "user" | "assistant"
    content: str


class RunRequest(BaseModel):
    """POST /api/studio/run body: which agent, the prompt, and optional overrides."""

    agent_path: str = Field(..., description="agent spec path, relative to the root")
    prompt: str
    provider: str | None = None
    model: str | None = None
    history: list[TurnInput] = []


def _sse(event: dict[str, Any]) -> str:
    """Encode one event dict as a Server-Sent-Events ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/run")
async def run(body: RunRequest) -> StreamingResponse:
    """Run an agent for one user turn, streaming GUI events over SSE.

    Loads the selected ``agent.yaml`` (with project defaults + skills applied),
    wires it exactly like ``himmy run``, and streams ``start``/``token``/``tool``/
    ``message``/``done`` frames. A load/build failure becomes a single ``error``
    frame so the client always gets a clean end to the stream.
    """
    # Resolve + load up front so a bad path is a clean 4xx, not a mid-stream error.
    try:
        spec = studio_service.load_studio_spec(
            body.agent_path, provider=body.provider, model=body.model
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _stream() -> AsyncIterator[str]:
        try:
            async for event in studio_service.stream_agent_run(
                spec,
                body.prompt,
                history=[t.model_dump() for t in body.history],
                provider=body.provider,
                model=body.model,
                agent_path=body.agent_path,
            ):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - surface as a terminal error frame
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class RunTeamRequest(BaseModel):
    """POST /api/studio/run-team body: which team and the request prompt."""

    team_path: str = Field(..., description="team spec path, relative to the root")
    prompt: str


@router.post("/run-team")
async def run_team(body: RunTeamRequest) -> StreamingResponse:
    """Run a multi-agent team, streaming the live routing/delegate/tool trail (SSE).

    Members run on their own providers (e.g. a Claude-CLI manager delegating to local
    Ollama workers). Frames: ``start`` → ``delegate``/``handoff``/``tool`` → ``message``
    → ``done`` (or a terminal ``error``).
    """
    try:
        spec = studio_service.load_team(body.team_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    teams = {t.path: t for t in studio_service.list_teams()}
    team_name = teams[body.team_path].name if body.team_path in teams else "team"

    async def _stream() -> AsyncIterator[str]:
        try:
            async for event in studio_service.stream_team_run(
                spec, body.prompt, team_name=team_name, team_path=body.team_path
            ):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - terminal error frame
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs", response_model=StudioRunListResponse)
async def list_runs(limit: int = 50, offset: int = 0) -> StudioRunListResponse:
    """List past Studio runs (newest first), paginated."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    store = get_run_store()
    items = store.list(limit=limit, offset=offset)
    total = store.count()
    return StudioRunListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        next_offset=(offset + len(items)) if offset + len(items) < total else None,
    )


@router.get("/runs/{run_id}", response_model=StudioRun)
async def get_run(run_id: str) -> StudioRun:
    """Fetch one run in full: transcript, tools, and the step-by-step timeline."""
    run = get_run_store().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


# ---- Agent authoring (the no-code builder) ------------------------------


@router.get("/tools", response_model=list[studio_agents.PackInfo])
async def tool_packs() -> list[studio_agents.PackInfo]:
    """The built-in tool packs an agent can switch on."""
    return studio_agents.list_tool_packs()


@router.get("/skills", response_model=list[studio_agents.SkillInfo])
async def skills() -> list[studio_agents.SkillInfo]:
    """Available skills (built-in + project-local)."""
    return studio_agents.list_skill_infos()


@router.get("/agent", response_model=studio_agents.AgentDetail)
async def get_agent(path: str) -> studio_agents.AgentDetail:
    """Load one agent's full editable spec (by project-relative path)."""
    try:
        return studio_agents.load_agent_detail(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/agents/validate", response_model=studio_agents.ValidationResult)
async def validate_agent(spec: dict) -> studio_agents.ValidationResult:
    """Validate a proposed spec without saving (live form feedback)."""
    errors = studio_agents.validate_spec(spec)
    return studio_agents.ValidationResult(ok=not errors, errors=errors)


@router.put("/agents", response_model=studio_service.AgentSummary)
async def save_agent(
    body: studio_agents.SaveAgentRequest,
) -> studio_service.AgentSummary:
    """Create or update an agent.yaml (validated; merges onto any existing file).

    Returns 409 when creating would overwrite an existing file without ``overwrite``.
    """
    try:
        return studio_agents.save_agent(body.path, body.spec, overwrite=body.overwrite)
    except studio_agents.AgentExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except studio_agents.SpecInvalid as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
