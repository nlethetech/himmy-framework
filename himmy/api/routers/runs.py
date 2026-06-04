"""API kernel: the /v1/runs router (async run create + read + replay).

Reads are tenant-scoped on ``workspace_id`` (AAEO-4), list is paginated with a
deterministic order + cap (AAEO-8), and routes declare typed responses + 404
shapes for accurate OpenAPI (AAEO-9).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from himmy.api.models import (
    NOT_FOUND_RESPONSE,
    RunListResponse,
)
from himmy.application.services import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from himmy.entities.lineage import DEFAULT_TRACE_DEPTH
from himmy.services.storage.models import RunRecord, RunStatus

router = APIRouter(prefix="/v1/runs", tags=["runs"])


class PersonaInput(BaseModel):
    """A simplified persona payload usable from curl."""

    name: str
    description: str = ""
    instructions: list[str] = []
    metadata: dict[str, Any] = {}


class TaskInput(BaseModel):
    """A simplified task payload usable from curl."""

    title: str
    prompt: str
    context: dict[str, Any] = {}


class CreateRunRequest(BaseModel):
    """The POST /v1/runs body: subject scope + persona + task + idempotency."""

    workspace_id: str
    subject_id: str
    persona: PersonaInput
    task: TaskInput
    idempotency_key: str | None = None


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


@router.post("", response_model=RunRecord)
async def create_run(body: CreateRunRequest, request: Request) -> RunRecord:
    """Create a run (idempotent) and execute it in the background; returns the record."""
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona

    persona = Persona(
        name=body.persona.name,
        description=body.persona.description,
        instructions=body.persona.instructions,
        metadata=body.persona.metadata,
    )
    task = Task(
        title=body.task.title,
        prompt=body.task.prompt,
        context=body.task.context,
    )
    return await _container(request).run_app.create_run(
        workspace_id=body.workspace_id,
        subject_id=body.subject_id,
        persona=persona,
        task=task,
        idempotency_key=body.idempotency_key,
    )


@router.get("", response_model=RunListResponse)
async def list_runs(
    request: Request,
    workspace_id: str | None = None,
    subject_id: str | None = None,
    status: RunStatus | None = None,
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> RunListResponse:
    """List runs (created_at desc), paginated. Returns a paged envelope (AAEO-8)."""
    run_app = _container(request).run_app
    items = await run_app.list_runs(
        workspace_id=workspace_id,
        subject_id=subject_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    total = await run_app.count_runs(
        workspace_id=workspace_id, subject_id=subject_id, status=status
    )
    return RunListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        next_offset=(offset + len(items)) if (offset + len(items)) < total else None,
    )


@router.get("/{run_id}", response_model=RunRecord, responses=NOT_FOUND_RESPONSE)
async def get_run(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> RunRecord:
    """Read one run record by id (404 when unknown/out-of-workspace, AAEO-4)."""
    run = await _container(request).run_app.get_run(run_id, workspace_id=workspace_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.get("/{run_id}/events")
async def get_run_events(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> list[Any]:
    """Replay the canonical event stream for one run (tenant-scoped, AAEO-4)."""
    return await _container(request).run_app.get_run_events(
        run_id, workspace_id=workspace_id
    )


@router.get("/{run_id}/thread", responses=NOT_FOUND_RESPONSE)
async def get_run_thread(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> Any:
    """Replay the full conversation thread for one run (404 when absent, AAEO-4)."""
    thread = await _container(request).run_app.get_run_thread(
        run_id, workspace_id=workspace_id
    )
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return thread


@router.get("/{run_id}/lineage", responses=NOT_FOUND_RESPONSE)
async def get_run_lineage(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
    max_depth: int = Query(DEFAULT_TRACE_DEPTH, ge=0, le=64),
    relations: str | None = Query(
        None,
        description="comma-separated relation allow-list (e.g. uses_persona,in_thread)",
    ),
    fmt: str = Query("json", alias="format", pattern="^(json|dot)$"),
) -> Any:
    """Trace a run's provenance subgraph: the persona, prompt, and evidence snapshot.

    Returns the typed lineage graph as JSON, or Graphviz DOT with ``?format=dot``.
    404 when the run is unknown / out-of-workspace or has no projected lineage.
    """
    rel = [r.strip() for r in relations.split(",") if r.strip()] if relations else None
    graph = await _container(request).run_app.get_run_lineage(
        run_id, workspace_id=workspace_id, max_depth=max_depth, relations=rel
    )
    if graph is None:
        raise HTTPException(status_code=404, detail="run lineage not found")
    if fmt == "dot":
        return PlainTextResponse(graph.to_dot())
    return graph


__all__ = ["router"]
