"""API kernel: the /v1/evaluation router (run a suite + read scorecards).

Makes the evaluation surface reachable over HTTP (AAEO-15): POST a suite +
actual outputs to score it, and GET runs by suite for the dashboard's
"scorecards on a dashboard" story.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from himmy.api.auth import (
    get_principal,
    require_permission,
    resolve_workspace,
    scoped_read,
)
from himmy.api.models import NOT_FOUND_RESPONSE
from himmy.services.evaluation.models import EvaluationRun, EvaluationSuite

_READ = [Depends(require_permission("evaluation", "read"))]
_WRITE = [Depends(require_permission("evaluation", "write"))]

router = APIRouter(
    prefix="/v1/evaluation", tags=["evaluation"], dependencies=[Depends(scoped_read)]
)


class RunSuiteRequest(BaseModel):
    """The POST /v1/evaluation/runs body: a suite + per-case actual outputs.

    ``workspace_id`` scopes the run to its owning tenant (resolved against the
    principal, AAEO-4); it is optional and ``None`` on the offline/single-tenant
    path, preserving legacy behavior.
    """

    suite: EvaluationSuite
    actual_outputs: dict[str, Any] = Field(default_factory=dict)
    workspace_id: str | None = None


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


def _evaluation_service(request: Request) -> Any:
    """Return the wired evaluation service from the container."""
    service = getattr(_container(request), "evaluation", None)
    if service is None:  # pragma: no cover - container always wires it
        raise HTTPException(status_code=503, detail="evaluation not configured")
    return service


@router.post("/runs", response_model=EvaluationRun, dependencies=_WRITE)
async def run_suite(body: RunSuiteRequest, request: Request) -> EvaluationRun:
    """Score a suite against actual outputs and persist + return the run.

    The run is stamped with the resolved ``workspace_id`` (AAEO-4) and the creating
    actor (audit spine, T1.1) so later reads can be tenant-scoped.
    """
    workspace_id = resolve_workspace(request, body.workspace_id)
    return cast(
        EvaluationRun,
        await _evaluation_service(request).run_suite(
            suite=body.suite,
            actual_outputs=body.actual_outputs,
            workspace_id=workspace_id,
            actor=get_principal(request).actor_metadata(),
        ),
    )


@router.get("/runs", response_model=list[EvaluationRun], dependencies=_READ)
async def list_evaluation_runs(
    request: Request, suite_id: str | None = None, workspace_id: str | None = None
) -> list[EvaluationRun]:
    """List evaluation runs, optionally filtered by suite id (tenant-scoped, AAEO-4)."""
    workspace_id = resolve_workspace(request, workspace_id)
    storage = _container(request).storage
    return cast(
        list[EvaluationRun],
        await storage.list_evaluation_runs(
            suite_id=suite_id, workspace_id=workspace_id
        ),
    )


@router.get(
    "/runs/{run_id}",
    response_model=EvaluationRun,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_evaluation_run(
    run_id: str, request: Request, workspace_id: str | None = None
) -> EvaluationRun:
    """Read one evaluation run by id (404 when unknown/out-of-workspace, AAEO-4)."""
    workspace_id = resolve_workspace(request, workspace_id)
    storage = _container(request).storage
    run = await storage.get_evaluation_run(run_id, workspace_id=workspace_id)
    if run is None:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    return cast(EvaluationRun, run)


__all__ = ["router"]
