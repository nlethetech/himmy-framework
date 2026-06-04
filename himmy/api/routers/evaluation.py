"""API kernel: the /v1/evaluation router (run a suite + read scorecards).

Makes the evaluation surface reachable over HTTP (AAEO-15): POST a suite +
actual outputs to score it, and GET runs by suite for the dashboard's
"scorecards on a dashboard" story.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from himmy.api.models import NOT_FOUND_RESPONSE
from himmy.services.evaluation.models import EvaluationRun, EvaluationSuite

router = APIRouter(prefix="/v1/evaluation", tags=["evaluation"])


class RunSuiteRequest(BaseModel):
    """The POST /v1/evaluation/runs body: a suite + per-case actual outputs."""

    suite: EvaluationSuite
    actual_outputs: dict[str, Any] = Field(default_factory=dict)


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


def _evaluation_service(request: Request) -> Any:
    """Return the wired evaluation service from the container."""
    service = getattr(_container(request), "evaluation", None)
    if service is None:  # pragma: no cover - container always wires it
        raise HTTPException(status_code=503, detail="evaluation not configured")
    return service


@router.post("/runs", response_model=EvaluationRun)
async def run_suite(body: RunSuiteRequest, request: Request) -> EvaluationRun:
    """Score a suite against actual outputs and persist + return the run."""
    return await _evaluation_service(request).run_suite(
        suite=body.suite, actual_outputs=body.actual_outputs
    )


@router.get("/runs", response_model=list[EvaluationRun])
async def list_evaluation_runs(
    request: Request, suite_id: str | None = None
) -> list[EvaluationRun]:
    """List evaluation runs, optionally filtered by suite id."""
    storage = _container(request).storage
    return await storage.list_evaluation_runs(suite_id=suite_id)


@router.get(
    "/runs/{run_id}", response_model=EvaluationRun, responses=NOT_FOUND_RESPONSE
)
async def get_evaluation_run(run_id: str, request: Request) -> EvaluationRun:
    """Read one evaluation run by id (404 when unknown)."""
    storage = _container(request).storage
    run = await storage.get_evaluation_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    return run


__all__ = ["router"]
