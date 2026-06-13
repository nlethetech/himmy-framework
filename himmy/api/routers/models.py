"""API kernel: the /v1/models router — model catalog + quick compare (T3d).

Two REST surfaces over the shared :mod:`himmy.services.inference.compare` seam — the
SAME helpers Himmy Studio's ``/api/studio/models`` + ``/api/studio/compare`` and the
``himmy models`` + ``himmy compare`` CLI render, so the three interfaces never drift:

  * ``GET  /v1/models``          — the available local providers + their models, with any
    cached ``himmy bench`` reliability stats. GLOBAL (not tenant-scoped — model
    availability is an infra fact, not per-tenant state) + read-gated (``model:read``).
    Carries NO secret material, only provider availability + model names + numbers.
  * ``POST /v1/models/compare``  — one prompt fanned out across N models, side-by-side.
    Workspace-scoped + write-gated (``model:write``), and — because it spawns N concurrent
    model calls — admitted under the per-workspace T0.4 quota (a tenant at its cap gets a
    429), so a fan-out cannot pin unbounded provider work.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from himmy.api.auth import require_permission, require_workspace

router = APIRouter(prefix="/v1/models", tags=["models"])

_READ = [Depends(require_permission("model", "read"))]
_WRITE = [Depends(require_permission("model", "write"))]


# ---- catalog --------------------------------------------------------------


class CatalogModel(BaseModel):
    """One model row in the catalog, decorated with any cached bench reliability."""

    name: str
    accuracy: float | None = None
    latency: float | None = None


class CatalogProvider(BaseModel):
    """One provider's availability + its models (the GET /v1/models row)."""

    provider: str
    available: bool
    models: list[CatalogModel] = []


@router.get("", response_model=list[CatalogProvider], dependencies=_READ)
async def list_models() -> list[CatalogProvider]:
    """The available local providers + models (with cached bench stats). Secrets-free."""
    from himmy.services.inference.compare import build_model_catalog

    catalog = await build_model_catalog()
    return [CatalogProvider.model_validate(entry) for entry in catalog]


# ---- compare --------------------------------------------------------------


class CompareTarget(BaseModel):
    """One provider+model cell in a comparison."""

    provider: str = Field(..., max_length=40)
    model: str = Field(..., max_length=80)


class CompareRequest(BaseModel):
    """A single prompt fanned out across several models, scoped to a workspace."""

    workspace_id: str
    prompt: str = Field(..., min_length=1, max_length=8000)
    system: str | None = Field(None, max_length=4000)
    targets: list[CompareTarget] = Field(..., min_length=1, max_length=6)
    timeout_seconds: float = Field(120.0, ge=1, le=600)


class CompareResult(BaseModel):
    """The outcome of running the prompt against one target."""

    provider: str
    model: str
    ok: bool
    output: str = ""
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    latency_ms: float | None = None


class CompareResponse(BaseModel):
    """The side-by-side comparison: the scoped workspace + one cell per target."""

    workspace_id: str
    results: list[CompareResult]


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


@router.post("/compare", response_model=CompareResponse, dependencies=_WRITE)
async def compare_models(body: CompareRequest, request: Request) -> CompareResponse:
    """Run one prompt across several models concurrently; return outputs + usage.

    Workspace-scoped (the requested workspace is authorized against the principal) and
    admitted under the per-workspace T0.4 concurrency quota — a tenant already at its cap
    is rejected with a 429 (``run_quota_exceeded``) rather than pinning more provider work.
    Each target is an isolated, tool-less model call so the cells are directly comparable;
    a bad provider/model is an ``ok=False`` cell, never a 500.
    """
    from himmy.services.inference.compare import CompareTargetSpec, run_compare

    workspace_id = require_workspace(request, body.workspace_id)
    run_app = _container(request).run_app
    async with run_app.workspace_run_slot(workspace_id):
        cells = await run_compare(
            [
                CompareTargetSpec(provider=t.provider, model=t.model)
                for t in body.targets
            ],
            prompt=body.prompt,
            system=body.system,
            timeout_seconds=body.timeout_seconds,
        )
    return CompareResponse(
        workspace_id=workspace_id,
        results=[
            CompareResult(
                provider=cell.provider,
                model=cell.model,
                ok=cell.ok,
                output=cell.output,
                error=cell.error,
                input_tokens=cell.input_tokens,
                output_tokens=cell.output_tokens,
                cost=cell.cost,
                latency_ms=cell.latency_ms,
            )
            for cell in cells
        ],
    )


__all__ = ["router"]
