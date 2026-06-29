"""API kernel: the /v1/dashboard router (operator overview summary)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from himmy.api.auth import (
    authorize_object,
    require_permission,
    require_workspace,
    scoped_read,
)
from himmy.api.models import DashboardSummary

_READ = [Depends(require_permission("dashboard", "read"))]

router = APIRouter(
    prefix="/v1/dashboard", tags=["dashboard"], dependencies=[Depends(scoped_read)]
)


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


@router.get("/summary", response_model=DashboardSummary, dependencies=_READ)
async def dashboard_summary(
    subject_id: str, workspace_id: str, request: Request
) -> DashboardSummary:
    """Return the operator overview: context + run/recommendation/eval counts.

    Object-level (BOLA, WS-bola): the summary aggregates a single data subject's
    context/run/recommendation counts, so a ``subject_scoped`` principal requesting ANOTHER
    data subject's aggregate within its own tenant gets a clean 404 (never an
    existence-confirming response) — :func:`authorize_object` is a no-op for offline /
    ``all_tenants`` / ``tenant_admin`` callers, so the zero-config path is byte-unchanged.
    """
    workspace_id = require_workspace(request, workspace_id)
    if not authorize_object(request, subject_id):
        raise HTTPException(status_code=404, detail="subject not found")
    summary = await _container(request).dashboard.summary(
        subject_id=subject_id, workspace_id=workspace_id
    )
    return DashboardSummary.model_validate(summary)


__all__ = ["router"]
