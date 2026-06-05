"""API kernel: the /v1/dashboard router (operator overview summary)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from himmy.api.auth import require_permission, require_workspace
from himmy.api.models import DashboardSummary

_READ = [Depends(require_permission("dashboard", "read"))]

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"])


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


@router.get("/summary", response_model=DashboardSummary, dependencies=_READ)
async def dashboard_summary(
    subject_id: str, workspace_id: str, request: Request
) -> DashboardSummary:
    """Return the operator overview: context + run/recommendation/eval counts."""
    workspace_id = require_workspace(request, workspace_id)
    summary = await _container(request).dashboard.summary(
        subject_id=subject_id, workspace_id=workspace_id
    )
    return DashboardSummary.model_validate(summary)


__all__ = ["router"]
