"""API kernel: the /v1/recommendations router (list + status transition).

List is paginated with a deterministic order + cap (AAEO-8); the PATCH transition
is tenant-scoped on ``workspace_id`` (AAEO-4) and declares a 404 shape (AAEO-9).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from opensims.api.models import (
    NOT_FOUND_RESPONSE,
    RecommendationListResponse,
)
from opensims.application.services import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from opensims.services.storage.models import RecommendationItem, RecommendationStatus

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])


class UpdateRecommendationRequest(BaseModel):
    """A status/notes transition payload for one recommendation."""

    status: RecommendationStatus
    notes: str | None = None


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


@router.get("", response_model=RecommendationListResponse)
async def list_recommendations(
    request: Request,
    workspace_id: str | None = None,
    subject_id: str | None = None,
    run_id: str | None = None,
    kind: str | None = None,
    status: RecommendationStatus | None = None,
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> RecommendationListResponse:
    """List recommendations (created_at desc), paginated into an envelope (AAEO-8)."""
    rec_app = _container(request).recommendation_app
    items = await rec_app.list(
        workspace_id=workspace_id,
        subject_id=subject_id,
        run_id=run_id,
        kind=kind,
        status=status,
        limit=limit,
        offset=offset,
    )
    total = await rec_app.count(
        workspace_id=workspace_id,
        subject_id=subject_id,
        run_id=run_id,
        kind=kind,
        status=status,
    )
    return RecommendationListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        next_offset=(offset + len(items)) if (offset + len(items)) < total else None,
    )


@router.patch(
    "/{recommendation_id}",
    response_model=RecommendationItem,
    responses=NOT_FOUND_RESPONSE,
)
async def update_recommendation(
    recommendation_id: str,
    body: UpdateRecommendationRequest,
    request: Request,
    workspace_id: str | None = None,
) -> RecommendationItem:
    """Transition a recommendation's status (404 when unknown/out-of-workspace)."""
    item = await _container(request).recommendation_app.update_status(
        recommendation_id,
        status=body.status,
        notes=body.notes,
        workspace_id=workspace_id,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="recommendation not found")
    return item


__all__ = ["router"]
