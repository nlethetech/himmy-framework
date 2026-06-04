"""API kernel: typed request/response/error models shared across routers.

These give the OpenAPI schema real shapes (AAEO-9): a stable error envelope, a
paginated-list envelope, and the dashboard summary. Routers import these so the
generated docs and clients are accurate, and so 404s/validation errors have a
consistent body.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """The consistent error envelope returned for handled failures (AAEO-9)."""

    detail: str
    code: str = "error"


class PageMeta(BaseModel):
    """Pagination metadata attached to every list response (AAEO-8)."""

    total: int
    limit: int
    offset: int
    next_offset: int | None = None


class RunListResponse(BaseModel):
    """A paginated list of run records."""

    items: list[Any] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


class RecommendationListResponse(BaseModel):
    """A paginated list of recommendation items."""

    items: list[Any] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


class ContextFieldListResponse(BaseModel):
    """A paginated list of context fields for a subject."""

    items: list[Any] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


class DashboardContext(BaseModel):
    """Context stats tile of the dashboard summary."""

    field_count: int = 0
    avg_confidence: float = 0.0
    avg_freshness_seconds: float | None = None


class DashboardCounts(BaseModel):
    """A total + by-status breakdown (runs or recommendations)."""

    total: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)


class DashboardEvaluation(BaseModel):
    """Latest evaluation aggregate folded into the dashboard (AAEO-15)."""

    total: int = 0
    latest_aggregate: float | None = None
    latest_suite_name: str | None = None
    latest_run_id: str | None = None


class DashboardSummary(BaseModel):
    """The operator overview returned by ``GET /v1/dashboard/summary``."""

    subject_id: str
    workspace_id: str
    context: DashboardContext
    runs: DashboardCounts
    recommendations: DashboardCounts
    evaluation: DashboardEvaluation = Field(default_factory=DashboardEvaluation)


class UpsertResult(BaseModel):
    """The result of a context-field bulk upsert."""

    upserted: int
    subject_id: str


#: The standard ``responses=`` map for routes that 404 (AAEO-9).
NOT_FOUND_RESPONSE: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Resource not found"},
}

#: The standard ``responses=`` map for routes behind the internal-key guard.
AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid internal API key"},
}


__all__ = [
    "ErrorResponse",
    "PageMeta",
    "RunListResponse",
    "RecommendationListResponse",
    "ContextFieldListResponse",
    "DashboardContext",
    "DashboardCounts",
    "DashboardEvaluation",
    "DashboardSummary",
    "UpsertResult",
    "NOT_FOUND_RESPONSE",
    "AUTH_RESPONSES",
]
