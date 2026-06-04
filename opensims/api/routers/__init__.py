"""API kernel: HTTP routers mapping transport shapes to application services."""

from __future__ import annotations

from opensims.api.routers import (
    context,
    dashboard,
    evaluation,
    recommendations,
    runs,
)

__all__ = ["context", "runs", "recommendations", "dashboard", "evaluation"]
