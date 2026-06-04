"""API kernel: the FastAPI BFF over the application services."""

from __future__ import annotations

from opensims.api.app import create_app, set_rate_limiter
from opensims.api.deps import ApiContainer

__all__ = ["create_app", "ApiContainer", "set_rate_limiter"]
