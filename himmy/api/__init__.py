"""API kernel: the FastAPI BFF over the application services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Importing ``himmy.api`` must NOT require fastapi (the ``[api]`` extra). The CLI's core,
# offline commands (``himmy run`` / ``init`` / ``chat``) reach submodules of ``himmy.api``
# (e.g. ``himmy.api.routines``, which is fastapi-free), so the package init stays
# dependency-light: ``create_app`` / ``set_rate_limiter`` (which need fastapi) and
# ``ApiContainer`` are resolved lazily on first access (PEP 562), never at import time.
if TYPE_CHECKING:  # type-checkers / IDEs see the names without a runtime import
    from himmy.api.app import create_app, set_rate_limiter
    from himmy.api.deps import ApiContainer

__all__ = ["create_app", "ApiContainer", "set_rate_limiter"]


def __getattr__(name: str) -> Any:
    if name in ("create_app", "set_rate_limiter"):
        from himmy.api import app

        return getattr(app, name)
    if name == "ApiContainer":
        from himmy.api.deps import ApiContainer

        return ApiContainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
