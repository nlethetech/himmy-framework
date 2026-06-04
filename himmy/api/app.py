"""API kernel: the FastAPI app factory (BFF for the application services).

``create_app`` wires an :class:`ApiContainer` onto ``app.state``, mounts the
routers, configures observability, registers a global exception handler, and —
when ``HIMMY_INTERNAL_API_KEY`` is set — guards every route behind a
constant-time trusted-boundary header (surfaced as an OpenAPI security scheme).
A FastAPI lifespan sweeps stuck runs on startup and drains in-flight background
runs on shutdown (AAEO-1). It is a thin transport layer: behavior lives below it.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from himmy.api.deps import ApiContainer
from himmy.api.models import ErrorResponse
from himmy.api.routers import (
    context,
    dashboard,
    evaluation,
    recommendations,
    runs,
)
from himmy.core.errors import HimmyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger("himmy.api")

#: Default header name carrying the internal trusted-boundary key.
_DEFAULT_HEADER = "x-himmy-internal-key"


def _expected_keys() -> set[str]:
    """Parse the configured internal key(s) (comma-separated supports rotation)."""
    raw = os.environ.get("HIMMY_INTERNAL_API_KEY", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _header_name() -> str:
    """Return the configured internal-key header name."""
    return os.environ.get("HIMMY_INTERNAL_HEADER", _DEFAULT_HEADER)


def _check_key(provided: str | None, expected: set[str]) -> bool:
    """Constant-time membership check of ``provided`` against ``expected`` (AAEO-13).

    Uses :func:`hmac.compare_digest` against every configured key so the
    comparison is timing-safe and key rotation (multiple keys) is supported.
    """
    if not provided:
        return False
    return any(hmac.compare_digest(provided, key) for key in expected)


# A no-op rate-limit hook point (AAEO-13). Replace via ``set_rate_limiter`` to
# enforce per-caller limits upstream of the application services. The default
# admits every request so the offline/default path is unchanged.
RateLimiter = Callable[[Request], None]
_RATE_LIMITER: RateLimiter | None = None


def set_rate_limiter(limiter: RateLimiter | None) -> None:
    """Install (or clear) the process-wide rate-limit hook (AAEO-13).

    The hook is called with the incoming :class:`Request` before routing; it
    should raise :class:`fastapi.HTTPException` (429) to reject. Off by default.
    """
    global _RATE_LIMITER
    _RATE_LIMITER = limiter


def _internal_key_dependency():
    """Build an optional auth + rate-limit dependency from env, or None when disabled.

    When ``HIMMY_INTERNAL_API_KEY`` is set, every request must carry a matching
    value in the header named by ``HIMMY_INTERNAL_HEADER`` (constant-time
    comparison, multi-key rotation supported). The rate-limit hook, if installed,
    runs regardless of the auth switch.
    """
    expected = _expected_keys()
    header_name = _header_name()
    api_key_scheme = APIKeyHeader(name=header_name, auto_error=False)

    if not expected:
        # No auth, but still expose the rate-limit hook point.
        async def _rate_only(request: Request) -> None:
            """Apply the rate-limit hook when installed; otherwise admit."""
            if _RATE_LIMITER is not None:
                _RATE_LIMITER(request)

        return _rate_only

    async def _guard(
        request: Request,
        provided: str | None = Depends(api_key_scheme),
    ) -> None:
        """Reject requests lacking a valid internal key (and apply rate limits)."""
        if _RATE_LIMITER is not None:
            _RATE_LIMITER(request)
        if not _check_key(provided, expected):
            raise HTTPException(
                status_code=401,
                detail="invalid internal api key",
                headers={"WWW-Authenticate": header_name},
            )

    return _guard


def _build_lifespan(container: ApiContainer):
    """Build a FastAPI lifespan that sweeps stuck runs + drains on shutdown (AAEO-1)."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # Startup: sweep runs left non-terminal by a previous process so they
        # reach a terminal state instead of hanging in QUEUED/RUNNING forever.
        run_app = getattr(container, "run_app", None)
        if run_app is not None:
            try:
                swept = await run_app.sweep_stuck_runs()
                if swept:
                    logger.info(
                        "swept %d stuck run(s) to FAILED on startup", len(swept)
                    )
            except Exception:  # pragma: no cover - startup sweep is best-effort
                logger.warning("startup run sweep failed", exc_info=True)
        try:
            yield
        finally:
            # Shutdown: cancel + await in-flight background runs (drain), then
            # release container resources (e.g. a Postgres pool).
            if run_app is not None:
                try:
                    await run_app.drain()
                except Exception:  # pragma: no cover - shutdown best-effort
                    logger.warning("run drain failed on shutdown", exc_info=True)
            try:
                await container.aclose()
            except Exception:  # pragma: no cover - shutdown best-effort
                pass

    return _lifespan


def create_app(container: ApiContainer | None = None) -> FastAPI:
    """Create and configure the Himmy FastAPI app.

    Pass a custom :class:`ApiContainer` to inject production backends; omit it to
    get the offline-first default (in-memory storage + stub inference). The app
    installs a startup run-sweep + shutdown drain (AAEO-1) and a global exception
    handler mapping :class:`HimmyError` to a structured 400 (AAEO-9).
    """
    from himmy.services.observability import (
        configure_observability,
        instrument_fastapi,
    )

    configure_observability()

    if container is None:
        container = ApiContainer.build_default()

    guard = _internal_key_dependency()
    dependencies = [Depends(guard)] if guard is not None else []

    app = FastAPI(
        title="Himmy API",
        version="0.1.0",
        description="Backend-for-frontend over the Himmy application services.",
        dependencies=dependencies,
        lifespan=_build_lifespan(container),
    )
    app.state.container = container

    instrument_fastapi(app)

    # AAEO-9: a global exception handler so HimmyError surfaces as a structured
    # error body instead of an opaque 500.
    @app.exception_handler(HimmyError)
    async def _himmy_error_handler(request: Request, exc: HimmyError) -> JSONResponse:
        """Map a domain error to a structured 400 with a stable envelope."""
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(detail=str(exc), code="himmy_error").model_dump(),
        )

    app.include_router(context.router)
    app.include_router(runs.router)
    app.include_router(recommendations.router)
    app.include_router(dashboard.router)
    app.include_router(evaluation.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    return app


__all__ = ["create_app", "set_rate_limiter"]
