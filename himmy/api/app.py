"""API kernel: the FastAPI app factory (BFF for the application services).

``create_app`` wires an :class:`ApiContainer` onto ``app.state``, mounts the
routers, configures observability, registers a global exception handler, and —
when ``HIMMY_INTERNAL_API_KEY`` is set — guards every route behind a
constant-time trusted-boundary header (surfaced as an OpenAPI security scheme).
A FastAPI lifespan sweeps stuck runs on startup and drains in-flight background
runs on shutdown (AAEO-1). It is a thin transport layer: behavior lives below it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from himmy.api.auth import (
    build_access_policy,
    build_authenticator,
    principal_dependency,
)
from himmy.api.deps import ApiContainer
from himmy.api.models import ErrorResponse
from himmy.api.ratelimit import build_rate_limiter
from himmy.api.routers import (
    audit,
    context,
    dashboard,
    evaluation,
    recommendations,
    runs,
)
from himmy.core.errors import HimmyError
from himmy.services.audit import SecurityAuditLog

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger("himmy.api")


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


async def _rate_limit_dependency(request: Request) -> None:
    """Apply the app's rate limiter (or the global hook); no-op by default."""
    limiter = getattr(request.app.state, "rate_limiter", None) or _RATE_LIMITER
    if limiter is not None:
        limiter(request)


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

    # Identity: authenticate every request → Principal (WS1); None ⇒ offline/no-auth
    # default where requests are ANONYMOUS (all tenants). Rate limiting runs either way.
    authenticator = build_authenticator()
    app = FastAPI(
        title="Himmy API",
        version="0.1.0",
        description="Backend-for-frontend over the Himmy application services.",
        # Authenticate first (so the limiter can key on the principal), then throttle.
        dependencies=[
            Depends(principal_dependency),
            Depends(_rate_limit_dependency),
        ],
        lifespan=_build_lifespan(container),
    )
    app.state.container = container
    app.state.authenticator = authenticator
    # Authorization: role → permission policy (data-driven via HIMMY_RBAC_FILE).
    # Enforced per-route via require_permission; bypassed when auth is off.
    app.state.access_policy = build_access_policy()
    # Security audit: auth/authz/access events as tamper-evident entities (WS1.4).
    app.state.security_audit = SecurityAuditLog(container.entity_registry)
    # Rate limiting: per-principal/IP token bucket (WS3.2), off unless configured.
    app.state.rate_limiter = build_rate_limiter()

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
    app.include_router(audit.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    _install_openapi_security(app, authenticator)
    return app


def _install_openapi_security(app: FastAPI, authenticator: object | None) -> None:
    """Advertise the active authenticator's security scheme in the OpenAPI doc.

    Pluggable: each authenticator describes its own scheme (API-key header today,
    OIDC bearer later), so the published contract reflects how to authenticate.
    """
    schemes_fn = getattr(authenticator, "openapi_security_scheme", None)
    if schemes_fn is None:
        return
    schemes = schemes_fn()

    def _custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {}).update(schemes)
        schema["security"] = [{name: []} for name in schemes]
        app.openapi_schema = schema
        return schema

    app.openapi = _custom_openapi  # type: ignore[method-assign]


__all__ = ["create_app", "set_rate_limiter"]


__all__ = ["create_app", "set_rate_limiter"]
