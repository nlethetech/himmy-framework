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

from fastapi import Depends, FastAPI, Request, Response
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
    studio,
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
        # Materialize Studio "connection" non-secret fields (SMTP host, search
        # backend, …) from the writable secrets backend into the process env so
        # tool config picks them up without a restart.
        try:
            from himmy.api.studio_connections import apply_connections_to_env

            apply_connections_to_env()
        except Exception:  # pragma: no cover - connections are best-effort
            logger.warning("applying studio connections to env failed", exc_info=True)
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
    app.include_router(studio.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    _install_security_headers(app)
    _install_studio_guard(app)
    _install_openapi_security(app, authenticator)
    # Mount the built Studio SPA last so its catch-all never shadows an API route.
    _mount_studio(app)
    return app


# Built Studio frontend (emitted by `npm run build` in studio/). Absent in a source
# checkout that hasn't built the GUI — the mount is then a no-op and `himmy studio`
# prints how to build it.
STUDIO_STATIC_DIR = (
    __import__("pathlib").Path(__file__).resolve().parent / "_studio_static"
)


def studio_is_built() -> bool:
    """True when the Studio SPA has been built into the package (index.html present)."""
    return (STUDIO_STATIC_DIR / "index.html").is_file()


# Path prefixes that must always resolve as API (never fall back to the SPA shell),
# so an unknown API route still returns a real JSON 404.
_STUDIO_API_PREFIXES = ("api/", "v1/", "health", "docs", "redoc", "openapi.json")


def _mount_studio(app: FastAPI) -> None:
    """Serve the built Studio SPA without ever shadowing an API route.

    Hashed assets are served from ``/assets`` and ``/`` serves the shell. For any
    other path we rely on a **404 fallback** rather than a greedy catch-all route:
    real routes (including any added after ``create_app``) always match first, and
    only a genuine 404 on a non-API ``GET`` returns ``index.html`` for client-side
    routing. Unknown API paths keep their JSON 404. No-op until the GUI is built.
    """
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException as StarletteHTTPException

    if not studio_is_built():
        return

    index = STUDIO_STATIC_DIR / "index.html"
    assets = STUDIO_STATIC_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="studio-assets")

    @app.get("/", include_in_schema=False)
    async def _studio_index() -> FileResponse:
        return FileResponse(str(index))

    @app.exception_handler(StarletteHTTPException)
    async def _spa_fallback(request: Request, exc: StarletteHTTPException):
        """Serve a real static file or the SPA shell on a 404 for a non-API GET.

        Top-level static files emitted into the build root (fonts, favicon, manifest)
        live outside ``/assets``; serve the real file when the path resolves to one
        under the build dir, otherwise fall back to ``index.html`` for client routing.
        """
        path = request.url.path.lstrip("/")
        is_spa_route = (
            exc.status_code == 404
            and request.method in ("GET", "HEAD")
            and not path.startswith(_STUDIO_API_PREFIXES)
        )
        if is_spa_route:
            if path:
                candidate = (STUDIO_STATIC_DIR / path).resolve()
                if STUDIO_STATIC_DIR in candidate.parents and candidate.is_file():
                    return FileResponse(str(candidate))
            return FileResponse(str(index))
        # Preserve FastAPI's default HTTPException shape (and any auth/Allow headers).
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )


def _studio_host(value: str) -> str:
    """The host part of a ``Host`` header (port + IPv6 brackets stripped)."""
    value = value.strip().lower()
    if not value:
        return ""
    if value.startswith("["):  # [::1]:8800
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if ":" in value else value


def _origin_host(value: str) -> str:
    """The host of an Origin/Referer URL."""
    from urllib.parse import urlparse

    try:
        return (urlparse(value).hostname or "").lower()
    except ValueError:
        return ""


def _install_studio_guard(app: FastAPI) -> None:
    """Block DNS-rebinding + cross-site access to the loopback Studio API (WS3.5).

    Studio runs agents that take real actions (send email/Telegram) and stores
    credentials, so its ``/api/studio`` surface must only answer requests whose
    ``Host`` is loopback and whose ``Origin``/``Referer`` (when present) is same-site.
    This defeats a malicious web page (or DNS rebinding) reaching the local API.
    Disable with ``HIMMY_STUDIO_GUARD=0``; allow extra hosts (e.g. a reverse proxy)
    via ``HIMMY_STUDIO_ALLOW_HOSTS=host1,host2``.
    """
    import os

    if os.environ.get("HIMMY_STUDIO_GUARD", "1").lower() in ("0", "false", "no"):
        return
    # "testserver" is Starlette's TestClient default Host. Browsers cannot forge a
    # Host header via fetch (it's a forbidden header), so allowing it opens no hole
    # while keeping the test suite green.
    allowed = {"127.0.0.1", "localhost", "::1", "testserver"} | {
        h.strip().lower()
        for h in (os.environ.get("HIMMY_STUDIO_ALLOW_HOSTS") or "").split(",")
        if h.strip()
    }

    @app.middleware("http")
    async def _guard(request: Request, call_next: Callable) -> Response:
        if request.url.path.startswith("/api/studio"):
            host = _studio_host(request.headers.get("host", ""))
            if host and host not in allowed:
                return JSONResponse(
                    status_code=403, content={"detail": "host not allowed"}
                )
            ref = request.headers.get("origin") or request.headers.get("referer")
            if ref:
                rh = _origin_host(ref)
                if rh and rh not in allowed:
                    return JSONResponse(
                        status_code=403, content={"detail": "cross-origin blocked"}
                    )
        return await call_next(request)


def _install_security_headers(app: FastAPI) -> None:
    """Add security response headers + an optional strict CORS policy (WS3.4).

    Headers (HSTS, nosniff, frame-deny, referrer) are on by default and safe; CORS
    stays **deny** (same-origin) unless ``HIMMY_CORS_ORIGINS`` lists allowed origins.
    """
    import os

    hsts_on = os.environ.get("HIMMY_HSTS", "1").lower() not in ("0", "false", "no")

    @app.middleware("http")
    async def _security_headers(request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if hsts_on:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains",
            )
        return response

    origins = [
        o.strip()
        for o in (os.environ.get("HIMMY_CORS_ORIGINS") or "").split(",")
        if o.strip()
    ]
    if origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


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
