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
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

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
    agents,
    audit,
    connectors,
    consent,
    context,
    dashboard,
    diagnostics,
    evaluation,
    models,
    privacy_audit,
    recommendations,
    runs,
    studio,
    studio_eval,
    studio_files,
    studio_knowledge_upload,
    studio_lineage,
    studio_mcp,
    studio_memory,
    studio_missions,
    studio_models,
    studio_notify,
    studio_privacy,
    studio_projects,
    studio_routines,
    studio_teams,
    threads,
)
from himmy.application.services import WorkspaceRunQuotaExceeded
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


def _is_loopback_host(host: str) -> bool:
    """True when a uvicorn bind ``host`` is loopback / unspecified-loopback only.

    ``127.0.0.1`` / ``::1`` / ``localhost`` are loopback. The empty string and the
    unspecified binds (``0.0.0.0`` / ``::``) reach the network, so they are NOT
    loopback. An unparseable value is treated as non-loopback (fail safe).
    """
    import ipaddress

    value = (host or "").strip().lower()
    if not value:
        return False
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _enforce_auth_posture(authenticator: object | None, bind_host: str) -> None:
    """Refuse to start an unauthenticated admin surface on a non-loopback bind.

    The zero-config default is a fully-unrestricted (all-tenants/admin) ANONYMOUS
    principal — safe only because the default bind is loopback. Binding off-loopback
    with no authenticator would expose that admin surface to the network. We fail
    closed: such a combination raises unless the operator explicitly opts in via
    ``HIMMY_ALLOW_UNAUTHENTICATED=1`` (e.g. auth is terminated at a trusted proxy).
    """
    import os

    if authenticator is not None or _is_loopback_host(bind_host):
        return
    opt_in = os.environ.get("HIMMY_ALLOW_UNAUTHENTICATED", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if opt_in:
        logger.warning(
            "binding to non-loopback host %r with NO authenticator configured: the "
            "BFF is exposed unauthenticated (admin/all-tenants) — HIMMY_ALLOW_"
            "UNAUTHENTICATED is set, so starting anyway",
            bind_host,
        )
        return
    raise HimmyError(
        f"refusing to start: bound to non-loopback host {bind_host!r} with no "
        "authenticator configured — this would expose an unauthenticated admin "
        "surface to the network. Configure auth (HIMMY_INTERNAL_API_KEY / "
        "HIMMY_API_KEYS_FILE / HIMMY_AUTH_MODE=oidc), bind to 127.0.0.1, or set "
        "HIMMY_ALLOW_UNAUTHENTICATED=1 to override (e.g. auth at a trusted proxy)."
    )


def _wants_durable_store() -> bool:
    """True when the server should wire the durable store instead of in-memory.

    Opt-in so ``create_app()`` stays offline-green by default: a durable backend is
    selected when a database DSN is configured (``HIMMY_DATABASE_URL``) or when the
    operator explicitly asks for it (``HIMMY_DURABLE_STORAGE=1``). Either way the
    backend is resolved by :class:`StoreFactory` (Postgres for a ``postgres://`` DSN,
    file-backed SQLite otherwise) — this entrypoint only decides *whether* to use it.
    """
    import os

    from himmy.config.secrets import get_secret

    if (get_secret("HIMMY_DATABASE_URL") or "").strip():
        return True
    return os.environ.get("HIMMY_DURABLE_STORAGE", "").lower() in ("1", "true", "yes")


def _rebind_container(app: FastAPI, container: ApiContainer) -> None:
    """Point ``app.state`` at a (re)built container's services.

    Mirrors the wiring ``create_app`` does after building the default container, so a
    durable container constructed in the lifespan replaces the in-memory one for every
    request-time reader (routers read ``app.state.container``; ``audit_event`` reads
    ``app.state.security_audit``).
    """
    app.state.container = container
    app.state.security_audit = SecurityAuditLog(container.entity_registry)
    app.state.consent_ledger = getattr(container, "consent_ledger", None)
    app.state.consent_policy = getattr(container, "consent_policy", None)
    app.state.retention_service = getattr(container, "retention_service", None)


def _build_lifespan(
    container: ApiContainer,
    *,
    upgrade_to_durable: bool = True,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build a FastAPI lifespan that sweeps stuck runs + drains on shutdown (AAEO-1).

    When a durable store is requested (item #7), the lifespan upgrades the in-memory
    default container to a durable one (Postgres/SQLite via
    :meth:`ApiContainer.build_default_async`) *before* serving any request, so
    background runs and the tamper-evident security-audit log survive restarts.

    ``upgrade_to_durable`` gates that upgrade: it is only ever attempted when
    ``create_app`` built the default in-memory container itself. A caller who injects
    a container via ``create_app(container=...)`` has already chosen its backends (the
    documented production recipe is to ``build_default_async`` and pass the result in),
    so we must NOT discard it and stand up a second, duplicate Postgres pool — the
    injected container is served as-is even when ``HIMMY_DATABASE_URL`` is set.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Durable-defaults: mark this process as a server context so any agent wired by
        # the shared spec-builder (himmy.runtime.from_spec) inside the app picks the
        # durable store instead of the one-shot in-memory one. Cleared on shutdown so a
        # later in-process CLI/test run reverts to the in-memory default.
        from himmy.services.storage.factory import (
            reset_server_context,
            set_server_context,
        )

        server_ctx_token = set_server_context(True)

        # Item #7: wire the durable store at startup (the sync default container is
        # in-memory). Build it asynchronously so a Postgres pool can be created +
        # migrated, then rebind app.state so every request-time reader uses it. If the
        # durable build fails we keep the in-memory container running (degraded but up).
        active = container
        built_durable = False
        rebuilt_for_spine = False
        if upgrade_to_durable and _wants_durable_store():
            try:
                active = await ApiContainer.build_default_async()
                _rebind_container(app, active)
                built_durable = True
                logger.info("durable storage wired for the server entrypoint")
            except Exception:  # pragma: no cover - defensive: stay up on store failure
                logger.warning(
                    "durable storage requested but could not be wired; "
                    "falling back to in-memory storage",
                    exc_info=True,
                )
                active = container
        elif upgrade_to_durable:
            # BUILD-ORDER trap (T2.1 reviewer must_fix): ``build_default`` ran BEFORE the
            # server-context flag was set above, so the container resolved its spine on the
            # CLI path. ``SpineFactory.for_cli`` is already durable (the canonical
            # ``.himmy/spine.db``), so the common zero-DSN case is correct at construction;
            # but if ``HIMMY_SPINE_DATABASE_URL`` is set the SERVER spine is Postgres, which
            # is ONLY resolved on the server path — the construction-time spine would be the
            # wrong (SQLite) backend. So when we are NOT building a durable container, ALWAYS
            # rebuild the in-memory default container under the now-active server context:
            # this re-resolves the spine through ``SpineFactory.for_context`` (server) while
            # keeping the in-memory run store the zero-config server expects. The
            # construction-time run store is empty (no request served yet), so discarding it
            # is safe, and a run's lineage now survives a restart on the durable spine.
            try:
                rebuilt = ApiContainer.build_default()
                _rebind_container(app, rebuilt)
                active = rebuilt
                rebuilt_for_spine = True
                logger.info("durable spine wired for the zero-DSN server entrypoint")
            except Exception:  # pragma: no cover - defensive: stay up on spine failure
                logger.warning(
                    "durable spine rebind failed; serving the construction-time spine",
                    exc_info=True,
                )
                active = container

        # The routine canonical-storage provider is wired in ``create_app`` to read
        # ``app.state.container.storage`` dynamically, so the rebind above (to the
        # durable store) is automatically reflected — no re-wiring needed here.

        # Startup: sweep runs left non-terminal by a previous process so they
        # reach a terminal state instead of hanging in QUEUED/RUNNING forever.
        run_app = getattr(active, "run_app", None)
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
                await active.aclose()
            except Exception:  # pragma: no cover - shutdown best-effort
                pass
            # Also close the original container if we swapped it out (for a durable
            # store OR a durable-spine rebind) so its construction-time spine/store
            # handle is released.
            if (built_durable or rebuilt_for_spine) and active is not container:
                try:
                    await container.aclose()
                except Exception:  # pragma: no cover - shutdown best-effort
                    pass
            # Clear the canonical-storage provider so a later in-process CLI/test run
            # does not mirror into this (now-closed) container's store.
            try:
                from himmy.api.studio_canonical import set_canonical_storage_provider

                set_canonical_storage_provider(None)
            except Exception:  # pragma: no cover - shutdown best-effort
                pass
            # Clear the server-context flag so a subsequent in-process CLI/test run
            # reverts to the in-memory one-shot default.
            try:
                reset_server_context(server_ctx_token)
            except Exception:  # pragma: no cover - shutdown best-effort
                pass

    return _lifespan


def create_app(
    container: ApiContainer | None = None, *, bind_host: str | None = None
) -> FastAPI:
    """Create and configure the Himmy FastAPI app.

    Pass a custom :class:`ApiContainer` to inject production backends; omit it to
    get the offline-first default (in-memory storage + stub inference). The app
    installs a startup run-sweep + shutdown drain (AAEO-1) and a global exception
    handler mapping :class:`HimmyError` to a structured 400 (AAEO-9).

    ``bind_host`` is the host uvicorn will bind to (the server entrypoints pass it).
    It gates the fail-closed posture check: starting an unauthenticated build on a
    non-loopback host is refused unless explicitly overridden. When ``None`` it is
    read from ``HIMMY_BIND_HOST`` and defaults to loopback, so a bare ``create_app()``
    (the in-process/test path) is always treated as a safe loopback bind.
    """
    import os

    from himmy.services.observability import (
        configure_observability,
        instrument_fastapi,
    )

    configure_observability()

    # Track whether we built the container ourselves: only an implicitly-built
    # in-memory default may be upgraded to the durable store in the lifespan. An
    # injected container has already chosen its backends and must be served as-is
    # (see ``_build_lifespan``'s ``upgrade_to_durable``).
    container_was_none = container is None
    if container is None:
        container = ApiContainer.build_default()

    # Identity: authenticate every request → Principal (WS1); None ⇒ offline/no-auth
    # default where requests are ANONYMOUS (all tenants). Rate limiting runs either way.
    authenticator = build_authenticator()
    # Fail closed: never silently expose an unauthenticated admin surface off-loopback.
    effective_host = (
        bind_host
        if bind_host is not None
        else os.environ.get("HIMMY_BIND_HOST", "127.0.0.1")
    )
    _enforce_auth_posture(authenticator, effective_host)
    app = FastAPI(
        title="Himmy API",
        version="0.1.0",
        description="Backend-for-frontend over the Himmy application services.",
        # Authenticate first (so the limiter can key on the principal), then throttle.
        dependencies=[
            Depends(principal_dependency),
            Depends(_rate_limit_dependency),
        ],
        lifespan=_build_lifespan(container, upgrade_to_durable=container_was_none),
    )
    app.state.container = container

    # T2.2/T3c: point the routine scheduler at the SAME canonical run store the Studio
    # runs reader uses (``app.state.container.storage``), resolved dynamically so a
    # lifespan rebind to the durable store is reflected without re-wiring. Set in
    # ``create_app`` (not only the lifespan) so even a bare ``TestClient(create_app())``
    # — which never enters the lifespan — mirrors a scheduled run into the one store the
    # GUI reads, keeping "all runs, one store" coherent.
    try:
        from himmy.api.studio_canonical import set_canonical_storage_provider

        set_canonical_storage_provider(
            lambda: getattr(getattr(app.state, "container", None), "storage", None)
        )
    except Exception:  # pragma: no cover - canonical wiring is best-effort
        logger.warning("wiring canonical run store failed", exc_info=True)

    app.state.authenticator = authenticator
    # Authorization: role → permission policy (data-driven via HIMMY_RBAC_FILE).
    # Enforced per-route via require_permission; bypassed when auth is off.
    app.state.access_policy = build_access_policy()
    # Security audit: auth/authz/access events as tamper-evident entities (WS1.4).
    app.state.security_audit = SecurityAuditLog(container.entity_registry)
    # Rate limiting: per-principal/IP token bucket (WS3.2), off unless configured.
    app.state.rate_limiter = build_rate_limiter()
    # Consent governance (WS4.6): only present in a governed deployment (HIMMY_CONSENT on).
    # The /v1/consent router reads these; they are None on the zero-config path so the
    # router is inert (404) and nothing about the offline surface changes.
    app.state.consent_ledger = getattr(container, "consent_ledger", None)
    app.state.consent_policy = getattr(container, "consent_policy", None)
    app.state.retention_service = getattr(container, "retention_service", None)

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

    # T0.4: a workspace at its outstanding-run cap is a backpressure signal, not a
    # server fault — surface it as a clean 429 (the exception's own contract) instead
    # of letting it fall through to the generic 500 handler.
    @app.exception_handler(WorkspaceRunQuotaExceeded)
    async def _run_quota_handler(
        request: Request, exc: WorkspaceRunQuotaExceeded
    ) -> JSONResponse:
        """Map a per-workspace run-concurrency quota breach to a structured 429."""
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                detail=str(exc), code="run_quota_exceeded"
            ).model_dump(),
        )

    app.include_router(context.router)
    app.include_router(runs.router)
    app.include_router(agents.router)
    app.include_router(threads.router)
    app.include_router(recommendations.router)
    app.include_router(dashboard.router)
    app.include_router(evaluation.router)
    app.include_router(models.router)
    app.include_router(diagnostics.router)
    app.include_router(connectors.router)
    app.include_router(privacy_audit.router)
    app.include_router(audit.router)
    app.include_router(consent.router)
    app.include_router(studio.router)
    app.include_router(studio_privacy.router)
    app.include_router(studio_lineage.router)
    app.include_router(studio_files.router)
    app.include_router(studio_knowledge_upload.router)
    app.include_router(studio_models.router)
    app.include_router(studio_memory.router)
    app.include_router(studio_teams.router)
    app.include_router(studio_eval.router)
    app.include_router(studio_missions.router)
    app.include_router(studio_routines.router)
    app.include_router(studio_mcp.router)
    app.include_router(studio_projects.router)
    app.include_router(studio_notify.router)

    # Mount enabled + capability-OK inbound connectors (webhook/Slack/Discord intake),
    # each wired to a configured agent.yaml. No-op unless an inbound agent is configured
    # AND a connector is enabled with its signing secret present — so the default offline
    # surface is unchanged and an unsigned public trigger is never exposed.
    _mount_inbound_connectors(app)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    _install_security_headers(app)
    _install_studio_guard(app)
    _install_request_context(app)
    _install_openapi_security(app, authenticator)
    # Mount the built Studio SPA last so its catch-all never shadows an API route.
    _mount_studio(app)
    return app


def _mount_inbound_connectors(app: FastAPI) -> None:
    """Mount enabled inbound connectors (best-effort; never abort app creation).

    Delegates to :func:`himmy.api.connector_inbound.mount_inbound_connectors`, which
    mounts a router only for a connector that is enabled for ``inbound`` AND whose signing
    secret is present (capability OK), wired to the configured inbound ``agent.yaml``. A
    failure to build any connector is logged and skipped so a misconfiguration cannot stop
    the BFF coming up.
    """
    try:
        from himmy.api.connector_inbound import mount_inbound_connectors

        mounted = mount_inbound_connectors(app)
        if mounted:
            logger.info("mounted inbound connector(s): %s", ", ".join(mounted))
    except Exception:  # pragma: no cover - defensive: inbound mount is best-effort
        logger.warning("mounting inbound connectors failed", exc_info=True)


# Built Studio frontend (emitted by `npm run build` in studio/). Absent in a source
# checkout that hasn't built the GUI — the mount is then a no-op and `himmy studio`
# prints how to build it.
STUDIO_STATIC_DIR = (
    __import__("pathlib").Path(__file__).resolve().parent / "_studio_static"
)


def studio_is_built() -> bool:
    """True when the Studio SPA has been built into the package (index.html present)."""
    return cast(bool, (STUDIO_STATIC_DIR / "index.html").is_file())


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
    async def _spa_fallback(request: Request, exc: StarletteHTTPException) -> Response:
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


def _install_request_context(app: FastAPI) -> None:
    """Tag every request with an id, echo it back, and log unhandled errors.

    Adds an ``X-Request-ID`` to every response (reusing an inbound one if present)
    and turns any *unhandled* exception into a clean 500 JSON carrying that id —
    while logging the full traceback under it, so a GUI error can be traced to a
    server log line. Framework errors (HimmyError) + HTTPException are still handled
    by their dedicated handlers; this only catches the truly-unexpected.
    """
    import uuid

    @app.middleware("http")
    async def _ctx(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - log + clean 500, never leak a traceback
            logger.exception(
                "unhandled error rid=%s %s %s",
                rid,
                request.method,
                request.url.path,
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "internal server error", "request_id": rid},
            )
        response.headers["X-Request-ID"] = rid
        return response


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


# Path prefixes the rebinding/cross-site guard protects. Every router that runs
# agents, reads runs/threads/context, or holds credentials lives under one of these:
# the Studio surface AND the unauthenticated tenant ``/v1/*`` surface (whose
# ``POST /v1/runs`` background-executes an agent). Liveness (``/health``), the SPA
# shell, hashed assets, and the OpenAPI docs are deliberately left open so probes and
# the browser app keep working. This is the default boundary for the whole loopback
# BFF, not just one prefix.
_GUARDED_PREFIXES = ("/api/studio", "/v1")


def _install_studio_guard(app: FastAPI) -> None:
    """Block DNS-rebinding + cross-site access to the loopback BFF (WS3.5).

    The BFF runs agents that take real actions (send email/Telegram), stores
    credentials, and exposes run/thread/context reads — so every sensitive surface
    (``/api/studio`` **and** the unauthenticated ``/v1/*`` tenant API, incl.
    ``POST /v1/runs`` which background-executes an agent) must only answer requests
    whose ``Host`` is loopback and whose ``Origin``/``Referer`` (when present) is
    same-site. This defeats a malicious web page (or DNS rebinding) reaching the local
    API for *any* guarded route, not just Studio. Disable with ``HIMMY_STUDIO_GUARD=0``;
    allow extra hosts (e.g. a reverse proxy) via ``HIMMY_STUDIO_ALLOW_HOSTS=host1,host2``.
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
    async def _guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path.startswith(_GUARDED_PREFIXES):
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
    async def _security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
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

    def _custom_openapi() -> dict[str, Any]:
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
