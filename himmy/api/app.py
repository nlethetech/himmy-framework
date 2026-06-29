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

from himmy import __version__
from himmy.api import runtime_bootstrap as _bootstrap
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
    knowledge,
    models,
    privacy_audit,
    recommendations,
    routines,
    runs,
    studio,
    studio_eval,
    studio_files,
    studio_guardrails,
    studio_knowledge_upload,
    studio_learning,
    studio_lineage,
    studio_mcp,
    studio_memory,
    studio_missions,
    studio_models,
    studio_notify,
    studio_privacy,
    studio_projects,
    studio_routines,
    studio_seclog,
    studio_teams,
    studio_telegram,
    teams,
    threads,
    workflows,
)
from himmy.api.runtime_bootstrap import (
    start_run_substrate,
    stop_run_substrate,
)
from himmy.api.runtime_bootstrap import (
    wants_durable_store as _bootstrap_wants_durable_store,
)
from himmy.application.services import WorkspaceRunQuotaExceeded
from himmy.core.errors import HimmyError
from himmy.services.audit import SecurityAuditLog, SecurityEvent
from himmy.services.observability.logging import (
    bind_request_id,
    configure_logging,
    reset_request_id,
)
from himmy.services.observability.metrics import install_metrics

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


def _enforce_multi_tenant_posture(authenticator: object | None) -> None:
    """Fail closed under the multi-tenant posture (G2), BEFORE the loopback shortcut.

    A multi-tenant deployment (``HIMMY_MULTI_TENANT`` truthy, or any non-empty
    ``HIMMY_AUTH_MODE``) MUST NOT run on an authenticator that mints every caller an
    all-tenants admin. This check runs ahead of the off-loopback shortcut in
    :func:`_enforce_auth_posture` precisely because a shared-key-only deploy
    (``authenticator is not None``) would otherwise take the early return and skip
    every guard — the named footgun would NOT be fixed.

    When multi-tenant we require an authenticator that BINDS callers to tenants
    (OIDC, or tenant-mapped API keys), read via ``getattr(..., "binds_tenants",
    False)`` so a custom/legacy authenticator lacking the member fails CLOSED rather
    than ``AttributeError``. We additionally HARD-REJECT the two unsafe escape hatches:
    ``HIMMY_ALLOW_UNAUTHENTICATED`` (would re-open the ANONYMOUS all-tenants surface)
    and a truthy ``HIMMY_ALLOW_OPERATOR_SPEC_TOOLS`` (would un-fail-close the RCE/SSRF
    spec sanitizer for tenant-submitted specs), plus ``HIMMY_STUDIO_AUTH=off`` (the
    Studio auth kill-switch — would re-open the network-isolated operator console to any
    authenticated principal instead of gating it behind ``studio:use``). RBAC is
    on-by-construction once an authenticator exists, but we also require
    ``build_access_policy()`` to resolve a non-None policy so a broken
    ``HIMMY_RBAC_FILE`` can't silently disarm authz.

    Note: ANY non-empty ``HIMMY_AUTH_MODE`` (incl. the ``apikey`` example in
    values.yaml) now engages strictness — a previously-working shared-key-ONLY deploy
    that also sets an auth mode will be refused until it configures tenant-binding auth
    (a mapped-keys file or OIDC).

    The posture also engages when the authenticator BINDS TENANTS even with NO env flag
    set: a per-tenant ``HIMMY_API_KEYS_FILE`` is multi-tenant IN FACT, so every guard
    below (the ``HIMMY_STUDIO_AUTH=off`` / ``HIMMY_ALLOW_UNAUTHENTICATED`` /
    ``HIMMY_ALLOW_OPERATOR_SPEC_TOOLS`` refusals) must fire for it too — otherwise a
    keys-file-only deploy that also flips one of those kill-switches would silently skip
    them (a fail-open the env-flag-only detector missed).
    """
    from himmy.api.auth import build_access_policy, is_multi_tenant
    from himmy.config.flags import env_falsy, env_truthy

    binds_tenants = bool(getattr(authenticator, "binds_tenants", False))
    if not (is_multi_tenant() or binds_tenants):
        return
    if not binds_tenants:
        raise HimmyError(
            "refusing to start: a multi-tenant posture is configured "
            "(HIMMY_MULTI_TENANT / HIMMY_AUTH_MODE) but the authenticator does not "
            "bind callers to tenants — a shared-key-only or unauthenticated build "
            "would run every caller as an all-tenants admin. Configure tenant-binding "
            "auth (HIMMY_API_KEYS_FILE with per-tenant keys, or HIMMY_AUTH_MODE=oidc)."
        )
    # Use the SAME truthy vocabulary as the consuming sanitizers / authenticator
    # (apikey._env_truthy, spec_sanitizer._truthy) — all now route through
    # himmy.config.flags.env_truthy — so a posture kill-switch can never be
    # half-honored: a value like ``on`` that enables the dangerous opt-in downstream MUST
    # also trip the startup refusal here. Diverging the detector from the consumer
    # silently fails the guard open (it accepts ``on`` but the refusal misses it).
    # Studio is a network-isolated OPERATOR console (lineage, privacy, governance,
    # raw run inspection) — never a tenant-facing surface. ``HIMMY_STUDIO_AUTH=off``
    # is its auth kill-switch (intended only for a trusted single-user box); under a
    # multi-tenant posture it would re-open every Studio surface to any authenticated
    # principal, regardless of role. Refuse to start: the operator console MUST stay
    # behind ``studio:use`` when callers are mutually-untrusted tenants. The off-switch
    # uses the canonical falsy reader (env_falsy) so this refusal and the runtime
    # ``_studio_auth_off`` reader recognise exactly the same off tokens.
    if env_falsy("HIMMY_STUDIO_AUTH"):
        raise HimmyError(
            "refusing to start: HIMMY_STUDIO_AUTH is disabled under a multi-tenant "
            "posture — that kill-switch re-opens the Studio operator console (lineage, "
            "privacy/governance, raw run inspection) to any authenticated principal "
            "instead of gating it behind the studio:use permission. Remove it for a "
            "multi-tenant deployment; Studio is a network-isolated operator surface."
        )
    if env_truthy("HIMMY_ALLOW_UNAUTHENTICATED"):
        raise HimmyError(
            "refusing to start: HIMMY_ALLOW_UNAUTHENTICATED is set under a "
            "multi-tenant posture — that override would re-expose the ANONYMOUS "
            "all-tenants admin surface. Remove it for a multi-tenant deployment."
        )
    if env_truthy("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS"):
        raise HimmyError(
            "refusing to start: HIMMY_ALLOW_OPERATOR_SPEC_TOOLS is set under a "
            "multi-tenant posture — that override un-fail-closes the RCE/SSRF spec "
            "sanitizer (tools_module/http_tools/mcp_servers) for tenant-submitted "
            "specs. Remove it for a multi-tenant deployment."
        )
    # RBAC must resolve a usable policy (a broken HIMMY_RBAC_FILE would otherwise raise
    # later or — worse — leave authz mis-wired). build_access_policy() returns the
    # built-in DEFAULT_POLICY or a loaded file; a None here means authz is disarmed.
    if build_access_policy() is None:  # pragma: no cover - defensive: build never None
        raise HimmyError(
            "refusing to start: no RBAC policy resolved under a multi-tenant posture "
            "(check HIMMY_RBAC_FILE)."
        )


def _enforce_auth_posture(authenticator: object | None, bind_host: str) -> None:
    """Refuse to start an unauthenticated admin surface on a non-loopback bind.

    The zero-config default is a fully-unrestricted (all-tenants/admin) ANONYMOUS
    principal — safe only because the default bind is loopback. Binding off-loopback
    with no authenticator would expose that admin surface to the network. We fail
    closed: such a combination raises unless the operator explicitly opts in via
    ``HIMMY_ALLOW_UNAUTHENTICATED=1`` (e.g. auth is terminated at a trusted proxy).

    The multi-tenant fail-closed posture (G2) is checked FIRST — before the
    off-loopback shortcut below — so a shared-key-only deploy (which would otherwise
    satisfy ``authenticator is not None`` and return early) is still refused.
    """
    from himmy.config.flags import env_truthy

    _enforce_multi_tenant_posture(authenticator)

    if authenticator is not None or _is_loopback_host(bind_host):
        return
    # Canonical truthy parse: the same vocabulary the multi-tenant refusal uses for this
    # exact flag, so ``HIMMY_ALLOW_UNAUTHENTICATED=on`` cannot be honored in one check
    # and ignored in the other (the prior ad-hoc tuple here omitted ``on``).
    opt_in = env_truthy("HIMMY_ALLOW_UNAUTHENTICATED")
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

    Thin alias over :func:`himmy.api.runtime_bootstrap.wants_durable_store` — the shared
    bootstrap is the single source of truth so the lifespan and the standalone worker
    decide durability identically. Kept under this name for backward-compat callers.
    """
    return _bootstrap_wants_durable_store()


#: How often the lifespan maintenance loop GC's the durable trigger-dedup ledger (s).
#: Backward-compat alias over the shared bootstrap's constant — kept so existing tests
#: that ``monkeypatch.setattr(app, "_DEDUP_SWEEP_INTERVAL_SECONDS", …)`` still resolve a
#: name here; the live value the lifespan uses comes from the bootstrap module.
_DEDUP_SWEEP_INTERVAL_SECONDS = _bootstrap.DEDUP_SWEEP_INTERVAL_SECONDS


#: Backward-compat re-exports: the dispatcher-knob resolver, the dedup-sweep loop, and the
#: durable-store predicate now live in :mod:`himmy.api.runtime_bootstrap` (the single source
#: of truth shared with the standalone ``himmy worker``). Re-exported under their old names
#: so existing importers/monkeypatchers keep working. Tests that need to intercept the LIVE
#: loop must patch the bootstrap module, since the lifespan calls the bootstrap directly.
_dispatch_env_overrides = _bootstrap.dispatch_env_overrides
_dedup_sweep_loop = _bootstrap._dedup_sweep_loop
_run_store_is_durable = _bootstrap.run_store_is_durable


def _active_backend_name(container: ApiContainer) -> str:
    """Name the container's storage backend for the readiness truth (G3): one of
    ``postgres`` / ``sqlite`` / ``memory`` / ``none``.

    Thin alias over :func:`himmy.api.runtime_bootstrap.active_backend_name` (single source
    of truth shared with the worker). Read by ``/readyz`` (G4): a ``postgres`` backend
    triggers the live ``SELECT 1`` + applied-migration check; ``sqlite``/``memory`` need no
    network probe.
    """
    return _bootstrap.active_backend_name(container)


def _record_durability_truth(
    app: FastAPI,
    container: ApiContainer,
    *,
    durable_requested: bool,
    built_durable: bool,
) -> None:
    """Stamp the durability truth onto ``app.state`` for ``/readyz`` (G3).

    Sets ``durable_requested`` (was a durable backend asked for), ``built_durable``
    (did it actually wire), and ``active_backend`` (the resolved backend name). Called
    in every lifespan branch — including the except-fallback that silently degrades to
    in-memory — AND in the create_app body so a bare ``TestClient(create_app())`` (which
    never enters the lifespan) does not ``AttributeError`` on ``/readyz``.
    """
    app.state.durable_requested = durable_requested
    app.state.built_durable = built_durable
    app.state.active_backend = _active_backend_name(container)


def _record_policy_loaded(app: FastAPI) -> None:
    """Emit a 'policy_loaded' audit event for the active RBAC policy (P1 observability).

    Captures the policy SOURCE (file path or ``<built-in>``) + a non-reversible content
    HASH + role count, so the forensic question "did the running authorization policy
    change?" is answerable from the trail — while the verbatim grants are NEVER copied
    into the (possibly wider-read) audit log.

    A NO-OP when no authenticator is configured, mirroring ``require_permission`` /
    ``audit_event``: the offline/zero-config path records nothing and is byte-unchanged.
    Called from the ``create_app`` body AND after a lifespan ``_rebind_container`` (which
    swaps the audit log onto the durable registry), so the event lands in whichever
    registry actually serves requests. Best-effort: a logging hiccup never blocks
    startup.
    """
    if getattr(app.state, "authenticator", None) is None:
        return
    audit = getattr(app.state, "security_audit", None)
    policy = getattr(app.state, "access_policy", None)
    if audit is None or policy is None:  # pragma: no cover - both wired together
        return
    try:
        from himmy.api.auth import policy_fingerprint, policy_source

        audit.record(
            SecurityEvent(
                event_type="policy_loaded",
                outcome="allow",
                actor={"subject": "system", "auth_method": "startup"},
                resource="rbac_policy",
                action="load",
                detail=(
                    f"source={policy_source()} "
                    f"hash={policy_fingerprint(policy)} "
                    f"roles={len(policy.role_permissions)}"
                ),
            )
        )
    except Exception:  # pragma: no cover - audit is best-effort, never fatal
        logger.warning("recording policy_loaded audit event failed", exc_info=True)


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
    # P0 tool authz (confused-deputy fix): re-wire the per-run tool-capability gate onto
    # the NEW container's run_app, mirroring the wiring ``create_app`` did onto the
    # throwaway in-memory container's run_app. The durable run_app rebuilt here in the
    # lifespan (``ApiContainer.build_default_async``) is constructed WITHOUT an access
    # policy, so without this re-wire its ``_access_policy`` stays ``None`` and
    # ``_build_tool_authorizer`` returns ``None`` — silently disabling the entire
    # capability authorizer for the durable auto-upgrade server path (the standard
    # multi-tenant production shape), letting any caller authorized merely to START a run
    # invoke EVERY tool the agent declares. We wire it ONLY when an authenticator is
    # configured, exactly as ``create_app`` does, so the offline/zero-config path is
    # byte-identical (no authenticator → no policy wired → no per-run authorizer).
    if getattr(app.state, "authenticator", None) is not None:
        policy = getattr(app.state, "access_policy", None)
        run_app = getattr(container, "run_app", None)
        if policy is not None and run_app is not None:
            run_app._access_policy = policy
    # Re-emit the policy_loaded marker into the NEW (durable) audit registry, so the
    # forensic record of the active policy survives the in-memory→durable swap. No-op
    # offline (no authenticator) — see :func:`_record_policy_loaded`.
    _record_policy_loaded(app)


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
            set_server_storage,
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

        # G3: record the durability truth on app.state for /readyz (G4), covering ALL
        # branches — the self-built durable success (built_durable), its silent
        # except-fallback to in-memory (durable WAS requested but NOT built → /readyz must
        # report not-ready), the zero-DSN spine-rebind (no durable RUN store requested),
        # AND the injected-container path (durability = whatever the injected container
        # actually is, inferred from its resolved backend). A degraded-ephemeral pod is
        # thus visible to the readiness probe instead of silently serving data-losing
        # traffic under a 200 /health.
        if upgrade_to_durable:
            durable_requested = _wants_durable_store()
            durable_built = built_durable
        else:
            # Injected container (the prod recipe): the operator already chose backends.
            # Treat a durable resolved backend as both requested and built.
            durable_requested = _active_backend_name(active) in ("postgres", "sqlite")
            durable_built = durable_requested
        _record_durability_truth(
            app,
            active,
            durable_requested=durable_requested,
            built_durable=durable_built,
        )

        # K1: publish the RESOLVED durable storage process-wide so the synchronous
        # ``build_runtime_for_spec`` wiring (``StoreFactory.for_context(server=True)``)
        # hands an in-server agent THIS backend — the actual Postgres/SQLite store the
        # rest of the server uses — instead of silently opening a second local
        # ``.himmy/storage.db`` under a ``postgres://`` DSN. ``active`` is the one resolved
        # container in EVERY branch: the self-built durable path (built_durable), the
        # zero-DSN spine-rebind path (rebuilt_for_spine), AND the injected-container path
        # (``upgrade_to_durable=False`` — the documented prod recipe — where neither branch
        # ran and ``active is container``). Resolving off ``active.storage`` therefore
        # covers all three lifespan branches the reviewer required, with no second pool.
        set_server_storage(getattr(active, "storage", None))

        # The routine canonical-storage provider is wired in ``create_app`` to read
        # ``app.state.container.storage`` dynamically, so the rebind above (to the
        # durable store) is automatically reflected — no re-wiring needed here.

        # Startup: bring up the durable run substrate (publish store, enable dispatch
        # BEFORE the recovery sweep, sweep + HITL reconcile, start dispatcher AFTER
        # recovery, start the dedup-ledger GC loop). This is the EXACT same sequence the
        # standalone ``himmy worker`` runs — both call the shared bootstrap so the wiring
        # never drifts. ``set_server_storage`` is called inside; ``install_providers=False``
        # because create_app's body already installed providers that read
        # ``app.state.container`` dynamically (so the durable rebind above is reflected) —
        # re-installing fixed-container lambdas would shadow that.
        substrate = await start_run_substrate(active, install_providers=False)
        try:
            yield
        finally:
            # Shutdown: tear the substrate down in reverse (stop dispatcher → drain inline
            # → close container → clear providers/aux loop). The lifespan owns the
            # server-context token, so it clears that itself below AFTER the substrate
            # teardown (no window leaves the flag set with stale storage). The original
            # container is closed too when it was swapped for a durable store / spine.
            await stop_run_substrate(
                substrate,
                original_container=(
                    container if (built_durable or rebuilt_for_spine) else None
                ),
                clear_providers=True,
            )
            # Clear the server-context flag so a subsequent in-process CLI/test run
            # reverts to the in-memory one-shot default. (The substrate already cleared
            # the published server storage before this.)
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
    # Docs gate: once auth is configured (or a multi-tenant posture is declared) the
    # interactive auto-docs (/docs, /redoc) and the raw schema (/openapi.json) are an
    # unauthenticated map of every route + model for an attacker, so we suppress them by
    # passing ``None`` URLs to FastAPI. ``/health`` and ``/readyz`` stay open (they are
    # added as explicit routes, unaffected by these knobs). In the offline zero-config
    # default (no authenticator, single-box) the docs stay ENABLED — byte-unchanged.
    from himmy.api.auth import is_multi_tenant as _is_multi_tenant

    _docs_locked = authenticator is not None or _is_multi_tenant()
    app = FastAPI(
        title="Himmy API",
        version=__version__,
        description="Backend-for-frontend over the Himmy application services.",
        docs_url=None if _docs_locked else "/docs",
        redoc_url=None if _docs_locked else "/redoc",
        openapi_url=None if _docs_locked else "/openapi.json",
        # Authenticate first (so the limiter can key on the principal), then throttle.
        dependencies=[
            Depends(principal_dependency),
            Depends(_rate_limit_dependency),
        ],
        lifespan=_build_lifespan(container, upgrade_to_durable=container_was_none),
    )
    app.state.container = container

    # G3: stamp a SAFE-DEFAULT durability truth on app.state in the create_app BODY so a
    # bare ``TestClient(create_app())`` WITHOUT a ``with`` block (a real test pattern that
    # never enters the lifespan) does not AttributeError on /readyz. The truth is finalized
    # by the lifespan when it actually wires the backend; here we report:
    #   * NOT-ready (built_durable=False) when a durable backend was requested/injected but
    #     not yet initialized — an uninitialized durable pod must not be marked ready;
    #   * READY (durable_requested=False) when nothing durable was asked for (the zero-config
    #     in-memory default and the loopback CLI/test path), so the offline surface is
    #     byte-unchanged and /readyz returns 200 with no DB call.
    _body_durable_requested = (
        _wants_durable_store()
        if container_was_none
        else _active_backend_name(container) in ("postgres", "sqlite")
    )
    _record_durability_truth(
        app,
        container,
        durable_requested=_body_durable_requested,
        built_durable=False,
    )

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

    # T3c: point the routine scheduler at the wired app container so a workspace-scoped
    # (agent_id) routine run dispatches through the SAME RunAppService the request-driven
    # /v1 surface uses — landing in the tenant's workspace of the canonical store under the
    # T0.4 quota + the HITL rails. Resolved dynamically so a lifespan rebind to the durable
    # container is reflected.
    try:
        from himmy.api.routines import set_routine_container_provider

        set_routine_container_provider(lambda: getattr(app.state, "container", None))
    except Exception:  # pragma: no cover - routine wiring is best-effort
        logger.warning("wiring routine container failed", exc_info=True)

    app.state.authenticator = authenticator
    # centralize-tool-gate: record the deployment's auth posture process-wide so the tool
    # chokepoint fails CLOSED when a path reaches it with NO authorizer in scope under a
    # configured authenticator (a future path that forgot BOTH the explicit arg AND the
    # ambient contextvar denies, instead of silently running un-gated). With no
    # authenticator (offline default) this stays False and the chokepoint is a pure pass —
    # byte-unchanged. Mirrors ``require_permission``: enforcement engages when auth does.
    from himmy.services.tools.ambient import mark_auth_configured

    mark_auth_configured(authenticator is not None)
    # Authorization: role → permission policy (data-driven via HIMMY_RBAC_FILE).
    # Enforced per-route via require_permission; bypassed when auth is off.
    app.state.access_policy = build_access_policy()
    # P0 tool authz (confused-deputy fix): hand the run service the RBAC policy ONLY when an
    # authenticator is configured, so it rebuilds a per-run tool-capability gate from each
    # run's actor metadata. With no authenticator (the offline/zero-config default) the
    # policy is NOT wired, so no per-run authorizer is built and tool dispatch is
    # byte-identical to before — enforcement engages exactly when auth does, mirroring
    # ``require_permission``. Best-effort: a missing run_app never blocks startup.
    if authenticator is not None:
        run_app = getattr(container, "run_app", None)
        if run_app is not None:
            run_app._access_policy = app.state.access_policy
    # Security audit: auth/authz/access events as tamper-evident entities (WS1.4).
    app.state.security_audit = SecurityAuditLog(container.entity_registry)
    # P1 observability: record a 'policy_loaded' audit event for the active RBAC policy
    # (source + non-reversible content hash + role count). No-op offline. Re-emitted by
    # _rebind_container if the lifespan later swaps in a durable audit registry.
    _record_policy_loaded(app)
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
    app.include_router(knowledge.router)
    app.include_router(routines.router)
    app.include_router(teams.router)
    app.include_router(workflows.router)
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
    app.include_router(studio_guardrails.router)
    app.include_router(studio_seclog.router)
    app.include_router(studio_learning.router)
    app.include_router(studio_lineage.router)
    app.include_router(studio_files.router)
    app.include_router(studio_knowledge_upload.router)
    app.include_router(studio_models.router)
    app.include_router(studio_memory.router)
    app.include_router(studio_teams.router)
    app.include_router(studio_eval.router)
    app.include_router(studio_missions.router)
    app.include_router(studio_routines.router)
    app.include_router(studio_telegram.router)
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
        """Liveness probe (process up). Stays 200 even when the pod is NOT ready."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    async def readyz(response: Response) -> dict[str, Any]:
        """Readiness probe (can serve durable traffic) — distinct from /health (G4).

        Returns 503 (with a JSON reason) when the pod cannot serve durable traffic:
        a durable backend was requested but is NOT wired (degraded-ephemeral fallback);
        OR the active backend is Postgres and a timeboxed ``SELECT 1`` fails (DB down);
        OR the applied ``schema_migrations`` version is behind the code's max (pending
        migration). The pure-SQLite / no-durable-requested path returns 200 with NO DB
        call, preserving the offline-first default. K8s pulls a 503 pod from the Service
        endpoints (it is NOT crash-looped — that's what the /health liveness probe is for).
        """
        ready, payload = await _readiness(app)
        if not ready:
            response.status_code = 503
        return payload

    _install_security_headers(app)
    _install_studio_guard(app)
    _install_request_context(app)
    # Production observability: in-process Prometheus metrics + GET /metrics, and
    # the optional JSON log format (HIMMY_LOG_FORMAT=json). Both are no-ops on the
    # zero-config/offline path (collecting counters is free; JSON logging stays off
    # by default). Metrics is installed AFTER request-context so its middleware is
    # outermost and observes every request, including inner short-circuits.
    configure_logging()
    install_metrics(app)
    _install_openapi_security(app, authenticator)
    # Mount the built Studio SPA last so its catch-all never shadows an API route.
    _mount_studio(app)
    return app


async def _readiness(app: FastAPI) -> tuple[bool, dict[str, Any]]:
    """Compute the readiness verdict + a JSON payload for ``/readyz`` (G4).

    Reads the durability truth stamped on ``app.state`` (G3) and, for a Postgres backend,
    probes the live pool via the PUBLIC :meth:`PostgresStorageService.ping` /
    :meth:`applied_migration_versions` accessors (never the private pool, never a fresh
    connection). Pure-SQLite / no-durable-requested short-circuits to ready with no DB call.
    """
    durable_requested = bool(getattr(app.state, "durable_requested", False))
    built_durable = bool(getattr(app.state, "built_durable", False))
    backend = str(getattr(app.state, "active_backend", "none"))

    # Durable was asked for but never wired (the degraded in-memory fallback): not ready.
    if durable_requested and not built_durable:
        return False, {
            "status": "not_ready",
            "reason": "durable storage requested but not initialized",
            "backend": backend,
        }

    # SQLite / in-memory / nothing durable: file-local or ephemeral, no network probe.
    if backend != "postgres":
        return True, {"status": "ready", "backend": backend}

    # Postgres: it must answer + be at (or ahead of) the code's migration version.
    storage = getattr(getattr(app.state, "container", None), "storage", None)
    from himmy.services.storage.postgres import PostgresStorageService

    if not isinstance(storage, PostgresStorageService):  # pragma: no cover - defensive
        return False, {
            "status": "not_ready",
            "reason": "postgres backend recorded but storage handle missing",
            "backend": backend,
        }
    if not await storage.ping():
        return False, {
            "status": "not_ready",
            "reason": "postgres unreachable",
            "backend": backend,
        }
    applied = await storage.applied_migration_versions()
    code_max = PostgresStorageService.code_migration_version()
    applied_max = max(applied) if applied else 0
    if applied_max < code_max:
        return False, {
            "status": "not_ready",
            "reason": "schema migrations behind code",
            "backend": backend,
            "applied_version": applied_max,
            "code_version": code_max,
        }
    return True, {
        "status": "ready",
        "backend": backend,
        "applied_version": applied_max,
        "code_version": code_max,
    }


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
_STUDIO_API_PREFIXES = (
    "api/",
    "v1/",
    "health",
    "readyz",
    "docs",
    "redoc",
    "openapi.json",
)


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
        # Bind the id into context so structured (JSON) log lines emitted anywhere
        # during this request carry the same request_id echoed back to the client.
        token = bind_request_id(rid)
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
        finally:
            reset_request_id(token)
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
            # Fail closed on a missing/empty Host too — an absent Host must NOT skip the
            # DNS-rebinding guard (mirrors the Origin/Referer fail-closed handling below).
            if not host or host not in allowed:
                return JSONResponse(
                    status_code=403, content={"detail": "host not allowed"}
                )
            origin = request.headers.get("origin")
            referer = request.headers.get("referer")
            ref = origin or referer
            if ref is not None:
                # A present Origin/Referer must parse to an allowed host. Opaque
                # values ("null", "file://...") yield an empty host: treat them as
                # cross-site and REJECT (fail closed) rather than skipping the check.
                rh = _origin_host(ref)
                if rh not in allowed:
                    return JSONResponse(
                        status_code=403, content={"detail": "cross-origin blocked"}
                    )
            elif request.method in ("POST", "PUT", "PATCH", "DELETE"):
                # A guarded state-changing request with NEITHER Origin nor Referer
                # cannot be proven same-site: fail closed to block CSRF.
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
