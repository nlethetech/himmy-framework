"""P1 — RBAC route-coverage gate: every non-public route carries a permission guard.

The companion to the ``/v1`` workspace-scoping AST guard
(``tests/api/test_v1_workspace_scoping_guard.py``): where that one defends against a
read forgetting to SCOPE its tenant, this one defends against a route shipping with NO
``require_permission`` guard at all — a new endpoint silently reachable by any
authenticated principal once auth is on.

Rather than a static AST pass (which cannot see router-level ``dependencies=[...]`` or
the ``studio_permission`` indirection), this walks the BUILT app's runtime route tree
and inspects each route's resolved dependency graph for the marker
:mod:`himmy.api.auth.rbac` stamps on every ``require_permission`` /
``studio_permission`` closure (``_rbac_permission``). A route that carries no such
guarded dependency — and is not on the small, explicit PUBLIC allow-list — fails the
gate, so a future unguarded route cannot ship unnoticed.

The analyzer is self-tested against a deliberately-unguarded synthetic route, so it
genuinely WOULD fail if a real route dropped its guard.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.routing import Mount

from himmy.api import ApiContainer
from himmy.api.app import create_app
from himmy.api.auth import require_permission
from himmy.api.connector_inbound import INBOUND_AGENT_PATH_ENV
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)
from himmy.connectors.manage import ConnectorService

#: Routes that are PUBLIC by design and therefore carry no RBAC guard. Each is infra,
#: not a data/operator surface: liveness/readiness probes, the Prometheus scrape, the
#: auto-docs + raw schema (themselves suppressed once auth is configured — see
#: ``create_app``'s ``_docs_locked``), and the Studio SPA shell ``/`` (static HTML/JS,
#: no data). A NEW entry here is a deliberate, reviewed decision to expose a route
#: without authorization.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",  # Studio single-page-app shell (static asset, no data)
        "/health",
        "/readyz",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


#: Conditionally-mounted INBOUND-connector routes that are UNAUTHENTICATED-BY-DESIGN
#: (signed-webhook surface). They mount only when ``HIMMY_INBOUND_AGENT_PATH`` names an
#: agent AND the connector is enabled + its signing secret is present, so the default
#: ``create_app()`` the gate inspects never sees them — the coverage gate had a blind
#: spot for this family. They carry NO ``require_permission`` guard ON PURPOSE: their
#: control is the connector's own HMAC signature + source allow-list + dedup gate
#: (``himmy.connectors.router``), an unauthenticated-by-design signed-trigger surface.
#: Enumerating them here makes that decision VISIBLE and regression-protected — a future
#: inbound route that should be RBAC-guarded would not be on this list and would fail
#: :func:`test_inbound_connector_routes_are_signed_not_rbac`.
_INBOUND_SIGNED_PATHS: frozenset[str] = frozenset(
    {
        "/v1/connectors/webhook",
        "/v1/connectors/slack",
        "/v1/connectors/discord",
    }
)

_INBOUND_SECRET = "whsec_aaaabbbbccccddddeeeeffff"


def _walk_dependency_calls(dependant: Any, out: list[Any]) -> None:
    """Collect every callable in a route's resolved dependency graph (DFS)."""
    out.append(getattr(dependant, "call", None))
    for sub in getattr(dependant, "dependencies", []):
        _walk_dependency_calls(sub, out)


def _route_is_rbac_guarded(route: Any) -> bool:
    """Whether ``route``'s dependency graph carries a ``require_permission`` guard.

    True iff some callable in the route's resolved dependant tree was stamped with the
    ``_rbac_permission`` marker — set on the closure returned by
    :func:`himmy.api.auth.rbac.require_permission` AND by
    :func:`himmy.api.routers.studio_common.studio_permission` (which wraps it). This
    catches BOTH a route-level ``Depends(require_permission(...))`` and a router-level
    ``dependencies=[...]`` baseline, regardless of nesting depth.
    """
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return False
    calls: list[Any] = []
    _walk_dependency_calls(dependant, calls)
    return any(getattr(call, "_rbac_permission", None) is not None for call in calls)


def _iter_api_routes(routes: list[Any], prefix: str = "") -> Any:
    """Yield ``(path, route)`` for every concrete leaf route on the app.

    Version-robust like :func:`himmy.api.route_introspection.collect_route_paths`:
    descends into ``Mount`` sub-apps (composing their prefix) and FastAPI 0.137+
    ``_IncludedRouter`` wrappers, so a route nested under an included router is still
    inspected (a top-level-only walk would silently skip — and thus pass — them).
    """
    for route in routes:
        sub = getattr(route, "routes", None)
        ctx = getattr(route, "include_context", None)
        included = getattr(ctx, "included_router", None) or getattr(
            route, "original_router", None
        )
        if sub:
            mount_prefix = getattr(route, "path", "") if isinstance(route, Mount) else ""
            yield from _iter_api_routes(sub, prefix + (mount_prefix or ""))
        elif included is not None:
            yield from _iter_api_routes(included.routes, prefix)
        else:
            yield prefix + getattr(route, "path", ""), route


def _candidate_routes() -> list[tuple[str, Any]]:
    """Every leaf route that has a dependant (i.e. an actual API endpoint)."""
    app = create_app()
    return [
        (path, route)
        for path, route in _iter_api_routes(app.routes)
        if getattr(route, "dependant", None) is not None
    ]


def test_every_non_public_route_is_rbac_guarded() -> None:
    """No route ships without a require_permission guard (or being on the allow-list)."""
    candidates = _candidate_routes()
    # Sanity: we are actually inspecting the real surface (a /v1 read + a Studio route).
    seen_paths = {path for path, _route in candidates}
    assert "/v1/runs" in seen_paths
    assert any(p.startswith("/api/studio/") for p in seen_paths)

    offenders: list[str] = []
    for path, route in candidates:
        if path in _PUBLIC_PATHS:
            continue
        if _route_is_rbac_guarded(route):
            continue
        methods = sorted(getattr(route, "methods", None) or [])
        offenders.append(f"{methods} {path}")
    assert not offenders, (
        "these routes carry no require_permission guard and are not on the public "
        f"allow-list (add a guard, or add to _PUBLIC_PATHS with a reason): {offenders}"
    )


def test_v1_runs_route_is_detected_as_guarded() -> None:
    """Regression: a known-guarded /v1 route is recognised by the analyzer."""
    guarded = {
        path: route
        for path, route in _candidate_routes()
        if path == "/v1/runs"
    }
    assert guarded, "expected /v1/runs to be present"
    assert all(_route_is_rbac_guarded(route) for route in guarded.values())


@pytest.fixture
def _inbound_env(tmp_path: Path):
    """A writable secrets backend + an inbound agent.yaml + an enabled webhook connector.

    Mirrors ``tests/api/test_connector_inbound_mount.py``: configures the webhook
    connector with a signing secret, enables it for ``inbound``, points
    ``HIMMY_INBOUND_AGENT_PATH`` at a stub agent, and disables the Studio rebinding guard
    (the connector's OWN HMAC gate is the control under test). Cleans up after.
    """
    touched = [
        "HIMMY_WEBHOOK_SIGNING_SECRET",
        "HIMMY_WEBHOOK_ALLOWED_SOURCES",
        "HIMMY_CONNECTOR_WEBHOOK_INBOUND_ENABLED",
        INBOUND_AGENT_PATH_ENV,
        "HIMMY_STUDIO_GUARD",
    ]
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    for name in touched:
        os.environ.pop(name, None)
    (tmp_path / "agent.yaml").write_text(
        "name: inbound-bot\ndescription: inbound test agent\nprovider: stub\n"
    )
    os.environ["HIMMY_STUDIO_GUARD"] = "0"
    svc = ConnectorService()
    svc.configure(
        "webhook",
        {
            "HIMMY_WEBHOOK_SIGNING_SECRET": _INBOUND_SECRET,
            "HIMMY_WEBHOOK_ALLOWED_SOURCES": "ci",
        },
    )
    svc.enable("webhook", "inbound")
    os.environ[INBOUND_AGENT_PATH_ENV] = str(tmp_path / "agent.yaml")
    yield tmp_path
    configure_secrets(None)
    for name in touched:
        os.environ.pop(name, None)


def test_inbound_connector_routes_are_signed_not_rbac(_inbound_env: Path) -> None:
    """The conditionally-mounted inbound family is RBAC-blind but signature-guarded.

    Red-team r5 (info): the default ``create_app()`` the gate inspects never mounts the
    inbound connector routes (they need ``HIMMY_INBOUND_AGENT_PATH`` + an enabled, signed
    connector), so the coverage gate could not see them — a blind spot. This drives the
    inbound mount explicitly and asserts the family is (a) actually present and (b) on the
    reasoned ``_INBOUND_SIGNED_PATHS`` allow-list rather than silently RBAC-unguarded, AND
    that its non-RBAC control (the HMAC signature gate) genuinely rejects an unsigned call.
    So a FUTURE inbound route that should be RBAC-guarded — but is not, and is not on this
    list — would surface here instead of shipping unguarded unnoticed.
    """
    app = create_app(ApiContainer.build_default())
    paths = {path for path, _route in _iter_api_routes(app.routes)}
    # (a) The family actually mounted (otherwise the assertion would be vacuous).
    assert "/v1/connectors/webhook" in paths

    # (b) Any mounted ``/v1/connectors/*`` route that is NOT RBAC-guarded must be on the
    # reasoned signed allow-list — its control is the signature gate, by design. The
    # always-mounted connectors-MANAGEMENT routes (``/v1/connectors``, ``/v1/connectors/{name}``)
    # are RBAC-guarded and so are not implicated; this catches an unguarded inbound family
    # (or a FUTURE inbound route that should be guarded but is not, and is not allow-listed).
    connector_routes = [
        (path, route)
        for path, route in _iter_api_routes(app.routes)
        if path.startswith("/v1/connectors")
        and getattr(route, "dependant", None) is not None
    ]
    unguarded = [
        (path, route)
        for path, route in connector_routes
        if not _route_is_rbac_guarded(route)
    ]
    assert unguarded, "expected at least one unguarded (signature-gated) inbound route"
    for path, _route in unguarded:
        assert path in _INBOUND_SIGNED_PATHS, (
            f"connector route {path} is neither RBAC-guarded nor on the reasoned signed "
            "allow-list — if it is meant to be RBAC-guarded, add a require_permission "
            "guard; otherwise add it to _INBOUND_SIGNED_PATHS with a reason"
        )

    # The non-RBAC control is live: an UNSIGNED delivery is rejected (401), so the
    # 'unauthenticated by design' surface is not actually open.
    body = json.dumps({"source": "ci", "text": "hi", "id": "x1"}).encode()
    with TestClient(app) as client:
        unsigned = client.post("/v1/connectors/webhook", content=body)
        assert unsigned.status_code == 401
        sig = "sha256=" + hmac.new(
            _INBOUND_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        signed = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={"X-Himmy-Signature": sig},
        )
        assert signed.status_code == 200


def test_gate_flags_a_deliberately_unguarded_route() -> None:
    """The analyzer genuinely FAILS for a route shipped without a permission guard.

    Builds a tiny synthetic app with two routes — one guarded by ``require_permission``,
    one bare — and asserts the analyzer distinguishes them. This proves the gate would
    catch a real unguarded route (and is not vacuously passing).
    """
    app = FastAPI()

    @app.get("/guarded", dependencies=[Depends(require_permission("run", "read"))])
    async def _guarded() -> dict[str, str]:  # pragma: no cover - never called
        return {"ok": "yes"}

    @app.get("/unguarded")
    async def _unguarded(request: Request) -> dict[str, str]:  # pragma: no cover
        return {"ok": "no"}

    by_path = {path: route for path, route in _iter_api_routes(app.routes)}
    assert _route_is_rbac_guarded(by_path["/guarded"])
    assert not _route_is_rbac_guarded(by_path["/unguarded"])
