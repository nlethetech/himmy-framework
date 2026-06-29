"""The whole-app tenant-scope coverage gate — the IDOR/BOLA recurrence guard.

Where ``tests/api/test_v1_workspace_scoping_guard.py`` defends the ``/v1`` REST surface
against a read that forgets to scope by tenant, that guard has a STRUCTURAL blind spot
the data-scoping MAP calls the root cause of each hardening round leaking one more
surface: it is a static AST pass that only inspects ``/v1``-prefixed routers
(``prefix.startswith("/v1")``), so the entire ``/api/studio/*`` family and the
process-local missions/notify registries are NEVER examined — a new Studio read silently
passes.

This gate closes that blind spot by mirroring the RBAC route-coverage gate
(``tests/api/test_rbac_route_coverage.py``): instead of a static AST walk it walks the
BUILT app's RUNTIME route tree (descending ``Mount`` + included routers exactly like
:func:`himmy.api.route_introspection.collect_route_paths`), so it sees EVERY leaf route —
``/v1`` AND ``/api/studio/*`` AND the missions/notify routers. For every GET/list/aggregate
leaf it requires that the route EITHER:

* carries the ``_scoped_read`` marker in its resolved dependency graph (stamped by
  :func:`himmy.api.auth.scoped_read` on a router/route that derives its scope from the
  verified principal and refuses out-of-scope rows) — detected exactly like the RBAC
  gate detects ``_rbac_permission``; OR
* sits on one of FOUR documented, reviewed allow-lists, each entry stating WHY:
  :data:`_PUBLIC` (infra, no data), :data:`_AUTHZ_ONLY` (subject-keyed, self-scope
  enforced), :data:`_ADMIN_CROSS_WORKSPACE` (deliberate admin export),
  :data:`_GLOBAL_NOT_TENANT_KEYED` (process-global / project-local operator resource with
  no tenant dimension), and the TIME-BOXED :data:`_STUDIO_TENANT_PENDING` (Studio data
  surfaces that genuinely carry no tenant/subject column yet — a known, reviewed gap
  awaiting a data-model stamp, NOT a permanently-safe exemption).

So a FUTURE read route that forgets scoping — on ``/v1`` OR Studio — fails this gate
unless someone deliberately adds it to a reasoned allow-list. The recurrence class dies.

The analyzer is self-tested against a deliberately-unscoped synthetic Studio route
(:func:`test_gate_flags_a_deliberately_unscoped_studio_route`), and carries sanity
assertions that a known ``/v1`` scoped route AND a known ``/api/studio`` route are actually
seen — so a FastAPI route-nesting change cannot make the gate vacuously pass.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.routing import Mount

from himmy.api import ApiContainer
from himmy.api.app import create_app
from himmy.api.auth import scoped_read
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal

# --------------------------------------------------------------------------- walk


def _walk_dependency_calls(dependant: Any, out: list[Any]) -> None:
    """Collect every callable in a route's resolved dependency graph (DFS)."""
    out.append(getattr(dependant, "call", None))
    for sub in getattr(dependant, "dependencies", []):
        _walk_dependency_calls(sub, out)


def _route_is_scoped(route: Any) -> bool:
    """Whether ``route``'s dependency graph carries the ``scoped_read`` marker.

    True iff some callable in the route's resolved dependant tree was stamped with the
    ``_scoped_read`` attribute — set on :func:`himmy.api.auth.scoped_read`. Catches BOTH a
    route-level ``Depends(scoped_read)`` and a router-level ``dependencies=[...]`` baseline,
    regardless of nesting depth (mirrors ``_route_is_rbac_guarded`` in the RBAC gate).
    """
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return False
    calls: list[Any] = []
    _walk_dependency_calls(dependant, calls)
    return any(getattr(call, "_scoped_read", None) is not None for call in calls)


def _iter_api_routes(routes: list[Any], prefix: str = "") -> Any:
    """Yield ``(path, route)`` for every concrete leaf route on the app.

    Version-robust like :func:`himmy.api.route_introspection.collect_route_paths`:
    descends into ``Mount`` sub-apps (composing their prefix) and FastAPI 0.137+
    ``_IncludedRouter`` wrappers, so a route nested under an included router (the whole
    ``/api/studio/*`` family) is still inspected — the blind spot the ``/v1`` AST guard has.
    """
    for route in routes:
        sub = getattr(route, "routes", None)
        ctx = getattr(route, "include_context", None)
        included = getattr(ctx, "included_router", None) or getattr(
            route, "original_router", None
        )
        if sub:
            mount_prefix = (
                getattr(route, "path", "") if isinstance(route, Mount) else ""
            )
            yield from _iter_api_routes(sub, prefix + (mount_prefix or ""))
        elif included is not None:
            yield from _iter_api_routes(included.routes, prefix)
        else:
            yield prefix + getattr(route, "path", ""), route


def _get_read_routes() -> list[tuple[str, Any]]:
    """Every GET leaf route on the built default app (a candidate data read)."""
    app = create_app()
    out: list[tuple[str, Any]] = []
    for path, route in _iter_api_routes(app.routes):
        if getattr(route, "dependant", None) is None:
            continue
        if "GET" in (getattr(route, "methods", None) or set()):
            out.append((path, route))
    return out


# ----------------------------------------------------------------------- allow-lists

#: Infra/static routes that serve NO tenant data: liveness/readiness probes, the
#: Prometheus scrape, the auto-docs, and the Studio SPA shell ``/`` (static HTML/JS).
_PUBLIC: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/readyz",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/studio/health",
    }
)

#: Subject-keyed reads whose resource is keyed by ``(subject_id, ...)`` not tenant, with
#: per-request self-scope enforcement in the handler — ``resolve_workspace`` runs purely
#: to authorize the tenant claim and its result is intentionally unused (mirrors the
#: ``/v1`` guard's ``_AUTHZ_ONLY_READS``).
_AUTHZ_ONLY: frozenset[str] = frozenset(
    {
        "/v1/consent/decision",  # subject-keyed; _enforce_self_scope + _authorize_subject_read
        "/v1/consent/latest",  # subject-keyed; _enforce_self_scope + _authorize_subject_read
        "/v1/consent/history",  # subject-keyed; _enforce_self_scope + _authorize_subject_read
    }
)

#: Admin-only reads that DELIBERATELY span all workspaces (gated by an elevated
#: permission, not data-scoped) — the security-event + privacy-report trail export.
_ADMIN_CROSS_WORKSPACE: frozenset[str] = frozenset(
    {
        "/v1/audit/bundle",  # admin-only cross-workspace export (audit:read)
        "/v1/audit/events",  # admin-only security-event trail (audit:read)
    }
)

#: Process-global / project-local OPERATOR resources with NO tenant dimension: the same
#: for every workspace (model/provider catalog, infra health, connector catalog, the MCP
#: server registry, the project-local agent/team/skill/tool specs read off disk, the file
#: sandbox, benchmarks, the eval-history operator store, the seclog console). RBAC-gated,
#: never tenant-keyed, never echoes secret material. Each is the models/connectors class.
_GLOBAL_NOT_TENANT_KEYED: frozenset[str] = frozenset(
    {
        # /v1 process-global (already documented in the /v1 AST guard)
        "/v1/models",
        "/v1/diagnostics",
        "/v1/connectors",
        "/v1/connectors/{name}",
        # Studio operator/infra posture (no tenant dimension; never echoes secrets)
        "/api/studio/models",
        "/api/studio/models/ollama",
        "/api/studio/models/openrouter",
        "/api/studio/models/providers",
        "/api/studio/doctor",
        "/api/studio/mcp/servers",
        "/api/studio/mcp/servers/{name}/tools",
        "/api/studio/mcp/servers/{name}/agents",
        "/api/studio/connections",
        "/api/studio/connections/{ctype}",
        "/api/studio/telegram/status",
        "/api/studio/google",
        "/api/studio/google/auth-url",
        "/api/studio/google/callback",
        "/api/studio/google/calendar",
        "/api/studio/google/gmail",
        "/api/studio/benchmarks",
        "/api/studio/guardrails",
        # Studio project-local SPEC reads (off the filesystem, not tenant data objects)
        "/api/studio/agents",
        "/api/studio/agent",
        "/api/studio/teams",
        "/api/studio/teams/detail",
        "/api/studio/skills",
        "/api/studio/tools",
        "/api/studio/workflows",
        "/api/studio/evals",
        "/api/studio/eval/suites",
        "/api/studio/eval/baseline",
        "/api/studio/eval/history",
        "/api/studio/knowledge",
        "/api/studio/kb/{kb_id}/documents",
        "/api/studio/files",
        "/api/studio/files/download",
        "/api/studio/seclog",
    }
)

#: TIME-BOXED, REVIEWED gap (NOT a permanently-safe exemption). These Studio data surfaces
#: genuinely carry NO tenant/subject column today — the process-local missions/notify
#: registries (no ``workspace_id`` on the Mission dataclass / notification record), the
#: cross-tenant entity-spine governance/lineage reads (``registry.list_by_kind`` has no
#: tenant dimension), the subject-memory enumerator, and the cwd-keyed singleton SQLite
#: stores (tasks/chats/notes/calendar/cookbook/projects). The data-scoping MAP's
#: by-construction fix for each is a DATA-MODEL change (stamp the registry/record, give the
#: spine a tenant-aware query path, add a workspace column + migration). Until that lands,
#: each is RBAC-gated only and is a KNOWN cross-tenant disclosure for any tenant-bound
#: principal holding the read grant ONCE AUTH IS ON. Enumerating them here makes the gap a
#: VISIBLE, regression-protected decision — a NEW Studio read that should be scoped, but is
#: not and is not on this list, fails the gate — rather than the silent leak each prior
#: round shipped. Entries here must be DRAINED by stamping the underlying store, not grown.
_STUDIO_TENANT_PENDING: frozenset[str] = frozenset(
    {
        # process-local registries (no tenant stamp on the in-memory record)
        "/api/studio/notify",
        # entity-spine governance + lineage (registry.list_by_kind/get has no tenant axis)
        "/api/studio/privacy/subjects",
        "/api/studio/privacy/consents",
        "/api/studio/lineage/graph",
        "/api/studio/lineage/entity/{record_id}",
        # subject memory enumerator (OrchestrationStore.list_memory has no workspace_id)
        "/api/studio/memory",
        "/api/studio/memory/subjects",
        # cwd-keyed singleton SQLite stores (no tenant/subject column)
        "/api/studio/tasks",
        "/api/studio/chats",
        "/api/studio/chats/{session_id}",
        "/api/studio/notes",
        "/api/studio/notes/{note_id}",
        "/api/studio/calendar",
        "/api/studio/cookbook",
        "/api/studio/projects",
        "/api/studio/projects/{project_id}",
        # HITL approvals queue (checkpoint store, not tenant-stamped)
        "/api/studio/approvals",
        "/api/studio/approvals/{checkpoint_id}",
    }
)

_ALL_ALLOWED = (
    _PUBLIC
    | _AUTHZ_ONLY
    | _ADMIN_CROSS_WORKSPACE
    | _GLOBAL_NOT_TENANT_KEYED
    | _STUDIO_TENANT_PENDING
)


# --------------------------------------------------------------------------- the gate


def test_every_get_read_route_is_scoped_or_documented() -> None:
    """No GET read ships unscoped — across ``/v1`` AND ``/api/studio/*`` — uncovered.

    Every GET leaf either carries the ``scoped_read`` marker (its scope is derived from
    the verified principal) or is on one of the four reasoned allow-lists. A new unscoped
    read fails here, closing the blind spot the ``/v1``-only AST guard left for Studio.
    """
    routes = _get_read_routes()
    seen = {path for path, _route in routes}

    # Sanity: the gate genuinely sees BOTH a /v1 scoped surface AND the Studio family, so
    # a route-nesting change cannot make it vacuously pass (mirrors the RBAC gate's check).
    assert "/v1/runs" in seen
    assert any(p.startswith("/api/studio/") for p in seen)
    # A known scoped /v1 route is actually detected as scoped (the marker is wired).
    runs_routes = [r for p, r in routes if p == "/v1/runs"]
    assert runs_routes and all(_route_is_scoped(r) for r in runs_routes)

    offenders: list[str] = []
    for path, route in routes:
        if _route_is_scoped(route):
            continue
        if path in _ALL_ALLOWED:
            continue
        methods = sorted(getattr(route, "methods", None) or [])
        offenders.append(f"{methods} {path}")
    assert not offenders, (
        "these GET read routes carry no scoped_read marker and are not on any documented "
        "tenant-scope allow-list (thread the scope + add Depends(scoped_read), or add to "
        f"the right allow-list with a reason): {offenders}"
    )


def test_allow_lists_are_disjoint_and_have_no_stale_entries() -> None:
    """The allow-lists do not overlap and every entry maps to a real GET route.

    Overlap would hide a mis-classification; a stale entry (a path that no longer exists)
    would silently let a future route by the same name skip the gate. Both fail loudly.
    """
    lists = {
        "_PUBLIC": _PUBLIC,
        "_AUTHZ_ONLY": _AUTHZ_ONLY,
        "_ADMIN_CROSS_WORKSPACE": _ADMIN_CROSS_WORKSPACE,
        "_GLOBAL_NOT_TENANT_KEYED": _GLOBAL_NOT_TENANT_KEYED,
        "_STUDIO_TENANT_PENDING": _STUDIO_TENANT_PENDING,
    }
    # Pairwise disjoint.
    items = list(lists.items())
    for i, (name_a, set_a) in enumerate(items):
        for name_b, set_b in items[i + 1 :]:
            overlap = set_a & set_b
            assert not overlap, f"{name_a} and {name_b} overlap: {sorted(overlap)}"

    # No stale entries: every allow-listed path (except the docs paths, suppressed when
    # auth is configured, and which the default app does expose) is a real GET route.
    seen = {path for path, _route in _get_read_routes()}
    docs = {"/docs", "/redoc", "/openapi.json"}
    for name, paths in lists.items():
        stale = {p for p in paths if p not in seen and p not in docs}
        assert not stale, f"{name} has stale entries (no such GET route): {sorted(stale)}"


def test_studio_runs_surface_is_detected_as_scoped() -> None:
    """Regression: the Studio runs reads (the GOOD scoped Studio surface) are detected."""
    routes = {path: route for path, route in _get_read_routes()}
    for path in ("/api/studio/runs", "/api/studio/runs/{run_id}"):
        assert path in routes, f"expected {path} present"
        assert _route_is_scoped(routes[path]), f"{path} should be a scoped reader"


def test_studio_learning_surface_is_detected_as_scoped() -> None:
    """Regression: ``/api/studio/learning`` is detected as a scoped reader, not exempt.

    The learning report read scopes via ``_panel_workspace`` / ``resolve_workspace`` (a
    tenant only reads its own learned reputation), so it carries the ``scoped_read`` marker
    and must NOT sit on ``_GLOBAL_NOT_TENANT_KEYED`` (whose rationale — "no tenant
    dimension" — is false for this route). This pins the correct classification.
    """
    routes = {path: route for path, route in _get_read_routes()}
    assert "/api/studio/learning" in routes, "expected /api/studio/learning present"
    assert _route_is_scoped(routes["/api/studio/learning"]), (
        "/api/studio/learning is tenant-scoped (resolve_workspace) and must carry the "
        "scoped_read marker, not be exempted as global"
    )
    # And it is no longer hiding on the global-not-tenant-keyed allow-list.
    assert "/api/studio/learning" not in _GLOBAL_NOT_TENANT_KEYED


def test_gate_flags_a_deliberately_unscoped_studio_route() -> None:
    """The analyzer genuinely FAILS for a Studio read shipped without the scope marker.

    Builds a tiny synthetic app with two GET routes under ``/api/studio`` — one stamped
    ``Depends(scoped_read)``, one bare — and asserts the analyzer distinguishes them. This
    proves the gate would catch a real unscoped Studio read (it is not vacuously passing),
    AND that the runtime walk reaches a route nested under an included router prefix.
    """
    inner = APIRouter()

    @inner.get("/scoped", dependencies=[Depends(scoped_read)])
    async def _scoped() -> dict[str, str]:  # pragma: no cover - never called
        return {"ok": "yes"}

    @inner.get("/unscoped")
    async def _unscoped(request: Request) -> dict[str, str]:  # pragma: no cover
        return {"ok": "no"}

    app = FastAPI()
    app.include_router(inner, prefix="/api/studio/synthetic")

    # The runtime walk reaches both nested leaves (the property the /v1 AST guard lacks);
    # match on the leaf suffix since the include-prefix baking is FastAPI-version-specific.
    by_suffix = {
        path.rsplit("/", 1)[-1]: route
        for path, route in _iter_api_routes(app.routes)
        if getattr(route, "dependant", None) is not None
    }
    assert _route_is_scoped(by_suffix["scoped"])
    assert not _route_is_scoped(by_suffix["unscoped"])


def test_scoped_read_marker_is_a_byte_unchanged_no_op_offline() -> None:
    """The ``scoped_read`` stamp changes NO behaviour on the offline default path.

    With no authenticator configured (the zero-config single-box default) a stamped read
    route behaves exactly as an unstamped one: the marker dependency returns ``None`` and
    the scoping it advertises (``resolve_workspace`` etc.) is itself a no-op for the
    ANONYMOUS / ``all_tenants`` principal. So a stamped ``/v1`` read still returns its data
    under a caller-supplied workspace, byte-unchanged.
    """
    import anyio

    # The marker dependency is a pure no-op (returns None, touches nothing).
    assert anyio.run(scoped_read) is None

    # End-to-end: an offline app still serves the stamped /v1/runs list (no auth → no
    # narrowing), proving the stamp did not start enforcing anything offline.
    client = TestClient(create_app(ApiContainer.build_default()))
    resp = client.get("/v1/runs", params={"workspace_id": "anything"})
    assert resp.status_code == 200, resp.text
    assert "items" in resp.json()


def test_scoped_studio_runs_invisibility_end_to_end() -> None:
    """A tenant-bound principal cannot see another tenant's run via the SCOPED Studio read.

    Ties the ``scoped_read`` marker to the property it advertises: ``/api/studio/runs`` is
    stamped scoped, and a run stamped to ``other`` is invisible (404 by-id, absent from the
    list) to a principal bound to tenant ``t`` — even though it holds ``studio.runs:read``.
    The marker is therefore not a hollow label: the surface it stamps genuinely refuses
    out-of-tenant rows.
    """
    import anyio

    from himmy.api.auth.rbac import AccessPolicy
    from himmy.services.storage.models import RunRecord, RunStatus

    container = ApiContainer.build_default()
    app = create_app(container)
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=["reader"], auth_method="apikey"
            )
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping(
        {"reader": ["studio.console:read", "studio.runs:read"]}
    )

    other = RunRecord(
        run_id="other-tenant-run",
        workspace_id="other",
        subject_id="s-other",
        status=RunStatus.SUCCEEDED,
    )
    anyio.run(container.storage.save_run, other)

    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    # By-id: foreign run is a uniform 404 (existence never leaked).
    assert client.get("/api/studio/runs/other-tenant-run").status_code == 404
    # List: the foreign run never appears.
    listed = client.get("/api/studio/runs").json()
    assert all(
        item.get("run_id") != "other-tenant-run" for item in listed.get("items", [])
    )
