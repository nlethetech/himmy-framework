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


def _route_is_subject_write_scoped(route: Any) -> bool:
    """Whether ``route``'s dependency graph carries the ``subject_write`` marker.

    The write-path companion to :func:`_route_is_scoped`. True iff some callable in the
    route's resolved dependant tree was stamped with the ``_subject_write`` attribute —
    set on :func:`himmy.api.auth.subject_write` — asserting the mutation gates its target
    ``subject_id`` against the verified principal (``enforce_subject_write`` /
    ``authorize_object``) so a ``subject_scoped`` caller may write only under its OWN
    subject. Catches both a route-level ``Depends(subject_write)`` and a router-level
    baseline, at any nesting depth.
    """
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return False
    calls: list[Any] = []
    _walk_dependency_calls(dependant, calls)
    return any(getattr(call, "_subject_write", None) is not None for call in calls)


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


#: POST/PUT/PATCH leaves that PERFORM A READ rather than (or as well as) a mutation —
#: the structural blind spot the GET-only gate above could not see. A read-shaped POST
#: (semantic recall, retrieval search) takes a query + a target selector (``subject_id`` /
#: ``kb_id``) in its BODY and returns rows; if that selector is honored over the verified
#: principal it is a cross-subject/cross-tenant BOLA the by-construction gate must catch.
#: Because there is no generic, body-shape-independent way to tell a read-POST from a pure
#: mutation by AST, these are enumerated by review: each MUST carry the ``scoped_read``
#: marker (its target derived from the principal) or sit on a documented allow-list. A new
#: read-shaped non-GET route should be ADDED here so the gate guards it too.
_NON_GET_READS: frozenset[str] = frozenset(
    {
        "/api/studio/memory/recall",  # POST recall: subject_id from body must not be honored
        "/api/studio/knowledge/{kb_id}/search",  # POST KB search (project-local, not tenant)
        # POST audit export/verify are READS over the governance evidence set: a tenant-bound
        # admin must NOT walk away with a signed bundle of every tenant's (or, subject_scoped,
        # every subject's) consent/erasure/security records (scope-r7). They route the
        # evidence set through _visible_records (tenant + subject axis) and carry scoped_read.
        "/api/studio/privacy/audit/export",
        "/api/studio/privacy/audit/verify",
    }
)


#: Path prefixes whose data store is keyed by ``subject_id`` (the BOLA axis), so EVERY
#: mutation under them (``POST`` / ``PUT`` / ``PATCH`` / ``DELETE``) must gate the target
#: subject against the verified principal. The memory store (``himmy.services.memory``) is
#: keyed by ``subject_id`` only — the subject axis is the sole isolation boundary — so an
#: ungated write is a cross-subject poisoning BOLA. The GET gate and the read-shaped-POST
#: gate are structurally BLIND to a ``PATCH`` / ``DELETE`` write (that is exactly how the
#: round-3 ``memory_edit`` in-place edit shipped ungated while ``memory_add`` /
#: ``memory_forget`` were closed). The subject-write gate below enumerates these write
#: leaves and requires each to carry the ``subject_write`` marker (its target gated via
#: ``enforce_subject_write`` / ``authorize_object``).
_SUBJECT_KEYED_WRITE_PREFIXES: tuple[str, ...] = ("/api/studio/memory",)

#: Subject-keyed write leaves that legitimately need NO per-subject gate — e.g. a bulk
#: admin-only purge already locked behind an elevated RBAC grant, or a write whose body
#: carries no subject selector. Empty today: every memory write is subject-gated. A NEW
#: subject-keyed write that cannot carry the marker must be added here WITH A REASON, so
#: the omission is a visible, reviewed decision rather than a silent BOLA.
_SUBJECT_WRITE_EXEMPT: frozenset[str] = frozenset()


def _subject_keyed_write_routes() -> list[tuple[str, Any]]:
    """Every non-GET (write) leaf under a subject-keyed surface (:data:`_SUBJECT_KEYED_WRITE_PREFIXES`).

    Mirrors :func:`_non_get_read_routes` but for the WRITE axis the read gates never see:
    ``POST`` / ``PUT`` / ``PATCH`` / ``DELETE`` leaves under a ``subject_id``-keyed store.
    Each must carry the ``subject_write`` marker so a ``subject_scoped`` caller cannot
    mutate another data subject's object — the recurrence class ``memory_edit`` exposed.
    """
    app = create_app()
    out: list[tuple[str, Any]] = []
    for path, route in _iter_api_routes(app.routes):
        if getattr(route, "dependant", None) is None:
            continue
        methods = getattr(route, "methods", None) or set()
        if "GET" in methods:
            continue
        if not any(path.startswith(p) for p in _SUBJECT_KEYED_WRITE_PREFIXES):
            continue
        # A read-shaped POST (e.g. semantic recall) is a READ, not a mutation — it is
        # guarded by the scoped_read gate (:data:`_NON_GET_READS`), not the write gate.
        if path in _NON_GET_READS:
            continue
        out.append((path, route))
    return out


def _non_get_read_routes() -> list[tuple[str, Any]]:
    """The reviewed read-shaped POST/PUT/PATCH leaves (:data:`_NON_GET_READS`).

    The GET-only :func:`_get_read_routes` is structurally blind to a read performed over
    a POST (e.g. ``POST /api/studio/memory/recall``), so a future cross-subject recall
    route could ship unguarded. This enumerates the reviewed read-shaped non-GET leaves
    on the built app so the coverage gate can require each to be scoped-or-documented too.
    """
    app = create_app()
    out: list[tuple[str, Any]] = []
    for path, route in _iter_api_routes(app.routes):
        if getattr(route, "dependant", None) is None:
            continue
        methods = getattr(route, "methods", None) or set()
        if "GET" in methods:
            continue
        if path in _NON_GET_READS:
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
#: permission, not data-scoped) — the security-event + privacy-report trail export, plus the
#: agent file sandbox (a single process-global ``fs_root`` with NO tenant dimension that
#: agent ``write_file`` outputs land in). The sandbox served tenant-PRODUCED data with no
#: per-tenant partition, so it was a cross-tenant BOLA when classified as "no tenant
#: dimension" below; it is now restricted to an UNSCOPED operator/offline (``all_tenants``)
#: principal (``_require_unscoped_sandbox_reader``), making the cross-workspace exposure an
#: EXPLICIT reviewed decision (mirroring ``reveal_host_posture``) rather than a hidden one.
_ADMIN_CROSS_WORKSPACE: frozenset[str] = frozenset(
    {
        "/v1/audit/bundle",  # admin-only cross-workspace export (audit:read)
        "/v1/audit/events",  # admin-only security-event trail (audit:read)
        "/api/studio/files",  # shared agent sandbox; all_tenants-only (operator)
        "/api/studio/files/download",  # shared agent sandbox; all_tenants-only (operator)
    }
)

#: Process-global / project-local OPERATOR resources with NO tenant dimension: the same
#: for every workspace (model/provider catalog, infra health, connector catalog, the MCP
#: server registry, the project-local agent/team/skill/tool specs read off disk,
#: benchmarks, the eval-history operator store). RBAC-gated, never tenant-keyed,
#: never echoes secret material. Each is the models/connectors class. (The seclog console is
#: NOT here: its events carry a ``workspace_id`` and it now tenant-scopes via ``scoped_read``.)
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
        "/api/studio/knowledge/{kb_id}/search",
        "/api/studio/kb/{kb_id}/documents",
    }
)

#: FULLY DRAINED. Every Studio singleton/shared store now carries a ``workspace_id`` stamp
#: and scopes every read/list/mutate through the principal-derived ``studio_tenant_filter`` /
#: ``studio_write_workspace`` — the singleton SQLite stores (tasks/notes/calendar/cookbook),
#: the notify ring, the HITL approvals queue, AND the unified conversation store (chats +
#: projects). The conversation store's tenant-column migration (additive + nullable on both
#: the SQLite ``ConversationStore`` and its Postgres mirror) keeps the offline CLI ``himmy
#: chat`` / ``/resume`` session path byte-unchanged: the CLI writer passes no ``workspace_id``
#: so its rows are ``NULL`` (visible to all, never re-stamped). This list is now EMPTY — any
#: NEW Studio read that should be tenant-scoped but is not fails the coverage gate.
_STUDIO_TENANT_PENDING: frozenset[str] = frozenset()

#: Studio memory reads that DO enforce the SUBJECT axis (``narrow_subject`` →
#: ``scoped_read``, the round-7 cross-subject-recall fix) but whose store
#: (``himmy/api/studio_memory.py`` → the process-wide memory DB) carries NO
#: ``workspace_id`` column, so the TENANT axis is structurally unenforceable on this
#: surface today. The ``scoped_read`` marker advertises tenant(+subject) scope; for these
#: three routes only the subject half is real, so without this list the tenant gap would be
#: INVISIBLE — fully hidden behind the marker and never drained like its siblings on
#: :data:`_STUDIO_TENANT_PENDING`. Enumerating them here makes the residual tenant gap a
#: VISIBLE, regression-protected decision: the same KNOWN cross-tenant disclosure the
#: process-global memory store already accepts (mirroring the tasks/chats/notes stores),
#: now tracked so a future workspace-column migration drains it. DRAIN by stamping the
#: memory store with a tenant column + intersecting every read, not by growing this list.
#: DRAINED. The Studio memory reads now enforce BOTH axes. The SUBJECT axis is the round-7
#: ``narrow_subject`` fix; the TENANT axis is closed WITHOUT a store-schema change by
#: NAMESPACING the subject the routes read/write/recall with the principal's resolved
#: workspace (``t:<workspace>:<subject>`` via ``_memory_tenant_prefix``) — the SAME scheme the
#: tool-layer memory pack uses (:meth:`ToolkitConfig.scoped_memory_subject`). So a tenant-bound
#: Studio console and that tenant's agents share one namespace and NO other tenant's, making a
#: cross-tenant recall structurally impossible on the shared process store. This set is EMPTY.
_STUDIO_MEMORY_TENANT_PENDING: frozenset[str] = frozenset()

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


def test_every_non_get_read_route_is_scoped_or_documented() -> None:
    """The GET-only blind spot is closed: read-shaped POSTs are gated too.

    ``POST /api/studio/memory/recall`` is a READ (semantic recall over a body-supplied
    ``subject_id``) that the GET-only gate above cannot see — a tenant/subject-bound
    principal could otherwise recall ANOTHER data subject's memories. Every reviewed
    read-shaped non-GET leaf (:data:`_NON_GET_READS`) must EITHER carry the ``scoped_read``
    marker (its target derived from the verified principal, never the body) OR sit on a
    documented allow-list — exactly the by-construction property the GET gate enforces.
    """
    routes = _non_get_read_routes()
    seen = {path for path, _route in routes}
    # The gate genuinely reaches the read-shaped POSTs (not vacuously empty).
    assert "/api/studio/memory/recall" in seen, (
        "expected the read-shaped POST /api/studio/memory/recall to be enumerated; a "
        "FastAPI nesting change must not make this gate silently empty"
    )
    # Every enumerated entry corresponds to a real route on the built app.
    assert seen == set(_NON_GET_READS), (
        "_NON_GET_READS has an entry that is no longer a real non-GET route (or vice "
        f"versa); reconcile: built={sorted(seen)} declared={sorted(_NON_GET_READS)}"
    )

    offenders: list[str] = []
    for path, route in routes:
        if _route_is_scoped(route):
            continue
        if path in _ALL_ALLOWED:
            continue
        methods = sorted(getattr(route, "methods", None) or [])
        offenders.append(f"{methods} {path}")
    assert not offenders, (
        "these read-shaped non-GET routes neither carry the scoped_read marker nor sit on "
        f"a documented tenant-scope allow-list: {offenders}"
    )


def test_memory_recall_post_read_is_detected_as_scoped() -> None:
    """Regression: the cross-subject recall fix carries the scoped_read marker.

    ``POST /api/studio/memory/recall`` derives its subject from the verified principal
    (``narrow_subject``), so it must be detected as a scoped reader — proving the
    by-construction gate now guards this read-shaped POST.
    """
    routes = {path: route for path, route in _non_get_read_routes()}
    assert "/api/studio/memory/recall" in routes
    assert _route_is_scoped(routes["/api/studio/memory/recall"]), (
        "/api/studio/memory/recall is subject-scoped (narrow_subject) and must carry the "
        "scoped_read marker so the non-GET gate enforces it"
    )


def test_studio_memory_reads_are_subject_and_tenant_scoped() -> None:
    """The Studio memory reads enforce BOTH the subject AND the tenant axis (gap CLOSED).

    The three memory read routes carry ``scoped_read`` (the subject axis via
    ``narrow_subject``) AND now namespace their effective subject by the principal's resolved
    workspace (``_memory_tenant_prefix`` → ``t:<workspace>:<subject>``), closing the former
    tenant gap on the process-global store WITHOUT a schema change. The pending set is now
    empty; this asserts the routes are still detected as scoped (the marker is wired) and
    that nothing remains pending.
    """
    get_routes = {path: route for path, route in _get_read_routes()}
    nonget_routes = {path: route for path, route in _non_get_read_routes()}
    for path in (
        "/api/studio/memory",
        "/api/studio/memory/subjects",
        "/api/studio/memory/recall",
    ):
        route = get_routes.get(path) or nonget_routes.get(path)
        assert route is not None, f"{path} is no longer a real memory read route"
        assert _route_is_scoped(route), (
            f"{path} must keep its scoped_read marker (subject + tenant axes)"
        )
    assert _STUDIO_MEMORY_TENANT_PENDING == frozenset()


def test_every_subject_keyed_write_route_is_subject_scoped() -> None:
    """No write on a ``subject_id``-keyed store ships WITHOUT a per-subject gate.

    The read gates (:func:`test_every_get_read_route_is_scoped_or_documented` and the
    read-shaped-POST gate) are structurally blind to a ``PATCH`` / ``DELETE`` mutation —
    which is exactly how the round-3 ``PATCH /api/studio/memory/{id}`` in-place edit shipped
    able to rewrite ANOTHER data subject's memory while the sibling ``DELETE`` and ``POST``
    were gated. This gate closes that recurrence class by construction: every non-GET leaf
    under a subject-keyed surface (:data:`_SUBJECT_KEYED_WRITE_PREFIXES`) must carry the
    ``subject_write`` marker (asserting it gates the target ``subject_id`` via
    ``enforce_subject_write`` / ``authorize_object``) OR sit on the reasoned
    :data:`_SUBJECT_WRITE_EXEMPT` list. A future ungated subject write fails the build.
    """
    routes = _subject_keyed_write_routes()
    seen = {path for path, _route in routes}
    # The gate genuinely reaches the memory writes (not vacuously empty after a nesting
    # change): the add (POST), forget (DELETE) and the round-3 edit (PATCH) are all present.
    for path in (
        "/api/studio/memory",
        "/api/studio/memory/{memory_id}",
    ):
        assert path in seen, (
            f"expected subject-keyed write route {path} to be enumerated; a FastAPI "
            "nesting change must not make this gate silently empty"
        )

    offenders: list[str] = []
    for path, route in routes:
        if _route_is_subject_write_scoped(route):
            continue
        if path in _SUBJECT_WRITE_EXEMPT:
            continue
        methods = sorted(getattr(route, "methods", None) or [])
        offenders.append(f"{methods} {path}")
    assert not offenders, (
        "these writes on a subject-keyed store carry no subject_write marker and are not on "
        "_SUBJECT_WRITE_EXEMPT (gate the target subject via enforce_subject_write / "
        f"authorize_object and add Depends(subject_write), or document an exemption): {offenders}"
    )


def test_memory_edit_patch_is_subject_write_scoped() -> None:
    """Regression (round-3 BOLA): the in-place memory edit gates the target subject.

    ``PATCH /api/studio/memory/{memory_id}`` rewrites a memory's text in place. Before the
    round-3 fix it called ``service.store.save`` with ZERO object-axis check, so a
    ``subject_scoped`` caller holding ``studio.memory:write`` could rewrite another data
    subject's memory by id. It now folds a cross-subject edit to 404 (``authorize_object``)
    and carries the ``subject_write`` marker so the coverage gate enforces it by construction.
    """
    routes = {path: route for path, route in _subject_keyed_write_routes()}
    edit = [
        route
        for path, route in routes.items()
        if path == "/api/studio/memory/{memory_id}"
        and "PATCH" in (getattr(route, "methods", None) or set())
    ]
    assert edit, "expected PATCH /api/studio/memory/{memory_id} present"
    assert all(_route_is_subject_write_scoped(r) for r in edit), (
        "PATCH /api/studio/memory/{memory_id} is subject-keyed (authorize_object) and must "
        "carry the subject_write marker so the subject-write gate enforces it"
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
    # auth is configured, and which the default app does expose) is a real read route —
    # GET or one of the reviewed read-shaped non-GET leaves (e.g. POST .../search).
    seen = {path for path, _route in _get_read_routes()}
    seen |= {path for path, _route in _non_get_read_routes()}
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
