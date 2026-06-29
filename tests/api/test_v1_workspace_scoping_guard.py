"""T1.2 — the workspace-scoping recurrence guard for the /v1 REST surface.

Two complementary defenses against re-introducing the T1.1 class of cross-tenant
IDOR (a /v1 read that forgets to scope by the resolved ``workspace_id``):

* A **static AST check** that walks every ``/v1``-prefixed router and asserts each
  GET handler either THREADS the resolved workspace into its data read (calls
  ``resolve_workspace``/``require_workspace`` AND passes ``workspace_id=`` down into
  a downstream call), or is in a small, *documented* exemption set — handlers that
  legitimately resolve-then-discard for authorization only (their resource is keyed
  by subject, not tenant) or perform an admin-only cross-workspace export. The
  analyzer is self-tested against a deliberately-unscoped synthetic route, so it
  genuinely WOULD fail if a future /v1 read dropped its scoping.
* A **runtime cross-tenant invisibility sweep** over every /v1 read surface: with
  tenant-bound API keys, data created in ``tenant-a`` is invisible (404/empty) to a
  ``tenant-b`` caller — the property the static check defends statically.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal

# --------------------------------------------------------------------------- AST

#: The directory holding the BFF routers.
_ROUTERS_DIR = Path(__file__).resolve().parents[2] / "himmy" / "api" / "routers"

#: Functions that, when called, resolve/authorize the request's workspace.
_RESOLVE_FUNCS = frozenset({"resolve_workspace", "require_workspace"})

#: Handlers that legitimately call ``resolve_workspace`` for AUTHORIZATION ONLY and
#: do not thread the result into a data read, keyed by ``(module, handler)``. Each
#: entry states WHY it is exempt — the resource it reads is keyed by ``subject_id``
#: (not tenant) with self-scope enforcement, so resolve_workspace runs purely to
#: authorize the caller's tenant claim and its result is intentionally unused. These
#: MUST still call resolve_workspace. New entries are a deliberate, reviewed decision.
_AUTHZ_ONLY_READS: dict[tuple[str, str], str] = {
    # consent decisions/records are keyed by (subject_id, purpose), not tenant; the
    # router additionally enforces _enforce_self_scope.
    ("consent", "get_decision"): "subject-keyed; self-scope enforced",
    ("consent", "get_latest"): "subject-keyed; self-scope enforced",
    ("consent", "get_history"): "subject-keyed; self-scope enforced",
    # privacy audit reports are a global posture trend, not tenant-keyed;
    # resolve_workspace enforces access (auditor/admin), not row filtering.
    ("privacy_audit", "list_privacy_audits"): "reports not tenant-keyed; authz only",
    ("privacy_audit", "get_privacy_audit"): "reports not tenant-keyed; authz only",
}

#: Admin-only reads that DELIBERATELY span all workspaces and therefore do not (and
#: must not) scope to a single tenant — gated by an elevated permission instead.
_ADMIN_CROSS_WORKSPACE_READS: dict[tuple[str, str], str] = {
    # the signed audit bundle is an admin-only export of the whole security-event +
    # privacy-report trail (gated by audit:read); cross-workspace by design.
    ("audit", "export_audit_bundle_route"): "admin-only cross-workspace export",
}

#: Reads of a PROCESS-GLOBAL operator resource that has no tenant dimension at all —
#: the data is the same for every workspace, so there is nothing to scope (and no
#: ``resolve_workspace`` to call). Gated by an RBAC permission instead. Each entry
#: states WHY the resource is not tenant-keyed. New entries are a reviewed decision.
_GLOBAL_NOT_TENANT_KEYED: dict[tuple[str, str], str] = {
    # the connector catalog is a process-level operator resource (like /v1/models):
    # the built-in connectors + their configured/enabled state are the same for every
    # workspace; read-only + RBAC-gated on connector:read, never echoes a secret.
    ("connectors", "list_connectors"): "process-global connector catalog; authz only",
    ("connectors", "get_connector"): "process-global connector catalog; authz only",
    # the model catalog is a process-level infra fact (which providers/models this host
    # can run): identical for every workspace; read-only + RBAC-gated on model:read,
    # carries no secret material. (T3d)
    ("models", "list_models"): "process-global model catalog; authz only",
    # global diagnostics (doctor + storage/spine/scheduler health) is infra, not
    # per-tenant: a per-tenant doctor is not meaningful; RBAC-gated on diagnostics:read,
    # secrets redacted. (T3d)
    ("diagnostics", "diagnostics"): "process-global infra health; authz only",
}


class _HandlerScope(ast.NodeVisitor):
    """Collects, for one handler body, whether it resolves + threads the workspace."""

    def __init__(self) -> None:
        self.calls_resolve = False
        self.threads_workspace = False

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast visitor name
        func_name = _called_name(node.func)
        if func_name in _RESOLVE_FUNCS:
            self.calls_resolve = True
        else:
            # A workspace_id=... keyword passed into ANY non-resolver call means the
            # resolved tenant is threaded into a downstream (data) read/write.
            for kw in node.keywords:
                if kw.arg == "workspace_id":
                    self.threads_workspace = True
        self.generic_visit(node)


def _called_name(func: ast.expr) -> str | None:
    """The bare callable name for ``f(...)`` or ``a.b.f(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_get_route(decorator: ast.expr) -> bool:
    """Whether a decorator is ``@router.get(...)``."""
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "get"
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "router"
    )


def _router_prefix(tree: ast.Module) -> str | None:
    """Extract the ``APIRouter(prefix=...)`` literal from a router module, if any."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "router" for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            for kw in node.value.keywords:
                if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
    return None


def _v1_get_handlers() -> list[tuple[str, str, ast.AsyncFunctionDef | ast.FunctionDef]]:
    """Every GET handler on a ``/v1``-prefixed router, as (module, name, node)."""
    found: list[tuple[str, str, ast.AsyncFunctionDef | ast.FunctionDef]] = []
    for path in sorted(_ROUTERS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        prefix = _router_prefix(tree)
        if prefix is None or not prefix.startswith("/v1"):
            continue
        module = path.stem
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and any(
                _is_get_route(dec) for dec in node.decorator_list
            ):
                found.append((module, node.name, node))
    return found


def _scope_of(node: ast.AST) -> _HandlerScope:
    scope = _HandlerScope()
    scope.visit(node)
    return scope


def test_every_v1_get_handler_threads_or_documents_its_workspace_scope() -> None:
    """No /v1 GET reads a tenant resource without scoping (or a documented authz-only)."""
    handlers = _v1_get_handlers()
    # Sanity: the known /v1 read surfaces are present, so we are actually checking them.
    seen_modules = {module for module, _name, _node in handlers}
    assert {"evaluation", "runs", "recommendations", "context"} <= seen_modules

    offenders: list[str] = []
    for module, name, node in handlers:
        scope = _scope_of(node)
        if scope.threads_workspace:
            continue  # threads the resolved workspace into a data read — compliant
        key = (module, name)
        if key in _AUTHZ_ONLY_READS:
            # Documented authz-only read: it MUST still resolve the workspace.
            assert scope.calls_resolve, (
                f"{module}.{name} is marked authz-only but never calls "
                f"resolve_workspace/require_workspace"
            )
            continue
        if key in _ADMIN_CROSS_WORKSPACE_READS:
            # Documented admin-only cross-workspace export: scoping is intentionally
            # absent (an elevated permission gates it instead).
            continue
        if key in _GLOBAL_NOT_TENANT_KEYED:
            # Documented process-global resource: it has no tenant dimension, so there
            # is nothing to scope (an RBAC permission gates it instead).
            continue
        offenders.append(f"{module}.{name}")
    assert not offenders, (
        "these /v1 GET handlers read without threading the resolved workspace and "
        f"are not documented as authz-only: {offenders}"
    )


def test_guard_flags_a_deliberately_unscoped_route() -> None:
    """The analyzer genuinely fails for a route that forgets to scope (recurrence guard)."""
    unscoped = ast.parse(
        "async def get_thing(thing_id, request):\n"
        "    storage = request.app.state.container.storage\n"
        "    return await storage.get_thing(thing_id)\n"
    ).body[0]
    scope = _scope_of(unscoped)
    assert not scope.threads_workspace
    assert not scope.calls_resolve

    scoped = ast.parse(
        "async def get_thing(thing_id, request, workspace_id=None):\n"
        "    workspace_id = resolve_workspace(request, workspace_id)\n"
        "    storage = request.app.state.container.storage\n"
        "    return await storage.get_thing(thing_id, workspace_id=workspace_id)\n"
    ).body[0]
    scope2 = _scope_of(scoped)
    assert scope2.threads_workspace
    assert scope2.calls_resolve


def test_evaluation_get_handler_is_scoped_post_fix() -> None:
    """Regression: the T1.1 eval handler now threads its workspace (was the IDOR)."""
    handlers = {
        (module, name): node for module, name, node in _v1_get_handlers()
    }
    node = handlers[("evaluation", "get_evaluation_run")]
    scope = _scope_of(node)
    assert scope.calls_resolve
    assert scope.threads_workspace


# ----------------------------------------------------------------------- runtime


def _app_with_mapped_keys() -> TestClient:
    """An app binding two API keys to two tenants (admin role: every read surface)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-a": Principal.build(
                "user-a",
                tenant_ids=["tenant-a"],
                roles=["admin"],
                auth_method="apikey",
            ),
            "key-b": Principal.build(
                "user-b",
                tenant_ids=["tenant-b"],
                roles=["admin"],
                auth_method="apikey",
            ),
        }
    )
    return TestClient(app)


def _client_as(key: str) -> TestClient:
    client = _app_with_mapped_keys()
    client.headers.update({"x-himmy-internal-key": key})
    return client


def _create_run(client: TestClient, workspace: str) -> str:
    body = {
        "workspace_id": workspace,
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    resp = client.post("/v1/runs", json=body)
    assert resp.status_code == 200, resp.text
    run_id = str(resp.json()["run_id"])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/runs/{run_id}").json().get("status") in (
            "SUCCEEDED",
            "FAILED",
        ):
            break
        time.sleep(0.02)
    return run_id


def _create_eval_run(client: TestClient, workspace: str) -> str:
    body = {
        "workspace_id": workspace,
        "suite": {
            "name": "qa",
            "cases": [
                {
                    "case_id": "c1",
                    "expected_output": {"answer": "yes"},
                    "metric_weights": {"accuracy": 1.0},
                }
            ],
        },
        "actual_outputs": {"c1": {"answer": "yes"}},
    }
    resp = client.post("/v1/evaluation/runs", json=body)
    assert resp.status_code == 200, resp.text
    return str(resp.json()["run_id"])


def _app_with_operator_keys() -> object:
    """An app binding two operator-role API keys to two tenants (attacker class).

    ``operator`` holds ``context:read`` + ``context:write`` (the perms the
    ``/v1/context`` field-upsert + snapshot-build routes require) and is NOT
    ``subject_scoped``, so ``enforce_subject_write``/``authorize_object`` are no-ops —
    exactly the principal in the cross-tenant context-field IDOR PoC. Returns the app
    (not a client) so both tenants share ONE backing store.
    """
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "op-a": Principal.build(
                "user-a",
                tenant_ids=["tenant-a"],
                roles=["operator"],
                auth_method="apikey",
            ),
            "op-b": Principal.build(
                "user-b",
                tenant_ids=["tenant-b"],
                roles=["operator"],
                auth_method="apikey",
            ),
        }
    )
    return app


def test_cross_tenant_context_field_not_leaked_via_snapshot_build() -> None:
    """red-team reattack-r1: snapshots:build must not resolve another tenant's field.

    ``context_fields`` are keyed globally by ``(subject_id, key)`` with no workspace
    column, and ``subject_id`` is a free-form attacker-chosen string. Tenant B writes
    a secret field; tenant A builds a snapshot for the SAME ``subject_id`` requesting
    that key. Before the fix, field RESOLUTION ignored the caller's workspace and
    returned tenant B's value (cross-tenant IDOR); after the fix the build path
    tenant-scopes resolution, so the foreign field is absent and lands in
    ``missing_required_keys`` instead of leaking.
    """
    app = _app_with_operator_keys()

    client_b = TestClient(app)
    client_b.headers.update({"x-himmy-internal-key": "op-b"})
    upsert = client_b.post(
        "/v1/context/fields:upsert",
        json={
            "workspace_id": "tenant-b",
            "subject_id": "alice@corp.com",
            "fields": [{"key": "ssn", "value": "123-45-6789"}],
        },
    )
    assert upsert.status_code == 200, upsert.text

    client_a = TestClient(app)
    client_a.headers.update({"x-himmy-internal-key": "op-a"})
    snap = client_a.post(
        "/v1/context/snapshots:build",
        json={
            "workspace_id": "tenant-a",
            "subject_id": "alice@corp.com",
            "build_spec": {"keys": [{"key": "ssn", "required": True}]},
        },
    )
    assert snap.status_code == 200, snap.text
    body = snap.json()
    # The foreign tenant's value must NOT be resolved into the snapshot.
    assert "ssn" not in body["fields"], body["fields"]
    assert "ssn" in body["missing_required_keys"]
    # And the persistent re-read path must not surface it either.
    snapshot_id = body["snapshot_id"]
    reread = client_a.get(
        f"/v1/context/snapshots/{snapshot_id}", params={"workspace_id": "tenant-a"}
    ).json()
    assert "ssn" not in reread.get("fields", {})


def test_same_tenant_context_field_still_resolves_in_snapshot_build() -> None:
    """The workspace-scoped build resolution still surfaces the caller's OWN field."""
    app = _app_with_operator_keys()
    client_a = TestClient(app)
    client_a.headers.update({"x-himmy-internal-key": "op-a"})

    up = client_a.post(
        "/v1/context/fields:upsert",
        json={
            "workspace_id": "tenant-a",
            "subject_id": "bob@corp.com",
            "fields": [{"key": "tier", "value": "gold"}],
        },
    )
    assert up.status_code == 200, up.text

    snap = client_a.post(
        "/v1/context/snapshots:build",
        json={
            "workspace_id": "tenant-a",
            "subject_id": "bob@corp.com",
            "build_spec": {"keys": [{"key": "tier", "required": True}]},
        },
    )
    assert snap.status_code == 200, snap.text
    body = snap.json()
    assert body["fields"].get("tier", {}).get("value") == "gold"
    assert "tier" not in body["missing_required_keys"]


def test_cross_tenant_eval_run_is_invisible() -> None:
    """T1.1 acceptance: a cross-workspace GET of an eval run is 404 (was a leak)."""
    client_a = _client_as("key-a")
    eval_id = _create_eval_run(client_a, "tenant-a")

    # Owner reads it under its own tenant.
    assert (
        client_a.get(
            f"/v1/evaluation/runs/{eval_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 200
    )

    # A genuinely different tenant cannot reach it: asking for tenant-a is 403
    # (authorization), and reading under its own tenant simply 404s (not a leak).
    client_b = _client_as("key-b")
    assert (
        client_b.get(
            f"/v1/evaluation/runs/{eval_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 403
    )
    assert (
        client_b.get(
            f"/v1/evaluation/runs/{eval_id}", params={"workspace_id": "tenant-b"}
        ).status_code
        == 404
    )
    # And B's list under its own tenant never surfaces A's eval run.
    listed_b = client_b.get(
        "/v1/evaluation/runs", params={"workspace_id": "tenant-b"}
    ).json()
    assert all(r["run_id"] != eval_id for r in listed_b)


def test_eval_run_stamps_workspace_and_actor() -> None:
    """The created eval run carries its tenant + the creating actor (audit spine)."""
    client_a = _client_as("key-a")
    eval_id = _create_eval_run(client_a, "tenant-a")
    run = client_a.get(
        f"/v1/evaluation/runs/{eval_id}", params={"workspace_id": "tenant-a"}
    ).json()
    assert run["workspace_id"] == "tenant-a"
    actor = run["metadata"].get("actor")
    assert actor and actor["subject"] == "user-a"
    assert actor["auth_method"] == "apikey"


def test_cross_tenant_invisibility_sweep_over_v1_read_surfaces() -> None:
    """Every /v1 read surface: tenant-a's data is invisible (404/empty) to tenant-b."""
    client_a = _client_as("key-a")

    # Seed one resource on each tenant-scoped read surface inside tenant-a.
    run_id = _create_run(client_a, "tenant-a")
    eval_id = _create_eval_run(client_a, "tenant-a")
    snapshot = client_a.post(
        "/v1/context/snapshots:build",
        json={
            "workspace_id": "tenant-a",
            "subject_id": "s1",
            "task_id": "task-1",
            "build_spec": {},
        },
    )
    assert snapshot.status_code == 200, snapshot.text
    snapshot_id = snapshot.json()["snapshot_id"]

    client_b = _client_as("key-b")
    ws_b = {"workspace_id": "tenant-b"}

    # Run by id + its sub-reads: 404 / empty under tenant-b.
    assert client_b.get(f"/v1/runs/{run_id}", params=ws_b).status_code == 404
    assert client_b.get(f"/v1/runs/{run_id}/events", params=ws_b).json() == []
    assert (
        client_b.get(f"/v1/runs/{run_id}/thread", params=ws_b).status_code == 404
    )
    assert (
        client_b.get(f"/v1/runs/{run_id}/lineage", params=ws_b).status_code == 404
    )
    listed_runs_b = client_b.get("/v1/runs", params=ws_b).json()
    assert all(it["run_id"] != run_id for it in listed_runs_b["items"])

    # Evaluation run by id + list.
    assert (
        client_b.get(f"/v1/evaluation/runs/{eval_id}", params=ws_b).status_code == 404
    )
    assert all(
        r["run_id"] != eval_id
        for r in client_b.get("/v1/evaluation/runs", params=ws_b).json()
    )

    # Context snapshot by id + fields list.
    assert (
        client_b.get(
            f"/v1/context/snapshots/{snapshot_id}", params=ws_b
        ).status_code
        == 404
    )
    fields_b = client_b.get(
        "/v1/context/fields", params={"subject_id": "s1", **ws_b}
    ).json()
    assert isinstance(fields_b, list)

    # Recommendations list is pinned to tenant-b (never surfaces tenant-a rows).
    recs_b = client_b.get("/v1/recommendations", params=ws_b).json()
    assert isinstance(recs_b["items"], list)

    # Dashboard tile: tenant-b's eval total excludes tenant-a's stamped run.
    summary_b = client_b.get(
        "/v1/dashboard/summary",
        params={"subject_id": "s1", **ws_b},
    ).json()
    # The only stamped eval run belongs to tenant-a, so tenant-b sees none.
    assert summary_b["evaluation"]["total"] == 0
    assert summary_b["runs"]["total"] == 0


def test_dashboard_eval_tile_does_not_leak_cross_tenant() -> None:
    """The dashboard eval tile shows a tenant only its own (+ unstamped) runs."""
    client_a = _client_as("key-a")
    _create_eval_run(client_a, "tenant-a")
    summary_a = client_a.get(
        "/v1/dashboard/summary",
        params={"subject_id": "s1", "workspace_id": "tenant-a"},
    ).json()
    assert summary_a["evaluation"]["total"] == 1

    client_b = _client_as("key-b")
    summary_b = client_b.get(
        "/v1/dashboard/summary",
        params={"subject_id": "s1", "workspace_id": "tenant-b"},
    ).json()
    assert summary_b["evaluation"]["total"] == 0


def test_offline_default_preserves_unscoped_eval_reads() -> None:
    """No authenticator → caller-supplied workspace honored; legacy reads unchanged."""
    client = TestClient(create_app(ApiContainer.build_default()))
    body = {
        "suite": {
            "name": "qa",
            "cases": [
                {
                    "case_id": "c1",
                    "expected_output": {"answer": "yes"},
                    "metric_weights": {"accuracy": 1.0},
                }
            ],
        },
        "actual_outputs": {"c1": {"answer": "yes"}},
    }
    # No workspace supplied → run is unstamped and readable without one (offline).
    eval_id = client.post("/v1/evaluation/runs", json=body).json()["run_id"]
    assert client.get(f"/v1/evaluation/runs/{eval_id}").status_code == 200
    # Asking for a workspace the unstamped run does not belong to is a 404, not a leak.
    assert (
        client.get(
            f"/v1/evaluation/runs/{eval_id}", params={"workspace_id": "w2"}
        ).status_code
        == 404
    )
