"""WS1.2 — RBAC: roles → permissions, enforced per route (deny-by-default)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import (
    DEFAULT_POLICY,
    AccessPolicy,
    RbacPolicyError,
    _parse_perm,
    lint_policy,
    load_policy,
)


# ------------------------------------------------------------ policy unit tests
def _p(*roles: str) -> Principal:
    return Principal.build("u", tenant_ids=["t"], roles=list(roles))


def test_default_role_matrix() -> None:
    pol = DEFAULT_POLICY
    # viewer: reads, no writes, no audit.
    assert pol.authorize(_p("viewer"), "run", "read")
    assert not pol.authorize(_p("viewer"), "run", "write")
    assert not pol.authorize(_p("viewer"), "audit", "read")
    # operator: reads + writes operational, still no audit.
    assert pol.authorize(_p("operator"), "run", "write")
    assert pol.authorize(_p("operator"), "context", "write")
    assert not pol.authorize(_p("operator"), "audit", "read")
    # auditor: reads incl. audit, no writes.
    assert pol.authorize(_p("auditor"), "audit", "read")
    assert not pol.authorize(_p("auditor"), "run", "write")
    # admin: everything via wildcard.
    assert pol.authorize(_p("admin"), "anything", "destroy")


def test_no_role_is_denied_by_default() -> None:
    assert not DEFAULT_POLICY.authorize(_p(), "run", "read")


def test_multiple_roles_union() -> None:
    assert DEFAULT_POLICY.authorize(_p("viewer", "operator"), "run", "write")


def test_load_policy_from_file(tmp_path: Path) -> None:
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({"custom": ["run:read", "widget:*"]}))
    pol = load_policy(f)
    assert pol.authorize(_p("custom"), "widget", "anything")
    assert not pol.authorize(_p("custom"), "run", "write")


def test_wildcard_action_and_resource() -> None:
    pol = AccessPolicy.from_mapping({"r": ["run:*"], "a": ["*:read"]})
    assert pol.authorize(_p("r"), "run", "write")
    assert not pol.authorize(_p("r"), "context", "read")
    assert pol.authorize(_p("a"), "context", "read")
    assert not pol.authorize(_p("a"), "context", "write")


# ------------------------------------- scopes NARROW role grants (least-privilege)
def _ps(*, roles: list[str], scopes: list[str]) -> Principal:
    return Principal.build("u", tenant_ids=["t"], roles=roles, scopes=scopes)


def test_scope_narrows_role_grant() -> None:
    """A token scoped to 'run:read' cannot 'run:write' even though its role grants it."""
    pol = DEFAULT_POLICY
    p = _ps(roles=["operator"], scopes=["run:read"])
    # role alone would allow run:write...
    assert pol.authorize(_p("operator"), "run", "write")
    # ...but the scope narrows it to read-only.
    assert pol.authorize(p, "run", "read")
    assert not pol.authorize(p, "run", "write")
    # a resource the scope never mentions is also denied (intersection, not union).
    assert not pol.authorize(p, "context", "write")


def test_empty_scopes_keep_full_role_reach() -> None:
    """An empty-scope token keeps its full role reach (no narrowing) — unchanged."""
    pol = DEFAULT_POLICY
    p = _ps(roles=["operator"], scopes=[])
    assert pol.authorize(p, "run", "write")
    assert pol.authorize(p, "context", "write")


def test_scope_cannot_widen_beyond_role() -> None:
    """A scope grants nothing the role lacks — it can only narrow."""
    pol = DEFAULT_POLICY
    # viewer has no run:write; a run:write SCOPE must not confer it.
    p = _ps(roles=["viewer"], scopes=["run:write"])
    assert not pol.authorize(p, "run", "write")
    # and the read the role does grant, scoped to write, is now also gone.
    assert not pol.authorize(p, "run", "read")


def test_unrecognized_scopes_do_not_narrow() -> None:
    """Standard OIDC scopes (openid/profile/email) are ignored → no narrowing."""
    pol = DEFAULT_POLICY
    p = _ps(roles=["operator"], scopes=["openid", "profile", "email"])
    # none of these parse as resource:action, so role reach is preserved.
    assert pol.authorize(p, "run", "write")
    assert pol.authorize(p, "context", "write")


def test_scope_wildcard_narrowing() -> None:
    """A wildcard scope ('run:*') narrows to a resource while keeping its actions."""
    pol = DEFAULT_POLICY
    p = _ps(roles=["operator"], scopes=["run:*"])
    assert pol.authorize(p, "run", "read")
    assert pol.authorize(p, "run", "write")
    # other resources the operator role grants are narrowed away.
    assert not pol.authorize(p, "context", "write")


def test_scope_mixed_recognized_and_standard() -> None:
    """A token mixing 'openid' with 'run:read' narrows on the grammar scope only."""
    pol = DEFAULT_POLICY
    p = _ps(roles=["operator"], scopes=["openid", "run:read"])
    assert pol.authorize(p, "run", "read")
    assert not pol.authorize(p, "run", "write")


def test_oidc_resource_scopes_do_not_lock_out() -> None:
    """Real colon-BEARING IdP resource scopes must NOT engage narrowing (no lock-out).

    Entra ID / Keycloak / ZITADEL emit scopes like ``api://himmy/access_as_user``,
    ``urn:zitadel:iam:org:project:role`` or ``https://graph.microsoft.com/User.Read``.
    These parse on the first colon (``('api','//…')`` etc.) but name no RBAC resource,
    so they must be dropped — a token carrying them keeps full role reach rather than
    being denied EVERYTHING.
    """
    pol = DEFAULT_POLICY
    p = _ps(
        roles=["operator"],
        scopes=[
            "openid",
            "profile",
            "api://himmy/access_as_user",
            "urn:zitadel:iam:org:project:role",
            "https://graph.microsoft.com/User.Read",
        ],
    )
    # No permission-scope is recognized → no narrowing → full operator reach.
    assert pol.authorize(p, "run", "read")
    assert pol.authorize(p, "run", "write")
    assert pol.authorize(p, "context", "write")


def test_admin_not_locked_out_by_oidc_resource_scope() -> None:
    """An admin token carrying a custom-API scope keeps its '*:*' reach (no lock-out)."""
    pol = DEFAULT_POLICY
    p = _ps(roles=["admin"], scopes=["api://himmy/access_as_user"])
    assert pol.authorize(p, "run", "write")
    assert pol.authorize(p, "anything", "at_all")


def test_oidc_resource_scope_mixed_with_real_perm_scope_narrows() -> None:
    """A garbage IdP scope alongside a real 'run:read' narrows on the real one only."""
    pol = DEFAULT_POLICY
    p = _ps(
        roles=["operator"],
        scopes=["api://himmy/access_as_user", "run:read"],
    )
    assert pol.authorize(p, "run", "read")
    assert not pol.authorize(p, "run", "write")  # narrowed by the real scope
    assert not pol.authorize(p, "context", "write")


def test_unknown_resource_scope_is_dropped() -> None:
    """A grammar-valid scope naming a resource NO policy defines is ignored (no narrow)."""
    pol = DEFAULT_POLICY
    # 'billing' is not an RBAC resource in DEFAULT_RBAC → dropped, not honored.
    p = _ps(roles=["operator"], scopes=["billing:read"])
    assert pol.authorize(p, "run", "write")  # full reach preserved


def test_known_resources_reflects_custom_policy() -> None:
    """known_resources() drives narrowing off the ACTIVE policy, not the default."""
    pol = AccessPolicy.from_mapping({"editor": ["doc:read", "doc:write"]})
    assert pol.known_resources() == frozenset({"doc"})
    p = _ps(roles=["editor"], scopes=["doc:read"])
    assert pol.authorize(p, "doc", "read")
    assert not pol.authorize(p, "doc", "write")  # narrowed
    # A scope naming a resource this custom policy never defines is dropped.
    p2 = _ps(roles=["editor"], scopes=["run:read"])
    assert pol.authorize(p2, "doc", "write")  # 'run' unknown here → no narrowing


# --------------------------------------------- fail-closed permission parsing (P0)
def test_trailing_colon_no_longer_widens() -> None:
    """A typo 'run:' must FAIL CLOSED, not silently widen to ('run', '*')."""
    with pytest.raises(RbacPolicyError):
        _parse_perm("run:")


def test_empty_half_rejected() -> None:
    """Both ':read' and an empty spec fail closed (no emptiness-as-wildcard)."""
    with pytest.raises(RbacPolicyError):
        _parse_perm(":read")
    with pytest.raises(RbacPolicyError):
        _parse_perm("")
    with pytest.raises(RbacPolicyError):
        _parse_perm("noseparator")


def test_literal_wildcard_still_parses() -> None:
    """A wildcard written as the literal '*' is fine; emptiness is not."""
    assert _parse_perm("run:*") == ("run", "*")
    assert _parse_perm("*:read") == ("*", "read")
    assert _parse_perm("*:*") == ("*", "*")


def test_from_mapping_names_offending_role_on_bad_perm() -> None:
    """A malformed perm names the role that carries it."""
    with pytest.raises(RbacPolicyError, match="broken"):
        AccessPolicy.from_mapping({"broken": ["run:"]})


def test_from_mapping_rejects_non_list_perms() -> None:
    with pytest.raises(RbacPolicyError, match="badrole"):
        AccessPolicy.from_mapping({"badrole": "run:read"})  # type: ignore[dict-item]


def test_from_mapping_rejects_non_string_perm() -> None:
    with pytest.raises(RbacPolicyError, match="r1"):
        AccessPolicy.from_mapping({"r1": [123]})  # type: ignore[list-item]


def test_from_mapping_rejects_non_dict_top_level() -> None:
    with pytest.raises(RbacPolicyError):
        AccessPolicy.from_mapping(["run:read"])  # type: ignore[arg-type]


def test_empty_policy_warns_lockout() -> None:
    """An empty {} policy parses but WARNS loudly (it locks everyone out)."""
    _policy, errors, warnings = lint_policy({})
    assert errors == []
    assert any("EMPTY" in w for w in warnings)


def test_nonadmin_wildcard_warns() -> None:
    """A non-admin role granted '*:*' / '*:<action>' warns (likely over-grant)."""
    _policy, errors, warnings = lint_policy({"weak": ["*:read"]})
    assert errors == []
    assert any("wildcard" in w and "weak" in w for w in warnings)
    # admin's wildcard is expected — no wildcard warning for it.
    _p2, _e2, w2 = lint_policy({"admin": ["*:*"]})
    assert not any("wildcard" in w and "admin" in w for w in w2)


def test_lint_unknown_token_warns_typo() -> None:
    """An unknown resource/action token is flagged as a likely typo (warning only)."""
    _policy, errors, warnings = lint_policy({"role": ["recommendaton:read"]})
    assert errors == []
    assert any("typo" in w for w in warnings)


def test_lint_collects_errors_without_raising() -> None:
    """lint_policy reports all errors at once and never raises."""
    policy, errors, _warnings = lint_policy({"r": ["run:", ":read"]})
    assert policy is None
    assert len(errors) == 2
    assert all("r" in e for e in errors)


def test_load_policy_malformed_json_raises_clean_error(tmp_path: Path) -> None:
    """A broken file raises a clean RbacPolicyError, never a raw JSONDecodeError 500."""
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")
    with pytest.raises(RbacPolicyError, match="invalid JSON"):
        load_policy(f)


def test_load_policy_non_object_raises_clean_error(tmp_path: Path) -> None:
    f = tmp_path / "list.json"
    f.write_text("[1, 2, 3]")
    with pytest.raises(RbacPolicyError, match="must be a JSON object"):
        load_policy(f)


def test_malformed_rbac_file_fails_startup_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken HIMMY_RBAC_FILE raises a clean HimmyError at create_app, not a 500."""
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({"role": ["run:"]}))  # trailing-colon widen attempt
    monkeypatch.setenv("HIMMY_RBAC_FILE", str(f))
    with pytest.raises(RbacPolicyError):
        create_app(ApiContainer.build_default())


# ------------------------------------------------------------- route enforcement
def _client(role: str) -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=[role], auth_method="apikey"
            )
        }
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


def _create_body() -> dict:
    return {
        "workspace_id": "t",
        "subject_id": "s",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }


def test_viewer_cannot_create_a_run() -> None:
    assert _client("viewer").post("/v1/runs", json=_create_body()).status_code == 403


def test_viewer_can_list_runs() -> None:
    assert (
        _client("viewer").get("/v1/runs", params={"workspace_id": "t"}).status_code
        == 200
    )


def test_operator_can_create_a_run() -> None:
    assert _client("operator").post("/v1/runs", json=_create_body()).status_code == 200


def test_roleless_principal_is_forbidden_everywhere() -> None:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={"k": Principal.build("u", tenant_ids=["t"], roles=[])}
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    assert c.get("/v1/runs", params={"workspace_id": "t"}).status_code == 403


def test_offline_default_bypasses_rbac() -> None:
    """No authenticator configured → RBAC off → zero-config behavior unchanged."""
    c = TestClient(create_app(ApiContainer.build_default()))
    assert c.post("/v1/runs", json=_create_body()).status_code == 200


# ---------------------------------------------- WP p1-studio-granular: Studio RBAC
@pytest.fixture
def studio_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate Studio's on-disk stores (``.himmy/``) in a fresh tmp cwd per test."""
    from himmy.api.studio_runs import reset_run_store

    monkeypatch.chdir(tmp_path)
    reset_run_store()  # the run-store cache is keyed on cwd
    return tmp_path


def _studio_client(
    *roles: str, tenant_ids: list[str] | None = None
) -> tuple[TestClient, object]:
    """A Studio client bound to ``roles`` (+ tenants); returns ``(client, app)``.

    Default-policy roles are used so the granular ``studio.*:read`` grants under test
    are the ones shipped in :data:`DEFAULT_RBAC`, not a hand-rolled fixture policy.
    """
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u",
                tenant_ids=tenant_ids if tenant_ids is not None else ["t"],
                roles=list(roles),
                auth_method="apikey",
            )
        }
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c, app


def test_readonly_studio_role_can_read(studio_cwd: Path) -> None:
    """A default read-only role (``viewer``) can browse Studio read surfaces."""
    c, _ = _studio_client("viewer")
    assert c.get("/api/studio/health").status_code == 200
    assert c.get("/api/studio/connections").status_code == 200
    assert c.get("/api/studio/mcp/servers").status_code == 200


def test_readonly_studio_role_is_403_on_mcp_mutation(studio_cwd: Path) -> None:
    """``viewer`` holds ``studio.mcp:read`` but NOT ``studio.mcp:manage`` → 403."""
    c, _ = _studio_client("viewer")
    body = {"name": "srv", "command": "echo"}
    assert c.post("/api/studio/mcp/servers", json=body).status_code == 403
    assert c.delete("/api/studio/mcp/servers/srv").status_code == 403


def test_readonly_studio_role_is_403_on_key_mutation(studio_cwd: Path) -> None:
    """``viewer`` cannot save a provider API key (``studio.models:write``)."""
    c, _ = _studio_client("viewer")
    body = {"provider": "openrouter", "api_key": "sk-x"}
    assert c.post("/api/studio/models/keys", json=body).status_code == 403


def test_readonly_studio_role_is_403_on_connection_mutation(studio_cwd: Path) -> None:
    """``viewer`` cannot write a connection's secrets (``studio.connections:write``)."""
    c, _ = _studio_client("viewer")
    r = c.put("/api/studio/connections/email", json={"fields": {}})
    assert r.status_code == 403


# ---- red-team r1: un-partitionable global-store Studio readers are admin-only -----
# The memory/notes/tasks/chats/cookbook/calendar/knowledge/projects/files readers all
# enumerate ONE process-wide, single-user-local store with NO tenant/subject axis, so a
# tenant-facing role (viewer/operator/auditor) granted ``studio.<surface>:read`` would
# read every OTHER tenant's/subject's private data (cross-tenant/cross-subject BOLA).
# These surfaces are withheld from the default browse roles (admin-only) once an
# authenticator is configured; the offline path is untouched (require_permission no-ops).
#
# red-team r2 extends the list with three PROCESS-WIDE SPINE readers that walk the shared
# ``.himmy/spine.db`` registry / security-audit log with no tenant filter and thus leak
# cross-tenant security events, lineage payloads, and privacy/consent records:
# ``seclog``/``lineage``/``privacy``. They were missing from the r1 withheld set, so a
# tenant-facing role could read every other tenant's data through them (a cross-tenant
# IDOR/BOLA). They are now in :data:`_STUDIO_GLOBAL_STORE_RESOURCES` (admin-only).
_GLOBAL_STORE_READS = [
    "/api/studio/memory/subjects",
    "/api/studio/notes",
    "/api/studio/tasks",
    "/api/studio/chats",
    "/api/studio/cookbook",
    "/api/studio/calendar",
    "/api/studio/projects",
    "/api/studio/files",
    "/api/studio/seclog",
    "/api/studio/privacy/subjects",
    "/api/studio/privacy/consents",
    "/api/studio/lineage/graph",
    "/api/studio/lineage/entity/anyid",
]


@pytest.mark.parametrize("role", ["viewer", "operator", "auditor"])
@pytest.mark.parametrize("path", _GLOBAL_STORE_READS)
def test_tenant_role_is_403_on_global_store_reader(
    studio_cwd: Path, role: str, path: str
) -> None:
    """A configured tenant-facing role is 403 reading an un-partitionable global store."""
    c, _ = _studio_client(role)
    assert c.get(path).status_code == 403, f"{role} reached {path}"


@pytest.mark.parametrize("path", _GLOBAL_STORE_READS)
def test_admin_can_read_global_store_reader(studio_cwd: Path, path: str) -> None:
    """``admin`` (``*:*``) still reads the operator-console global stores (not 403)."""
    c, _ = _studio_client("admin")
    assert c.get(path).status_code != 403, f"admin blocked on {path}"


@pytest.mark.parametrize("path", _GLOBAL_STORE_READS)
def test_offline_reads_global_store_reader(studio_cwd: Path, path: str) -> None:
    """INVARIANT: offline / no-auth reads every global-store surface (byte-unchanged)."""
    c = TestClient(create_app(ApiContainer.build_default()))
    assert c.get(path).status_code != 403, f"offline blocked on {path}"


def test_tenant_role_is_403_on_mcp_test_launch(studio_cwd: Path) -> None:
    """``viewer`` cannot launch an MCP server subprocess via /test (now studio.mcp:manage).

    Probing /test spawns the server with admin secrets resolved into env AND mutates the
    persisted registry — both privileged. Previously it rode only the ``studio.mcp:read``
    baseline; a read-only role must now be 403'd BEFORE any subprocess is spawned.
    """
    c, _ = _studio_client("viewer")
    assert c.post("/api/studio/mcp/servers/any/test").status_code == 403
    op, _ = _studio_client("operator")
    assert op.post("/api/studio/mcp/servers/any/test").status_code == 403


def test_admin_passes_mcp_test_rbac(studio_cwd: Path) -> None:
    """``admin`` clears the /test manage guard (404 on a missing server, not 403)."""
    c, _ = _studio_client("admin")
    assert c.post("/api/studio/mcp/servers/missing/test").status_code == 404


def test_admin_can_mutate_studio(studio_cwd: Path) -> None:
    """``admin`` (``*:*``) clears every granular Studio write/manage guard."""
    c, _ = _studio_client("admin")
    body = {"name": "srv", "command": "echo"}
    assert c.post("/api/studio/mcp/servers", json=body).status_code == 201


def test_auditor_can_read_studio_but_not_mutate(studio_cwd: Path) -> None:
    """``auditor`` is read-only: ``studio.*:read`` yes, ``:manage`` no."""
    c, _ = _studio_client("auditor")
    assert c.get("/api/studio/mcp/servers").status_code == 200
    body = {"name": "srv", "command": "echo"}
    assert c.post("/api/studio/mcp/servers", json=body).status_code == 403


def _seed_run(app: object, *, workspace_id: str, run_id: str) -> None:
    """Persist one canonical run in ``workspace_id`` via the app's container storage."""
    import asyncio

    from himmy.services.storage.models import RunRecord, RunStatus

    storage = app.state.container.storage  # type: ignore[attr-defined]
    rec = RunRecord(
        run_id=run_id,
        workspace_id=workspace_id,
        subject_id="s",
        status=RunStatus.SUCCEEDED,
        output_text="hi",
    )
    asyncio.run(storage.save_run(rec))


def test_tenant_bound_principal_cannot_list_cross_tenant_runs(studio_cwd: Path) -> None:
    """A tenant-bound principal sees only its own workspace's runs in Studio."""
    c, app = _studio_client("admin", tenant_ids=["t"])
    _seed_run(app, workspace_id="t", run_id="run-mine")
    _seed_run(app, workspace_id="other", run_id="run-theirs")

    listing = c.get("/api/studio/runs").json()
    ids = {item["id"] for item in listing["items"]}
    assert "run-mine" in ids
    assert "run-theirs" not in ids
    assert listing["total"] == 1


def test_tenant_bound_principal_cannot_get_cross_tenant_run(studio_cwd: Path) -> None:
    """Fetching another tenant's run by id is 404 (never leak existence)."""
    c, app = _studio_client("admin", tenant_ids=["t"])
    _seed_run(app, workspace_id="t", run_id="run-mine")
    _seed_run(app, workspace_id="other", run_id="run-theirs")

    assert c.get("/api/studio/runs/run-mine").status_code == 200
    assert c.get("/api/studio/runs/run-theirs").status_code == 404


def test_offline_studio_runs_unscoped(studio_cwd: Path) -> None:
    """No authenticator → Studio shows ALL runs (byte-unchanged single-box path)."""
    app = create_app(ApiContainer.build_default())
    _seed_run(app, workspace_id="t", run_id="run-a")
    _seed_run(app, workspace_id="other", run_id="run-b")
    c = TestClient(app)

    ids = {item["id"] for item in c.get("/api/studio/runs").json()["items"]}
    assert ids == {"run-a", "run-b"}


# ---- review fix: every mutating sub-router rejects a read-only role -----------
#: One representative mutation per sub-router annotated in this review pass. A
#: read-only ``viewer`` holds ``studio.<seg>:read`` but NOT the route's ``:write``/
#: ``:manage`` permission, so each MUST 403 (never reach the handler for a 404/422).
_VIEWER_DENIED_MUTATIONS = [
    ("post", "/api/studio/routines", {"name": "r", "agent_path": "a", "schedule": "@daily"}),
    ("delete", "/api/studio/routines/x", None),
    ("post", "/api/studio/routines/x/run-now", None),
    ("post", "/api/studio/projects", {"name": "p"}),
    ("delete", "/api/studio/projects/x", None),
    ("post", "/api/studio/privacy/erase", {"subject_id": "s", "confirm": "s"}),
    ("post", "/api/studio/telegram/start", {}),
    ("post", "/api/studio/telegram/stop", {}),
    ("post", "/api/studio/missions", {}),
    ("post", "/api/studio/missions/x/steer", {}),
    ("post", "/api/studio/missions/x/interrupt", {}),
    ("post", "/api/studio/notify/read-all", {}),
    ("post", "/api/studio/notify/settings", {}),
    ("post", "/api/studio/learning/overrides", {}),
    ("post", "/api/studio/learning/trust-feedback", {}),
    ("patch", "/api/studio/memory/x", {"text": "x"}),
    ("post", "/api/studio/eval/run", {}),
    ("post", "/api/studio/teams", {}),
    ("post", "/api/studio/workflows/run-stream", {}),
    ("post", "/api/studio/kb/x/upload", None),
    ("post", "/api/studio/benchmarks/probe", None),
]


@pytest.mark.parametrize(
    ("method", "path", "body"),
    _VIEWER_DENIED_MUTATIONS,
    ids=[f"{m}-{p}" for m, p, _ in _VIEWER_DENIED_MUTATIONS],
)
def test_readonly_role_is_403_on_every_studio_mutation(
    studio_cwd: Path, method: str, path: str, body: object
) -> None:
    """A ``viewer`` is 403 (not 404/422) on a representative mutation per sub-router.

    Guards the deny-by-default regression: before this pass these POST/PUT/PATCH/DELETE
    routes carried only the router-level ``:read`` baseline, so the read grant alone
    authorized the mutation. A 403 proves RBAC rejects BEFORE the handler runs.
    """
    c, _ = _studio_client("viewer")
    resp = c.request(method, path, json=body)
    assert resp.status_code == 403, f"{method} {path} -> {resp.status_code}"


def test_admin_passes_studio_mutation_rbac(studio_cwd: Path) -> None:
    """``admin`` (``*:*``) clears the write/manage guard — reaches the handler.

    The complement to the viewer-403 sweep: a representative mutation per newly-annotated
    sub-router must NOT 403 for admin (it may 404/422 on missing data/validation, proving
    the request passed RBAC and entered the handler).
    """
    c, _ = _studio_client("admin")
    for method, path, body in _VIEWER_DENIED_MUTATIONS:
        resp = c.request(method, path, json=body)
        assert resp.status_code != 403, f"admin blocked on {method} {path}"


def test_every_studio_non_get_route_carries_a_write_guard() -> None:
    """Structural: every Studio non-GET route declares a non-``read`` studio permission.

    The deny-by-default backstop — a future mutating route added without a write/manage
    dependency would be authorized by the router's read baseline alone (the exact gap
    this pass closed). Read-style POST helpers (search/recall/validate) are exempted by
    name since they mutate nothing.
    """
    from himmy.api.routers.studio_common import STUDIO_RESOURCE_PREFIX

    app = create_app(ApiContainer.build_default())
    #: POST routes that are reads in disguise (a body-carrying query), so a read guard
    #: is correct for them.
    read_post_suffixes = (
        "/memory/recall",
        "/knowledge/{kb_id}/search",
        "/agents/validate",
        "/teams/validate",
    )
    offenders: list[str] = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", "")
        if not path.startswith("/api/studio/"):
            continue
        if methods <= {"GET", "HEAD", "OPTIONS"}:
            continue
        if path.endswith(read_post_suffixes):
            continue
        # Walk the route's dependant tree for a studio_permission with a non-read action.
        deps = getattr(getattr(route, "dependant", None), "dependencies", [])

        def _names(dependant: object) -> list[str]:
            out: list[str] = []
            call = getattr(dependant, "call", None)
            if call is not None:
                out.append(getattr(call, "__qualname__", repr(call)))
            for sub in getattr(dependant, "dependencies", []) or []:
                out.extend(_names(sub))
            return out

        all_names = []
        for d in deps:
            all_names.extend(_names(d))
        # studio_permission's inner closure is named ``studio_permission.<locals>._dep``.
        has_write_guard = any(
            "studio_permission" in n and "_dep" in n for n in all_names
        )
        if not has_write_guard:
            offenders.append(f"{sorted(methods)} {path}")
    assert not offenders, (
        "Studio non-GET routes missing a write/manage "
        f"{STUDIO_RESOURCE_PREFIX} permission dep: {offenders}"
    )


# ---- review fix: cross-tenant feedback/lineage/analytics readers -------------
def _seed_cache_row(workspace_id: str, run_id: str) -> None:
    """Insert a studio.db presentation-cache row (the leak surface) for ``run_id``."""
    from himmy.api.studio_runs import StudioRun, get_run_store

    get_run_store().save(
        StudioRun(id=run_id, created_at="2026-06-28T00:00:00Z", prompt="hi")
    )


def test_tenant_bound_principal_cannot_read_cross_tenant_feedback(
    studio_cwd: Path,
) -> None:
    """Reading another tenant's run feedback by id is 404 (cache row notwithstanding)."""
    c, app = _studio_client("admin", tenant_ids=["t"])
    _seed_run(app, workspace_id="other", run_id="run-theirs")
    _seed_cache_row("other", "run-theirs")

    assert c.get("/api/studio/runs/run-theirs/feedback").status_code == 404
    assert (
        c.post(
            "/api/studio/runs/run-theirs/feedback", json={"verdict": "up"}
        ).status_code
        == 404
    )


def test_tenant_bound_principal_cannot_read_cross_tenant_lineage(
    studio_cwd: Path,
) -> None:
    """Reading another tenant's run lineage by id is 404 (cache row notwithstanding)."""
    c, app = _studio_client("admin", tenant_ids=["t"])
    _seed_run(app, workspace_id="other", run_id="run-theirs")
    _seed_cache_row("other", "run-theirs")

    assert c.get("/api/studio/runs/run-theirs/lineage").status_code == 404


def test_tenant_bound_principal_can_read_own_feedback(studio_cwd: Path) -> None:
    """The gate does not over-block: a principal's OWN run feedback is reachable (200)."""
    c, app = _studio_client("admin", tenant_ids=["t"])
    _seed_run(app, workspace_id="t", run_id="run-mine")
    _seed_cache_row("t", "run-mine")

    assert c.get("/api/studio/runs/run-mine/feedback").status_code == 200


def test_tenant_bound_principal_gets_empty_analytics(studio_cwd: Path) -> None:
    """Analytics returns empty (not a cross-tenant total) for a tenant-bound principal."""
    c, app = _studio_client("admin", tenant_ids=["t"])
    _seed_run(app, workspace_id="other", run_id="run-theirs")
    _seed_cache_row("other", "run-theirs")

    data = c.get("/api/studio/runs/analytics").json()
    assert data["total_runs"] == 0


def test_offline_analytics_and_lineage_unscoped(studio_cwd: Path) -> None:
    """No authenticator → by-id readers and analytics are unscoped (byte-unchanged)."""
    app = create_app(ApiContainer.build_default())
    _seed_run(app, workspace_id="other", run_id="run-b")
    _seed_cache_row("other", "run-b")
    c = TestClient(app)

    assert c.get("/api/studio/runs/run-b/feedback").status_code == 200
    assert c.get("/api/studio/runs/run-b/lineage").status_code == 200


# ---------------------------------- WP p1-observability-gate: policy_loaded + metrics
def _authed_app_via_env(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build an app whose authenticator is configured AT create_app time (via env).

    The route-enforcement fixtures above set ``app.state.authenticator`` AFTER
    ``create_app`` returns, so the startup ``policy_loaded`` emit (which only fires when
    ``build_authenticator()`` is non-None inside ``create_app``) is intentionally NOT
    exercised by them. Here we configure a shared internal key via the environment so
    ``create_app`` builds the authenticator itself and the startup event fires.
    """
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "k")
    app = create_app(ApiContainer.build_default())
    assert app.state.authenticator is not None  # auth really is configured
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


def test_policy_loaded_event_fires_with_a_hash_not_raw_perms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup records a 'policy_loaded' audit event with a content HASH + role count."""
    c = _authed_app_via_env(monkeypatch)
    events = c.app.state.security_audit.recent(event_type="policy_loaded")  # type: ignore[attr-defined]
    assert len(events) == 1
    ev = events[0]
    assert ev.outcome == "allow"
    assert ev.resource == "rbac_policy"
    assert ev.action == "load"
    # The detail carries source + a sha256 fingerprint + role count.
    assert "source=<built-in>" in ev.detail
    assert "hash=sha256:" in ev.detail
    assert "roles=" in ev.detail
    # CRITICAL: no verbatim permission string is leaked into the audit trail.
    from himmy.api.auth.rbac import DEFAULT_RBAC

    for perms in DEFAULT_RBAC.values():
        for perm in perms:
            assert perm not in ev.detail


def test_policy_loaded_records_the_file_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom HIMMY_RBAC_FILE is named (config, not secret) as the policy source."""
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({"viewer": ["run:read"]}))
    monkeypatch.setenv("HIMMY_RBAC_FILE", str(f))
    c = _authed_app_via_env(monkeypatch)
    ev = c.app.state.security_audit.recent(event_type="policy_loaded")[0]  # type: ignore[attr-defined]
    assert f"source={f}" in ev.detail
    assert "roles=1" in ev.detail


def test_policy_loaded_is_noop_offline() -> None:
    """No authenticator → no policy_loaded event (offline/zero-config is unchanged)."""
    app = create_app(ApiContainer.build_default())
    assert app.state.authenticator is None
    # Query INSIDE the lifespan (the registry is closed on exit); no policy_loaded
    # event is ever recorded on the offline path.
    with TestClient(app):
        assert app.state.security_audit.recent(event_type="policy_loaded") == []


def test_policy_fingerprint_is_stable_and_order_independent() -> None:
    """The fingerprint is a non-reversible digest, identical for equal policies."""
    from himmy.api.auth.rbac import policy_fingerprint

    a = AccessPolicy.from_mapping({"x": ["run:read", "context:write"], "y": ["run:*"]})
    b = AccessPolicy.from_mapping({"y": ["run:*"], "x": ["context:write", "run:read"]})
    fp = policy_fingerprint(a)
    assert fp.startswith("sha256:")
    assert fp == policy_fingerprint(b)  # order-independent
    # A different policy yields a different fingerprint (drift is detectable).
    c = AccessPolicy.from_mapping({"x": ["run:read"]})
    assert policy_fingerprint(c) != fp
    # And it never embeds a verbatim permission (non-reversible).
    assert "run:read" not in fp


def test_authz_denied_metric_increments_on_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 increments the himmy_authz_denied_total counter for that resource:action."""
    from himmy.services.observability.metrics import get_registry, reset_registry

    reset_registry()
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "k")
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build("u", tenant_ids=["t"], roles=["viewer"], auth_method="apikey")
        }
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    # viewer cannot create a run (run:write) → 403 → denied counter ticks.
    assert c.post("/v1/runs", json=_create_body()).status_code == 403
    assert get_registry().authz_denied_total.value(("run", "write")) >= 1.0


def test_authz_granted_metric_sampled_for_privileged_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grant on a privileged resource (audit) is sampled; an ordinary read is not."""
    from himmy.services.observability.metrics import get_registry, reset_registry

    reset_registry()
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "k")
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=["auditor"], auth_method="apikey"
            )
        }
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    reg = get_registry()
    # An ordinary high-volume read (run:read) is granted but NOT sampled into grants.
    assert c.get("/v1/runs", params={"workspace_id": "t"}).status_code == 200
    assert reg.authz_granted_total.value(("run", "read")) == 0.0
    # A grant on the PRIVILEGED 'audit' resource IS sampled.
    assert c.get("/v1/audit/events").status_code in (200, 404)
    assert reg.authz_granted_total.value(("audit", "read")) >= 1.0
