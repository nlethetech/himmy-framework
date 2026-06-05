"""WS1.2 — RBAC: roles → permissions, enforced per route (deny-by-default)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import DEFAULT_POLICY, AccessPolicy, load_policy


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
