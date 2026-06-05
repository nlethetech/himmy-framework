"""WS1.4 — security audit through the BFF: auth/authz/access events + audit read."""

from __future__ import annotations

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal

_BODY = {
    "workspace_id": "t",
    "subject_id": "s",
    "persona": {"name": "A"},
    "task": {"title": "t", "prompt": "hi"},
}


def _app() -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "op": Principal.build(
                "op", tenant_ids=["t"], roles=["operator"], auth_method="apikey"
            ),
            "view": Principal.build(
                "v", tenant_ids=["t"], roles=["viewer"], auth_method="apikey"
            ),
            "admin": Principal.build(
                "root", all_tenants=True, roles=["admin"], auth_method="apikey"
            ),
        }
    )
    return TestClient(app)


def _events(client: TestClient, **params: str) -> list[dict]:
    resp = client.get(
        "/v1/audit/events", headers={"x-himmy-internal-key": "admin"}, params=params
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_access_event_recorded_on_run_create() -> None:
    client = _app()
    client.post("/v1/runs", headers={"x-himmy-internal-key": "op"}, json=_BODY)
    events = _events(client, event_type="access")
    assert any(
        e["resource"] == "run"
        and e["action"] == "create"
        and e["actor"]["subject"] == "op"
        and e["outcome"] == "allow"
        for e in events
    )


def test_authz_denied_event_recorded() -> None:
    client = _app()
    # viewer cannot create a run → 403 + an authz_denied event.
    assert (
        client.post(
            "/v1/runs", headers={"x-himmy-internal-key": "view"}, json=_BODY
        ).status_code
        == 403
    )
    events = _events(client, event_type="authz_denied")
    assert any(
        e["resource"] == "run"
        and e["action"] == "write"
        and e["actor"]["subject"] == "v"
        for e in events
    )


def test_auth_failure_event_recorded() -> None:
    client = _app()
    assert (
        client.get(
            "/v1/runs",
            headers={"x-himmy-internal-key": "bad"},
            params={"workspace_id": "t"},
        ).status_code
        == 401
    )
    events = _events(client, event_type="auth_failure")
    assert any(e["outcome"] == "deny" for e in events)


def test_audit_read_requires_audit_permission() -> None:
    client = _app()
    # operator lacks audit:read.
    assert (
        client.get(
            "/v1/audit/events", headers={"x-himmy-internal-key": "op"}
        ).status_code
        == 403
    )
    # admin can read.
    assert (
        client.get(
            "/v1/audit/events", headers={"x-himmy-internal-key": "admin"}
        ).status_code
        == 200
    )


def test_offline_records_no_security_events() -> None:
    """No authenticator → audit is off → the trail stays empty (zero-config)."""
    client = TestClient(create_app(ApiContainer.build_default()))
    client.post("/v1/runs", json=_BODY)
    assert client.get("/v1/audit/events").json() == []
