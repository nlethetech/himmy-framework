"""WS4.5 — the GET /v1/audit/bundle route (signed, auditor-gated)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.entities.integrity import AuditBundle, verify_audit_bundle


def _app() -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "op": Principal.build(
                "op", tenant_ids=["t"], roles=["operator"], auth_method="apikey"
            ),
            "admin": Principal.build(
                "root", all_tenants=True, roles=["admin"], auth_method="apikey"
            ),
        }
    )
    return TestClient(app)


def test_bundle_requires_audit_permission() -> None:
    client = _app()
    assert (
        client.get(
            "/v1/audit/bundle", headers={"x-himmy-internal-key": "op"}
        ).status_code
        == 403
    )


def test_bundle_503_without_signing_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_AUDIT_SECRET", raising=False)
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    client = _app()
    resp = client.get("/v1/audit/bundle", headers={"x-himmy-internal-key": "admin"})
    assert resp.status_code == 503


def test_hmac_bundle_exports_and_verifies(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "audit-signing-secret")
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    client = _app()
    # Generate a security event (a denied write by the operator).
    client.post(
        "/v1/runs",
        headers={"x-himmy-internal-key": "op"},
        json={
            "workspace_id": "t",
            "subject_id": "s",
            "persona": {"name": "A"},
            "task": {"title": "t", "prompt": "hi"},
        },
    )
    resp = client.get("/v1/audit/bundle", headers={"x-himmy-internal-key": "admin"})
    assert resp.status_code == 200
    bundle = AuditBundle.model_validate(resp.json())
    assert bundle.algorithm == "HMAC-SHA256"
    assert bundle.records  # at least the authz_denied event is covered
    # The exported bundle is internally consistent (signature over its own root).
    result = verify_audit_bundle(bundle, [], [], secret="audit-signing-secret")
    assert result.signature_valid
