"""WS4.7 B4 — the /v1/audit/privacy router + the widened /v1/audit/bundle coverage.

Builds an authenticated app (so RBAC is enforced) and asserts:

* an ``auditor`` may ``POST`` (``audit:run``) and ``GET`` (``audit:read``) the audit;
* a ``viewer`` (no ``audit:*``) is 403 on both;
* a tenant-bound principal cannot run an audit in a workspace it is not entitled to;
* an unknown report id is 404;
* requesting a per-report bundle with no signing key is a 503 (mirrors the security bundle);
* the registered ``privacy_audit_report`` appears in ``GET /v1/audit/bundle`` (B4 union).
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.entities.integrity import AuditBundle, verify_audit_bundle


def _app() -> TestClient:
    """An app with an auditor (tenant-bound), a viewer, and an all-tenant admin."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "auditor": Principal.build(
                "aud", tenant_ids=["t"], roles=["auditor"], auth_method="apikey"
            ),
            "viewer": Principal.build(
                "vw", tenant_ids=["t"], roles=["viewer"], auth_method="apikey"
            ),
            "admin": Principal.build(
                "root", all_tenants=True, roles=["admin"], auth_method="apikey"
            ),
        }
    )
    return TestClient(app)


# --------------------------------------------------------------- RBAC matrix
def test_auditor_can_run_and_read() -> None:
    client = _app()
    run = client.post(
        "/v1/audit/privacy?workspace_id=t", headers={"x-himmy-internal-key": "auditor"}
    )
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["report"]["passed"] is True  # clean store ⇒ passing posture
    assert body["bundle"] is None  # no bundle requested

    trend = client.get(
        "/v1/audit/privacy?workspace_id=t", headers={"x-himmy-internal-key": "auditor"}
    )
    assert trend.status_code == 200
    reports = trend.json()
    assert len(reports) >= 1
    assert reports[0]["report_id"] == body["report"]["report_id"]


def test_viewer_cannot_run_or_read() -> None:
    client = _app()
    assert (
        client.post(
            "/v1/audit/privacy?workspace_id=t",
            headers={"x-himmy-internal-key": "viewer"},
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/v1/audit/privacy?workspace_id=t",
            headers={"x-himmy-internal-key": "viewer"},
        ).status_code
        == 403
    )


def test_cross_tenant_run_denied() -> None:
    """An auditor bound to tenant 't' cannot run an audit scoped to 'other' → 403."""
    client = _app()
    resp = client.post(
        "/v1/audit/privacy?workspace_id=other",
        headers={"x-himmy-internal-key": "auditor"},
    )
    assert resp.status_code == 403


def test_unknown_report_id_is_404() -> None:
    client = _app()
    resp = client.get(
        "/v1/audit/privacy/does-not-exist?workspace_id=t",
        headers={"x-himmy-internal-key": "auditor"},
    )
    assert resp.status_code == 404


def test_get_report_by_id() -> None:
    client = _app()
    run = client.post(
        "/v1/audit/privacy?workspace_id=t", headers={"x-himmy-internal-key": "auditor"}
    )
    rid = run.json()["report"]["report_id"]
    got = client.get(
        f"/v1/audit/privacy/{rid}?workspace_id=t",
        headers={"x-himmy-internal-key": "auditor"},
    )
    assert got.status_code == 200
    assert got.json()["report_id"] == rid


# --------------------------------------------------------------- bundle (503 / union)
def test_run_with_bundle_503_without_key(monkeypatch: Any) -> None:
    """Requesting a per-report bundle with no signing key configured ⇒ 503."""
    import himmy.config.secrets as secrets

    monkeypatch.delenv("HIMMY_AUDIT_SECRET", raising=False)
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    secrets._PROVIDER = None  # noqa: SLF001 - reset the cached provider
    client = _app()
    resp = client.post(
        "/v1/audit/privacy?workspace_id=t&bundle=1",
        headers={"x-himmy-internal-key": "auditor"},
    )
    assert resp.status_code == 503


def test_privacy_report_appears_in_audit_bundle(monkeypatch: Any) -> None:
    """The registered privacy_audit_report is genuinely covered by GET /v1/audit/bundle."""
    import himmy.config.secrets as secrets

    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "audit-signing-secret")
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    secrets._PROVIDER = None  # noqa: SLF001 - reset the cached provider
    client = _app()

    # Run an audit so a privacy_audit_report is registered on the spine.
    run = client.post(
        "/v1/audit/privacy?workspace_id=t", headers={"x-himmy-internal-key": "auditor"}
    )
    assert run.status_code == 200, run.text

    # The privacy_audit_report record the audit registered on the spine.
    registry = client.app.state.container.entity_registry  # type: ignore[attr-defined]
    report_ids = {r.record_id for r in registry.list_by_kind("privacy_audit_report")}
    assert report_ids  # an audit ran

    bundle_resp = client.get(
        "/v1/audit/bundle", headers={"x-himmy-internal-key": "admin"}
    )
    assert bundle_resp.status_code == 200
    bundle = AuditBundle.model_validate(bundle_resp.json())
    # B4 union: every registered privacy_audit_report is covered by the signed bundle.
    assert report_ids <= set(bundle.records)
    # The bundle is internally consistent (signature over its own root).
    assert verify_audit_bundle(
        bundle, [], [], secret="audit-signing-secret"
    ).signature_valid


def test_run_with_bundle_succeeds_with_key(monkeypatch: Any) -> None:
    """With a signing key, ?bundle=1 returns a verifiable bundle over the report."""
    import himmy.config.secrets as secrets

    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "audit-signing-secret")
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    secrets._PROVIDER = None  # noqa: SLF001 - reset the cached provider
    client = _app()
    resp = client.post(
        "/v1/audit/privacy?workspace_id=t&bundle=1",
        headers={"x-himmy-internal-key": "auditor"},
    )
    assert resp.status_code == 200, resp.text
    bundle = AuditBundle.model_validate(resp.json()["bundle"])
    registry = client.app.state.container.entity_registry  # type: ignore[attr-defined]
    report_ids = {r.record_id for r in registry.list_by_kind("privacy_audit_report")}
    assert report_ids <= set(bundle.records)
