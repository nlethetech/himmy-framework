"""Finding #2 regression: GET /v1/audit/bundle must be tenant-scoped.

The bundle export used to union every ``security_event`` / ``privacy_audit_report``
record with no tenant filter, so a tenant-bound auditor could export every other
tenant's security trail. The export must mirror the sibling ``/events`` route: a
tenant-bound principal only sees its own workspace's records (plus global/ops
records with no ``workspace_id``), while an ``all_tenants`` principal still gets the
full export.

``AuditBundle.records`` is a ``record_id -> content_hash`` map, and the audit log
registers each event under a deterministic ``record_id`` derived from its
``event_id`` (stable_id) + kind + version, so we assert on record-id membership.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.entities.integrity import AuditBundle
from himmy.entities.records import record_id_for
from himmy.services.audit.log import SECURITY_EVENT_KIND
from himmy.services.audit.models import SecurityEvent
from himmy.services.evaluation.privacy import PRIVACY_AUDIT_REPORT_KIND
from himmy.services.evaluation.privacy.models import PrivacyAuditReport


def _record_id(event: SecurityEvent) -> str:
    return record_id_for(stable_id=event.event_id, version=1, kind=SECURITY_EVENT_KIND)


def _seed_privacy_report(client: TestClient, workspace_id: str | None) -> str:
    """Register a per-tenant privacy_audit_report and return its bundle record id."""
    report = PrivacyAuditReport(suite_name=f"{workspace_id or 'global'}-suite")
    report.workspace_id = workspace_id
    record = client.app.state.container.entity_registry.register(report.to_record())
    return record_id_for(
        stable_id=record.stable_id,
        version=record.version,
        kind=PRIVACY_AUDIT_REPORT_KIND,
    )


def _seed(monkeypatch: Any) -> tuple[TestClient, dict[str, str]]:
    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "audit-signing-secret")
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "alpha": Principal.build(
                "alpha-auditor",
                tenant_ids=["alpha"],
                roles=["auditor"],
                auth_method="apikey",
            ),
            "root": Principal.build(
                "root-auditor",
                all_tenants=True,
                roles=["auditor"],
                auth_method="apikey",
            ),
        }
    )
    client = TestClient(app)
    log = client.app.state.security_audit
    alpha_evt = log.record(
        SecurityEvent(event_type="access", workspace_id="alpha", outcome="allow")
    )
    beta_evt = log.record(
        SecurityEvent(event_type="access", workspace_id="beta", outcome="allow")
    )
    global_evt = log.record(
        SecurityEvent(event_type="admin", workspace_id=None, outcome="allow")
    )
    ids = {
        "alpha": _record_id(alpha_evt),
        "beta": _record_id(beta_evt),
        "global": _record_id(global_evt),
    }
    return client, ids


def test_tenant_auditor_bundle_excludes_other_tenants(monkeypatch: Any) -> None:
    client, ids = _seed(monkeypatch)
    resp = client.get("/v1/audit/bundle", headers={"x-himmy-internal-key": "alpha"})
    assert resp.status_code == 200
    record_ids = set(AuditBundle.model_validate(resp.json()).records)
    # Own tenant + global ops events are present; another tenant's is NOT.
    assert ids["alpha"] in record_ids
    assert ids["global"] in record_ids
    assert ids["beta"] not in record_ids


def test_all_tenants_auditor_bundle_includes_everything(monkeypatch: Any) -> None:
    client, ids = _seed(monkeypatch)
    resp = client.get("/v1/audit/bundle", headers={"x-himmy-internal-key": "root"})
    assert resp.status_code == 200
    record_ids = set(AuditBundle.model_validate(resp.json()).records)
    assert {ids["alpha"], ids["beta"], ids["global"]} <= record_ids


def test_tenant_auditor_bundle_excludes_other_tenants_privacy_report(
    monkeypatch: Any,
) -> None:
    """The proven live bypass: privacy_audit_report records carried no workspace_id in
    their metadata, so ``metadata.get("workspace_id")`` was always ``None`` and EVERY
    tenant's report leaked through the ``(None, workspace_id)`` filter. With the report
    now carrying its workspace, tenant alpha must NOT see tenant beta's BETA-SECRET
    report, while an all_tenants auditor still exports both.
    """
    client, _ = _seed(monkeypatch)
    alpha_report = _seed_privacy_report(client, "alpha")
    beta_report = _seed_privacy_report(client, "beta")

    resp = client.get("/v1/audit/bundle", headers={"x-himmy-internal-key": "alpha"})
    assert resp.status_code == 200
    record_ids = set(AuditBundle.model_validate(resp.json()).records)
    assert alpha_report in record_ids
    assert beta_report not in record_ids

    resp = client.get("/v1/audit/bundle", headers={"x-himmy-internal-key": "root"})
    assert resp.status_code == 200
    record_ids = set(AuditBundle.model_validate(resp.json()).records)
    assert {alpha_report, beta_report} <= record_ids
