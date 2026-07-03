"""WP enterprise-console — the /v1/enterprise/overview governance rollup.

Builds an authenticated app (so RBAC is enforced), seeds runs + inference cost events +
security events across two tenants, and asserts:

* the overview is ENTITLEMENT-gated — a community install is refused **402** with the
  actionable upgrade message; an entitled enterprise install returns data;
* the overview is RBAC-gated — an entitled but unauthorized principal (``data_subject``,
  no ``dashboard:read``) is **403**;
* the rollups are correct — runs count-by-status, cost total + per-agent + per-model, the
  agents summary, and the recent security events;
* the overview is TENANT-SCOPED — tenant B's runs / cost / agents / events never appear in
  tenant A's overview (and vice versa).
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

import himmy.licensing.core as _lic
from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.core.events import EventType, RunEvent
from himmy.services.audit.models import SecurityEvent
from himmy.services.storage.models import RunRecord, RunStatus


# --------------------------------------------------------------- EE license fixture
@pytest.fixture
def ee_console(monkeypatch: Any) -> Any:
    """Grant the ``enterprise_console`` Enterprise entitlement via a locally-signed license."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    pub_hex = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    payload = {
        "license_id": "lic-test",
        "edition": "enterprise",
        "features": ["enterprise_console"],
        "customer": "Test",
        "issued_at": int(time.time()),
        "expires_at": 0,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    token = f"{_lic._b64url_encode(raw)}.{_lic._b64url_encode(priv.sign(raw))}"
    monkeypatch.setattr(_lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    monkeypatch.setenv(_lic.HIMMY_LICENSE_KEY_ENV, token)
    _lic.reset_license_cache()
    yield
    _lic.reset_license_cache()


@pytest.fixture
def community(monkeypatch: Any) -> Any:
    """Force the community (no-license) posture — the fail-closed default."""
    monkeypatch.delenv(_lic.HIMMY_LICENSE_KEY_ENV, raising=False)
    monkeypatch.setattr(_lic, "license_file_path", lambda: "/nonexistent/license")
    monkeypatch.setattr("himmy.config.secrets.get_secret", lambda *a, **k: None)
    _lic.reset_license_cache()
    yield
    _lic.reset_license_cache()


# --------------------------------------------------------------- app + seeding
def _app() -> TestClient:
    """An app with tenant-'a' / tenant-'b' operators, a data_subject, and an admin."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "op-a": Principal.build(
                "op-a", tenant_ids=["a"], roles=["operator"], auth_method="apikey"
            ),
            "op-b": Principal.build(
                "op-b", tenant_ids=["b"], roles=["operator"], auth_method="apikey"
            ),
            "subject": Principal.build(
                "sub", tenant_ids=["a"], roles=["data_subject"], auth_method="apikey"
            ),
            "admin": Principal.build(
                "root", all_tenants=True, roles=["admin"], auth_method="apikey"
            ),
        }
    )
    return TestClient(app)


async def _seed(client: TestClient) -> None:
    """Seed runs + inference cost events + security events for tenants 'a' and 'b'."""
    storage = client.app.state.container.storage  # type: ignore[attr-defined]
    audit = client.app.state.security_audit  # type: ignore[attr-defined]

    # Tenant 'a': two runs (one SUCCEEDED / one FAILED), two agents, two cost events.
    await storage.save_run(
        RunRecord(
            run_id="a1",
            workspace_id="a",
            subject_id="s1",
            status=RunStatus.SUCCEEDED,
            trace_id="ta1",
            model_key="gpt-4o",
            metadata={},
        )
    )
    await storage.save_run(
        RunRecord(
            run_id="a2",
            workspace_id="a",
            subject_id="s1",
            status=RunStatus.FAILED,
            trace_id="ta2",
            model_key="llama3",
            metadata={},
        )
    )
    await storage.append_event(
        RunEvent(
            event_type=EventType.INFERENCE_SUCCEEDED,
            trace_id="ta1",
            agent_id="agent-a",
            workspace_id="a",
            cost=1.5,
            payload={"input_tokens": 100, "output_tokens": 50},
        )
    )
    await storage.append_event(
        RunEvent(
            event_type=EventType.INFERENCE_SUCCEEDED,
            trace_id="ta2",
            agent_id="agent-a",
            workspace_id="a",
            cost=0.5,
            payload={"input_tokens": 20, "output_tokens": 10},
        )
    )

    # Tenant 'b': one run + one (expensive) cost event — must NOT leak into 'a'.
    await storage.save_run(
        RunRecord(
            run_id="b1",
            workspace_id="b",
            subject_id="s2",
            status=RunStatus.SUCCEEDED,
            trace_id="tb1",
            model_key="claude",
            metadata={},
        )
    )
    await storage.append_event(
        RunEvent(
            event_type=EventType.INFERENCE_SUCCEEDED,
            trace_id="tb1",
            agent_id="agent-b",
            workspace_id="b",
            cost=99.0,
            payload={"input_tokens": 1, "output_tokens": 1},
        )
    )

    # Security events, one per tenant.
    audit.record(
        SecurityEvent(
            event_type="authz_denied",
            outcome="deny",
            workspace_id="a",
            actor={"subject": "mallory"},
            resource="run",
            action="write",
            detail="denied for a",
        )
    )
    audit.record(
        SecurityEvent(
            event_type="authz_denied",
            outcome="deny",
            workspace_id="b",
            actor={"subject": "eve"},
            resource="run",
            action="write",
            detail="denied for b",
        )
    )


# --------------------------------------------------------------- entitlement gate
def test_community_is_402_with_upgrade_message(community: None) -> None:
    client = _app()
    resp = client.get(
        "/v1/enterprise/overview?workspace_id=a",
        headers={"x-himmy-internal-key": "op-a"},
    )
    assert resp.status_code == 402, resp.text
    assert "Enterprise Edition feature" in resp.json()["detail"]
    assert "himmy license install" in resp.json()["detail"]


def test_enterprise_returns_data(ee_console: None) -> None:
    import anyio

    client = _app()
    anyio.run(_seed, client)
    resp = client.get(
        "/v1/enterprise/overview?workspace_id=a",
        headers={"x-himmy-internal-key": "op-a"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["workspace_id"] == "a"
    assert body["health"]["edition"] == "enterprise"


# --------------------------------------------------------------- RBAC gate
def test_unauthorized_role_is_403(ee_console: None) -> None:
    """An entitled but unauthorized principal (no ``dashboard:read``) is 403, not data."""
    client = _app()
    resp = client.get(
        "/v1/enterprise/overview?workspace_id=a",
        headers={"x-himmy-internal-key": "subject"},
    )
    assert resp.status_code == 403


# --------------------------------------------------------------- rollup correctness
def test_rollups_are_correct(ee_console: None) -> None:
    import anyio

    client = _app()
    anyio.run(_seed, client)
    body = client.get(
        "/v1/enterprise/overview?workspace_id=a",
        headers={"x-himmy-internal-key": "op-a"},
    ).json()

    # Runs rollup.
    assert body["runs"]["total"] == 2
    assert body["runs"]["by_status"]["SUCCEEDED"] == 1
    assert body["runs"]["by_status"]["FAILED"] == 1

    # Cost total = 1.5 + 0.5 (tenant 'a' only; 'b' 99.0 excluded).
    assert body["cost_total"] == pytest.approx(2.0)

    # Cost per agent: all of tenant 'a' cost is on 'agent-a'.
    by_agent = {b["key"]: b for b in body["cost_by_agent"]}
    assert by_agent["agent-a"]["cost"] == pytest.approx(2.0)
    assert by_agent["agent-a"]["calls"] == 2
    assert by_agent["agent-a"]["input_tokens"] == 120

    # Cost per model: gpt-4o (trace ta1) = 1.5, llama3 (trace ta2) = 0.5.
    by_model = {b["key"]: b for b in body["cost_by_model"]}
    assert by_model["gpt-4o"]["cost"] == pytest.approx(1.5)
    assert by_model["llama3"]["cost"] == pytest.approx(0.5)

    # Recent events (tenant 'a' only).
    assert len(body["recent_events"]) == 1
    assert body["recent_events"][0]["detail"] == "denied for a"

    # Agents summary + RBAC + health present.
    assert body["rbac"]["roles"]  # non-empty policy
    assert "operator" in body["rbac"]["roles"]
    assert body["health"]["ready"] is True


# --------------------------------------------------------------- tenant scoping
def test_tenant_scoped_no_cross_leak(ee_console: None) -> None:
    """Tenant A's overview never carries tenant B's runs / cost / events (and vice versa)."""
    import anyio

    client = _app()
    anyio.run(_seed, client)

    a = client.get(
        "/v1/enterprise/overview?workspace_id=a",
        headers={"x-himmy-internal-key": "op-a"},
    ).json()
    # B's expensive 99.0 cost and B's model/agent must be absent from A.
    assert a["cost_total"] == pytest.approx(2.0)
    assert "agent-b" not in {b["key"] for b in a["cost_by_agent"]}
    assert "claude" not in {b["key"] for b in a["cost_by_model"]}
    assert all(e["detail"] != "denied for b" for e in a["recent_events"])

    b = client.get(
        "/v1/enterprise/overview?workspace_id=b",
        headers={"x-himmy-internal-key": "op-b"},
    ).json()
    assert b["runs"]["total"] == 1
    assert b["cost_total"] == pytest.approx(99.0)
    assert {x["key"] for x in b["cost_by_agent"]} == {"agent-b"}
    assert len(b["recent_events"]) == 1
    assert b["recent_events"][0]["detail"] == "denied for b"


def test_cross_tenant_workspace_denied(ee_console: None) -> None:
    """A tenant-'a' principal cannot request tenant 'b's overview → 403."""
    client = _app()
    resp = client.get(
        "/v1/enterprise/overview?workspace_id=b",
        headers={"x-himmy-internal-key": "op-a"},
    )
    assert resp.status_code == 403
