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


# ---------------------------------------------- subject scoping (BOLA, ee sec-r1)
def _app_subject_scoped() -> TestClient:
    """An app whose tenant-'a' viewer is ``subject_scoped`` to its OWN data subject 's1'.

    Mirrors an OIDC + ``HIMMY_OIDC_SUBJECT_SCOPED=1`` deployment: a per-user principal that
    still holds ``dashboard:read`` (via the viewer role) but must only see its own subject.
    """
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            # 'alice' is subject-scoped to subject 's1' (owns runs a1/a2 seeded below).
            "alice": Principal.build(
                "s1",
                tenant_ids=["a"],
                roles=["viewer"],
                auth_method="apikey",
                subject_scoped=True,
            ),
            # An unscoped tenant-'a' operator (full-workspace baseline).
            "op-a": Principal.build(
                "op-a", tenant_ids=["a"], roles=["operator"], auth_method="apikey"
            ),
        }
    )
    return TestClient(app)


async def _seed_two_subjects(client: TestClient) -> None:
    """Two data subjects (s1, s2) in the SAME tenant 'a', with distinct runs/cost/audit."""
    storage = client.app.state.container.storage  # type: ignore[attr-defined]
    audit = client.app.state.security_audit  # type: ignore[attr-defined]

    # Subject s1 (alice's own): one run + one cost event.
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
    await storage.append_event(
        RunEvent(
            event_type=EventType.INFERENCE_SUCCEEDED,
            trace_id="ta1",
            agent_id="agent-alice",
            workspace_id="a",
            cost=2.0,
            payload={"input_tokens": 100, "output_tokens": 50},
        )
    )
    # Subject s2 (a SIBLING in the same tenant): must NOT leak to alice.
    await storage.save_run(
        RunRecord(
            run_id="a2",
            workspace_id="a",
            subject_id="s2",
            status=RunStatus.SUCCEEDED,
            trace_id="ta2",
            model_key="claude",
            metadata={},
        )
    )
    await storage.append_event(
        RunEvent(
            event_type=EventType.INFERENCE_SUCCEEDED,
            trace_id="ta2",
            agent_id="agent-bob",
            workspace_id="a",
            cost=77.0,
            payload={"input_tokens": 999, "output_tokens": 999},
        )
    )
    # Audit rows attributed to each subject.
    audit.record(
        SecurityEvent(
            event_type="access",
            outcome="allow",
            workspace_id="a",
            actor={"subject": "s1"},
            resource="run",
            action="read",
            detail="alice own action",
        )
    )
    audit.record(
        SecurityEvent(
            event_type="access",
            outcome="allow",
            workspace_id="a",
            actor={"subject": "s2"},
            resource="run",
            action="read",
            detail="bob private action",
        )
    )


def test_subject_scoped_console_sees_only_own_subject(ee_console: None) -> None:
    """ee sec-r1: a ``subject_scoped`` viewer's overview is pinned to its OWN subject.

    A sibling subject's runs / cost / tokens / audit detail in the SAME tenant must never
    appear — the console must mirror the ``/v1/runs`` subject narrowing, not just the
    tenant axis.
    """
    import anyio

    client = _app_subject_scoped()
    anyio.run(_seed_two_subjects, client)

    body = client.get(
        "/v1/enterprise/overview",
        headers={"x-himmy-internal-key": "alice"},
    ).json()

    # Only s1's single run — s2's run is invisible.
    assert body["runs"]["total"] == 1
    # Only s1's cost (2.0); s2's 77.0 must be excluded.
    assert body["cost_total"] == pytest.approx(2.0)
    assert "agent-bob" not in {b["key"] for b in body["cost_by_agent"]}
    assert "claude" not in {b["key"] for b in body["cost_by_model"]}
    # Audit: only s1's own actor rows — bob's private detail never leaks.
    details = {e["detail"] for e in body["recent_events"]}
    assert "alice own action" in details
    assert "bob private action" not in details


def test_unscoped_operator_still_sees_full_workspace(ee_console: None) -> None:
    """The non-subject-scoped operator keeps the full-workspace rollup (byte-unchanged)."""
    import anyio

    client = _app_subject_scoped()
    anyio.run(_seed_two_subjects, client)

    body = client.get(
        "/v1/enterprise/overview",
        headers={"x-himmy-internal-key": "op-a"},
    ).json()
    # Both subjects' runs + cost are visible to the unscoped operator.
    assert body["runs"]["total"] == 2
    assert body["cost_total"] == pytest.approx(79.0)


def test_overview_windows_event_scan(ee_console: None, monkeypatch: Any) -> None:
    """ee sec-r1: the cost pass is WINDOWED, not an unbounded full-history load.

    Asserts ``list_events`` is called with a finite ``limit`` (the cap) rather than
    ``None`` (load-everything), so a single request cannot materialize an unbounded set.
    """
    import anyio

    import himmy.api.routers.enterprise as ent

    client = _app()
    anyio.run(_seed, client)

    storage = client.app.state.container.storage  # type: ignore[attr-defined]
    seen: dict[str, Any] = {}
    orig = storage.list_events

    async def _spy(*args: Any, **kwargs: Any) -> Any:
        seen["limit"] = kwargs.get("limit")
        return await orig(*args, **kwargs)

    monkeypatch.setattr(storage, "list_events", _spy)

    resp = client.get(
        "/v1/enterprise/overview?workspace_id=a",
        headers={"x-himmy-internal-key": "op-a"},
    )
    assert resp.status_code == 200
    # A finite, non-None cap was passed (the DoS window), matching the module constant.
    assert seen["limit"] == ent._MAX_COST_EVENTS
    assert seen["limit"] is not None


def test_overview_audit_scan_is_bounded(ee_console: None) -> None:
    """ee sec-r2: the overview's audit pass NEVER materializes the full history.

    The recent-events step used to load+pydantic-validate every ``security_event`` row
    (``list_by_kind``) before slicing to ``events_limit``, so per-request work grew with
    lifetime audit history while output stayed ~20 rows — an asymmetric DoS. This asserts
    the hot path now takes the DB-side windowed reader (``list_recent_by_kind`` with a
    finite limit) and NEVER touches the unbounded ``list_by_kind``.
    """
    import anyio

    client = _app()
    anyio.run(_seed, client)

    audit = client.app.state.security_audit  # type: ignore[attr-defined]
    registry = audit._registry
    calls: dict[str, Any] = {"list_by_kind": 0, "windowed_limit": None}

    orig_full = registry.list_by_kind
    orig_window = registry.list_recent_by_kind

    def _spy_full(kind: str) -> Any:
        calls["list_by_kind"] += 1
        return orig_full(kind)

    def _spy_window(kind: str, *, limit: int, metadata_filters: Any = None) -> Any:
        calls["windowed_limit"] = limit
        return orig_window(kind, limit=limit, metadata_filters=metadata_filters)

    registry.list_by_kind = _spy_full  # type: ignore[method-assign]
    registry.list_recent_by_kind = _spy_window  # type: ignore[method-assign]

    resp = client.get(
        "/v1/enterprise/overview?workspace_id=a&events_limit=20",
        headers={"x-himmy-internal-key": "op-a"},
    )
    assert resp.status_code == 200, resp.text
    # The unbounded full-history scan was NOT used on the overview audit path...
    assert calls["list_by_kind"] == 0
    # ...and the bounded reader ran with the finite output window, not load-everything.
    assert calls["windowed_limit"] == 20
