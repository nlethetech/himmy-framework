"""WS4.6 A3 — /v1/consent router: RBAC matrix, self-scope, tenant isolation, withdrawal.

The router only exists in a governed deployment (``HIMMY_CONSENT=on``); these tests build a
governed container, attach an :class:`ApiKeyAuthenticator`, and assert:

* operator/admin may write (grant/deny), viewer is 403, auditor reads but cannot write;
* a ``data_subject`` principal may read/withdraw ONLY its own subject_id (cross-subject 403);
* withdraw crypto-shreds (records a tombstone + flips the decision to DENY);
* a tenant-bound principal cannot grant in a workspace it is not entitled to (IDOR);
* the zero-config (no consent) app exposes the route as inert (404).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal


def _governed_client(
    monkeypatch: pytest.MonkeyPatch, principals: dict[str, Principal]
) -> TestClient:
    monkeypatch.setenv("HIMMY_CONSENT", "on")
    monkeypatch.delenv("HIMMY_CONSENT_FILE", raising=False)
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(key_principals=principals)
    return TestClient(app)


# A tiny helper so each client carries its key header.
def _keyed(client: TestClient) -> TestClient:
    client.headers.update({"x-himmy-internal-key": "k"})
    return client


def _grant_body(subject: str = "teacher_a", purpose: str = "retain") -> dict:
    return {"subject_id": subject, "purpose": purpose, "workspace_id": "t"}


# --------------------------------------------------------------- RBAC write matrix
def test_operator_can_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed(_governed_client(monkeypatch, _principals("operator")))
    r = c.post("/v1/consent/grant", json=_grant_body())
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "granted"


def test_operator_can_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed(_governed_client(monkeypatch, _principals("operator")))
    r = c.post("/v1/consent/deny", json=_grant_body(purpose="train"))
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "denied"


def test_viewer_cannot_write_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed(_governed_client(monkeypatch, _principals("viewer")))
    assert c.post("/v1/consent/grant", json=_grant_body()).status_code == 403


def test_viewer_cannot_read_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    # viewer has no consent:read either.
    c = _keyed(_governed_client(monkeypatch, _principals("viewer")))
    r = c.get(
        "/v1/consent/decision",
        params={"subject_id": "teacher_a", "purpose": "retain"},
    )
    assert r.status_code == 403


def test_auditor_can_read_but_not_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed(_governed_client(monkeypatch, _principals("auditor")))
    assert (
        c.get(
            "/v1/consent/decision",
            params={"subject_id": "teacher_a", "purpose": "retain"},
        ).status_code
        == 200
    )
    assert c.post("/v1/consent/grant", json=_grant_body()).status_code == 403


def test_admin_can_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed(_governed_client(monkeypatch, _principals("admin")))
    assert c.post("/v1/consent/grant", json=_grant_body()).status_code == 200


def _principals(role: str) -> dict[str, Principal]:
    return {
        "k": Principal.build(
            f"{role}-user", tenant_ids=["t"], roles=[role], auth_method="apikey"
        )
    }


# --------------------------------------------------------------- data_subject self-scope
def _data_subject_client(monkeypatch: pytest.MonkeyPatch, subject: str) -> TestClient:
    return _keyed(
        _governed_client(
            monkeypatch,
            {
                "k": Principal.build(
                    subject,
                    tenant_ids=["t"],
                    roles=["data_subject"],
                    auth_method="apikey",
                )
            },
        )
    )


def test_data_subject_can_read_own_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _data_subject_client(monkeypatch, "teacher_a")
    r = c.get(
        "/v1/consent/decision",
        params={"subject_id": "teacher_a", "purpose": "retain"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["subject_id"] == "teacher_a"


def test_data_subject_cannot_read_another_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _data_subject_client(monkeypatch, "teacher_a")
    r = c.get(
        "/v1/consent/decision",
        params={"subject_id": "teacher_b", "purpose": "retain"},
    )
    assert r.status_code == 403


def test_data_subject_cannot_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    # data_subject holds only consent:read, so write is forbidden even for itself.
    c = _data_subject_client(monkeypatch, "teacher_a")
    assert c.post("/v1/consent/grant", json=_grant_body("teacher_a")).status_code == 403


def test_data_subject_can_withdraw_own(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _data_subject_client(monkeypatch, "teacher_a")
    # Withdraw is consent:read-gated so a self-scoped subject can exercise erasure.
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "teacher_a", "purpose": "retain", "workspace_id": "t"},
    )
    assert r.status_code == 200, r.text


def test_data_subject_cannot_withdraw_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _data_subject_client(monkeypatch, "teacher_a")
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "teacher_b", "purpose": "retain", "workspace_id": "t"},
    )
    assert r.status_code == 403


# ------------------------------------------------- withdrawal flips decision (no shred)
def test_workspace_scoped_withdraw_flips_decision_without_shredding_global_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A workspace-scoped withdrawal flips the decision to WITHDRAWN but does NOT shred.

    SECURITY FAIL-SAFE (Phase-4): in production every subject key is minted via
    ``vault.encryptor_for(subject)`` with NO workspace_id (consent_registry.py:100), so
    the key is GLOBAL/unbound. A multi-tenant caller passing ``workspace_id`` therefore
    does not provably own the key, and crypto-shred is REFUSED — otherwise a tenant-A
    operator could forge a withdrawal and permanently destroy a key shared by tenant B.
    The ledger state still flips to WITHDRAWN (the decision correctly becomes DENY); only
    the destructive crypto-shred + erasure tombstone are withheld. (Single-tenant erasure,
    ``workspace_id=None``, still shreds — see test_offline_unscoped_withdraw_shreds.)
    """
    c = _keyed(_governed_client(monkeypatch, _principals("operator")))
    c.post("/v1/consent/grant", json=_grant_body())
    assert (
        c.get(
            "/v1/consent/decision",
            params={"subject_id": "teacher_a", "purpose": "retain"},
        ).json()["allowed"]
        is True
    )
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "teacher_a", "workspace_id": "t"},
    )
    assert r.status_code == 200
    # Decision now DENY (withdrawn) — ledger state flips regardless of the shred gate.
    after = c.get(
        "/v1/consent/decision",
        params={"subject_id": "teacher_a", "purpose": "retain"},
    ).json()
    assert after["allowed"] is False
    assert after["state"] == "withdrawn"
    # No erasure tombstone: the global key was NOT shredded by a workspace-scoped caller.
    container = c.app.state.container  # type: ignore[attr-defined]
    assert not container.entity_registry.list_by_kind("erasure_tombstone")


# ------------------------------------------- withdraw is :write for non-self callers
def test_auditor_cannot_withdraw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read-only auditor (consent:read, no consent:write) cannot erase a subject → 403.

    Closes the privilege hole where the destructive /withdraw (crypto-shred + erasure
    tombstone) sat behind consent:read; it now follows the :write-for-mutation convention.
    """
    c = _keyed(_governed_client(monkeypatch, _principals("auditor")))
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "teacher_a", "purpose": "retain", "workspace_id": "t"},
    )
    assert r.status_code == 403
    # And no erasure tombstone was written (the destructive op never ran).
    container = c.app.state.container  # type: ignore[attr-defined]
    assert not container.entity_registry.list_by_kind("erasure_tombstone")


def test_operator_can_withdraw(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator holds consent:write, so it may withdraw (mutating state change)."""
    c = _keyed(_governed_client(monkeypatch, _principals("operator")))
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "teacher_a", "purpose": "retain", "workspace_id": "t"},
    )
    assert r.status_code == 200, r.text


def test_admin_can_withdraw(monkeypatch: pytest.MonkeyPatch) -> None:
    """admin (``*:*``) may withdraw."""
    c = _keyed(_governed_client(monkeypatch, _principals("admin")))
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "teacher_a", "purpose": "retain", "workspace_id": "t"},
    )
    assert r.status_code == 200, r.text


def test_data_subject_with_operator_role_can_withdraw_any_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A data_subject that ALSO holds operator keeps operator's reach (consent:write).

    The self-erasure shortcut only applies to a pure data_subject acting on its OWN
    subject; a principal carrying a broader role is authorized via that role's write
    permission and the self-scope guard yields to it.
    """
    c = _keyed(
        _governed_client(
            monkeypatch,
            {
                "k": Principal.build(
                    "teacher_a",
                    tenant_ids=["t"],
                    roles=["data_subject", "operator"],
                    auth_method="apikey",
                )
            },
        )
    )
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "teacher_b", "purpose": "retain", "workspace_id": "t"},
    )
    assert r.status_code == 200, r.text


# --------------------------------------------------------------- tenant isolation (IDOR)
def test_cross_tenant_grant_is_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    # operator bound to tenant 't' tries to grant in workspace 'other' → 403.
    c = _keyed(_governed_client(monkeypatch, _principals("operator")))
    r = c.post(
        "/v1/consent/grant",
        json={"subject_id": "x", "purpose": "retain", "workspace_id": "other"},
    )
    assert r.status_code == 403


# --------------------------------------------------------------- history endpoint
def test_history_returns_version_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed(_governed_client(monkeypatch, _principals("operator")))
    c.post("/v1/consent/grant", json=_grant_body())
    c.post("/v1/consent/deny", json=_grant_body())
    r = c.get(
        "/v1/consent/history",
        params={"subject_id": "teacher_a", "purpose": "retain"},
    )
    assert r.status_code == 200
    states = [rec["state"] for rec in r.json()]
    assert states == ["granted", "denied"]


# --------------------------------------------------------------- offline inert (404)
def test_offline_app_has_inert_consent_route() -> None:
    """Zero-config app: the route exists but governance is off → 404 (no auth → no RBAC)."""
    c = TestClient(create_app(ApiContainer.build_default()))
    r = c.get(
        "/v1/consent/decision",
        params={"subject_id": "x", "purpose": "retain"},
    )
    assert r.status_code == 404
    assert "not enabled" in r.json()["detail"]
