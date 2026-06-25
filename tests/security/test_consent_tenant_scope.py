"""Security regression — consent ledger must be tenant-scoped + crypto-shred fail-safe.

Two layers are pinned here:

1. **Identity scoping (Finding #1).** ``consent_stable_id`` keys version chains by
   ``(workspace_id, subject_id, purpose)`` so the same ``subject_id`` in two tenants is two
   independent chains — a tenant-A operator can neither read nor withdraw a tenant-B subject's
   ledger state.

2. **Crypto-shred fail-safe (Finding #1, re-attack).** A brutal live re-attack proved the prior
   fix used the WRONG primitive. In production every subject key is minted via
   ``vault.encryptor_for(subject)`` with NO ``workspace_id`` (see consent_registry.py:100 and
   sidecar_storage.py), so ``workspace_of(subject)`` is ALWAYS ``None`` (a GLOBAL key). The prior
   gate ``owns_subject = bound_ws is None or bound_ws == workspace_id`` therefore degraded to a
   no-op: it ALWAYS allowed the shred, so a tenant-A operator could forge a grant+withdraw and
   permanently crypto-shred a GLOBAL key shared by tenant B. The fix FAILS SAFE: a multi-tenant
   caller (``workspace_id`` set) may shred ONLY a key provably bound to its own workspace; a
   global/unbound key is NEVER shredded per-tenant (that would destroy other tenants' data).
   Single-tenant/offline callers (``workspace_id is None``) still shred — legacy erasure is
   unchanged.

These tests mint keys the PRODUCTION way (``encryptor_for`` with no workspace) so they reproduce
the real bug, in an ISOLATED keyvault (``HIMMY_HOME`` tmp / explicit ``SubjectKeyVault(tmp)``) —
NEVER the repo ``.himmy/keyvault.db``, which a synthetic in-repo read both contaminates and
mis-reports (that contamination produced the prior FALSE "held" result).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.entities.registry import EntityRegistry
from himmy.services.governance.consent import (
    ConsentPolicy,
    ConsentState,
    Effect,
    Purpose,
    consent_stable_id,
)
from himmy.services.governance.consent_ledger import ConsentLedger
from himmy.services.governance.retention import RetentionService, SubjectKeyVault


# ----------------------------------------------------------------- unit: identity
def test_stable_id_is_workspace_scoped() -> None:
    """Same (subject, purpose) in two workspaces must not collide; legacy id preserved."""
    a = consent_stable_id("alice", Purpose.RETAIN, workspace_id="t_a")
    b = consent_stable_id("alice", Purpose.RETAIN, workspace_id="t_b")
    legacy = consent_stable_id("alice", Purpose.RETAIN)
    assert a != b  # two tenants → two independent chains
    assert a != legacy and b != legacy
    # Backward-compat: omitting workspace reproduces the old, unscoped id exactly.
    assert legacy == consent_stable_id("alice", Purpose.RETAIN, workspace_id=None)


# ----------------------------------------------------------------- unit: ledger reads
def test_ledger_reads_do_not_leak_across_tenants() -> None:
    registry = EntityRegistry()
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
    ledger.grant("alice", Purpose.RETAIN, workspace_id="t_a")

    # Tenant A sees the grant; tenant B sees nothing for the same subject id.
    assert ledger.state("alice", Purpose.RETAIN, workspace_id="t_a") is (
        ConsentState.GRANTED
    )
    assert ledger.state("alice", Purpose.RETAIN, workspace_id="t_b") is (
        ConsentState.UNKNOWN
    )
    assert ledger.decision("alice", Purpose.RETAIN, workspace_id="t_a").effect is (
        Effect.ALLOW
    )
    # Governed default for an unknown subject in tenant B → DENY (no cross-read).
    assert ledger.decision("alice", Purpose.RETAIN, workspace_id="t_b").effect is (
        Effect.DENY
    )
    assert ledger.latest("alice", Purpose.RETAIN, workspace_id="t_b") is None
    assert ledger.history("alice", Purpose.RETAIN, workspace_id="t_b") == []


# --------------------------------------------- unit: cross-tenant withdraw (no shred)
def test_ledger_withdraw_is_tenant_scoped_and_does_not_shred_other_tenant(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A tenant-B caller withdrawing tenant-A's subject is a no-op and never shreds.

    The victim key is minted the PRODUCTION way — ``vault.encryptor_for("alice")`` with NO
    workspace_id — so ``workspace_of("alice")`` is ``None`` (a global key). The withdrawal
    scoped to tenant B finds no record in B (empty version chain) → no WITHDRAWN record and
    ``erase_subject`` is never reached.
    """
    registry = EntityRegistry()
    vault = SubjectKeyVault(
        str(tmp_path / "keyvault.db")
    )  # isolated, never the repo vault
    ledger = ConsentLedger(
        registry,
        policy=ConsentPolicy(governed=True),
        retention_service=RetentionService(registry, key_vault=vault),
    )
    # The subject exists ONLY in tenant A's ledger; its vault key is GLOBAL (production mint).
    ledger.grant("alice", Purpose.RETAIN, workspace_id="t_a")
    vault.encryptor_for(
        "alice"
    )  # mints the key the production way (no workspace binding)
    assert vault.workspace_of("alice") is None
    alice_key_version = vault.key_version_of("alice")
    assert alice_key_version is not None

    # A tenant-B caller withdraws the SAME subject_id: no record in B → no-op, NO shred.
    records = ledger.withdraw("alice", Purpose.RETAIN, workspace_id="t_b")
    assert records == []
    assert not registry.list_by_kind("erasure_tombstone")
    # Tenant A's consent is untouched (still GRANTED → ALLOW) and the key is alive.
    assert ledger.decision("alice", Purpose.RETAIN, workspace_id="t_a").effect is (
        Effect.ALLOW
    )
    assert vault.key_version_of("alice") == alice_key_version


# ----------------------- unit: multi-tenant withdraw on a GLOBAL key must NOT shred (re-attack)
def test_multi_tenant_withdraw_does_not_shred_global_key(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The proven live re-attack: forge grant+withdraw in tenant A to shred a GLOBAL key.

    In production keys are minted via ``encryptor_for(subject)`` with NO workspace, so
    ``workspace_of`` is ``None``. The OLD gate ``bound_ws is None or ...`` treated that as
    "unbound → erase proceeds", letting a tenant-A operator crypto-shred a key shared by every
    tenant. The FIX fails safe: a multi-tenant caller (``workspace_id`` set) on an unbound/global
    key does NOT shred. The caller still gets its OWN WITHDRAWN record, but the key SURVIVES.
    """
    registry = EntityRegistry()
    vault = SubjectKeyVault(str(tmp_path / "keyvault.db"))
    ledger = ConsentLedger(
        registry,
        policy=ConsentPolicy(governed=True),
        retention_service=RetentionService(registry, key_vault=vault),
    )

    # 'bob' has a GLOBAL key (the production default: no workspace_id on the mint). Data for
    # potentially MANY tenants is encrypted under it; per-tenant shredding it is unsound.
    vault.encryptor_for("bob")
    assert vault.workspace_of("bob") is None
    bob_key_version = vault.key_version_of("bob")
    assert bob_key_version is not None

    # Attacker = a tenant-A operator. They forge a grant for 'bob' in their OWN workspace A
    # (so the old "records non-empty" guard would pass) ...
    ledger.grant("bob", Purpose.RETAIN, workspace_id="t_a")
    # ... then withdraw, attempting to trigger the (formerly tenant-blind) crypto-shred.
    records = ledger.withdraw("bob", Purpose.RETAIN, workspace_id="t_a")

    # The caller STILL gets a WITHDRAWN record for their own (forged) tenant-A consent ...
    assert [r.payload["state"] for r in records] == ["withdrawn"]
    # ... but the GLOBAL vault key SURVIVES — the multi-tenant shred is refused (fail safe).
    assert vault.has("bob")
    assert vault.key_version_of("bob") == bob_key_version
    assert not registry.list_by_kind("erasure_tombstone")
    # Defense-in-depth: a direct workspace-scoped destroy of the unbound key REFUSES ...
    assert vault.destroy("bob", workspace_id="t_a") is False
    assert vault.has("bob")
    # ... while the legacy unscoped destroy (single-tenant/offline) still shreds.
    assert vault.destroy("bob", workspace_id=None) is True
    assert not vault.has("bob")


# -------------------------------------- unit: single-tenant (offline) withdraw still shreds
def test_single_tenant_withdraw_still_shreds(tmp_path: pytest.TempPathFactory) -> None:
    """The legacy single-tenant/offline path (``workspace_id=None``) must still crypto-shred.

    There is no other tenant to harm when the caller is single-tenant, so erasure is unchanged:
    a ``workspace_id=None`` withdraw appends WITHDRAWN, calls ``erase_subject``, and the key is
    destroyed (tombstone written).
    """
    registry = EntityRegistry()
    vault = SubjectKeyVault(str(tmp_path / "keyvault.db"))
    ledger = ConsentLedger(
        registry,
        policy=ConsentPolicy(governed=True),
        retention_service=RetentionService(registry, key_vault=vault),
    )
    ledger.grant("carol", Purpose.RETAIN)  # no workspace → legacy/offline chain
    vault.encryptor_for("carol")  # global key, production mint
    assert vault.has("carol")

    records = ledger.withdraw("carol", Purpose.RETAIN, workspace_id=None)
    assert [r.payload["state"] for r in records] == ["withdrawn"]
    assert registry.list_by_kind("erasure_tombstone")  # erasure ran
    assert not vault.has("carol")  # key crypto-shredded


# ----------------------------------------------------------------- HTTP integration
def _two_tenant_operator_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A governed client whose operator is entitled to BOTH tenants 'a' and 'b'."""
    monkeypatch.setenv("HIMMY_CONSENT", "on")
    monkeypatch.delenv("HIMMY_CONSENT_FILE", raising=False)
    # The cross-site/DNS-rebinding guard is orthogonal to the tenant-scoping under test;
    # disable it (its own escape hatch) so TestClient POSTs reach the router.
    monkeypatch.setenv("HIMMY_STUDIO_GUARD", "0")
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "op", tenant_ids=["a", "b"], roles=["operator"], auth_method="apikey"
            )
        }
    )
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    return client


@pytest.fixture(autouse=True)
def _isolated_keyvault(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """Point ALL durable himmy state (incl. the keyvault) at a throwaway dir.

    ``ApiContainer.build_default`` wires ``SubjectKeyVault(keyvault_db_path())`` — the REAL
    repo ``.himmy/keyvault.db``. Without this, the HTTP tests below read AND mutate the
    production-critical vault (which is exactly what masked the bug in the prior FALSE pass).
    ``HIMMY_HOME`` reroots ``himmy_dir`` (and every default db path) at a tmp dir.
    """
    monkeypatch.setenv("HIMMY_HOME", str(tmp_path))


def test_cross_tenant_read_is_isolated_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _two_tenant_operator_client(monkeypatch)
    # Grant for subject 'sub' in tenant 'a'.
    assert (
        c.post(
            "/v1/consent/grant",
            json={"subject_id": "sub", "purpose": "retain", "workspace_id": "a"},
        ).status_code
        == 200
    )
    # Same subject id, read scoped to tenant 'b' → no leak (UNKNOWN, not allowed).
    r = c.get(
        "/v1/consent/decision",
        params={"subject_id": "sub", "purpose": "retain", "workspace_id": "b"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "unknown"
    assert body["allowed"] is False
    # Tenant 'b' latest/history are empty for the same subject id.
    assert (
        c.get(
            "/v1/consent/latest",
            params={"subject_id": "sub", "purpose": "retain", "workspace_id": "b"},
        ).json()
        is None
    )
    assert (
        c.get(
            "/v1/consent/history",
            params={"subject_id": "sub", "purpose": "retain", "workspace_id": "b"},
        ).json()
        == []
    )


def test_cross_tenant_withdraw_does_not_shred_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _two_tenant_operator_client(monkeypatch)
    # Subject 'sub' is granted in tenant 'a' only.
    c.post(
        "/v1/consent/grant",
        json={"subject_id": "sub", "purpose": "retain", "workspace_id": "a"},
    )

    # A withdrawal scoped to tenant 'b' must NOT crypto-shred tenant 'a''s subject.
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "sub", "purpose": "retain", "workspace_id": "b"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == []  # nothing withdrawn in tenant 'b'
    container = c.app.state.container  # type: ignore[attr-defined]
    assert not container.entity_registry.list_by_kind("erasure_tombstone")
    # Tenant 'a' consent survives intact.
    assert (
        c.get(
            "/v1/consent/decision",
            params={"subject_id": "sub", "purpose": "retain", "workspace_id": "a"},
        ).json()["allowed"]
        is True
    )

    # The rightful tenant 'a' withdrawal flips state to WITHDRAWN end-to-end ...
    r = c.post(
        "/v1/consent/withdraw",
        json={"subject_id": "sub", "purpose": "retain", "workspace_id": "a"},
    )
    assert r.status_code == 200, r.text
    assert [rec["state"] for rec in r.json()] == ["withdrawn"]
    assert (
        c.get(
            "/v1/consent/decision",
            params={"subject_id": "sub", "purpose": "retain", "workspace_id": "a"},
        ).json()["allowed"]
        is False
    )
    # ... but because the subject's vault key is GLOBAL (minted with no workspace), the
    # workspace-scoped withdraw does NOT crypto-shred it (fail safe): no erasure tombstone.
    # A per-tenant erasure of a shared global key would destroy other tenants' data; per-tenant
    # keys (an owner design decision) are the proper long-term fix.
    assert not container.entity_registry.list_by_kind("erasure_tombstone")
