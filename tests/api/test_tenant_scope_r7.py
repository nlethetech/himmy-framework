"""red-team scope-r7: by-construction tenant/subject scoping for the next leak class.

CONFIRMED cross-tenant / cross-subject holes a TENANT-BOUND (or ``subject_scoped``) principal
could reach once auth is configured — each closed by routing the read/erase path through the
centralized scoping the rest of the app enforces, and each asserted byte-unchanged on the
offline / ``all_tenants`` path:

* ``POST /api/studio/privacy/audit/export`` + ``/verify`` — a tenant-bound admin walked away
  with a SIGNED bundle of EVERY tenant's governance evidence (consent chains, erasure
  tombstones, security events, privacy reports) because ``_evidence_records`` iterated the
  whole spine with NO tenant filter. Now the evidence set routes through ``_visible_records``
  (tenant + subject axis), exactly like the ``/subjects`` + ``/consents`` readers.
* ``GET /api/studio/privacy/subjects`` + ``/consents`` — enforced only the TENANT axis. A
  ``subject_scoped`` admin enumerated every OTHER subject's consent chains + footprint WITHIN
  its tenant. Now ``_visible_records`` pins each record's ``subject_id`` against
  ``studio_subject_filter`` too.
* ``GET /api/studio/lineage/graph?run_id=`` + ``/entity/{id}`` — gated only the TENANT axis.
  A ``subject_scoped`` caller read another subject's run/entity lineage payloads WITHIN its
  tenant. Now the run path gates the run's ``subject_id`` (``_run_subject_blocked``), the
  entity path gates the TARGET record's OWN subject + workspace (anchor-confusion fix), and
  per-node/link filters drop foreign-subject records.
* ``POST /api/studio/privacy/erase`` — the un-hardened twin of ``/v1/consent/withdraw``: it
  called ``ledger.withdraw()`` with NO ``workspace_id``, so a tenant-bound caller's
  crypto-shred + hard-delete fired on a GLOBAL/foreign-bound key across ALL tenants. Now it
  threads the verified principal's ``resolve_workspace`` tenant in, so the ledger's fail-safe
  refuses to shred a key the caller does not provably own.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy
from himmy.entities.records import EntityRecord

_ADMIN_POLICY = AccessPolicy.from_mapping({"admin": ["*:*"]})


def _admin_app(
    subject: str,
    tenant: str | None,
    *,
    subject_scoped: bool = False,
    consent: bool = False,
    monkeypatch: pytest.MonkeyPatch | None = None,
    audit_secret: str | None = None,
) -> TestClient:
    """A built app authenticated as an admin (``*:*``), tenant- and/or subject-scoped.

    ``tenant=None`` builds the OFFLINE / ``all_tenants`` default (no authenticator), so the
    same fixtures can prove the byte-unchanged invariant.
    """
    if consent and monkeypatch is not None:
        monkeypatch.setenv("HIMMY_CONSENT", "on")
        monkeypatch.delenv("HIMMY_CONSENT_FILE", raising=False)
    if audit_secret is not None and monkeypatch is not None:
        monkeypatch.setenv("HIMMY_AUDIT_SECRET", audit_secret)
        monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    app = create_app(ApiContainer.build_default())
    if tenant is not None:
        app.state.authenticator = ApiKeyAuthenticator(
            key_principals={
                "k": Principal.build(
                    subject,
                    tenant_ids=[tenant],
                    roles=["admin"],
                    auth_method="apikey",
                    subject_scoped=subject_scoped,
                )
            }
        )
        app.state.access_policy = _ADMIN_POLICY
    client = TestClient(app)
    if tenant is not None:
        client.headers.update({"x-himmy-internal-key": "k"})
    return client


def _grant(client: TestClient, subject: str, *, workspace_id: str | None) -> None:
    """Record a GRANTED consent stamped to ``workspace_id`` (the tenant that recorded it)."""
    from himmy.services.governance.consent import Purpose

    ledger = client.app.state.consent_ledger  # type: ignore[attr-defined]
    ledger.grant(
        subject, Purpose("retain"), workspace_id=workspace_id, actor="t", source="t"
    )


# ============ vuln 1/7: audit export/verify leaks every tenant's evidence ============


def test_tenant_admin_audit_export_is_tenant_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-A admin's signed audit bundle commits to ONLY A's governance records.

    FAILS before scope-r7 (``_evidence_records`` called ``list_by_kind`` directly, so the
    bundle's ``record_count`` + ``bundle.records`` ids covered tenant B too); PASSES after.
    """
    monkeypatch.chdir(tmp_path)
    client = _admin_app(
        "admin-a", "A", consent=True, monkeypatch=monkeypatch, audit_secret="x"
    )
    _grant(client, "alice", workspace_id="A")
    _grant(client, "bob", workspace_id="B")  # another tenant's subject

    exported = client.post("/api/studio/privacy/audit/export", json={})
    assert exported.status_code == 200, exported.text
    envelope = exported.json()
    # Exactly one consent record (alice@A); tenant B's bob is never committed to the bundle.
    assert envelope["record_count"] == 1, envelope
    assert "bob" not in exported.text


def test_offline_audit_export_sees_all_tenants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline / all_tenants: the bundle covers EVERY tenant's evidence — byte-unchanged."""
    monkeypatch.chdir(tmp_path)
    client = _admin_app(
        "x", None, consent=True, monkeypatch=monkeypatch, audit_secret="x"
    )
    _grant(client, "alice", workspace_id="A")
    _grant(client, "bob", workspace_id="B")

    envelope = client.post("/api/studio/privacy/audit/export", json={}).json()
    assert envelope["record_count"] == 2, envelope


# ============ vuln 2: privacy subjects/consents enforce only the TENANT axis ============


def test_subject_scoped_admin_privacy_consents_is_subject_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subject_scoped admin sees ONLY its OWN subject's consents WITHIN its tenant.

    All three subjects live in tenant A (tenant axis passes); the SUBJECT axis must still
    hide bob/carol from alice. FAILS before scope-r7 (no subject narrowing on this surface).
    """
    monkeypatch.chdir(tmp_path)
    client = _admin_app(
        "alice", "A", subject_scoped=True, consent=True, monkeypatch=monkeypatch
    )
    _grant(client, "alice", workspace_id="A")
    _grant(client, "bob", workspace_id="A")
    _grant(client, "carol", workspace_id="A")

    body = client.get("/api/studio/privacy/consents").json()
    assert {e["subject_id"] for e in body["items"]} == {"alice"}
    # The free ?subject= param cannot reach another subject either.
    probed = client.get("/api/studio/privacy/consents", params={"subject": "bob"}).json()
    assert probed["items"] == []


def test_subject_scoped_admin_privacy_subjects_is_subject_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subject_scoped admin's /subjects roster lists ONLY its own subject."""
    monkeypatch.chdir(tmp_path)
    client = _admin_app(
        "alice", "A", subject_scoped=True, consent=True, monkeypatch=monkeypatch
    )
    _grant(client, "alice", workspace_id="A")
    _grant(client, "bob", workspace_id="A")

    body = client.get("/api/studio/privacy/subjects").json()
    assert {s["subject_id"] for s in body["subjects"]} == {"alice"}
    assert "bob" not in client.get("/api/studio/privacy/subjects").text


def test_tenant_admin_privacy_still_crosses_subjects_in_its_tenant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NON-subject-scoped tenant admin still sees every subject in its tenant — unchanged.

    The subject axis only narrows an opt-in ``subject_scoped`` principal; a plain tenant-bound
    admin keeps the historical within-tenant cross-subject reach.
    """
    monkeypatch.chdir(tmp_path)
    client = _admin_app("admin-a", "A", consent=True, monkeypatch=monkeypatch)
    _grant(client, "alice", workspace_id="A")
    _grant(client, "bob", workspace_id="A")

    body = client.get("/api/studio/privacy/subjects").json()
    assert {s["subject_id"] for s in body["subjects"]} == {"alice", "bob"}


# ============ vuln 3/8: lineage enforces TENANT but not SUBJECT axis ============


def _registry(client: TestClient) -> Any:
    return client.app.state.container.entity_registry  # type: ignore[attr-defined]


def _seed_subject_record(
    registry: Any, *, ws: str, subject: str, secret: str
) -> EntityRecord:
    """A workspace-stamped record attributed to ``subject`` carrying a secret prompt."""
    rec = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="prompt",
        payload={"prompt": secret},
        metadata={"workspace_id": ws, "subject_id": subject},
    )
    registry.register(rec)
    return rec


def test_subject_scoped_lineage_entity_hides_foreign_subject_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subject_scoped caller cannot read another subject's entity payload in its tenant.

    Both records live in tenant A (tenant axis passes), but bob's record is attributed to a
    different subject — the entity detail must 404 on its OWN-subject gate. FAILS before
    scope-r7 (the detail handler gated only the tenant/anchor axis).
    """
    monkeypatch.chdir(tmp_path)
    client = _admin_app("alice", "A", subject_scoped=True)
    bob_rec = _seed_subject_record(
        _registry(client), ws="A", subject="bob", secret="BOB-SECRET-PROMPT"
    )

    r = client.get(f"/api/studio/lineage/entity/{bob_rec.record_id}")
    assert r.status_code == 404, r.text
    assert "BOB-SECRET-PROMPT" not in r.text


def test_subject_scoped_lineage_entity_allows_own_subject_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same caller CAN read its OWN subject's record — the gate is narrowing, not a wall."""
    monkeypatch.chdir(tmp_path)
    client = _admin_app("alice", "A", subject_scoped=True)
    own = _seed_subject_record(
        _registry(client), ws="A", subject="alice", secret="ALICE-OWN-PROMPT"
    )

    r = client.get(f"/api/studio/lineage/entity/{own.record_id}")
    assert r.status_code == 200, r.text
    assert "ALICE-OWN-PROMPT" in r.text


def test_lineage_entity_anchor_confusion_does_not_leak_foreign_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign-tenant record reachable via a SHARED node is NOT disclosed by id (vuln 8).

    The attacker (tenant A) owns a thread linked to a shared (unstamped) persona that also
    links a tenant-B record. Querying the B record directly must 404 — the TARGET record's
    OWN workspace gate must fire, not merely the traced subgraph's in-tenant anchor.
    """
    monkeypatch.chdir(tmp_path)
    client = _admin_app("admin-a", "A")
    reg = _registry(client)
    own_thread = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="chat_thread",
        payload={"name": "own", "metadata": {"workspace_id": "A"}},
        metadata={"workspace_id": "A"},
    )
    shared = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="persona",
        payload={"name": "shared persona"},  # no workspace stamp
        metadata={},
    )
    foreign = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="prompt",
        payload={"prompt": "TENANT-B-SECRET-PROMPT"},
        metadata={"workspace_id": "B"},
    )
    for rec in (own_thread, shared, foreign):
        reg.register(rec)
    # own_thread -> shared <- foreign : the shared node bridges A's thread into B's record.
    reg.link(
        from_record_id=own_thread.record_id,
        to_record_id=shared.record_id,
        relation="uses_persona",
    )
    reg.link(
        from_record_id=foreign.record_id,
        to_record_id=shared.record_id,
        relation="uses_persona",
    )

    r = client.get(f"/api/studio/lineage/entity/{foreign.record_id}")
    assert r.status_code == 404, r.text
    assert "TENANT-B-SECRET-PROMPT" not in r.text


def test_offline_lineage_entity_discloses_any_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline / all_tenants: any subject's entity payload IS disclosed — byte-unchanged."""
    monkeypatch.chdir(tmp_path)
    client = _admin_app("x", None)
    bob_rec = _seed_subject_record(
        _registry(client), ws="A", subject="bob", secret="BOB-SECRET-PROMPT"
    )
    r = client.get(f"/api/studio/lineage/entity/{bob_rec.record_id}")
    assert r.status_code == 200, r.text
    assert "BOB-SECRET-PROMPT" in r.text


# ============ vuln 4: Studio erase crypto-shreds a global key cross-tenant ============


def _register_keyed_message(client: TestClient, subject: str) -> None:
    """Register a subject-bearing message through the GATED registry (creates the vault key)."""
    registry = client.app.state.container.runtime.entity_registry  # type: ignore[attr-defined]
    registry.register(
        EntityRecord.create(
            stable_id=str(uuid.uuid4()),
            version=1,
            kind="message",
            payload={"content": "secret"},
            metadata={"subject_id": subject},
        )
    )


def test_tenant_admin_erase_refuses_to_shred_global_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-bound admin's /erase must NOT crypto-shred a GLOBAL/unbound subject key.

    The subject's vault key is minted globally (no workspace binding — the production
    default). Before scope-r7 the Studio erase called ``withdraw`` with NO ``workspace_id``,
    so the fail-safe was bypassed (``workspace_id is None`` => owns_subject True) and the
    global key was shredded + the subject hard-deleted across ALL tenants. After the fix the
    route threads the caller's tenant in, so ``withdraw``'s ownership gate refuses the shred.
    """
    monkeypatch.chdir(tmp_path)
    client = _admin_app("admin-a", "A", consent=True, monkeypatch=monkeypatch)
    # alice's consent is recorded under tenant A (so the 404 pre-check passes for tenant A),
    # but her vault key is GLOBAL/unbound (minted without a workspace).
    _grant(client, "alice", workspace_id="A")
    _register_keyed_message(client, "alice")

    r = client.post(
        "/api/studio/privacy/erase", json={"subject_id": "alice", "confirm": "alice"}
    )
    assert r.status_code == 200, r.text
    # The destructive crypto-shred of the global key was REFUSED (fail-safe), so it is False —
    # not a cross-tenant key destruction.
    assert r.json()["crypto_shredded"] is False


def test_offline_erase_still_shreds_global_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline / all_tenants: the single-box erase legitimately shreds the key — byte-unchanged."""
    monkeypatch.chdir(tmp_path)
    client = _admin_app("x", None, consent=True, monkeypatch=monkeypatch)
    _grant(client, "alice", workspace_id=None)
    _register_keyed_message(client, "alice")

    r = client.post(
        "/api/studio/privacy/erase", json={"subject_id": "alice", "confirm": "alice"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["crypto_shredded"] is True
