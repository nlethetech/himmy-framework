"""Studio Privacy router — subjects/consents ledger, erasure, audit export/verify.

In-process FastAPI app, no network. Covers:

* the ungoverned (zero-config) path: reads answer ``governed: false`` and the
  destructive erase is 404-inert;
* the governed path: subject counts, consent version chains + PDP effects,
  typed-confirmation erasure (withdraw + crypto-shred + tombstone);
* audit export/verify: 503 without a signing key, HMAC round-trip, per-check
  verdicts incl. a tampered-bundle failure and bad-upload 400s.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.core.ids import new_uuid
from himmy.entities.records import EntityRecord


def _ungoverned_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("HIMMY_CONSENT", raising=False)
    return TestClient(create_app(ApiContainer.build_default()))


def _governed_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("HIMMY_CONSENT", "on")
    monkeypatch.delenv("HIMMY_CONSENT_FILE", raising=False)
    return TestClient(create_app(ApiContainer.build_default()))


def _grant(client: TestClient, subject: str, purpose: str = "retain") -> None:
    ledger = client.app.state.consent_ledger  # type: ignore[attr-defined]
    ledger.grant(subject, _purpose(purpose), actor="test", source="test")


def _purpose(value: str) -> Any:
    from himmy.services.governance.consent import Purpose

    return Purpose(value)


def _register_message(client: TestClient, subject: str, content: str) -> None:
    """Register a subject-bearing message through the GATED registry (the runtime's).

    Going through :class:`ConsentAwareRegistry` is what creates the subject's
    shreddable key (the record's fields are encrypted under it on ALLOW).
    """
    registry = client.app.state.container.runtime.entity_registry  # type: ignore[attr-defined]
    registry.register(
        EntityRecord.create(
            stable_id=new_uuid(),
            version=1,
            kind="message",
            payload={"content": content},
            metadata={"subject_id": subject},
        )
    )


# ------------------------------------------------------------- ungoverned path


def test_subjects_ungoverned_is_empty_not_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _ungoverned_client(monkeypatch)
    r = c.get("/api/studio/privacy/subjects")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["governed"] is False
    assert body["subjects"] == []


def test_consents_ungoverned_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _ungoverned_client(monkeypatch)
    r = c.get("/api/studio/privacy/consents")
    assert r.status_code == 200
    assert r.json() == {"governed": False, "items": []}


def test_erase_ungoverned_is_404_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _ungoverned_client(monkeypatch)
    r = c.post("/api/studio/privacy/erase", json={"subject_id": "s1", "confirm": "s1"})
    assert r.status_code == 404
    assert "HIMMY_CONSENT" in r.json()["detail"]


# --------------------------------------------------------------- governed reads


def test_subjects_counts_and_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")
    _grant(c, "alice", "train")
    _grant(c, "bob", "retain")
    _register_message(c, "alice", "hello")  # ALLOW → persisted (encrypted)

    body = c.get("/api/studio/privacy/subjects").json()
    assert body["governed"] is True
    by_id = {s["subject_id"]: s for s in body["subjects"]}
    assert by_id["alice"]["consent_records"] == 2
    assert by_id["alice"]["purposes"] == 2
    assert by_id["alice"]["data_records"] == 1
    assert by_id["alice"]["erased"] is False
    assert by_id["alice"]["last_activity"]
    assert by_id["bob"]["consent_records"] == 1
    assert by_id["bob"]["data_records"] == 0


def test_consents_filter_and_version_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")
    ledger = c.app.state.consent_ledger  # type: ignore[attr-defined]
    ledger.deny("alice", _purpose("retain"), actor="test", source="test")
    _grant(c, "bob", "retain")

    body = c.get("/api/studio/privacy/consents", params={"subject": "alice"}).json()
    assert body["governed"] is True
    assert {i["subject_id"] for i in body["items"]} == {"alice"}
    versions = sorted(i["version"] for i in body["items"])
    assert versions == [1, 2]
    latest = next(i for i in body["items"] if i["version"] == 2)
    assert latest["state"] == "denied"
    assert latest["effect"] == "deny"
    first = next(i for i in body["items"] if i["version"] == 1)
    assert first["state"] == "granted"
    assert first["effect"] == "allow"

    everyone = c.get("/api/studio/privacy/consents").json()
    assert {i["subject_id"] for i in everyone["items"]} == {"alice", "bob"}


def test_consents_subject_param_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _governed_client(monkeypatch)
    r = c.get("/api/studio/privacy/consents", params={"subject": "x" * 201})
    assert r.status_code == 422


# -------------------------------------------------------------------- erasure


def test_erase_requires_matching_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")
    r = c.post(
        "/api/studio/privacy/erase",
        json={"subject_id": "alice", "confirm": "alic"},
    )
    assert r.status_code == 400
    assert "confirmation" in r.json()["detail"]


def test_erase_unknown_subject_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _governed_client(monkeypatch)
    r = c.post(
        "/api/studio/privacy/erase",
        json={"subject_id": "ghost", "confirm": "ghost"},
    )
    assert r.status_code == 404


def test_erase_withdraws_shreds_and_tombstones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")
    _grant(c, "alice", "train")
    _register_message(c, "alice", "secret")  # creates alice's shreddable key

    r = c.post(
        "/api/studio/privacy/erase",
        json={"subject_id": "alice", "confirm": "alice"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(body["consents_withdrawn"]) == ["retain", "train"]
    assert body["data_records"] == 1
    assert body["crypto_shredded"] is True
    assert body["tombstone_id"]
    assert body["erased_at"]

    # The ledger now shows the subject as erased and every state WITHDRAWN.
    subjects = c.get("/api/studio/privacy/subjects").json()["subjects"]
    alice = next(s for s in subjects if s["subject_id"] == "alice")
    assert alice["erased"] is True
    consents = c.get(
        "/api/studio/privacy/consents", params={"subject": "alice"}
    ).json()["items"]
    latest_by_purpose: dict[str, dict[str, Any]] = {}
    for item in consents:
        cur = latest_by_purpose.get(item["purpose"])
        if cur is None or item["version"] > cur["version"]:
            latest_by_purpose[item["purpose"]] = item
    assert all(i["state"] == "withdrawn" for i in latest_by_purpose.values())


def test_erase_input_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _governed_client(monkeypatch)
    r = c.post(
        "/api/studio/privacy/erase",
        json={"subject_id": "x" * 201, "confirm": "x" * 201},
    )
    assert r.status_code == 422


# ----------------------------------------------------------- audit export/verify


def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIMMY_AUDIT_SECRET", raising=False)
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)


def test_export_503_without_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_keys(monkeypatch)
    c = _ungoverned_client(monkeypatch)
    r = c.post("/api/studio/privacy/audit/export", json={})
    assert r.status_code == 503
    assert "HIMMY_AUDIT" in r.json()["detail"]


def test_hmac_export_then_verify_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_keys(monkeypatch)
    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "studio-secret")
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")  # ensure there is evidence on the spine

    exported = c.post("/api/studio/privacy/audit/export", json={})
    assert exported.status_code == 200, exported.text
    envelope = exported.json()
    assert envelope["algorithm"] == "HMAC-SHA256"
    assert envelope["record_count"] >= 1
    assert envelope["bundle"]["signature"]

    verified = c.post("/api/studio/privacy/audit/verify", json=envelope)
    assert verified.status_code == 200, verified.text
    verdict = verified.json()
    assert verdict["ok"] is True
    names = {chk["name"]: chk["ok"] for chk in verdict["checks"]}
    assert names["signature"] is True
    assert names["records intact"] is True
    assert names["records present"] is True
    assert names["no records added"] is True


def test_verify_flags_tampered_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_keys(monkeypatch)
    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "studio-secret")
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")
    envelope = c.post("/api/studio/privacy/audit/export", json={}).json()
    envelope["bundle"]["signature"] = "00" * 32  # forged

    verdict = c.post("/api/studio/privacy/audit/verify", json=envelope).json()
    assert verdict["ok"] is False
    sig = next(chk for chk in verdict["checks"] if chk["name"] == "signature")
    assert sig["ok"] is False


def test_verify_detects_records_added_after_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_keys(monkeypatch)
    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "studio-secret")
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")
    envelope = c.post("/api/studio/privacy/audit/export", json={}).json()
    _grant(c, "bob", "retain")  # new consent evidence after export

    verdict = c.post("/api/studio/privacy/audit/verify", json=envelope).json()
    assert verdict["ok"] is False  # strict verdict: the live graph diverged
    added = next(c2 for c2 in verdict["checks"] if c2["name"] == "no records added")
    assert added["ok"] is False
    assert "since export" in added["detail"]
    # but nothing was tampered with or lost
    names = {chk["name"]: chk["ok"] for chk in verdict["checks"]}
    assert names["records intact"] is True
    assert names["records present"] is True


def test_verify_accepts_raw_bundle_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_keys(monkeypatch)
    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "studio-secret")
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")
    envelope = c.post("/api/studio/privacy/audit/export", json={}).json()

    r = c.post("/api/studio/privacy/audit/verify", json=envelope["bundle"])
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


def test_verify_rejects_garbage_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_keys(monkeypatch)
    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "studio-secret")
    c = _ungoverned_client(monkeypatch)
    r = c.post("/api/studio/privacy/audit/verify", json={"hello": "world"})
    assert r.status_code == 400
    r2 = c.post(
        "/api/studio/privacy/audit/verify",
        json={"bundle": {"records": "not-a-map", "signature": "x", "merkle_root": "y"}},
    )
    assert r2.status_code == 400


def test_verify_hmac_503_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_keys(monkeypatch)
    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "studio-secret")
    c = _ungoverned_client(monkeypatch)
    envelope = c.post("/api/studio/privacy/audit/export", json={}).json()
    monkeypatch.delenv("HIMMY_AUDIT_SECRET")
    r = c.post("/api/studio/privacy/audit/verify", json=envelope)
    assert r.status_code == 503


def test_ed25519_export_then_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cryptography")
    from himmy.entities.integrity import generate_ed25519_keypair

    _no_keys(monkeypatch)
    private_pem, _public = generate_ed25519_keypair()
    monkeypatch.setenv("HIMMY_AUDIT_PRIVATE_KEY", private_pem)
    c = _governed_client(monkeypatch)
    _grant(c, "alice", "retain")

    envelope = c.post("/api/studio/privacy/audit/export", json={}).json()
    assert envelope["algorithm"] == "Ed25519"
    verdict = c.post("/api/studio/privacy/audit/verify", json=envelope).json()
    assert verdict["ok"] is True
    assert all(chk["ok"] for chk in verdict["checks"])
