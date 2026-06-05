"""WS4.5 — Ed25519-signed audit bundles (verifier needs only the public key)."""

from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from himmy.entities.integrity import (  # noqa: E402
    export_audit_bundle_ed25519,
    generate_ed25519_keypair,
    verify_audit_bundle_ed25519,
)
from himmy.entities.records import EntityRecord  # noqa: E402


def _records() -> list[EntityRecord]:
    return [
        EntityRecord.create(
            stable_id="s1", version=1, kind="security_event", payload={"a": 1}
        ),
        EntityRecord.create(
            stable_id="s2", version=1, kind="security_event", payload={"a": 2}
        ),
    ]


def test_ed25519_bundle_verifies_with_public_key() -> None:
    private_pem, public_pem = generate_ed25519_keypair()
    records = _records()
    bundle = export_audit_bundle_ed25519(records, [], private_pem=private_pem)
    assert bundle.algorithm == "Ed25519"
    result = verify_audit_bundle_ed25519(bundle, records, [], public_pem=public_pem)
    assert result.ok and result.signature_valid


def test_tampering_a_record_is_detected() -> None:
    private_pem, public_pem = generate_ed25519_keypair()
    records = _records()
    bundle = export_audit_bundle_ed25519(records, [], private_pem=private_pem)
    tampered = list(records)
    tampered[0] = tampered[0].model_copy(update={"payload": {"a": 999}})
    result = verify_audit_bundle_ed25519(bundle, tampered, [], public_pem=public_pem)
    assert not result.ok
    assert tampered[0].record_id in result.tampered_record_ids


def test_wrong_public_key_fails_signature() -> None:
    private_pem, _ = generate_ed25519_keypair()
    _, other_public = generate_ed25519_keypair()
    records = _records()
    bundle = export_audit_bundle_ed25519(records, [], private_pem=private_pem)
    result = verify_audit_bundle_ed25519(bundle, records, [], public_pem=other_public)
    assert not result.signature_valid and not result.ok
