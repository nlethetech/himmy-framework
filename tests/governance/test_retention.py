"""WS4.2 — retention + right-to-erasure (crypto-shred + tombstone)."""

from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from datetime import UTC

from cryptography.exceptions import InvalidTag

from himmy.entities.registry import EntityRegistry  # noqa: E402
from himmy.services.governance.retention import (  # noqa: E402
    ERASURE_KIND,
    RetentionService,
    SubjectKeyVault,
)


def test_erase_subject_crypto_shreds_and_tombstones() -> None:
    vault = SubjectKeyVault()
    registry = EntityRegistry()
    service = RetentionService(registry, key_vault=vault)

    # The subject's data is encrypted under their per-subject key.
    enc = vault.encryptor_for("alice")
    token = enc.encrypt("alice's medical record")
    assert enc.decrypt(token) == "alice's medical record"

    tombstone = service.erase_subject("alice", reason="GDPR request #42")

    # Crypto-shredded: the key is gone, so the ciphertext is unrecoverable. A later
    # re-onboarding of the same subject mints a FRESH key (so re-consent doesn't crash), but
    # that new key CANNOT decrypt the pre-erasure ciphertext — the shred is irrecoverable.
    assert vault.has("alice") is False
    with pytest.raises(Exception):  # noqa: B017,PT011 - re-minted key never decrypts old token
        vault.encryptor_for("alice").decrypt(token)

    # The tombstone is an immutable, audit-covered proof of erasure.
    assert tombstone.kind == ERASURE_KIND
    assert tombstone.payload["subject_id"] == "alice"
    assert tombstone.payload["crypto_shredded"] is True
    assert registry.list_by_kind(ERASURE_KIND)[0].record_id == tombstone.record_id


def test_rotate_subject_key_rewraps_without_decrypting_plaintext() -> None:
    import os

    from himmy.services.storage.encryption import FieldEncryptor, LocalKekProvider

    vault = SubjectKeyVault()
    enc = vault.encryptor_for("carol")
    token = enc.encrypt("carol's ssn", aad=b"carol")

    # Rotate the subject's DEK wrapping onto a brand-new KEK provider.
    new_provider = LocalKekProvider(os.urandom(32))
    (rotated,) = vault.rotate_subject_key("carol", new_provider, [token])

    # The plaintext was never re-encrypted: the data nonce + ciphertext tail are identical.
    _v_old, blob_old = FieldEncryptor._parse_token(token)
    _v_new, blob_new = FieldEncryptor._parse_token(rotated)
    _w_old, nonce_old, ct_old = FieldEncryptor._split_blob(blob_old)
    _w_new, nonce_new, ct_new = FieldEncryptor._split_blob(blob_new)
    assert (nonce_old, ct_old) == (nonce_new, ct_new)

    # The rotated token decrypts under the NEW provider (and not the subject's old key).
    assert FieldEncryptor.from_provider(new_provider).decrypt(
        rotated, aad=b"carol"
    ) == ("carol's ssn")
    with pytest.raises(InvalidTag):
        vault.encryptor_for("carol").decrypt(rotated, aad=b"carol")


def test_rotate_subject_key_rejects_erased_subject() -> None:
    import os

    from himmy.services.storage.encryption import LocalKekProvider

    vault = SubjectKeyVault()
    vault.key_for("dave")
    assert vault.destroy("dave") is True
    with pytest.raises(KeyError, match="dave"):
        vault.rotate_subject_key("dave", LocalKekProvider(os.urandom(32)), [])


def test_erase_without_vault_still_tombstones() -> None:
    registry = EntityRegistry()
    service = RetentionService(registry)
    tombstone = service.erase_subject("bob")
    assert tombstone.payload["crypto_shredded"] is False


def test_expired_identifies_old_records() -> None:
    from datetime import datetime

    class _Rec:
        def __init__(self, created_at: str) -> None:
            self.created_at = created_at

    now = datetime(2026, 1, 10, tzinfo=UTC)
    old = _Rec("2026-01-01T00:00:00+00:00")  # 9 days
    fresh = _Rec("2026-01-09T00:00:00+00:00")  # 1 day
    expired = RetentionService.expired(
        [old, fresh], max_age_seconds=5 * 86400, now_epoch=now.timestamp()
    )
    assert expired == [old]
