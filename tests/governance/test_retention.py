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

    # Crypto-shredded: the key is gone, so the ciphertext is unrecoverable.
    assert vault.has("alice") is False
    with pytest.raises(InvalidTag):
        vault.encryptor_for("alice").decrypt(token)  # a fresh (different) key

    # The tombstone is an immutable, audit-covered proof of erasure.
    assert tombstone.kind == ERASURE_KIND
    assert tombstone.payload["subject_id"] == "alice"
    assert tombstone.payload["crypto_shredded"] is True
    assert registry.list_by_kind(ERASURE_KIND)[0].record_id == tombstone.record_id


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
