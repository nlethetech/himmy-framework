"""Retention & right-to-erasure for an append-only audit spine (WS4.2).

An immutable, content-addressed entity store can't simply *delete* a person's records —
that's the whole point of the audit trail. The reconciliation is **crypto-shredding**:
each subject's sensitive data is encrypted under a per-subject key; erasing the subject
destroys that key, rendering their ciphertext permanently unrecoverable, while the
(now-undecryptable) records and a signed **erasure tombstone** remain so the audit bundle
still verifies and you can prove the subject existed and was erased.

* :class:`SubjectKeyVault` — per-subject encryption keys (create / shred).
* :class:`RetentionService` — ``erase_subject`` (crypto-shred + tombstone) and an
  age-based ``expired`` helper for time-bound retention of operational records.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from himmy.core.ids import new_uuid, utc_now_iso
from himmy.entities.records import EntityRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable

    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.services.storage.encryption import FieldEncryptor, KekProvider

#: EntityRecord kind for an erasure tombstone (the immutable proof of erasure).
ERASURE_KIND = "erasure_tombstone"


class SubjectKeyVault:
    """Per-subject data keys; destroying one crypto-shreds that subject's data."""

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}

    def key_for(self, subject_id: str) -> bytes:
        """Return (creating if needed) the subject's 32-byte key."""
        key = self._keys.get(subject_id)
        if key is None:
            key = os.urandom(32)
            self._keys[subject_id] = key
        return key

    def encryptor_for(self, subject_id: str) -> FieldEncryptor:
        """A :class:`FieldEncryptor` bound to the subject's key (local KEK provider)."""
        from himmy.services.storage.encryption import FieldEncryptor

        return FieldEncryptor(self.key_for(subject_id))

    def rotate_subject_key(
        self, subject_id: str, new_provider: KekProvider, tokens: Iterable[str]
    ) -> list[str]:
        """Re-wrap a subject's ciphertext DEKs under ``new_provider`` (no plaintext touch).

        The subject's per-subject key currently wraps each token's DEK (the local KEK).
        This unwraps each token's DEK with that subject key and re-wraps it under
        ``new_provider`` (e.g. a cloud KMS), returning the rotated tokens in order. The
        protected plaintext is never decrypted — only the ``wrapped_dek`` and version
        change — so the audit/ciphertext is untouched and crypto-shred semantics survive.

        The subject must still hold a key (i.e. not be erased); rotating onto a cloud KEK
        moves custody of the wrapping key without re-encrypting any data.
        """
        if not self.has(subject_id):
            raise KeyError(f"subject {subject_id!r} has no key (erased?)")
        enc = self.encryptor_for(subject_id)
        return [enc.rotate_kek(token, new_provider) for token in tokens]

    def has(self, subject_id: str) -> bool:
        """Whether the subject still has a key (i.e. is not yet erased)."""
        return subject_id in self._keys

    def destroy(self, subject_id: str) -> bool:
        """Crypto-shred: forget the subject's key (returns True if one existed)."""
        return self._keys.pop(subject_id, None) is not None


class RetentionService:
    """Right-to-erasure (crypto-shred + tombstone) + age-based retention."""

    def __init__(
        self,
        entity_registry: EntityRegistryProtocol,
        *,
        key_vault: SubjectKeyVault | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._registry = entity_registry
        self._keys = key_vault
        self._clock = clock or utc_now_iso

    def erase_subject(self, subject_id: str, *, reason: str = "") -> EntityRecord:
        """Crypto-shred a subject's data and register an immutable erasure tombstone."""
        shredded = self._keys.destroy(subject_id) if self._keys is not None else False
        tombstone = EntityRecord.create(
            stable_id=new_uuid(),
            version=1,
            kind=ERASURE_KIND,
            payload={
                "subject_id": subject_id,
                "reason": reason,
                "crypto_shredded": shredded,
                "erased_at": self._clock(),
            },
        )
        return self._registry.register(tombstone)

    @staticmethod
    def expired(
        records: Iterable[Any], *, max_age_seconds: float, now_epoch: float
    ) -> list[Any]:
        """Return the records whose ``created_at`` is older than ``max_age_seconds``."""
        out: list[Any] = []
        for record in records:
            created = getattr(record, "created_at", None)
            if not created:
                continue
            try:
                ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if now_epoch - ts.timestamp() > max_age_seconds:
                out.append(record)
        return out


__all__ = ["SubjectKeyVault", "RetentionService", "ERASURE_KIND"]
