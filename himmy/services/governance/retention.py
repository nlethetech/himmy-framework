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
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError
from himmy.core.ids import new_uuid, utc_now_iso
from himmy.core.sqlite_util import connect_hardened
from himmy.entities.records import EntityRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable

    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.services.storage.encryption import (
        FieldEncryptor,
        KekProvider,
    )

#: EntityRecord kind for an erasure tombstone (the immutable proof of erasure).
ERASURE_KIND = "erasure_tombstone"

#: Sentinel meta-KEK ``key_version`` recorded for an unwrapped (raw-on-disk) subject key —
#: i.e. when no ``HIMMY_KEK_PROVIDER`` is configured (the offline default). The keyvault.db
#: file IS the secret in that case (documented in :class:`SubjectKeyVault`).
_RAW_KEY_VERSION = "RAW"

#: Durable subject-key table. ``wrapped_dek`` holds the subject's 32-byte KEK, wrapped under
#: the META-KEK (``build_kek_provider()``) when one is configured, else the raw key bytes.
#: ``key_version`` records the meta-KEK version (or :data:`_RAW_KEY_VERSION`). A row's
#: ``shredded_at`` is set (and ``wrapped_dek`` nulled) by :meth:`SubjectKeyVault.destroy` so
#: the destruction is durable + auditable; an erased subject's row is retained as a tombstone
#: of WHEN the key was destroyed.
_KEYVAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS subject_keys (
    subject_id  TEXT PRIMARY KEY,
    wrapped_dek BLOB,
    key_version TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    shredded_at TEXT
);
"""


class _Unset:
    """Sentinel distinguishing 'resolve meta-KEK from env' from an explicit ``None``."""


#: Module-level singleton sentinel (so the default arg is a stable identity).
_UNSET = _Unset()


class SubjectKeyVault:
    """Per-subject crypto-shred keys, durably persisted (S2).

    The historical vault was an in-RAM ``dict`` (gone on restart), so ``erase_subject``
    reported ``crypto_shredded=True`` while the key was already volatile — a restart before
    erasure left nothing to destroy. This vault persists each subject's 32-byte key in a
    hardened SQLite table at ``path`` so the key (and its later destruction) survive a
    restart.

    **Key model (reviewer must_fix — subject-key-as-KEK, persisted).** The subject key *is*
    the KEK: :meth:`encryptor_for` returns ``FieldEncryptor(subject_key)``, exactly as before,
    so previously-written ciphertext still decrypts and NO token re-encryption is required.
    What is new is that the subject key is now persisted — wrapped under a META-KEK from
    :func:`~himmy.services.storage.encryption.build_kek_provider` when one is configured
    (``HIMMY_KEK_PROVIDER`` / a cloud KMS), else stored raw. When no meta-KEK is configured
    (the offline default), the raw key lives in the file, so **keyvault.db itself is the
    secret** and MUST share ``spine.db``'s backup/residency posture.

    ``path`` defaults to ``:memory:`` so the zero-config CLI/eval path keeps the old ephemeral
    behavior byte-for-byte; the durable wiring (``api/deps.py``) passes the canonical
    ``.himmy/keyvault.db``.
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        meta_kek: KekProvider | None | _Unset = _UNSET,
    ) -> None:
        """Open (or create) the keyvault DB at ``path`` and apply the schema.

        ``meta_kek`` wraps each persisted subject key (the envelope around the KEK). By
        default it is resolved from :func:`build_kek_provider` (``None`` ⇒ raw-on-disk, the
        offline path); pass an explicit provider (or ``None``) to override. The parent dir
        of a file path is created on first use, like the other durable stores.
        """
        self._path = str(path)
        if self._path != ":memory:":
            from pathlib import Path

            Path(self._path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._conn = connect_hardened(self._path)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_KEYVAULT_SCHEMA)
            self._conn.commit()
        if isinstance(meta_kek, _Unset):
            from himmy.services.storage.encryption import build_kek_provider

            self._meta_kek = build_kek_provider()
        else:
            self._meta_kek = meta_kek
        #: Per-process cache of unwrapped subject keys (avoids a KMS round-trip per call).
        #: Authoritative state is the DB row; the cache is invalidated on destroy.
        self._cache: dict[str, bytes] = {}

    # ----------------------------------------------------------------- wrapping
    def _wrap(self, key: bytes) -> tuple[bytes, str]:
        """Wrap a raw subject key under the meta-KEK (or pass through raw)."""
        if self._meta_kek is None:
            return key, _RAW_KEY_VERSION
        wrapped, version = self._meta_kek.wrap_dek(key)
        return wrapped, version

    def _unwrap(self, wrapped: bytes, key_version: str) -> bytes:
        """Unwrap a stored subject key (raw pass-through when stored raw).

        A row stored RAW is returned verbatim. A row stored under a meta-KEK requires the
        meta-KEK to be present: if it was removed after the key was wrapped, fail loud rather
        than silently hand back the wrapped bytes as if they were the key (which would
        produce wrong-key ciphertext, not a clean error).
        """
        if key_version == _RAW_KEY_VERSION:
            return bytes(wrapped)
        if self._meta_kek is None:
            raise HimmyError(
                f"subject key was wrapped under meta-KEK version {key_version!r} but no "
                "meta-KEK is configured (set HIMMY_KEK_PROVIDER / HIMMY_ENCRYPTION_KEY to "
                "the same provider that wrote keyvault.db)."
            )
        return self._meta_kek.unwrap_dek(bytes(wrapped), key_version)

    # -------------------------------------------------------------------- reads
    def key_for(self, subject_id: str) -> bytes:
        """Return (creating + persisting if needed) the subject's 32-byte key.

        Raises :class:`KeyError` if the subject was already erased (its row exists with a
        nulled key) — re-creating a key for an erased subject would silently undo the shred.
        """
        cached = self._cache.get(subject_id)
        if cached is not None:
            return cached
        with self._lock:
            row = self._conn.execute(
                "SELECT wrapped_dek, key_version, shredded_at "
                "FROM subject_keys WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
            if row is not None:
                wrapped, key_version, shredded_at = row
                if shredded_at is not None or wrapped is None:
                    raise KeyError(
                        f"subject {subject_id!r} was erased; its key is destroyed."
                    )
                key = self._unwrap(wrapped, key_version)
                self._cache[subject_id] = key
                return key
            # First sight: generate, wrap, persist atomically.
            key = os.urandom(32)
            wrapped, key_version = self._wrap(key)
            self._conn.execute(
                "INSERT INTO subject_keys "
                "(subject_id, wrapped_dek, key_version, created_at) "
                "VALUES (?, ?, ?, ?)",
                (subject_id, wrapped, key_version, utc_now_iso()),
            )
            self._conn.commit()
            self._cache[subject_id] = key
            return key

    def encryptor_for(self, subject_id: str) -> FieldEncryptor:
        """A :class:`FieldEncryptor` bound to the subject's key (subject-key-as-KEK)."""
        from himmy.services.storage.encryption import FieldEncryptor

        return FieldEncryptor(self.key_for(subject_id))

    def rotate_subject_key(
        self, subject_id: str, new_provider: KekProvider, tokens: Iterable[str]
    ) -> list[str]:
        """Re-wrap a subject's ciphertext DEKs under ``new_provider`` (no plaintext touch).

        The subject's per-subject key wraps each token's DEK (the subject-key-as-KEK). This
        unwraps each token's DEK with that subject key and re-wraps it under ``new_provider``
        (e.g. a cloud KMS), returning the rotated tokens in order. The protected plaintext is
        never decrypted — only the ``wrapped_dek`` and version change — so the
        audit/ciphertext is untouched and crypto-shred semantics survive.

        The subject must still hold a key (i.e. not be erased).
        """
        if not self.has(subject_id):
            raise KeyError(f"subject {subject_id!r} has no key (erased?)")
        enc = self.encryptor_for(subject_id)
        return [enc.rotate_kek(token, new_provider) for token in tokens]

    def has(self, subject_id: str) -> bool:
        """Whether the subject still has a (non-shredded) key, i.e. is not yet erased."""
        if subject_id in self._cache:
            return True
        with self._lock:
            row = self._conn.execute(
                "SELECT wrapped_dek, shredded_at FROM subject_keys "
                "WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
        return row is not None and row[1] is None and row[0] is not None

    def destroy(self, subject_id: str) -> bool:
        """Crypto-shred durably: null the stored key + stamp ``shredded_at``.

        Returns ``True`` if a live key existed (and was destroyed), ``False`` if the subject
        was unknown or already shredded. The row is RETAINED (with a nulled ``wrapped_dek``)
        as a durable tombstone of WHEN the key was destroyed — the destruction survives a
        restart and is irrecoverable.
        """
        self._cache.pop(subject_id, None)
        with self._lock:
            row = self._conn.execute(
                "SELECT wrapped_dek, shredded_at FROM subject_keys "
                "WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
            if row is None or row[1] is not None or row[0] is None:
                return False
            self._conn.execute(
                "UPDATE subject_keys SET wrapped_dek = NULL, shredded_at = ? "
                "WHERE subject_id = ?",
                (utc_now_iso(), subject_id),
            )
            self._conn.commit()
        return True

    def close(self) -> None:
        """Close the backing connection (idempotent)."""
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - best-effort teardown
            pass

    def __enter__(self) -> SubjectKeyVault:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


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
