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
import sqlite3
import threading
from dataclasses import dataclass
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
        WorkspaceKekProvider,
    )

#: EntityRecord kind for an erasure tombstone (the immutable proof of erasure).
ERASURE_KIND = "erasure_tombstone"

#: EntityRecord kind for the structured, spine-signed deletion certificate (S4) — the
#: machine-verifiable per-store record of WHAT was erased (action + count per store), the
#: destroyed key_version, and ``erased_at``. Registered on the spine so the audit bundle's
#: Merkle root + HMAC signature cover it (i.e. it is tamper-evident from the DB alone).
DELETION_CERTIFICATE_KIND = "deletion_certificate"

#: Sentinel meta-KEK ``key_version`` recorded for an unwrapped (raw-on-disk) subject key —
#: i.e. when no ``HIMMY_KEK_PROVIDER`` is configured (the offline default). The keyvault.db
#: file IS the secret in that case (documented in :class:`SubjectKeyVault`).
_RAW_KEY_VERSION = "RAW"

#: Durable subject-key table. ``wrapped_dek`` holds the subject's 32-byte KEK, wrapped under
#: the META-KEK (``build_kek_provider()``) when one is configured, else the raw key bytes.
#: ``key_version`` records the meta-KEK version (or :data:`_RAW_KEY_VERSION`). A row's
#: ``shredded_at`` is set (and ``wrapped_dek`` nulled) by :meth:`SubjectKeyVault.destroy` so
#: the destruction is durable + auditable; an erased subject's row is retained as a tombstone
#: of WHEN the key was destroyed. ``workspace_id`` (S5) records the tenant a subject's key
#: belongs to so a per-workspace KEK can wrap it and a single-tenant crypto-shred can null
#: every key in one workspace (NULL ⇒ the legacy single-process-KEK path, unchanged).
#: Base DDL. ``CREATE TABLE IF NOT EXISTS`` lays down the full table for a FRESH db (with the
#: S5 ``workspace_id`` column) but is a no-op on a legacy table that lacks the column — the
#: ``workspace_id`` index is therefore created in :meth:`_migrate_legacy_schema_locked` AFTER
#: the column is ensured, never here, so it cannot reference a missing column on a legacy file.
_KEYVAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS subject_keys (
    subject_id   TEXT PRIMARY KEY,
    wrapped_dek  BLOB,
    key_version  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    shredded_at  TEXT,
    workspace_id TEXT
);

CREATE TABLE IF NOT EXISTS disabled_workspaces (
    workspace_id TEXT PRIMARY KEY,
    disabled_at  TEXT NOT NULL
);
"""

#: Durable disabled-workspaces record name used in operator messaging.
_WORKSPACE_DISABLED_TOMBSTONE = "disabled_workspaces"


class _Unset:
    """Sentinel distinguishing 'resolve meta-KEK from env' from an explicit ``None``."""


#: Module-level singleton sentinel (so the default arg is a stable identity).
_UNSET = _Unset()


class _DbDisabledWorkspaces:
    """A durable disabled-workspaces set backed by a :class:`SubjectKeyVault`'s db (S5).

    Implements the :class:`~himmy.services.storage.encryption._DisabledWorkspaces` protocol so
    a :class:`WorkspaceKekProvider`'s per-tenant shred survives a restart: ``disable`` inserts
    a row into ``disabled_workspaces`` and ``is_disabled`` reads it back. The provider consults
    this on EVERY :meth:`for_workspace`, so a shredded tenant's KEK can never be re-derived even
    after the process restarts.
    """

    def __init__(self, vault: SubjectKeyVault) -> None:
        self._vault = vault

    def is_disabled(self, workspace_id: str) -> bool:
        return self._vault._is_workspace_disabled(workspace_id)

    def disable(self, workspace_id: str) -> None:
        self._vault._mark_workspace_disabled(workspace_id)


def _rebind_disabled_set(
    provider: WorkspaceKekProvider, disabled: _DbDisabledWorkspaces
) -> WorkspaceKekProvider:
    """Re-point an injected workspace provider's disabled-set at the durable db one.

    A caller may inject a workspace provider built with an ephemeral in-memory disabled-set
    (the offline default of :func:`build_workspace_kek_provider`). To make a per-tenant shred
    durable, swap its ``_disabled`` to the db-backed set. Implementations keep the set on a
    ``_disabled`` attribute; if a custom provider does not, it is left untouched (its own
    disabled-set is then authoritative).
    """
    if hasattr(provider, "_disabled"):
        provider._disabled = disabled  # type: ignore[attr-defined]
    return provider


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

    **Re-onboarding after erasure.** Erasing a subject (:meth:`destroy`) nulls the key
    durably and irrecoverably. If the SAME external ``subject_id`` later re-consents and
    needs a key again, :meth:`key_for` mints a FRESH key and re-activates the row in place
    rather than refusing — the prior key bytes are already gone, so any pre-erasure
    ciphertext stays permanently unrecoverable, while new data is protected under the new
    key. This keeps an erase-then-re-onboard flow working without ever un-shredding the past.

    ``path`` defaults to ``:memory:`` so the zero-config CLI/eval path keeps the old ephemeral
    behavior byte-for-byte; the durable wiring (``api/deps.py``) passes the canonical
    ``.himmy/keyvault.db``.
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        meta_kek: KekProvider | None | _Unset = _UNSET,
        workspace_kek: WorkspaceKekProvider | None | _Unset = _UNSET,
    ) -> None:
        """Open (or create) the keyvault DB at ``path`` and apply the schema.

        ``meta_kek`` wraps each persisted subject key (the envelope around the KEK). By
        default it is resolved from :func:`build_kek_provider` (``None`` ⇒ raw-on-disk, the
        offline path); pass an explicit provider (or ``None``) to override.

        ``workspace_kek`` (S5) selects a PER-WORKSPACE KEK so a single tenant is independently
        crypto-shreddable. When set, a subject registered with a ``workspace_id`` has its key
        wrapped under that workspace's KEK (not the single process meta-KEK), and
        :meth:`shred_workspace` disables the workspace KEK + nulls every key in it — rendering
        only that tenant's ciphertext undecryptable. Default (``_UNSET`` ⇒ resolved from
        :func:`build_workspace_kek_provider`, normally ``None``) keeps the single process KEK
        with NO behavior change. The durable disabled-workspaces set is backed by THIS db so a
        per-tenant shred survives a restart. The parent dir of a file path is created on first
        use, like the other durable stores.
        """
        self._path = str(path)
        if self._path != ":memory:":
            from pathlib import Path

            Path(self._path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._conn = connect_hardened(self._path)
        # Re-entrant: the wrap/unwrap path (inside the held lock) calls back into the workspace
        # KEK provider, whose durable disabled-set adapter re-enters the vault on the SAME
        # thread. A plain Lock would self-deadlock; RLock keeps the single-connection
        # serialisation across threads while allowing this same-thread re-entry (S5).
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_KEYVAULT_SCHEMA)
            self._migrate_legacy_schema_locked()
            self._conn.commit()
        if isinstance(meta_kek, _Unset):
            from himmy.services.storage.encryption import build_kek_provider

            self._meta_kek = build_kek_provider()
        else:
            self._meta_kek = meta_kek
        if isinstance(workspace_kek, _Unset):
            from himmy.services.storage.encryption import (
                build_workspace_kek_provider,
            )

            # Back the provider's disabled-set with THIS db so a tenant shred is durable.
            self._workspace_kek = build_workspace_kek_provider(
                disabled=_DbDisabledWorkspaces(self)
            )
        elif workspace_kek is None:
            self._workspace_kek = None
        else:
            self._workspace_kek = _rebind_disabled_set(
                workspace_kek, _DbDisabledWorkspaces(self)
            )
        #: Per-process cache of unwrapped subject keys (avoids a KMS round-trip per call).
        #: Authoritative state is the DB row; the cache is invalidated on destroy.
        self._cache: dict[str, bytes] = {}

    def _migrate_legacy_schema_locked(self) -> None:
        """Forward-migrate a pre-S5 keyvault.db in place (caller holds the lock).

        A keyvault.db written before S5 has a ``subject_keys`` table WITHOUT the
        ``workspace_id`` column (and no ``disabled_workspaces`` table — that one is created by
        the idempotent base DDL above). ``CREATE TABLE IF NOT EXISTS`` never adds a column to
        an existing table, so add it here when absent. Legacy rows get ``workspace_id`` NULL,
        i.e. the single-process-KEK path — exactly their original behavior.
        """
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(subject_keys)").fetchall()
        }
        if "workspace_id" not in cols:
            self._conn.execute(
                "ALTER TABLE subject_keys ADD COLUMN workspace_id TEXT"
            )
        # The index is created here (not in the base DDL) so it always runs AFTER the column
        # exists — on a fresh db the column came from the base DDL, on a legacy db from the
        # ALTER above. Idempotent either way.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS subject_keys_workspace_idx "
            "ON subject_keys (workspace_id)"
        )

    # ----------------------------------------------------------------- wrapping
    def _wrap(self, key: bytes, *, workspace_id: str | None) -> tuple[bytes, str]:
        """Wrap a raw subject key under the right KEK; return ``(wrapped, key_version)``.

        Selection order: (1) when a per-workspace KEK provider is configured AND the subject
        carries a ``workspace_id``, wrap under that workspace's KEK (so the tenant is
        independently shreddable — S5); (2) else the single process meta-KEK; (3) else raw
        (the offline default). The returned ``key_version`` records WHICH KEK wrapped it so the
        matching one is located on read.
        """
        provider = self._workspace_kek_for(workspace_id)
        if provider is not None:
            return provider.wrap_dek(key)
        if self._meta_kek is None:
            return key, _RAW_KEY_VERSION
        return self._meta_kek.wrap_dek(key)

    def _unwrap(
        self, wrapped: bytes, key_version: str, *, workspace_id: str | None
    ) -> bytes:
        """Unwrap a stored subject key (raw pass-through when stored raw).

        A row stored RAW is returned verbatim. A workspace-scoped row (the subject carries a
        ``workspace_id`` and a workspace KEK provider is configured) is unwrapped under that
        workspace's KEK — which RAISES :class:`WorkspaceShredded` when the workspace has been
        disabled (the per-tenant crypto-shred). Otherwise the single process meta-KEK is
        required: if it was removed after the key was wrapped, fail loud rather than silently
        hand back the wrapped bytes as if they were the key.
        """
        if key_version == _RAW_KEY_VERSION:
            return bytes(wrapped)
        provider = self._workspace_kek_for(workspace_id)
        if provider is not None:
            # for_workspace already raised if the workspace is shredded; the derived/selected
            # provider unwraps under the workspace KEK.
            return provider.unwrap_dek(bytes(wrapped), key_version)
        if self._meta_kek is None:
            raise HimmyError(
                f"subject key was wrapped under meta-KEK version {key_version!r} but no "
                "meta-KEK is configured (set HIMMY_KEK_PROVIDER / HIMMY_ENCRYPTION_KEY to "
                "the same provider that wrote keyvault.db)."
            )
        return self._meta_kek.unwrap_dek(bytes(wrapped), key_version)

    def _workspace_kek_for(self, workspace_id: str | None) -> KekProvider | None:
        """The per-workspace KEK for ``workspace_id`` (``None`` ⇒ fall back to meta-KEK).

        Returns ``None`` when no workspace KEK provider is configured OR the subject has no
        ``workspace_id`` (the legacy single-process-KEK path, unchanged). When a provider IS
        configured and the workspace is disabled, :meth:`WorkspaceKekProvider.for_workspace`
        raises :class:`WorkspaceShredded` — that is the tenant shred surfacing on any read.
        """
        if self._workspace_kek is None or workspace_id is None:
            return None
        return self._workspace_kek.for_workspace(workspace_id)

    def _workspace_of(self, subject_id: str) -> str | None:
        """The stored ``workspace_id`` of a subject's row (caller holds the lock)."""
        row = self._conn.execute(
            "SELECT workspace_id FROM subject_keys WHERE subject_id = ?",
            (subject_id,),
        ).fetchone()
        return row[0] if row is not None else None

    def _is_workspace_disabled(self, workspace_id: str) -> bool:
        """Whether ``workspace_id`` is durably recorded as shredded.

        The lock is re-entrant (:class:`threading.RLock`), so this is safe both standalone
        (the :class:`_DbDisabledWorkspaces` adapter calls it outside the vault's other work)
        and re-entered from the wrap/unwrap path that already holds the lock.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM disabled_workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        return row is not None

    def _mark_workspace_disabled(self, workspace_id: str) -> None:
        """Durably record ``workspace_id`` as shredded (idempotent; acquires the lock)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO disabled_workspaces (workspace_id, disabled_at) "
                "VALUES (?, ?) ON CONFLICT(workspace_id) DO NOTHING",
                (workspace_id, utc_now_iso()),
            )
            self._conn.commit()

    # -------------------------------------------------------------------- reads
    def key_for(self, subject_id: str, *, workspace_id: str | None = None) -> bytes:
        """Return (creating + persisting if needed) the subject's 32-byte key.

        ``workspace_id`` (S5) binds a FIRST-SEEN subject to a tenant so its key is wrapped
        under that workspace's KEK (and the tenant becomes independently shreddable). It is
        recorded once on creation; passing a DIFFERENT ``workspace_id`` for an existing
        subject raises (a subject must not silently jump tenants — that would mis-target a
        per-workspace shred). Omitting it for an existing subject reuses the stored linkage.
        When no workspace KEK provider is configured, ``workspace_id`` is still recorded for
        the reach-map linkage but wrapping stays under the single process KEK (unchanged).

        If the subject was previously erased (its row exists with a nulled key), a FRESH
        key is minted and the row is re-activated (``shredded_at`` cleared, a new
        ``wrapped_dek`` + ``key_version`` written, ``created_at`` restamped). This restores
        a legitimate erase-then-re-onboard flow (same external ``subject_id`` re-grants
        consent) WITHOUT undoing the original shred: the old key bytes were already nulled
        on :meth:`destroy`, so any ciphertext written under the previous key stays
        permanently unrecoverable. The new key only protects data written AFTER re-consent.
        """
        cached = self._cache.get(subject_id)
        if cached is not None:
            # Guard the cache hit too: a caller passing a DIFFERENT workspace for a known
            # subject must still be rejected (the cache must not bypass the tenant binding).
            if workspace_id is not None:
                with self._lock:
                    self._assert_same_workspace(
                        subject_id, self._workspace_of(subject_id), workspace_id
                    )
            return cached
        with self._lock:
            row = self._conn.execute(
                "SELECT wrapped_dek, key_version, shredded_at, workspace_id "
                "FROM subject_keys WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
            if row is not None:
                wrapped, key_version, shredded_at, stored_ws = row
                self._assert_same_workspace(subject_id, stored_ws, workspace_id)
                if shredded_at is not None or wrapped is None:
                    # Erased subject re-onboarding: re-mint under a fresh key + re-activate
                    # the row in place. The prior key bytes are gone (nulled on destroy),
                    # so the shred remains irrecoverable for pre-erasure ciphertext.
                    return self._remint_locked(
                        subject_id, workspace_id=workspace_id or stored_ws
                    )
                key = self._unwrap(wrapped, key_version, workspace_id=stored_ws)
                self._cache[subject_id] = key
                return key
            # First sight: generate, wrap, persist atomically.
            key = os.urandom(32)
            wrapped, key_version = self._wrap(key, workspace_id=workspace_id)
            self._conn.execute(
                "INSERT INTO subject_keys "
                "(subject_id, wrapped_dek, key_version, created_at, workspace_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (subject_id, wrapped, key_version, utc_now_iso(), workspace_id),
            )
            self._conn.commit()
            self._cache[subject_id] = key
            return key

    @staticmethod
    def _assert_same_workspace(
        subject_id: str, stored: str | None, requested: str | None
    ) -> None:
        """Refuse a subject silently changing tenants (would mis-target a per-tenant shred)."""
        if requested is not None and stored is not None and requested != stored:
            raise HimmyError(
                f"subject {subject_id!r} is bound to workspace {stored!r} but was "
                f"requested under {requested!r}; a subject cannot move between tenants "
                "(this would mis-target a per-workspace crypto-shred)."
            )

    def _remint_locked(
        self, subject_id: str, *, workspace_id: str | None = None
    ) -> bytes:
        """Mint a fresh key for an erased subject and re-activate its row in place.

        Caller must hold ``self._lock``. The old (nulled) key bytes are unrecoverable, so
        re-minting cannot un-shred prior ciphertext — it only provisions a NEW key for data
        written after re-consent. ``shredded_at`` is cleared and ``created_at`` restamped so
        the row reflects the active re-onboarded key. ``workspace_id`` re-binds the tenant
        (defaults to the row's existing one).
        """
        key = os.urandom(32)
        wrapped, key_version = self._wrap(key, workspace_id=workspace_id)
        self._conn.execute(
            "UPDATE subject_keys "
            "SET wrapped_dek = ?, key_version = ?, created_at = ?, shredded_at = NULL, "
            "    workspace_id = ? "
            "WHERE subject_id = ?",
            (wrapped, key_version, utc_now_iso(), workspace_id, subject_id),
        )
        self._conn.commit()
        self._cache[subject_id] = key
        return key

    def encryptor_for(
        self, subject_id: str, *, workspace_id: str | None = None
    ) -> FieldEncryptor:
        """A :class:`FieldEncryptor` bound to the subject's key (subject-key-as-KEK).

        ``workspace_id`` (S5) binds a first-seen subject to a tenant (see :meth:`key_for`); it
        is recorded once and reused thereafter, so existing callers that omit it are unchanged.
        """
        from himmy.services.storage.encryption import FieldEncryptor

        return FieldEncryptor(self.key_for(subject_id, workspace_id=workspace_id))

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

    def key_version_of(self, subject_id: str) -> str | None:
        """The meta-KEK ``key_version`` of a subject's live key (``None`` if unknown/erased).

        Read BEFORE :meth:`destroy` so a deletion certificate can record which key version
        was destroyed (the row's ``key_version`` is retained as a tombstone, but reading it
        before the shred keeps the certificate's value the one that protected live data).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT wrapped_dek, key_version, shredded_at FROM subject_keys "
                "WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
        if row is None or row[2] is not None or row[0] is None:
            return None
        return str(row[1])

    def _checkpoint_locked(self) -> None:
        """Force the WAL into the main DB + fsync after a crypto-shred (caller holds ``_lock``).

        The vault opens with ``synchronous=NORMAL``, so a plain commit only appends to the WAL
        and a power-cut before the next checkpoint can lose the shred — RESURRECTING a key the
        deletion certificate already swore was destroyed. ``wal_checkpoint(TRUNCATE)`` merges
        the WAL into the main file and fsyncs it, so the destruction is durable before
        :meth:`RetentionService.erase_subject` signs the certificate. Best-effort: a memory DB
        has no WAL, and a checkpoint blocked by a concurrent reader is a transient durability
        delay, not a correctness break (the commit itself already landed).
        """
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:  # pragma: no cover - checkpoint is best-effort
            pass

    def destroy(self, subject_id: str, *, workspace_id: str | None = None) -> bool:
        """Crypto-shred durably: null the stored key + stamp ``shredded_at``.

        Returns ``True`` if a live key existed (and was destroyed), ``False`` if the subject
        was unknown or already shredded. The row is RETAINED (with a nulled ``wrapped_dek``)
        as a durable tombstone of WHEN the key was destroyed — the destruction survives a
        restart and is irrecoverable. After committing the null, the WAL is checkpoint-truncated
        (:meth:`_checkpoint_locked`) so the shred is fsynced to the main DB and cannot be lost
        to a power-cut before the deletion certificate is written.

        ``workspace_id`` (S5, defense-in-depth) gates the shred to the caller's tenant: when
        given, the row's key must be provably bound to that SAME workspace; a row bound to a
        different tenant OR an UNBOUND/global row (``workspace_id`` NULL — the production
        default) is REFUSED (returns ``False``, destroying nothing), so a cross-tenant
        withdrawal can never crypto-shred a key it does not own. ``None`` keeps the legacy
        single-tenant/offline path: any row is shreddable, exactly as before.
        """
        self._cache.pop(subject_id, None)
        with self._lock:
            row = self._conn.execute(
                "SELECT wrapped_dek, shredded_at, workspace_id FROM subject_keys "
                "WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
            if row is None or row[1] is not None or row[0] is None:
                return False
            # Cross-tenant guard (FAIL SAFE): a workspace-scoped destroy may ONLY shred a key
            # provably bound to that same workspace. When the caller scopes the destroy
            # (workspace_id given), an UNBOUND key (row[2] is None — the production default,
            # since keys are minted without a workspace) or a key bound to a DIFFERENT tenant
            # is REFUSED (None != workspace_id => refuse). A multi-tenant caller must not
            # crypto-shred a global key per-tenant. The legacy unscoped path (workspace_id=None)
            # is unchanged: any row is shreddable.
            if workspace_id is not None and row[2] != workspace_id:
                return False
            self._conn.execute(
                "UPDATE subject_keys SET wrapped_dek = NULL, shredded_at = ? "
                "WHERE subject_id = ?",
                (utc_now_iso(), subject_id),
            )
            self._conn.commit()
            self._checkpoint_locked()
        return True

    # --------------------------------------------------------- workspace (S5)
    @property
    def supports_workspace_shred(self) -> bool:
        """Whether a per-workspace KEK is configured (so a single tenant is shreddable)."""
        return self._workspace_kek is not None

    def workspace_of(self, subject_id: str) -> str | None:
        """The ``workspace_id`` a subject is bound to (``None`` if unbound/unknown).

        Used by the reach-map to assert subject→workspace linkage so a per-workspace shred
        neither misses a subject (wrong tenant) nor over-deletes (a shared subject).
        """
        with self._lock:
            return self._workspace_of(subject_id)

    def subjects_in_workspace(self, workspace_id: str) -> list[str]:
        """Every LIVE (non-shredded) subject bound to ``workspace_id``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT subject_id FROM subject_keys "
                "WHERE workspace_id = ? AND shredded_at IS NULL AND wrapped_dek IS NOT NULL",
                (workspace_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def is_workspace_shredded(self, workspace_id: str) -> bool:
        """Whether ``workspace_id`` has been cryptographically shredded (its KEK disabled)."""
        return self._is_workspace_disabled(workspace_id)

    def shred_workspace(self, workspace_id: str) -> int:
        """Cryptographically shred ONE tenant: disable its KEK + null every key in it (S5).

        This is the single per-tenant erasure operation: it (1) disables the workspace KEK
        durably (via the configured :class:`WorkspaceKekProvider` — the local HKDF path marks
        the durable disabled-set so the KEK can no longer be DERIVED; the AWS path ALSO disables
        the real CMK), and (2) nulls + tombstones every live ``subject_keys`` row in the
        workspace (so the subject keys are gone even crypto-shredded). After this, ALL of the
        workspace's subject ciphertext is permanently undecryptable while every OTHER workspace
        still decrypts. Returns the number of subject keys destroyed.

        Requires a workspace KEK provider (otherwise there is nothing tenant-scoped to disable);
        raises :class:`HimmyError` when none is configured. Idempotent: a second call disables
        an already-disabled workspace and destroys any remaining keys (returns the new count,
        usually 0).
        """
        if self._workspace_kek is None:
            raise HimmyError(
                "shred_workspace requires a per-workspace KEK provider "
                "(set HIMMY_WORKSPACE_KEK_PROVIDER); the single process KEK cannot erase "
                "one tenant without shredding everyone."
            )
        # Disable the workspace KEK FIRST (durable + HSM-side), so even a concurrent reader
        # that races the row-null below cannot unwrap under the now-disabled KEK.
        self._workspace_kek.disable_workspace(workspace_id)
        destroyed = 0
        with self._lock:
            rows = self._conn.execute(
                "SELECT subject_id FROM subject_keys "
                "WHERE workspace_id = ? AND shredded_at IS NULL "
                "AND wrapped_dek IS NOT NULL",
                (workspace_id,),
            ).fetchall()
            now = utc_now_iso()
            for (subject_id,) in rows:
                self._cache.pop(subject_id, None)
                self._conn.execute(
                    "UPDATE subject_keys SET wrapped_dek = NULL, shredded_at = ? "
                    "WHERE subject_id = ?",
                    (now, subject_id),
                )
                destroyed += 1
            self._conn.commit()
            self._checkpoint_locked()
        return destroyed

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


@dataclass(frozen=True)
class StoreErasureResult:
    """The outcome of erasing one subject from one mutable sidecar store (S4).

    ``store`` names the durable location (e.g. ``"memory"`` / ``"conversations"`` /
    ``"knowledge"`` / ``"spine_storage"``); ``deleted`` is the row count hard-removed;
    ``ok`` is ``False`` (with ``error`` set) when the store raised, so a partial mid-erase
    failure is recorded — and re-running ``erase_subject`` is idempotent (a re-run deletes
    whatever survived and flips ``ok`` to ``True`` once the store is clean).
    """

    store: str
    action: str
    deleted: int = 0
    ok: bool = True
    error: str = ""


@dataclass(frozen=True)
class DeletionCertificate:
    """A structured, provable record of a subject's erasure across every store (S4).

    Emitted by :meth:`RetentionService.erase_subject` and persisted as a spine
    :data:`DELETION_CERTIFICATE_KIND` record (so the audit bundle's signature covers it).
    ``per_store`` lists the action+count for the durable vault key plus every mutable
    sidecar; ``key_version`` is the meta-KEK version of the destroyed subject key;
    ``complete`` is ``True`` only when every store reported ``ok`` (a partial failure leaves
    it ``False`` and the erase is safely re-runnable).
    """

    subject_id: str
    reason: str
    crypto_shredded: bool
    key_version: str | None
    erased_at: str
    per_store: tuple[StoreErasureResult, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether every store erasure step succeeded (idempotent re-run otherwise)."""
        return all(r.ok for r in self.per_store)

    def to_payload(self) -> dict[str, Any]:
        """The JSON-serialisable payload registered onto the spine."""
        return {
            "subject_id": self.subject_id,
            "reason": self.reason,
            "crypto_shredded": self.crypto_shredded,
            "key_version": self.key_version,
            "erased_at": self.erased_at,
            "complete": self.complete,
            "per_store": [
                {
                    "store": r.store,
                    "action": r.action,
                    "deleted": r.deleted,
                    "ok": r.ok,
                    "error": r.error,
                }
                for r in self.per_store
            ],
        }


@dataclass
class SubjectReachMap:
    """Every durable location a subject's data may live, as injectable store handles (S4).

    ``RetentionService`` historically held ONLY the registry + key vault, so its erase walk
    reached the (undecryptable) spine alone. Erasing the MUTABLE sidecars requires the store
    handles + a subject-scoped DELETE on each — neither of which existed. This injects them:
    each handle is optional (``None`` ⇒ that store is absent in this deployment and skipped),
    so the offline path that wires no sidecars is unchanged, and a governed server wires the
    durable singletons.

    * ``memory_store`` — the memory module's ``MemoryStore`` (``delete_by_subject``).
    * ``conversation_store`` — the unified ``ConversationStore`` (``delete_by_subject``).
    * ``spine_eraser`` — a callable ``(subject_id) -> int`` hard-deleting the subject's
      ``storage.db`` ``chat_threads`` + ``run_events`` rows. The runtime persists every
      governed run's full transcript + event stream there encrypted under the store-WIDE
      KEK (NOT the per-subject key), so a crypto-shred alone leaves them recoverable; this
      reaches them. The wiring layer adapts the storage backend's ``delete_by_subject``
      (sync on SQLite/in-memory, async on Postgres) to this callable; ``None`` offline.
    * ``knowledge_eraser`` — a callable ``(subject_id) -> int`` deleting a subject-scoped KB's
      rows (the pgvector backend is async + scoped by ``client_id == subject``, so the wiring
      layer adapts it to this sync callable; ``None`` when no knowledge backend is wired).
    """

    memory_store: Any = None
    conversation_store: Any = None
    spine_eraser: Callable[[str], int] | None = None
    knowledge_eraser: Callable[[str], int] | None = None

    def erase(self, subject_id: str) -> list[StoreErasureResult]:
        """Hard-DELETE the subject from every wired mutable store; collect per-store results.

        Each store is erased independently and a raised store records ``ok=False`` rather than
        aborting the whole walk, so one failing store never blocks shredding the rest — and the
        certificate's ``complete`` flag (plus an idempotent re-run) covers the partial case.
        """
        results: list[StoreErasureResult] = []
        if self.memory_store is not None:
            results.append(
                _safe_delete("memory", self.memory_store.delete_by_subject, subject_id)
            )
        if self.conversation_store is not None:
            results.append(
                _safe_delete(
                    "conversations",
                    self.conversation_store.delete_by_subject,
                    subject_id,
                )
            )
        if self.spine_eraser is not None:
            results.append(
                _safe_delete("spine_storage", self.spine_eraser, subject_id)
            )
        if self.knowledge_eraser is not None:
            results.append(
                _safe_delete("knowledge", self.knowledge_eraser, subject_id)
            )
        return results


def _safe_delete(
    store: str, fn: Callable[[str], int], subject_id: str
) -> StoreErasureResult:
    """Run a subject-scoped DELETE, capturing a failure as ``ok=False`` (never raising)."""
    try:
        deleted = int(fn(subject_id))
    except Exception as exc:  # noqa: BLE001 - one failing store must not abort the walk
        return StoreErasureResult(
            store=store, action="delete", ok=False, error=str(exc)
        )
    return StoreErasureResult(store=store, action="delete", deleted=deleted)


class RetentionService:
    """Right-to-erasure (whole-system reach + signed certificate) + age-based retention."""

    def __init__(
        self,
        entity_registry: EntityRegistryProtocol,
        *,
        key_vault: SubjectKeyVault | None = None,
        reach_map: SubjectReachMap | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._registry = entity_registry
        self._keys = key_vault
        self._reach = reach_map
        self._clock = clock or utc_now_iso

    def workspace_of(self, subject_id: str) -> str | None:
        """The ``workspace_id`` a subject's vault key is bound to (``None`` if unbound).

        Exposed so a caller (e.g. :meth:`ConsentLedger.withdraw`) can gate a destructive
        crypto-shred on tenant ownership BEFORE invoking :meth:`erase_subject`. Returns
        ``None`` when no key vault is wired or the subject is unbound/unknown.
        """
        if self._keys is None:
            return None
        return self._keys.workspace_of(subject_id)

    def erase_subject(
        self, subject_id: str, *, reason: str = "", workspace_id: str | None = None
    ) -> DeletionCertificate:
        """Erase a subject everywhere and register a signed deletion certificate (S4).

        The order is: (1) destroy the durable vault key (crypto-shred — renders the subject's
        spine ciphertext permanently unrecoverable), (2) hard-DELETE the subject's rows from
        every wired MUTABLE sidecar via the :class:`SubjectReachMap` (the append-only spine is
        NOT deleted — it stays as undecryptable ciphertext + the tombstone), (3) register the
        existing :data:`ERASURE_KIND` tombstone AND a structured
        :data:`DELETION_CERTIFICATE_KIND` record. The returned certificate lists the per-store
        action+count, the destroyed ``key_version`` and ``erased_at``; if any store failed it
        is marked incomplete and re-running ``erase_subject`` is idempotent.

        Backwards-compatible: with no reach map wired the per-store list is empty and the
        behavior reduces to the historical crypto-shred + tombstone (now also accompanied by
        the certificate).

        ``workspace_id`` (defense-in-depth) is forwarded to :meth:`SubjectKeyVault.destroy`
        so the crypto-shred is REFUSED when the subject's key is bound to a different tenant;
        ``None`` keeps the legacy single-tenant/offline path.
        """
        key_version = (
            self._keys.key_version_of(subject_id) if self._keys is not None else None
        )
        shredded = (
            self._keys.destroy(subject_id, workspace_id=workspace_id)
            if self._keys is not None
            else False
        )
        per_store = self._reach.erase(subject_id) if self._reach is not None else []
        erased_at = self._clock()

        # The immutable tombstone (historical proof-of-erasure) stays exactly as before so the
        # existing privacy audit / bundle paths keep verifying.
        tombstone = EntityRecord.create(
            stable_id=new_uuid(),
            version=1,
            kind=ERASURE_KIND,
            payload={
                "subject_id": subject_id,
                "reason": reason,
                "crypto_shredded": shredded,
                "erased_at": erased_at,
            },
        )
        self._registry.register(tombstone)

        certificate = DeletionCertificate(
            subject_id=subject_id,
            reason=reason,
            crypto_shredded=shredded,
            key_version=key_version,
            erased_at=erased_at,
            per_store=tuple(per_store),
        )
        cert_record = EntityRecord.create(
            stable_id=new_uuid(),
            version=1,
            kind=DELETION_CERTIFICATE_KIND,
            payload=certificate.to_payload(),
        )
        self._registry.register(cert_record)
        return certificate

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


__all__ = [
    "DELETION_CERTIFICATE_KIND",
    "ERASURE_KIND",
    "DeletionCertificate",
    "RetentionService",
    "StoreErasureResult",
    "SubjectKeyVault",
    "SubjectReachMap",
]
