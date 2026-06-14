"""S2: durable SubjectKeyVault (.himmy/keyvault.db, KEK-wrapped subject keys).

The historical vault was an in-RAM ``dict`` (gone on restart), so ``erase_subject`` could
report ``crypto_shredded=True`` while the key was already volatile — a restart before
erasure left nothing to destroy. This vault persists each subject's KEK (the subject key IS
the KEK — no token re-encryption) to a hardened SQLite table so BOTH the key and its later
destruction survive a restart. These tests pin: stability across restart, durable +
irrecoverable shred, the meta-KEK-wrapped-on-disk path (with an injected provider, no cloud
key required), and the preserved zero-config ``:memory:`` default.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from himmy.services.governance.retention import (  # noqa: E402
    _RAW_KEY_VERSION,
    SubjectKeyVault,
)
from himmy.services.storage.encryption import LocalKekProvider  # noqa: E402


def test_subject_key_is_stable_across_restart(tmp_path: Path) -> None:
    """A subject's key (and any ciphertext it wrapped) survives a process restart."""
    path = str(tmp_path / "keyvault.db")
    first = SubjectKeyVault(path, meta_kek=None)
    enc = first.encryptor_for("alice")
    token = enc.encrypt("alice's record", aad=b"alice")
    key = first.key_for("alice")
    first.close()

    second = SubjectKeyVault(path, meta_kek=None)  # restart
    assert second.key_for("alice") == key
    assert second.encryptor_for("alice").decrypt(token, aad=b"alice") == "alice's record"
    second.close()


def test_destroy_is_durable_and_irrecoverable(tmp_path: Path) -> None:
    """Write data, restart, erase, restart: the key is destroyed durably + a control survives.

    This is the S2 acceptance: the in-RAM vault could not satisfy it because the key was
    gone the moment the process exited.
    """
    path = str(tmp_path / "keyvault.db")
    v1 = SubjectKeyVault(path, meta_kek=None)
    v1.key_for("alice")
    v1.key_for("bob")  # control subject
    v1.close()

    # Restart (proving the keys are durable, not just in-RAM), THEN erase alice.
    v2 = SubjectKeyVault(path, meta_kek=None)
    assert v2.has("alice") is True
    assert v2.destroy("alice") is True
    v2.close()

    # Restart again: alice is irrecoverable, bob still decrypts.
    v3 = SubjectKeyVault(path, meta_kek=None)
    assert v3.has("alice") is False
    with pytest.raises(KeyError, match="alice"):
        v3.encryptor_for("alice")  # refuses to re-mint (would silently un-shred)
    assert v3.has("bob") is True
    v3.encryptor_for("bob")  # control subject still works
    v3.close()


def test_destroy_returns_false_for_unknown_or_already_shredded(tmp_path: Path) -> None:
    path = str(tmp_path / "keyvault.db")
    vault = SubjectKeyVault(path, meta_kek=None)
    assert vault.destroy("never_seen") is False
    vault.key_for("carol")
    assert vault.destroy("carol") is True
    assert vault.destroy("carol") is False  # already shredded
    vault.close()


def test_shred_retains_tombstone_row(tmp_path: Path) -> None:
    """A shredded subject's row is RETAINED with a null key + a shredded_at stamp."""
    path = str(tmp_path / "keyvault.db")
    vault = SubjectKeyVault(path, meta_kek=None)
    vault.key_for("dave")
    assert vault.destroy("dave") is True
    vault.close()

    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT wrapped_dek, shredded_at FROM subject_keys WHERE subject_id = ?",
            ("dave",),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None  # row retained as a destruction tombstone
    assert row[0] is None  # key bytes nulled
    assert row[1] is not None  # shredded_at stamped


def test_meta_kek_wraps_key_on_disk_and_unwraps_stably(tmp_path: Path) -> None:
    """With a meta-KEK provider the on-disk key is WRAPPED (not raw) yet unwraps stably.

    Uses an INJECTED LocalKekProvider so no env/cloud key is required (offline-testable).
    """
    path = str(tmp_path / "keyvault.db")
    meta = LocalKekProvider(os.urandom(32))
    first = SubjectKeyVault(path, meta_kek=meta)
    key = first.key_for("erin")
    first.close()

    conn = sqlite3.connect(path)
    try:
        wrapped, key_version = conn.execute(
            "SELECT wrapped_dek, key_version FROM subject_keys WHERE subject_id = ?",
            ("erin",),
        ).fetchone()
    finally:
        conn.close()
    # The stored blob is the WRAPPED key, not the raw key bytes, and tagged with the
    # meta-KEK's version (not the raw sentinel).
    assert key_version != _RAW_KEY_VERSION
    assert key_version == meta.key_version
    assert bytes(wrapped) != key

    # Reopen with the SAME meta-KEK: the wrapped key unwraps to the identical bytes.
    second = SubjectKeyVault(path, meta_kek=meta)
    assert second.key_for("erin") == key
    second.close()


def test_raw_path_stores_unwrapped_key(tmp_path: Path) -> None:
    """With no meta-KEK the key is stored raw (keyvault.db itself is the secret)."""
    path = str(tmp_path / "keyvault.db")
    vault = SubjectKeyVault(path, meta_kek=None)
    key = vault.key_for("frank")
    vault.close()

    conn = sqlite3.connect(path)
    try:
        wrapped, key_version = conn.execute(
            "SELECT wrapped_dek, key_version FROM subject_keys WHERE subject_id = ?",
            ("frank",),
        ).fetchone()
    finally:
        conn.close()
    assert key_version == _RAW_KEY_VERSION
    assert bytes(wrapped) == key  # raw-on-disk


def test_memory_default_preserves_ephemeral_contract() -> None:
    """The zero-config ``:memory:`` default keeps the old ephemeral semantics."""
    vault = SubjectKeyVault()  # default path :memory:, meta-KEK resolved from env (None)
    key = vault.key_for("g")
    assert vault.key_for("g") == key  # stable within the instance
    assert vault.has("g") is True
    assert vault.destroy("g") is True
    assert vault.has("g") is False
    vault.close()


def test_unwrap_fails_loud_when_meta_kek_removed(tmp_path: Path) -> None:
    """A key wrapped under a meta-KEK that is later absent fails loud, not silently wrong."""
    from himmy.core.errors import HimmyError

    path = str(tmp_path / "keyvault.db")
    meta = LocalKekProvider(os.urandom(32))
    first = SubjectKeyVault(path, meta_kek=meta)
    first.key_for("ivan")
    first.close()

    # Reopen WITHOUT the meta-KEK: the wrapped row can't be unwrapped — must raise, not
    # hand back the wrapped bytes as if they were the key.
    second = SubjectKeyVault(path, meta_kek=None)
    with pytest.raises(HimmyError, match="meta-KEK"):
        second.key_for("ivan")
    second.close()


def test_meta_kek_resolved_from_env_by_default(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """An unset ``meta_kek`` arg resolves via build_kek_provider (HIMMY_KEK_PROVIDER)."""
    import base64

    monkeypatch.setenv("HIMMY_KEK_PROVIDER", "local")
    monkeypatch.setenv("HIMMY_ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())
    path = str(tmp_path / "keyvault.db")
    vault = SubjectKeyVault(path)  # meta_kek defaulted -> resolved from env
    vault.key_for("heidi")
    vault.close()

    conn = sqlite3.connect(path)
    try:
        key_version = conn.execute(
            "SELECT key_version FROM subject_keys WHERE subject_id = ?",
            ("heidi",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert key_version == "LOCAL"  # the env meta-KEK wrapped it (not RAW)
