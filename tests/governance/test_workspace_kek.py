"""S5: per-workspace/tenant KEK — a single tenant is cryptographically shreddable.

A single process-wide KEK (the historical default) cannot erase ONE tenant: destroying it
shreds everyone. S5 wraps each subject DEK under a PER-WORKSPACE KEK so disabling one
workspace's KEK renders ALL that workspace's subject ciphertext undecryptable while every
OTHER workspace still decrypts. Two backends: the portable local ``HKDF(master,
salt=workspace_id)`` path and the AWS-only one-CMK-per-workspace path (exercised offline via
the same fake KMS client the rest of the suite uses; GCP/Azure stay seams). These tests pin:
isolation (shred A, B survives), durability across a restart, the subject->workspace linkage
guard, and the unchanged single-process-KEK default.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cryptography")

from himmy.core.errors import HimmyError  # noqa: E402
from himmy.services.governance.retention import SubjectKeyVault  # noqa: E402
from himmy.services.storage.encryption import (  # noqa: E402
    HkdfWorkspaceKekProvider,
    WorkspaceShredded,
    build_workspace_kek_provider,
)
from himmy.services.storage.kms_providers import (  # noqa: E402
    AwsKmsWorkspaceKekProvider,
)
from tests.storage.test_kms_providers import FakeAwsKmsClient  # noqa: E402

# --------------------------------------------------------------- HKDF provider unit


def test_hkdf_provider_derives_distinct_keys_per_workspace() -> None:
    provider = HkdfWorkspaceKekProvider.generate()
    kek_a = provider.for_workspace("wsA")
    kek_b = provider.for_workspace("wsB")
    # Distinct workspaces => distinct derived KEKs and distinct (workspace-fingerprint) versions.
    assert kek_a.key_version != kek_b.key_version
    assert kek_a.key_version.startswith("local-ws.")
    dek = os.urandom(32)
    wrapped_a, ver_a = kek_a.wrap_dek(dek)
    # A's wrapped DEK only unwraps under A's KEK, not B's.
    assert kek_a.unwrap_dek(wrapped_a, ver_a) == dek
    with pytest.raises(Exception):  # noqa: B017,PT011 - wrong workspace KEK can't unwrap
        kek_b.unwrap_dek(wrapped_a, ver_a)


def test_hkdf_provider_is_deterministic_for_same_master_and_workspace() -> None:
    master = os.urandom(32)
    p1 = HkdfWorkspaceKekProvider(master)
    p2 = HkdfWorkspaceKekProvider(master)
    dek = os.urandom(32)
    wrapped, ver = p1.for_workspace("ws").wrap_dek(dek)
    # A fresh provider over the SAME master re-derives the SAME workspace KEK (restart-safe).
    assert p2.for_workspace("ws").unwrap_dek(wrapped, ver) == dek


def test_hkdf_disabled_workspace_refuses_derivation() -> None:
    provider = HkdfWorkspaceKekProvider.generate()
    provider.disable_workspace("ws")
    assert provider.is_disabled("ws") is True
    with pytest.raises(WorkspaceShredded, match="shredded"):
        provider.for_workspace("ws")
    # Other workspaces are unaffected.
    assert provider.for_workspace("other") is not None


def test_hkdf_rejects_short_master() -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        HkdfWorkspaceKekProvider(os.urandom(8))


# --------------------------------------------------- vault end-to-end (local HKDF)


def test_per_workspace_shred_isolates_one_tenant(tmp_path: Path) -> None:
    """The S5 acceptance: shred ONE workspace; its ciphertext dies, another survives."""
    path = str(tmp_path / "keyvault.db")
    ws = HkdfWorkspaceKekProvider.generate()
    vault = SubjectKeyVault(path, meta_kek=None, workspace_kek=ws)

    tok_a = vault.encryptor_for("alice", workspace_id="wsA").encrypt("a", aad=b"alice")
    tok_b = vault.encryptor_for("bob", workspace_id="wsB").encrypt("b", aad=b"bob")
    assert vault.supports_workspace_shred is True
    assert vault.subjects_in_workspace("wsA") == ["alice"]

    destroyed = vault.shred_workspace("wsA")
    assert destroyed == 1
    assert vault.is_workspace_shredded("wsA") is True

    # wsB still decrypts symmetrically; wsA is permanently undecryptable.
    assert vault.encryptor_for("bob").decrypt(tok_b, aad=b"bob") == "b"
    with pytest.raises(WorkspaceShredded):
        vault.encryptor_for("alice").decrypt(tok_a, aad=b"alice")
    vault.close()


def test_per_workspace_shred_is_durable_across_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "keyvault.db")
    master = os.urandom(32)
    vault = SubjectKeyVault(
        path, meta_kek=None, workspace_kek=HkdfWorkspaceKekProvider(master)
    )
    tok = vault.encryptor_for("alice", workspace_id="wsA").encrypt("a", aad=b"alice")
    vault.shred_workspace("wsA")
    vault.close()

    # Restart with the SAME master: the shred (disabled-set) is durable, so wsA stays dead.
    restarted = SubjectKeyVault(
        path, meta_kek=None, workspace_kek=HkdfWorkspaceKekProvider(master)
    )
    assert restarted.is_workspace_shredded("wsA") is True
    with pytest.raises(WorkspaceShredded):
        restarted.encryptor_for("alice").decrypt(tok, aad=b"alice")
    restarted.close()


def test_subject_cannot_silently_change_workspace(tmp_path: Path) -> None:
    """A subject is bound to ONE tenant; re-requesting under another raises (no mis-shred)."""
    path = str(tmp_path / "keyvault.db")
    ws = HkdfWorkspaceKekProvider.generate()
    vault = SubjectKeyVault(path, meta_kek=None, workspace_kek=ws)
    vault.key_for("alice", workspace_id="wsA")
    # Same workspace (or omitted) is fine; a DIFFERENT one is rejected.
    assert vault.key_for("alice", workspace_id="wsA") == vault.key_for("alice")
    with pytest.raises(HimmyError, match="cannot move between tenants"):
        vault.key_for("alice", workspace_id="wsB")
    vault.close()


def test_workspace_linkage_is_recorded_and_queryable(tmp_path: Path) -> None:
    path = str(tmp_path / "keyvault.db")
    ws = HkdfWorkspaceKekProvider.generate()
    vault = SubjectKeyVault(path, meta_kek=None, workspace_kek=ws)
    vault.key_for("alice", workspace_id="wsA")
    vault.key_for("carol", workspace_id="wsA")
    vault.key_for("bob", workspace_id="wsB")
    assert vault.workspace_of("alice") == "wsA"
    assert vault.workspace_of("bob") == "wsB"
    assert vault.workspace_of("nobody") is None
    assert sorted(vault.subjects_in_workspace("wsA")) == ["alice", "carol"]
    vault.close()


def test_shredded_workspace_blocks_reonboarding(tmp_path: Path) -> None:
    """Re-onboarding a subject into a SHREDDED workspace fails loud (KEK gone)."""
    path = str(tmp_path / "keyvault.db")
    ws = HkdfWorkspaceKekProvider.generate()
    vault = SubjectKeyVault(path, meta_kek=None, workspace_kek=ws)
    vault.key_for("alice", workspace_id="wsA")
    vault.shred_workspace("wsA")
    # alice's row was nulled by the shred; re-minting needs the (now disabled) workspace KEK.
    with pytest.raises(WorkspaceShredded):
        vault.key_for("alice", workspace_id="wsA")
    vault.close()


def test_shred_workspace_requires_provider(tmp_path: Path) -> None:
    """Without a per-workspace KEK there is nothing tenant-scoped to disable."""
    path = str(tmp_path / "keyvault.db")
    vault = SubjectKeyVault(path, meta_kek=None, workspace_kek=None)
    vault.key_for("alice", workspace_id="wsA")
    with pytest.raises(HimmyError, match="requires a per-workspace KEK"):
        vault.shred_workspace("wsA")
    vault.close()


def test_default_is_single_process_kek_no_behavior_change(tmp_path: Path) -> None:
    """No workspace provider => the single-process-KEK path, unchanged (workspace_id recorded)."""
    path = str(tmp_path / "keyvault.db")
    vault = SubjectKeyVault(path, meta_kek=None, workspace_kek=None)
    assert vault.supports_workspace_shred is False
    tok = vault.encryptor_for("alice", workspace_id="wsA").encrypt("a", aad=b"alice")
    # Decrypts fine (raw single-process path); linkage is still recorded for the reach-map.
    assert vault.encryptor_for("alice").decrypt(tok, aad=b"alice") == "a"
    assert vault.workspace_of("alice") == "wsA"
    vault.close()


# --------------------------------------------------------- AWS KMS workspace path


def test_aws_workspace_provider_one_cmk_per_workspace_via_fake_client() -> None:
    """One CMK per workspace; a wrap targets the workspace-templated KeyId (offline fake)."""
    client = FakeAwsKmsClient()
    provider = AwsKmsWorkspaceKekProvider(
        key_id_template="alias/himmy-ws-{workspace_id}", client=client
    )
    dek = os.urandom(32)
    kek_a = provider.for_workspace("wsA")
    wrapped, ver = kek_a.wrap_dek(dek)
    assert kek_a.unwrap_dek(wrapped, ver) == dek
    # The CMK id is the workspace-templated alias.
    assert client.encrypt_calls[-1]["KeyId"] == "alias/himmy-ws-wsA"


def test_aws_workspace_disable_calls_kms_and_refuses_locally() -> None:
    class DisablingClient(FakeAwsKmsClient):
        def __init__(self) -> None:
            super().__init__()
            self.disabled: list[str] = []

        def disable_key(self, *, KeyId: str) -> dict[str, Any]:  # noqa: N803
            self.disabled.append(KeyId)
            return {}

    client = DisablingClient()
    provider = AwsKmsWorkspaceKekProvider(
        key_id_template="alias/ws-{workspace_id}", client=client
    )
    provider.disable_workspace("wsA")
    # The HSM-side CMK was disabled AND local derivation now refuses.
    assert client.disabled == ["alias/ws-wsA"]
    assert provider.is_disabled("wsA") is True
    with pytest.raises(WorkspaceShredded):
        provider.for_workspace("wsA")


def test_aws_workspace_schedule_deletion_escalates_kms_call() -> None:
    class DeletingClient(FakeAwsKmsClient):
        def __init__(self) -> None:
            super().__init__()
            self.scheduled: list[tuple[str, int]] = []

        def schedule_key_deletion(  # noqa: N802
            self, *, KeyId: str, PendingWindowInDays: int  # noqa: N803
        ) -> dict[str, Any]:
            self.scheduled.append((KeyId, PendingWindowInDays))
            return {}

    client = DeletingClient()
    provider = AwsKmsWorkspaceKekProvider(
        key_id_template="alias/ws-{workspace_id}",
        client=client,
        schedule_deletion_days=7,
    )
    provider.disable_workspace("wsA")
    assert client.scheduled == [("alias/ws-wsA", 7)]


def test_aws_workspace_template_must_contain_placeholder() -> None:
    with pytest.raises(ValueError, match=r"\{workspace_id\}"):
        AwsKmsWorkspaceKekProvider(key_id_template="alias/fixed")


def test_aws_workspace_vault_shred_disables_cmk(tmp_path: Path) -> None:
    """End-to-end: a vault over the AWS workspace provider shreds via the CMK + nulls keys."""

    class DisablingClient(FakeAwsKmsClient):
        def __init__(self) -> None:
            super().__init__()
            self.disabled: list[str] = []

        def disable_key(self, *, KeyId: str) -> dict[str, Any]:  # noqa: N803
            self.disabled.append(KeyId)
            return {}

    client = DisablingClient()
    provider = AwsKmsWorkspaceKekProvider(
        key_id_template="alias/ws-{workspace_id}", client=client
    )
    path = str(tmp_path / "keyvault.db")
    vault = SubjectKeyVault(path, meta_kek=None, workspace_kek=provider)
    tok = vault.encryptor_for("alice", workspace_id="wsA").encrypt("a", aad=b"alice")
    assert vault.shred_workspace("wsA") == 1
    assert client.disabled == ["alias/ws-wsA"]
    with pytest.raises(WorkspaceShredded):
        vault.encryptor_for("alice").decrypt(tok, aad=b"alice")
    vault.close()


# ----------------------------------------------------- builder (env-selected)


def _with_secrets(mapping: dict[str, str]) -> None:
    from himmy.config.secrets import configure_secrets

    class _Secrets:
        def get(self, name: str) -> str | None:
            return mapping.get(name)

    configure_secrets(_Secrets())


def test_build_workspace_provider_default_is_none(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_WORKSPACE_KEK_PROVIDER", raising=False)
    assert build_workspace_kek_provider() is None


def test_build_workspace_provider_local(monkeypatch: Any) -> None:
    import base64

    monkeypatch.setenv("HIMMY_WORKSPACE_KEK_PROVIDER", "local")
    _with_secrets(
        {"HIMMY_WORKSPACE_KEK_MASTER": base64.b64encode(os.urandom(32)).decode()}
    )
    try:
        provider = build_workspace_kek_provider()
        assert isinstance(provider, HkdfWorkspaceKekProvider)
    finally:
        from himmy.config.secrets import configure_secrets

        configure_secrets(None)


def test_build_workspace_provider_local_needs_master(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_WORKSPACE_KEK_PROVIDER", "local")
    _with_secrets({})
    try:
        with pytest.raises(ValueError, match="HIMMY_WORKSPACE_KEK_MASTER"):
            build_workspace_kek_provider()
    finally:
        from himmy.config.secrets import configure_secrets

        configure_secrets(None)


def test_build_workspace_provider_aws_needs_template(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_WORKSPACE_KEK_PROVIDER", "aws-kms")
    _with_secrets({})
    try:
        with pytest.raises(ValueError, match="HIMMY_AWS_KMS_KEY_ID_TEMPLATE"):
            build_workspace_kek_provider()
    finally:
        from himmy.config.secrets import configure_secrets

        configure_secrets(None)


def test_build_workspace_provider_unknown(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_WORKSPACE_KEK_PROVIDER", "nope")
    try:
        with pytest.raises(ValueError, match="unknown HIMMY_WORKSPACE_KEK_PROVIDER"):
            build_workspace_kek_provider()
    finally:
        pass


def test_vault_resolves_workspace_provider_from_env_with_durable_disabled_set(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An env-configured local workspace provider shreds durably via the vault's own db."""
    import base64

    master_b64 = base64.b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("HIMMY_WORKSPACE_KEK_PROVIDER", "local")
    _with_secrets({"HIMMY_WORKSPACE_KEK_MASTER": master_b64})
    try:
        path = str(tmp_path / "keyvault.db")
        vault = SubjectKeyVault(path, meta_kek=None)  # workspace_kek resolved from env
        assert vault.supports_workspace_shred is True
        tok = vault.encryptor_for("alice", workspace_id="wsA").encrypt("a", aad=b"alice")
        vault.shred_workspace("wsA")
        vault.close()

        # A NEW vault over the SAME env master must see the durable disabled-set => still dead.
        vault2 = SubjectKeyVault(path, meta_kek=None)
        assert vault2.is_workspace_shredded("wsA") is True
        with pytest.raises(WorkspaceShredded):
            vault2.encryptor_for("alice").decrypt(tok, aad=b"alice")
        vault2.close()
    finally:
        from himmy.config.secrets import configure_secrets

        configure_secrets(None)


def test_legacy_keyvault_db_migrates_forward(tmp_path: Path) -> None:
    """A pre-S5 keyvault.db (no workspace_id column) gains it forward without data loss."""
    path = str(tmp_path / "keyvault.db")
    # Simulate a legacy file: the OLD subject_keys schema with no workspace_id column.
    raw = sqlite3.connect(path)
    raw.executescript(
        """
        CREATE TABLE subject_keys (
            subject_id  TEXT PRIMARY KEY,
            wrapped_dek BLOB,
            key_version TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            shredded_at TEXT
        );
        """
    )
    raw.execute(
        "INSERT INTO subject_keys (subject_id, wrapped_dek, key_version, created_at) "
        "VALUES ('legacy', ?, 'RAW', '2020-01-01T00:00:00Z')",
        (os.urandom(32),),
    )
    raw.commit()
    raw.close()

    vault = SubjectKeyVault(path, meta_kek=None, workspace_kek=None)
    cols = {
        r[1]
        for r in vault._conn.execute("PRAGMA table_info(subject_keys)").fetchall()
    }
    assert "workspace_id" in cols  # the migration added it
    # The legacy row still resolves (workspace_id NULL => single-process path) and has a key.
    assert vault.has("legacy") is True
    assert vault.workspace_of("legacy") is None
    vault.close()
