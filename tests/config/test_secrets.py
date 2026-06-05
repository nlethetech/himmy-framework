"""WS3.1 — secret provider abstraction (env / file / vault / cloud)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from himmy.config.secrets import (
    AwsSecretsManager,
    AzureKeyVault,
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    GcpSecretManager,
    VaultSecrets,
    build_secret_provider,
    configure_secrets,
    get_secret,
)
from himmy.toolkit.config import ToolkitConfig


@pytest.fixture(autouse=True)
def _reset_provider() -> Iterator[None]:
    yield
    configure_secrets(None)  # don't leak a configured provider into other tests


def test_env_secrets(monkeypatch: Any) -> None:
    monkeypatch.setenv("MY_SECRET", "from-env")
    assert EnvSecrets().get("MY_SECRET") == "from-env"
    assert EnvSecrets().get("ABSENT") is None


def test_file_secrets_file_suffix_convention(tmp_path: Path, monkeypatch: Any) -> None:
    secret_file = tmp_path / "db.txt"
    secret_file.write_text("super-dsn\n")
    monkeypatch.setenv("HIMMY_SQL_DSN_FILE", str(secret_file))
    assert (
        FileSecrets().get("HIMMY_SQL_DSN") == "super-dsn"
    )  # trailing newline stripped


def test_file_secrets_directory(tmp_path: Path) -> None:
    (tmp_path / "HIMMY_SMTP_PASSWORD").write_text("pw123")
    fs = FileSecrets(tmp_path)
    assert fs.get("HIMMY_SMTP_PASSWORD") == "pw123"
    assert fs.get("ABSENT") is None


def test_chain_first_non_none_wins(tmp_path: Path) -> None:
    (tmp_path / "A").write_text("from-file")
    chain = ChainSecretProvider([FileSecrets(tmp_path), EnvSecrets()])
    assert chain.get("A") == "from-file"
    assert chain.get("MISSING") is None


def test_vault_secrets_with_injected_loader() -> None:
    vault = VaultSecrets(
        addr="https://vault",
        token="t",
        path="himmy",
        loader=lambda: {"HIMMY_SQL_DSN": "vault-dsn", "K": "v"},
    )
    assert vault.get("HIMMY_SQL_DSN") == "vault-dsn"
    assert vault.get("missing") is None


def test_aws_secrets_manager_with_fake_client() -> None:
    class _Client:
        def get_secret_value(self, SecretId: str) -> dict[str, str]:  # noqa: N803
            return {"SecretString": f"aws:{SecretId}"}

    assert AwsSecretsManager(client=_Client()).get("db") == "aws:db"


def test_gcp_secret_manager_with_fake_client() -> None:
    class _Payload:
        data = b"gcp-secret"

    class _Resp:
        payload = _Payload()

    class _Client:
        def access_secret_version(self, name: str) -> _Resp:
            return _Resp()

    mgr = GcpSecretManager(client=_Client(), project="proj")
    assert mgr.get("db") == "gcp-secret"


def test_azure_key_vault_with_fake_client() -> None:
    class _Secret:
        value = "azure-secret"

    class _Client:
        def get_secret(self, name: str) -> _Secret:
            return _Secret()

    kv = AzureKeyVault(client=_Client(), vault_url="https://kv")
    assert kv.get("db") == "azure-secret"


def test_build_secret_provider_modes(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_SECRETS", raising=False)
    assert isinstance(build_secret_provider(), EnvSecrets)
    monkeypatch.setenv("HIMMY_SECRETS", "file")
    assert isinstance(build_secret_provider(), ChainSecretProvider)


def test_get_secret_uses_configured_provider(tmp_path: Path) -> None:
    (tmp_path / "HIMMY_SMTP_PASSWORD").write_text("file-pw")
    configure_secrets(ChainSecretProvider([FileSecrets(tmp_path), EnvSecrets()]))
    assert get_secret("HIMMY_SMTP_PASSWORD") == "file-pw"
    assert get_secret("ABSENT", "fallback") == "fallback"


def test_toolkit_config_reads_secrets_from_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """End-to-end: ToolkitConfig.from_env sources a secret from a file, not env."""
    (tmp_path / "HIMMY_SMTP_PASSWORD").write_text("vaulted-pw")
    monkeypatch.delenv("HIMMY_SMTP_PASSWORD", raising=False)
    configure_secrets(ChainSecretProvider([FileSecrets(tmp_path), EnvSecrets()]))
    cfg = ToolkitConfig.from_env()
    assert cfg.smtp_password == "vaulted-pw"
