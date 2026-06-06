"""WS3.1 — secret provider abstraction (env / file / vault / cloud)."""

from __future__ import annotations

import subprocess
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
    KeychainSecrets,
    VaultSecrets,
    WritableSecretProvider,
    build_secret_provider,
    configure_secrets,
    get_secret,
    get_writable_provider,
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


def test_file_secrets_writable_round_trip(tmp_path: Path) -> None:
    fs = FileSecrets(tmp_path / "secrets")
    assert isinstance(fs, WritableSecretProvider)
    fs.set("HIMMY_TELEGRAM_BOT_TOKEN", "123:abc")
    assert fs.get("HIMMY_TELEGRAM_BOT_TOKEN") == "123:abc"
    # written 0600 and removable
    mode = (tmp_path / "secrets" / "HIMMY_TELEGRAM_BOT_TOKEN").stat().st_mode & 0o777
    assert mode == 0o600
    fs.delete("HIMMY_TELEGRAM_BOT_TOKEN")
    assert fs.get("HIMMY_TELEGRAM_BOT_TOKEN") is None
    fs.delete("HIMMY_TELEGRAM_BOT_TOKEN")  # idempotent


def test_keychain_secrets_with_fake_runner() -> None:
    store: dict[str, str] = {}
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        sub = argv[1]
        name = argv[argv.index("-a") + 1]
        if sub == "find-generic-password":
            if name in store:
                return subprocess.CompletedProcess(argv, 0, store[name] + "\n", "")
            return subprocess.CompletedProcess(argv, 44, "", "not found")
        if sub == "add-generic-password":
            store[name] = argv[argv.index("-w") + 1]
            return subprocess.CompletedProcess(argv, 0, "", "")
        if sub == "delete-generic-password":
            existed = store.pop(name, None) is not None
            return subprocess.CompletedProcess(argv, 0 if existed else 44, "", "")
        return subprocess.CompletedProcess(argv, 1, "", "")

    kc = KeychainSecrets(runner=runner)
    assert isinstance(kc, WritableSecretProvider)
    assert kc.get("HIMMY_SMTP_PASSWORD") is None  # exit 44 → None
    kc.set("HIMMY_SMTP_PASSWORD", "pw")
    assert kc.get("HIMMY_SMTP_PASSWORD") == "pw"  # newline stripped
    kc.delete("HIMMY_SMTP_PASSWORD")
    kc.delete("HIMMY_SMTP_PASSWORD")  # exit 44 tolerated
    assert kc.get("HIMMY_SMTP_PASSWORD") is None
    assert calls[0][:2] == ["security", "find-generic-password"]


def test_build_keychain_mode_selects_writable(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_SECRETS", "keychain")
    # Force the non-macOS path so the test is platform-independent.
    monkeypatch.setattr(KeychainSecrets, "available", staticmethod(lambda: False))
    provider = build_secret_provider()
    configure_secrets(provider)
    writable = get_writable_provider()
    assert isinstance(writable, FileSecrets)  # file fallback is writable


def test_get_writable_provider_none_for_readonly_backend(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_SECRETS", raising=False)
    configure_secrets(build_secret_provider())  # plain EnvSecrets
    assert get_writable_provider() is None


def test_toolkit_config_reads_secrets_from_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """End-to-end: ToolkitConfig.from_env sources a secret from a file, not env."""
    (tmp_path / "HIMMY_SMTP_PASSWORD").write_text("vaulted-pw")
    monkeypatch.delenv("HIMMY_SMTP_PASSWORD", raising=False)
    configure_secrets(ChainSecretProvider([FileSecrets(tmp_path), EnvSecrets()]))
    cfg = ToolkitConfig.from_env()
    assert cfg.smtp_password == "vaulted-pw"
