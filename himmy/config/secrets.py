"""Secret provider abstraction: read credentials from env, files, or a vault (WS3.1).

Every secret (DB DSNs, SMTP/Telegram/search keys, the internal API key, the gateway
key) is read through :func:`get_secret` rather than straight from ``os.environ``, so an
operator can source secrets from Docker/K8s secret files or a managed vault without code
changes. The default provider is :class:`EnvSecrets` — **identical to the old behavior**,
so offline/zero-config is unchanged. Select another backend with ``HIMMY_SECRETS``:

* ``env`` *(default)* — environment variables.
* ``file`` — Docker/K8s secret files: a ``<NAME>_FILE`` env pointing at a file, or a
  file named ``<NAME>`` under ``HIMMY_SECRETS_DIR`` (e.g. ``/run/secrets``). Falls back
  to env for anything not present as a file.
* ``vault`` — HashiCorp Vault KV v2 over HTTP (no SDK). Env fallback.
* ``aws`` / ``gcp`` / ``azure`` — the cloud secret managers (SDK lazily imported; client
  injectable for tests). Env fallback.

All non-``env`` backends are chained with an env fallback, so a partially-migrated
deployment keeps working.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable


@runtime_checkable
class SecretProvider(Protocol):
    """Resolves a secret name to its value (or ``None`` if absent)."""

    def get(self, name: str) -> str | None:
        """Return the secret's value, or ``None`` when this provider lacks it."""
        ...


class EnvSecrets:
    """Read secrets from environment variables (the default, unchanged behavior)."""

    def get(self, name: str) -> str | None:
        return os.environ.get(name)


class FileSecrets:
    """Read secrets from files (the ``*_FILE`` convention + a secrets directory)."""

    def __init__(self, base_dir: str | os.PathLike[str] | None = None) -> None:
        self._dir = Path(base_dir) if base_dir else None

    def get(self, name: str) -> str | None:
        file_env = os.environ.get(f"{name}_FILE")
        if file_env and Path(file_env).is_file():
            return Path(file_env).read_text(encoding="utf-8").strip()
        if self._dir is not None:
            candidate = self._dir / name
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
        return None


class ChainSecretProvider:
    """Try each provider in order; the first non-``None`` value wins."""

    def __init__(self, providers: Iterable[SecretProvider]) -> None:
        self._providers = list(providers)

    def get(self, name: str) -> str | None:
        for provider in self._providers:
            value = provider.get(name)
            if value is not None:
                return value
        return None


class VaultSecrets:
    """HashiCorp Vault KV v2 over HTTP (one KV path holding many keys), no SDK."""

    def __init__(
        self,
        *,
        addr: str,
        token: str,
        path: str,
        mount: str = "secret",
        timeout: float = 10.0,
        loader: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self._addr = addr.rstrip("/")
        self._token = token
        self._path = path.strip("/")
        self._mount = mount.strip("/")
        self._timeout = timeout
        self._loader = loader
        self._cache: dict[str, str] | None = None

    @classmethod
    def from_env(cls) -> VaultSecrets:
        """Build from ``VAULT_ADDR`` / ``VAULT_TOKEN`` / ``HIMMY_VAULT_*``."""
        addr = os.environ.get("VAULT_ADDR")
        token = os.environ.get("VAULT_TOKEN")
        path = os.environ.get("HIMMY_VAULT_PATH")
        if not (addr and token and path):
            raise ValueError(
                "vault secrets need VAULT_ADDR, VAULT_TOKEN and HIMMY_VAULT_PATH"
            )
        return cls(
            addr=addr,
            token=token,
            path=path,
            mount=os.environ.get("HIMMY_VAULT_MOUNT", "secret"),
        )

    def _load(self) -> dict[str, str]:
        if self._loader is not None:
            return self._loader()
        import httpx

        url = f"{self._addr}/v1/{self._mount}/data/{self._path}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(url, headers={"X-Vault-Token": self._token})
            resp.raise_for_status()
            body = resp.json()
        data = ((body or {}).get("data") or {}).get("data") or {}
        return {str(k): str(v) for k, v in data.items()}

    def get(self, name: str) -> str | None:
        if self._cache is None:
            self._cache = self._load()
        return self._cache.get(name)


class AwsSecretsManager:
    """AWS Secrets Manager (boto3 lazily imported; client injectable for tests)."""

    def __init__(
        self, *, client: object | None = None, region: str | None = None
    ) -> None:
        self._client = client
        self._region = region

    def _ensure(self) -> object:
        if self._client is None:
            import boto3

            self._client = boto3.client("secretsmanager", region_name=self._region)
        return self._client

    def get(self, name: str) -> str | None:
        try:
            resp = self._ensure().get_secret_value(SecretId=name)  # type: ignore[attr-defined]
        except Exception:
            return None
        return resp.get("SecretString")


class GcpSecretManager:
    """GCP Secret Manager (SDK lazily imported; client injectable for tests)."""

    def __init__(
        self, *, client: object | None = None, project: str | None = None
    ) -> None:
        self._client = client
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")

    def _ensure(self) -> object:
        if self._client is None:
            from google.cloud import secretmanager

            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get(self, name: str) -> str | None:
        if not self._project:
            return None
        path = f"projects/{self._project}/secrets/{name}/versions/latest"
        try:
            resp = self._ensure().access_secret_version(name=path)  # type: ignore[attr-defined]
        except Exception:
            return None
        return resp.payload.data.decode("utf-8")


class AzureKeyVault:
    """Azure Key Vault (SDK lazily imported; client injectable for tests)."""

    def __init__(
        self, *, client: object | None = None, vault_url: str | None = None
    ) -> None:
        self._client = client
        self._vault_url = vault_url or os.environ.get("HIMMY_AZURE_VAULT_URL")

    def _ensure(self) -> object:
        if self._client is None:
            from azure.identity import (
                DefaultAzureCredential,  # type: ignore[import-not-found]
            )
            from azure.keyvault.secrets import (
                SecretClient,  # type: ignore[import-not-found]
            )

            self._client = SecretClient(
                vault_url=self._vault_url, credential=DefaultAzureCredential()
            )
        return self._client

    def get(self, name: str) -> str | None:
        if not self._vault_url:
            return None
        try:
            return self._ensure().get_secret(name).value  # type: ignore[attr-defined]
        except Exception:
            return None


def build_secret_provider() -> SecretProvider:
    """Select the secret backend from ``HIMMY_SECRETS`` (env fallback for all)."""
    mode = os.environ.get("HIMMY_SECRETS", "env").strip().lower()
    env = EnvSecrets()
    if mode == "file":
        return ChainSecretProvider(
            [FileSecrets(os.environ.get("HIMMY_SECRETS_DIR")), env]
        )
    if mode == "vault":
        return ChainSecretProvider([VaultSecrets.from_env(), env])
    if mode == "aws":
        return ChainSecretProvider([AwsSecretsManager(), env])
    if mode == "gcp":
        return ChainSecretProvider([GcpSecretManager(), env])
    if mode == "azure":
        return ChainSecretProvider([AzureKeyVault(), env])
    return env


_PROVIDER: SecretProvider | None = None


def configure_secrets(provider: SecretProvider | None) -> None:
    """Install (or clear) the process-wide secret provider (mainly for embedding/tests)."""
    global _PROVIDER
    _PROVIDER = provider


def get_secret(name: str, default: str | None = None) -> str | None:
    """Resolve secret ``name`` through the configured provider (env by default)."""
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = build_secret_provider()
    value = _PROVIDER.get(name)
    return value if value is not None else default


__all__ = [
    "SecretProvider",
    "EnvSecrets",
    "FileSecrets",
    "ChainSecretProvider",
    "VaultSecrets",
    "AwsSecretsManager",
    "GcpSecretManager",
    "AzureKeyVault",
    "build_secret_provider",
    "configure_secrets",
    "get_secret",
]
