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
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable


@runtime_checkable
class SecretProvider(Protocol):
    """Resolves a secret name to its value (or ``None`` if absent)."""

    def get(self, name: str) -> str | None:
        """Return the secret's value, or ``None`` when this provider lacks it."""
        ...


@runtime_checkable
class WritableSecretProvider(SecretProvider, Protocol):
    """A secret provider that can also persist + remove secrets (for a GUI).

    Most backends are read-only (env/vault/cloud). The Studio Connections UI needs
    to *write* a secret a user pastes in, so it resolves the active provider and
    only offers connect/disconnect when a writable one is in play.
    """

    def set(self, name: str, value: str) -> None:
        """Persist ``value`` under ``name`` (upsert)."""
        ...

    def delete(self, name: str) -> None:
        """Remove ``name`` (idempotent — absent is success)."""
        ...


class EnvSecrets:
    """Read secrets from environment variables (the default, unchanged behavior)."""

    def get(self, name: str) -> str | None:
        return os.environ.get(name)


class FileSecrets:
    """Read+write secrets as files (the ``*_FILE`` convention + a secrets directory).

    Reading honors a ``<NAME>_FILE`` env pointer or a ``<NAME>`` file under the base
    dir. Writing (``set``/``delete``) only targets the base dir, so a writable
    instance must be constructed with one — the keychain-fallback path does this.
    """

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

    def set(self, name: str, value: str) -> None:
        if self._dir is None:
            raise RuntimeError("FileSecrets has no base dir; cannot write")
        self._dir.mkdir(parents=True, exist_ok=True)
        target = self._dir / name
        # Atomic write with a 0600 temp file so the secret is never world-readable
        # nor half-written.
        fd, tmp = tempfile.mkstemp(dir=str(self._dir))
        try:
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(value)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        os.chmod(target, 0o600)

    def delete(self, name: str) -> None:
        if self._dir is not None:
            (self._dir / name).unlink(missing_ok=True)


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
        return cast("str | None", resp.get("SecretString"))


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
        return cast(str, resp.payload.data.decode("utf-8"))


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
            return cast("str | None", self._ensure().get_secret(name).value)  # type: ignore[attr-defined]
        except Exception:
            return None


class KeychainSecrets:
    """macOS login keychain via the ``security`` CLI (read + write + delete).

    One generic-password item per secret name (``service="himmy"``,
    ``account=<NAME>``). Write-capable, so the Studio Connections UI can store a
    pasted token. macOS-only — :func:`build_secret_provider` falls back to a
    writable :class:`FileSecrets` on other platforms.

    NOTE: ``set`` passes the value as a CLI argument, so it is briefly visible in
    the process table. Acceptable for a local, single-user, loopback GUI; do not
    use this backend on a shared host.
    """

    _NOT_FOUND = 44  # `security`'s exit code for "item not found"

    def __init__(
        self,
        *,
        service: str = "himmy",
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._service = service
        self._runner = runner or self._default_runner

    @staticmethod
    def available() -> bool:
        """True only on macOS with the ``security`` binary present."""
        return platform.system() == "Darwin" and shutil.which("security") is not None

    @staticmethod
    def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        # Fixed `security` argv (no shell, no untrusted executable) — the only
        # variable parts are the secret name/value passed as discrete args.
        return subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, check=False
        )

    def get(self, name: str) -> str | None:
        result = self._runner(
            ["security", "find-generic-password", "-s", self._service, "-a", name, "-w"]
        )
        if result.returncode != 0:
            return None
        return (result.stdout or "").rstrip("\n")

    def set(self, name: str, value: str) -> None:
        # -U upserts (replaces an existing item rather than erroring on duplicate).
        result = self._runner(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                self._service,
                "-a",
                name,
                "-w",
                value,
            ]
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"keychain write failed for {name!r} (exit {result.returncode})"
            )

    def delete(self, name: str) -> None:
        result = self._runner(
            ["security", "delete-generic-password", "-s", self._service, "-a", name]
        )
        if result.returncode not in (0, self._NOT_FOUND):
            raise RuntimeError(
                f"keychain delete failed for {name!r} (exit {result.returncode})"
            )


def _default_secrets_dir() -> str:
    """Where the writable file backend stores secrets when no dir is configured."""
    return os.environ.get("HIMMY_SECRETS_DIR") or str(Path(".himmy") / "secrets")


def build_secret_provider() -> SecretProvider:
    """Select the secret backend from ``HIMMY_SECRETS`` (env fallback for all)."""
    mode = os.environ.get("HIMMY_SECRETS", "env").strip().lower()
    env = EnvSecrets()
    if mode == "keychain":
        if KeychainSecrets.available():
            return ChainSecretProvider([KeychainSecrets(), env])
        # Non-macOS: a writable file store keeps the Connections UI functional.
        return ChainSecretProvider([FileSecrets(_default_secrets_dir()), env])
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


def get_writable_provider() -> WritableSecretProvider | None:
    """The active provider if it can persist secrets (keychain/file), else ``None``.

    Unwraps a :class:`ChainSecretProvider` to find the first writable member. The
    Connections service uses this to decide whether connect/disconnect is possible.
    """
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = build_secret_provider()
    candidates: list[SecretProvider]
    if isinstance(_PROVIDER, ChainSecretProvider):
        candidates = list(_PROVIDER._providers)
    else:
        candidates = [_PROVIDER]
    for provider in candidates:
        if isinstance(provider, WritableSecretProvider):
            return provider
    return None


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
    "WritableSecretProvider",
    "EnvSecrets",
    "FileSecrets",
    "KeychainSecrets",
    "ChainSecretProvider",
    "VaultSecrets",
    "AwsSecretsManager",
    "GcpSecretManager",
    "AzureKeyVault",
    "build_secret_provider",
    "get_writable_provider",
    "configure_secrets",
    "get_secret",
]
