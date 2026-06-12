"""Cloud KMS KEK providers for envelope field encryption (WS4.4 / cloud-kms).

These implement the :class:`~himmy.services.storage.encryption.KekProvider` protocol by
delegating DEK wrap/unwrap to a managed Key Management Service, so the master key (KEK)
never leaves the cloud HSM — only the wrapped DEK and ciphertext are stored locally.

* :class:`AwsKmsKekProvider` — the reference implementation (AWS KMS ``encrypt`` /
  ``decrypt``). boto3 is lazily imported (the ``kms-aws`` extra) and the client is
  injectable so tests never touch AWS.
* :class:`GcpKmsKekProvider` / :class:`AzureKeyVaultKekProvider` — documented seams that
  raise :class:`NotImplementedError`; the docstrings point at the SDK call to wire in.

The provider's ``key_version`` embeds a stable, *non-secret* fingerprint of the key id
(hashed) so a token can be matched to the KEK that wrapped it without revealing the ARN.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast


class AwsKmsKekProvider:
    """Wrap/unwrap a DEK with AWS KMS (the reference cloud KEK provider).

    Wrapping calls KMS ``encrypt`` on the plaintext DEK; unwrapping calls KMS ``decrypt``
    on the returned blob (KMS records which CMK produced a blob, so ``KeyId`` is not
    required to decrypt). boto3 is imported lazily so the dependency stays optional, and
    ``client`` may be injected for offline tests. (botocore exposes snake_case operation
    methods — ``client.encrypt`` / ``client.decrypt`` — not the PascalCase API names.)

    ``key_version`` is ``"aws-kms.<fp>"`` where ``<fp>`` is the first 16 hex chars of the
    SHA-256 of the key id — a stable, non-secret fingerprint that ties a token to its CMK
    without storing the ARN. (The tag is colon-free so it round-trips the token's
    ``:``-delimited version segment.)
    """

    def __init__(
        self,
        *,
        key_id: str,
        client: Any | None = None,
        region: str | None = None,
    ) -> None:
        """Use CMK ``key_id`` (ARN / key-id / alias); ``client`` overrides boto3."""
        if not key_id:
            raise ValueError("AwsKmsKekProvider needs a key_id")
        self._key_id = key_id
        self._client = client
        self._region = region

    def _ensure(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - only without boto3
                raise RuntimeError(
                    "the 'aws-kms' KEK provider needs the 'kms-aws' extra "
                    "(install it: pip install 'himmy[kms-aws]')"
                ) from exc

            self._client = boto3.client("kms", region_name=self._region)
        return self._client

    @property
    def key_version(self) -> str:
        fingerprint = hashlib.sha256(self._key_id.encode("utf-8")).hexdigest()[:16]
        return f"aws-kms.{fingerprint}"

    def wrap_dek(self, plaintext_dek: bytes) -> tuple[bytes, str]:
        """Wrap the DEK via KMS ``encrypt``; return ``(ciphertext_blob, key_version)``."""
        resp = self._ensure().encrypt(KeyId=self._key_id, Plaintext=plaintext_dek)
        return cast(bytes, resp["CiphertextBlob"]), self.key_version

    def unwrap_dek(self, wrapped_dek: bytes, key_version: str) -> bytes:
        """Unwrap the DEK via KMS ``decrypt`` (``key_version`` is informational)."""
        resp = self._ensure().decrypt(CiphertextBlob=wrapped_dek)
        return cast(bytes, resp["Plaintext"])


class GcpKmsKekProvider:
    """Documented seam for Google Cloud KMS (not yet implemented).

    To implement: use ``google.cloud.kms`` (the ``kms-gcp`` extra). Wrap with
    ``KeyManagementServiceClient.encrypt(name=<crypto-key>, plaintext=dek)`` and unwrap
    with ``.decrypt(name=<crypto-key>, ciphertext=wrapped)``. Set ``key_version`` to a
    non-secret fingerprint of the crypto-key resource name (mirroring
    :class:`AwsKmsKekProvider`).
    """

    def __init__(
        self, *, key_name: str | None = None, client: Any | None = None
    ) -> None:
        self._key_name = key_name
        self._client = client

    @property
    def key_version(self) -> str:
        raise NotImplementedError(
            "GcpKmsKekProvider is a documented seam; wire up google.cloud.kms "
            "(KeyManagementServiceClient.encrypt/decrypt) to implement it."
        )

    def wrap_dek(self, plaintext_dek: bytes) -> tuple[bytes, str]:
        raise NotImplementedError(
            "GcpKmsKekProvider is a documented seam; wire up google.cloud.kms "
            "(KeyManagementServiceClient.encrypt) to implement it."
        )

    def unwrap_dek(self, wrapped_dek: bytes, key_version: str) -> bytes:
        raise NotImplementedError(
            "GcpKmsKekProvider is a documented seam; wire up google.cloud.kms "
            "(KeyManagementServiceClient.decrypt) to implement it."
        )


class AzureKeyVaultKekProvider:
    """Documented seam for Azure Key Vault keys (not yet implemented).

    To implement: use ``azure.keyvault.keys.crypto`` (the ``kms-azure`` extra). Build a
    ``CryptographyClient(key, credential=DefaultAzureCredential())`` and wrap with
    ``.wrap_key(KeyWrapAlgorithm.rsa_oaep, dek)`` / unwrap with
    ``.unwrap_key(KeyWrapAlgorithm.rsa_oaep, wrapped)``. Set ``key_version`` to a
    non-secret fingerprint of the key identifier (mirroring :class:`AwsKmsKekProvider`).
    """

    def __init__(self, *, key_id: str | None = None, client: Any | None = None) -> None:
        self._key_id = key_id
        self._client = client

    @property
    def key_version(self) -> str:
        raise NotImplementedError(
            "AzureKeyVaultKekProvider is a documented seam; wire up "
            "azure.keyvault.keys.crypto (CryptographyClient.wrap_key/unwrap_key)."
        )

    def wrap_dek(self, plaintext_dek: bytes) -> tuple[bytes, str]:
        raise NotImplementedError(
            "AzureKeyVaultKekProvider is a documented seam; wire up "
            "azure.keyvault.keys.crypto (CryptographyClient.wrap_key)."
        )

    def unwrap_dek(self, wrapped_dek: bytes, key_version: str) -> bytes:
        raise NotImplementedError(
            "AzureKeyVaultKekProvider is a documented seam; wire up "
            "azure.keyvault.keys.crypto (CryptographyClient.unwrap_key)."
        )


__all__ = [
    "AwsKmsKekProvider",
    "GcpKmsKekProvider",
    "AzureKeyVaultKekProvider",
]
