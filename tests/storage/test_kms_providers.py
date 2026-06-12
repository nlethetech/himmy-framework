"""cloud-kms — pluggable KEK providers (local + AWS KMS) and key rotation.

All offline: AWS is exercised through a :class:`FakeAwsKmsClient` that mimics the
``Encrypt``/``Decrypt`` API; the real boto3 client is never constructed.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("cryptography")

from cryptography.exceptions import InvalidTag  # noqa: E402

from himmy.config.secrets import configure_secrets  # noqa: E402
from himmy.services.storage.encryption import (  # noqa: E402
    ENC_PREFIX,
    LOCAL_VERSION,
    FieldEncryptor,
    LocalKekProvider,
    build_field_encryptor,
    build_kek_provider,
)
from himmy.services.storage.kms_providers import (  # noqa: E402
    AwsKmsKekProvider,
    AzureKeyVaultKekProvider,
    GcpKmsKekProvider,
)


class FakeAwsKmsClient:
    """Offline stand-in for a boto3 KMS client (reversible XOR + a per-key tag).

    botocore exposes operation methods in snake_case (``encrypt`` / ``decrypt``), so the
    fake mirrors that surface — a PascalCase ``Encrypt`` would not match the real client
    and would let the provider's method names silently drift. ``encrypt`` prefixes the
    blob with a key-id tag so ``decrypt`` can recover the DEK without being told the key
    (mirroring real KMS, which records the producing CMK).
    """

    def __init__(self) -> None:
        self.encrypt_calls: list[dict[str, Any]] = []
        self.decrypt_calls: list[dict[str, Any]] = []

    @staticmethod
    def _mask(blob: bytes) -> bytes:
        return bytes(b ^ 0x5A for b in blob)

    def encrypt(self, *, KeyId: str, Plaintext: bytes) -> dict[str, bytes]:  # noqa: N803
        self.encrypt_calls.append({"KeyId": KeyId, "Plaintext": Plaintext})
        tag = KeyId.encode("utf-8")
        framed = bytes([len(tag)]) + tag + self._mask(Plaintext)
        return {"CiphertextBlob": framed}

    def decrypt(self, *, CiphertextBlob: bytes) -> dict[str, bytes]:  # noqa: N803
        self.decrypt_calls.append({"CiphertextBlob": CiphertextBlob})
        tag_len = CiphertextBlob[0]
        key_id = CiphertextBlob[1 : 1 + tag_len].decode("utf-8")
        plaintext = self._mask(CiphertextBlob[1 + tag_len :])
        return {"Plaintext": plaintext, "KeyId": key_id}


# --- local provider ---------------------------------------------------------------


def test_local_provider_roundtrip_and_version() -> None:
    provider = LocalKekProvider.generate()
    assert provider.key_version == LOCAL_VERSION
    dek = os.urandom(32)
    wrapped, version = provider.wrap_dek(dek)
    assert version == LOCAL_VERSION
    assert wrapped != dek
    assert provider.unwrap_dek(wrapped, version) == dek


def test_field_encryptor_token_has_version_segment() -> None:
    enc = FieldEncryptor.generate()
    token = enc.encrypt("secret value")
    assert token.startswith(f"{ENC_PREFIX}{LOCAL_VERSION}:")
    assert "secret value" not in token
    assert enc.decrypt(token) == "secret value"


# --- AWS KMS provider (fake client) -----------------------------------------------


def test_aws_kms_wrap_unwrap_via_fake_client() -> None:
    client = FakeAwsKmsClient()
    provider = AwsKmsKekProvider(key_id="arn:aws:kms:...:key/abc", client=client)
    dek = os.urandom(32)

    wrapped, version = provider.wrap_dek(dek)
    assert version.startswith("aws-kms.")
    assert dek not in wrapped  # the raw DEK is not stored in the wrapped blob
    assert client.encrypt_calls[0]["KeyId"] == "arn:aws:kms:...:key/abc"

    assert provider.unwrap_dek(wrapped, version) == dek
    assert client.decrypt_calls  # Decrypt was actually exercised


def test_aws_kms_calls_botocore_snake_case_methods() -> None:
    """Regression: the provider must call ``client.encrypt`` / ``client.decrypt``.

    botocore KMS clients expose snake_case operation methods; the historical PascalCase
    ``Encrypt`` / ``Decrypt`` raise ``AttributeError`` against a real boto3 client. A
    recording client that ONLY implements the snake_case surface (and explodes on the
    PascalCase names) pins the wire so signature drift is caught offline.
    """

    class SnakeCaseOnlyClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def encrypt(self, *, KeyId: str, Plaintext: bytes) -> dict[str, bytes]:  # noqa: N803
            self.calls.append("encrypt")
            return {"CiphertextBlob": b"wrapped:" + Plaintext}

        def decrypt(self, *, CiphertextBlob: bytes) -> dict[str, bytes]:  # noqa: N803
            self.calls.append("decrypt")
            return {"Plaintext": CiphertextBlob[len(b"wrapped:") :]}

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(
                f"provider called non-botocore method {name!r}; real KMS clients only "
                "expose snake_case operations (encrypt/decrypt)"
            )

    client = SnakeCaseOnlyClient()
    provider = AwsKmsKekProvider(key_id="key-1", client=client)
    dek = os.urandom(32)
    wrapped, _version = provider.wrap_dek(dek)
    assert provider.unwrap_dek(wrapped, _version) == dek
    assert client.calls == ["encrypt", "decrypt"]


def test_aws_kms_version_is_stable_non_secret_fingerprint() -> None:
    client = FakeAwsKmsClient()
    p1 = AwsKmsKekProvider(key_id="arn:key/abc", client=client)
    p2 = AwsKmsKekProvider(key_id="arn:key/abc", client=FakeAwsKmsClient())
    p3 = AwsKmsKekProvider(key_id="arn:key/xyz", client=client)
    assert p1.key_version == p2.key_version  # stable for a given key id
    assert p1.key_version != p3.key_version  # differs across keys
    assert "arn:key/abc" not in p1.key_version  # the ARN is not leaked


def test_aws_kms_full_field_encryption_roundtrip() -> None:
    client = FakeAwsKmsClient()
    provider = AwsKmsKekProvider(key_id="key-1", client=client)
    enc = FieldEncryptor.from_provider(provider)
    token = enc.encrypt("aws-protected", aad=b"rec-1")
    assert token.startswith(f"{ENC_PREFIX}aws-kms.")
    assert enc.decrypt(token, aad=b"rec-1") == "aws-protected"


def test_aws_kms_requires_key_id() -> None:
    with pytest.raises(ValueError, match="key_id"):
        AwsKmsKekProvider(key_id="", client=FakeAwsKmsClient())


# --- GCP / Azure seams ------------------------------------------------------------


def test_gcp_seam_raises_not_implemented() -> None:
    provider = GcpKmsKekProvider()
    with pytest.raises(NotImplementedError, match="google.cloud.kms"):
        provider.wrap_dek(os.urandom(32))
    with pytest.raises(NotImplementedError):
        provider.unwrap_dek(b"x", "v")
    with pytest.raises(NotImplementedError):
        _ = provider.key_version


def test_azure_seam_raises_not_implemented() -> None:
    provider = AzureKeyVaultKekProvider()
    with pytest.raises(NotImplementedError, match="azure.keyvault.keys"):
        provider.wrap_dek(os.urandom(32))
    with pytest.raises(NotImplementedError):
        provider.unwrap_dek(b"x", "v")
    with pytest.raises(NotImplementedError):
        _ = provider.key_version


# --- build_kek_provider / build_field_encryptor -----------------------------------


def _with_secrets(mapping: dict[str, str]) -> None:
    class _Secrets:
        def get(self, name: str) -> str | None:
            return mapping.get(name)

    configure_secrets(_Secrets())


def test_build_default_is_local(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_KEK_PROVIDER", raising=False)
    key = LocalKekProvider.generate().key_b64()
    _with_secrets({"HIMMY_ENCRYPTION_KEY": key})
    try:
        provider = build_kek_provider()
        assert isinstance(provider, LocalKekProvider)
        enc = build_field_encryptor()
        assert enc is not None
        assert enc.decrypt(enc.encrypt("v")) == "v"
    finally:
        configure_secrets(None)


def test_build_default_off_without_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_KEK_PROVIDER", raising=False)
    monkeypatch.delenv("HIMMY_ENCRYPTION_KEY", raising=False)
    _with_secrets({})
    try:
        assert build_kek_provider() is None
        assert build_field_encryptor() is None
    finally:
        configure_secrets(None)


def test_build_aws_kms_with_fake_client(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_KEK_PROVIDER", "aws-kms")
    _with_secrets({"HIMMY_AWS_KMS_KEY_ID": "arn:aws:kms:key/abc"})
    try:
        provider = build_kek_provider()
        assert isinstance(provider, AwsKmsKekProvider)
        # Inject a fake client so no AWS call happens, then prove a full roundtrip.
        provider._client = FakeAwsKmsClient()
        enc = FieldEncryptor.from_provider(provider)
        assert enc.decrypt(enc.encrypt("kms value")) == "kms value"
    finally:
        configure_secrets(None)


def test_build_aws_kms_requires_key_id(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_KEK_PROVIDER", "aws-kms")
    _with_secrets({})
    try:
        with pytest.raises(ValueError, match="HIMMY_AWS_KMS_KEY_ID"):
            build_kek_provider()
    finally:
        configure_secrets(None)


def test_build_unknown_provider_raises(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_KEK_PROVIDER", "nope-kms")
    _with_secrets({})
    try:
        with pytest.raises(ValueError, match="unknown HIMMY_KEK_PROVIDER"):
            build_kek_provider()
    finally:
        configure_secrets(None)


# --- rotation: re-wrap DEK without changing ciphertext ----------------------------


def test_rotate_kek_rewraps_without_changing_ciphertext() -> None:
    old = FieldEncryptor.generate()
    token = old.encrypt("rotate me", aad=b"ctx")

    client = FakeAwsKmsClient()
    new_provider = AwsKmsKekProvider(key_id="key-1", client=client)
    rotated = old.rotate_kek(token, new_provider)

    # Version changed local -> aws-kms; the rotated token is decryptable with the new KEK.
    assert rotated.startswith(f"{ENC_PREFIX}aws-kms.")
    assert FieldEncryptor.from_provider(new_provider).decrypt(rotated, aad=b"ctx") == (
        "rotate me"
    )
    # The data nonce + ciphertext tail are byte-for-byte unchanged: only the wrapped DEK
    # framing (length-prefixed at the head of the blob) moved.
    old_ct = _ciphertext_tail(token)
    new_ct = _ciphertext_tail(rotated)
    assert old_ct == new_ct
    # The old KEK can no longer read the rotated token (its DEK is wrapped by KMS now).
    with pytest.raises(InvalidTag):
        old.decrypt(rotated, aad=b"ctx")


def _ciphertext_tail(token: str) -> bytes:
    """Extract (data_nonce || ciphertext) from a token, dropping the wrapped-DEK frame."""
    version, blob = FieldEncryptor._parse_token(token)
    _wrapped, data_nonce, ciphertext = FieldEncryptor._split_blob(blob)
    return data_nonce + ciphertext


def test_rotate_kek_local_to_local_preserves_plaintext() -> None:
    old = FieldEncryptor.generate()
    token = old.encrypt("hello")
    new = LocalKekProvider.generate()
    rotated = old.rotate_kek(token, new)
    assert FieldEncryptor.from_provider(new).decrypt(rotated) == "hello"
