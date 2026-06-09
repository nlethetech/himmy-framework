"""Field-level encryption at rest: envelope AES-GCM for sensitive payloads (WS4.4).

A :class:`FieldEncryptor` encrypts a string with a fresh random **data key** (DEK),
then wraps that DEK with a long-lived **key-encryption key** (KEK) — the envelope
pattern, so rotating the KEK never requires re-encrypting the data. AES-GCM is
authenticated, and an optional ``aad`` (additional authenticated data — e.g. the record
id or tenant) binds a ciphertext to its context, so it can't be moved between records.

The KEK is sourced through a pluggable :class:`KekProvider`. The default is
:class:`LocalKekProvider` (a raw key from ``HIMMY_ENCRYPTION_KEY``, base64), which keeps
the offline/zero-key path unchanged. A cloud KMS (AWS KMS today; GCP/Azure as documented
seams) can wrap/unwrap the DEK remotely so the master key never leaves the HSM. Encryption
is **opt-in** — no key/provider configured ⇒ ``build_field_encryptor`` returns ``None`` and
storage stays plaintext (unchanged).

The produced token is a prefixed, URL-safe string — ``himmy:enc:v1:<VERSION>:<BLOB>`` —
where ``<VERSION>`` identifies the KEK that wrapped the DEK (so a rotated/cloud key can be
located at read time). For backward compatibility, a legacy ``himmy:enc:v1:<BLOB>`` token
(no version segment) is still decryptable and is treated as version ``LOCAL``. Tokens
round-trip through JSON/JSONB and the tamper-evident audit hash simply covers the
ciphertext.

``cryptography`` is required (the ``encryption`` extra; also pulled in by ``auth``). Cloud
providers need their own optional extra (``kms-aws`` → boto3), lazily imported.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

#: Marker so we can recognize (and not double-encrypt) our own ciphertext.
ENC_PREFIX = "himmy:enc:v1:"

#: Version tag for the built-in local (raw-key) KEK provider.
LOCAL_VERSION = "LOCAL"


@runtime_checkable
class KekProvider(Protocol):
    """Wraps/unwraps a data key (DEK) under a key-encryption key (KEK).

    The DEK is the per-value AES-256 key that actually encrypts the plaintext; the KEK
    is the long-lived master key that wraps it. Implementations may keep the KEK locally
    (:class:`LocalKekProvider`) or delegate to a cloud KMS so the master key never leaves
    the HSM. ``key_version`` is an opaque tag stored alongside the ciphertext so the right
    KEK can be located at read time (enabling rotation across versions).
    """

    @property
    def key_version(self) -> str:
        """Opaque tag identifying the KEK this provider wraps with (stored in tokens)."""
        ...

    def wrap_dek(self, plaintext_dek: bytes) -> tuple[bytes, str]:
        """Wrap ``plaintext_dek``; return ``(wrapped_dek, key_version)``."""
        ...

    def unwrap_dek(self, wrapped_dek: bytes, key_version: str) -> bytes:
        """Unwrap a DEK previously wrapped under ``key_version`` (raises on failure)."""
        ...


class LocalKekProvider:
    """Wrap/unwrap a DEK with a locally-held AES-GCM KEK (the default, offline path).

    This is the historical behavior: the raw KEK lives in the process (from
    ``HIMMY_ENCRYPTION_KEY``) and wraps the DEK with AES-GCM. Its ``key_version`` is the
    constant ``"LOCAL"``, which is also how legacy version-less tokens are interpreted.
    """

    def __init__(self, kek: bytes) -> None:
        """Use ``kek`` (16/24/32 bytes) as the local key-encryption key."""
        if len(kek) not in (16, 24, 32):
            raise ValueError("KEK must be 16, 24, or 32 bytes (AES-128/192/256)")
        self._kek = kek

    @classmethod
    def from_key_b64(cls, key_b64: str) -> LocalKekProvider:
        """Build from a base64-encoded key."""
        return cls(base64.b64decode(key_b64))

    @classmethod
    def generate(cls) -> LocalKekProvider:
        """Generate a fresh AES-256 KEK (for bootstrapping / tests)."""
        return cls(os.urandom(32))

    def key_b64(self) -> str:
        """The KEK as base64 (to persist in a secret store)."""
        return base64.b64encode(self._kek).decode("ascii")

    @property
    def key_version(self) -> str:
        return LOCAL_VERSION

    def wrap_dek(self, plaintext_dek: bytes) -> tuple[bytes, str]:
        """Wrap the DEK as ``nonce(12) || AES-GCM(wrapped_dek)``."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        wrap_nonce = os.urandom(12)
        wrapped = AESGCM(self._kek).encrypt(wrap_nonce, plaintext_dek, b"")
        return wrap_nonce + wrapped, LOCAL_VERSION

    def unwrap_dek(self, wrapped_dek: bytes, key_version: str) -> bytes:
        """Unwrap ``nonce(12) || AES-GCM(wrapped_dek)`` (``key_version`` is informational)."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        wrap_nonce, blob = wrapped_dek[:12], wrapped_dek[12:]
        return AESGCM(self._kek).decrypt(wrap_nonce, blob, b"")


class FieldEncryptor:
    """Envelope AES-GCM encryptor: per-value DEK wrapped by a :class:`KekProvider`."""

    def __init__(
        self, kek: bytes | None = None, *, provider: KekProvider | None = None
    ) -> None:
        """Use a :class:`KekProvider`, or a raw ``kek`` (wrapped in a local provider).

        Passing ``kek`` directly preserves the original constructor signature: it builds a
        :class:`LocalKekProvider`. Passing ``provider`` selects any KEK backend (e.g. a
        cloud KMS). Exactly one of the two must be supplied.
        """
        if provider is not None:
            if kek is not None:
                raise ValueError("pass either kek or provider, not both")
            self._provider = provider
        elif kek is not None:
            self._provider = LocalKekProvider(kek)
        else:
            raise ValueError("FieldEncryptor needs a kek or a provider")

    @classmethod
    def from_provider(cls, provider: KekProvider) -> FieldEncryptor:
        """Build from any :class:`KekProvider` (e.g. a cloud KMS provider)."""
        return cls(provider=provider)

    @classmethod
    def from_key_b64(cls, key_b64: str) -> FieldEncryptor:
        """Build from a base64-encoded key (local KEK provider)."""
        return cls(provider=LocalKekProvider.from_key_b64(key_b64))

    @classmethod
    def generate(cls) -> FieldEncryptor:
        """Generate a fresh AES-256 local KEK (for bootstrapping / tests)."""
        return cls(provider=LocalKekProvider.generate())

    @property
    def provider(self) -> KekProvider:
        """The KEK provider backing this encryptor."""
        return self._provider

    def key_b64(self) -> str:
        """The KEK as base64 (local provider only; raises for cloud KEKs)."""
        provider = self._provider
        if isinstance(provider, LocalKekProvider):
            return provider.key_b64()
        raise ValueError("key_b64 is only available for a local KEK provider")

    @staticmethod
    def is_encrypted(value: Any) -> bool:
        """Whether ``value`` is a himmy ciphertext token."""
        return isinstance(value, str) and value.startswith(ENC_PREFIX)

    @staticmethod
    def _is_legacy_token(token: str) -> bool:
        """Whether ``token`` uses the historical version-less wire format.

        New tokens are ``himmy:enc:v1:<VERSION>:<b64-blob>`` (a ``:`` after the prefix);
        legacy tokens are ``himmy:enc:v1:<b64-blob>`` with no version segment. URL-safe
        base64 never contains ``:``, so the presence of a ``:`` after the prefix is an
        unambiguous discriminator. Legacy tokens use the historical blob framing (see
        :meth:`_split_blob`) and resolve to version ``LOCAL``.
        """
        if not token.startswith(ENC_PREFIX):
            raise ValueError("not a himmy ciphertext token")
        return ":" not in token[len(ENC_PREFIX) :]

    @staticmethod
    def _parse_token(token: str) -> tuple[str, bytes]:
        """Split a token into ``(key_version, blob)``, handling the legacy form.

        New tokens are ``himmy:enc:v1:<VERSION>:<b64-blob>``. Legacy tokens are
        ``himmy:enc:v1:<b64-blob>`` (no version) and resolve to version ``LOCAL``. The
        version segment never contains URL-safe-base64-illegal characters and never a
        ``:``, so a single split off the prefix is unambiguous.
        """
        if not token.startswith(ENC_PREFIX):
            raise ValueError("not a himmy ciphertext token")
        rest = token[len(ENC_PREFIX) :]
        version, sep, b64 = rest.partition(":")
        if sep:
            return version, base64.urlsafe_b64decode(b64)
        # Legacy version-less token: the whole remainder is the blob, version is LOCAL.
        return LOCAL_VERSION, base64.urlsafe_b64decode(rest)

    @staticmethod
    def _split_blob(blob: bytes, *, legacy: bool = False) -> tuple[bytes, bytes, bytes]:
        """Split a blob into ``(wrapped_dek, data_nonce, ciphertext)``.

        New framing is ``[edek_len] || wrapped_dek(edek_len) || data_nonce(12) || ct``,
        where ``wrapped_dek`` already carries the wrap nonce folded in (the local provider
        prepends it). The historical (legacy) framing instead kept the wrap nonce as a
        separate field: ``[edek_len] || wrap_nonce(12) || wrapped_dek(edek_len) ||
        data_nonce(12) || ct``. For legacy blobs we re-fold ``wrap_nonce || wrapped_dek``
        into the contiguous ``wrapped_dek`` the local provider's ``unwrap_dek`` expects
        (verified equivalent: the two AES-GCM wrap layouts are byte-identical).
        """
        pos = 0
        edek_len = blob[pos]
        pos += 1
        if legacy:
            wrap_nonce = blob[pos : pos + 12]
            pos += 12
            wrapped_dek = wrap_nonce + blob[pos : pos + edek_len]
            pos += edek_len
        else:
            wrapped_dek = blob[pos : pos + edek_len]
            pos += edek_len
        data_nonce = blob[pos : pos + 12]
        pos += 12
        ciphertext = blob[pos:]
        return wrapped_dek, data_nonce, ciphertext

    @staticmethod
    def _assemble_token(
        key_version: str, wrapped_dek: bytes, data_nonce: bytes, ciphertext: bytes
    ) -> str:
        """Assemble a ``himmy:enc:v1:<VERSION>:<blob>`` token."""
        if ":" in key_version:
            raise ValueError("key_version must not contain ':' (token delimiter)")
        if len(wrapped_dek) > 255:
            raise ValueError("wrapped DEK too large to frame")
        blob = b"".join(
            [bytes([len(wrapped_dek)]), wrapped_dek, data_nonce, ciphertext]
        )
        b64 = base64.urlsafe_b64encode(blob).decode("ascii")
        return f"{ENC_PREFIX}{key_version}:{b64}"

    def encrypt(self, plaintext: str, *, aad: bytes = b"") -> str:
        """Encrypt ``plaintext`` (envelope); return a versioned URL-safe token."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        dek = AESGCM.generate_key(bit_length=256)
        data_nonce = os.urandom(12)
        ciphertext = AESGCM(dek).encrypt(data_nonce, plaintext.encode("utf-8"), aad)
        wrapped_dek, key_version = self._provider.wrap_dek(dek)
        return self._assemble_token(key_version, wrapped_dek, data_nonce, ciphertext)

    def decrypt(self, token: str, *, aad: bytes = b"") -> str:
        """Decrypt a token produced by :meth:`encrypt` (raises on tamper/wrong key).

        Both the versioned and the legacy version-less token forms are accepted.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        legacy = self._is_legacy_token(token)
        key_version, blob = self._parse_token(token)
        wrapped_dek, data_nonce, ciphertext = self._split_blob(blob, legacy=legacy)
        dek = self._provider.unwrap_dek(wrapped_dek, key_version)
        return AESGCM(dek).decrypt(data_nonce, ciphertext, aad).decode("utf-8")

    def rotate_kek(self, token: str, new_provider: KekProvider) -> str:
        """Re-wrap a token's DEK under ``new_provider`` without touching the plaintext.

        Unwraps the DEK with this encryptor's provider, re-wraps it under
        ``new_provider``, and re-assembles the token with the new ``key_version``. The
        ciphertext and its data nonce are left byte-for-byte unchanged — only the
        ``wrapped_dek`` and the version segment differ — so rotating the master key never
        decrypts/re-encrypts the protected value.

        Returns the rotated token. Use :meth:`from_provider` on ``new_provider`` to read it.
        """
        legacy = self._is_legacy_token(token)
        key_version, blob = self._parse_token(token)
        wrapped_dek, data_nonce, ciphertext = self._split_blob(blob, legacy=legacy)
        dek = self._provider.unwrap_dek(wrapped_dek, key_version)
        new_wrapped, new_version = new_provider.wrap_dek(dek)
        return self._assemble_token(new_version, new_wrapped, data_nonce, ciphertext)


class RecordCipher:
    """Encrypt/decrypt named fields of a record dict (transparent at-rest helper)."""

    def __init__(self, encryptor: FieldEncryptor) -> None:
        self._enc = encryptor

    def encrypt_fields(
        self, data: dict[str, Any], fields: Iterable[str], *, aad: bytes = b""
    ) -> dict[str, Any]:
        """Return a copy with the named string fields encrypted (idempotent)."""
        out = dict(data)
        for field in fields:
            value = out.get(field)
            if isinstance(value, str) and not self._enc.is_encrypted(value):
                out[field] = self._enc.encrypt(value, aad=aad)
        return out

    def decrypt_fields(
        self, data: dict[str, Any], fields: Iterable[str], *, aad: bytes = b""
    ) -> dict[str, Any]:
        """Return a copy with the named ciphertext fields decrypted."""
        out = dict(data)
        for field in fields:
            value = out.get(field)
            if isinstance(value, str) and self._enc.is_encrypted(value):
                out[field] = self._enc.decrypt(value, aad=aad)
        return out


def build_kek_provider() -> KekProvider | None:
    """Build a :class:`KekProvider` from ``HIMMY_KEK_PROVIDER`` (+ secrets), or ``None``.

    ``HIMMY_KEK_PROVIDER`` selects the backend (default ``local``):

    * ``local`` *(default)* — a raw KEK from ``HIMMY_ENCRYPTION_KEY`` (base64). Absent ⇒
      ``None`` (encryption off, offline path unchanged).
    * ``aws-kms`` — :class:`~himmy.services.storage.kms_providers.AwsKmsKekProvider`
      keyed by ``HIMMY_AWS_KMS_KEY_ID`` (needs the ``kms-aws`` extra → boto3).
    * ``gcp-kms`` / ``azure-kms`` — documented seams (not yet implemented).

    A cloud provider is only instantiated once its extra is importable; otherwise a clear
    error names the missing extra.
    """
    from himmy.config.secrets import get_secret

    mode = (os.environ.get("HIMMY_KEK_PROVIDER") or "local").strip().lower()
    if mode in ("", "local"):
        key_b64 = get_secret("HIMMY_ENCRYPTION_KEY")
        return LocalKekProvider.from_key_b64(key_b64) if key_b64 else None
    if mode == "aws-kms":
        _require_extra("boto3", "kms-aws", "aws-kms")
        from himmy.services.storage.kms_providers import AwsKmsKekProvider

        key_id = get_secret("HIMMY_AWS_KMS_KEY_ID")
        if not key_id:
            raise ValueError("aws-kms KEK provider needs HIMMY_AWS_KMS_KEY_ID")
        return AwsKmsKekProvider(key_id=key_id, region=get_secret("AWS_REGION"))
    if mode == "gcp-kms":
        _require_extra("google.cloud.kms", "kms-gcp", "gcp-kms")
        from himmy.services.storage.kms_providers import GcpKmsKekProvider

        return GcpKmsKekProvider()
    if mode == "azure-kms":
        _require_extra("azure.keyvault.keys", "kms-azure", "azure-kms")
        from himmy.services.storage.kms_providers import AzureKeyVaultKekProvider

        return AzureKeyVaultKekProvider()
    raise ValueError(f"unknown HIMMY_KEK_PROVIDER: {mode!r}")


def _require_extra(module: str, extra: str, provider_name: str) -> None:
    """Raise a clear error if ``module`` (the extra's SDK) is not importable."""
    import importlib.util

    if importlib.util.find_spec(module.split(".")[0]) is None:
        raise RuntimeError(
            f"the {provider_name!r} KEK provider requires the {extra!r} extra "
            f"(install it: pip install 'himmy[{extra}]')"
        )


def build_field_encryptor() -> FieldEncryptor | None:
    """Build a :class:`FieldEncryptor` from the configured KEK provider, or ``None`` (off).

    Selects the backend via ``HIMMY_KEK_PROVIDER`` (default ``local``, sourced from
    ``HIMMY_ENCRYPTION_KEY``). Returns ``None`` when no key/provider is configured, so the
    offline plaintext path is unchanged.
    """
    provider = build_kek_provider()
    return FieldEncryptor.from_provider(provider) if provider is not None else None


__all__ = [
    "FieldEncryptor",
    "RecordCipher",
    "KekProvider",
    "LocalKekProvider",
    "build_field_encryptor",
    "build_kek_provider",
    "ENC_PREFIX",
    "LOCAL_VERSION",
]
