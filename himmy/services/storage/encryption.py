"""Field-level encryption at rest: envelope AES-GCM for sensitive payloads (WS4.4).

A :class:`FieldEncryptor` encrypts a string with a fresh random **data key** (DEK),
then wraps that DEK with a long-lived **key-encryption key** (KEK) — the envelope
pattern, so rotating the KEK never requires re-encrypting the data. AES-GCM is
authenticated, and an optional ``aad`` (additional authenticated data — e.g. the record
id or tenant) binds a ciphertext to its context, so it can't be moved between records.

The KEK comes from the secret provider (``HIMMY_ENCRYPTION_KEY``, base64; ideally a cloud
KMS-managed key). Encryption is **opt-in** — no key configured ⇒ ``build_field_encryptor``
returns ``None`` and storage stays plaintext (unchanged). The produced token is a
prefixed, URL-safe string, so it round-trips through JSON/JSONB and the tamper-evident
audit hash simply covers the ciphertext.

``cryptography`` is required (the ``encryption`` extra; also pulled in by ``auth``).
"""

from __future__ import annotations

import base64
import os
from collections.abc import Iterable
from typing import Any

#: Marker so we can recognize (and not double-encrypt) our own ciphertext.
ENC_PREFIX = "himmy:enc:v1:"


class FieldEncryptor:
    """Envelope AES-GCM encryptor: per-value DEK wrapped by a configured KEK."""

    def __init__(self, kek: bytes) -> None:
        """Use ``kek`` (16/24/32 bytes) as the key-encryption key."""
        if len(kek) not in (16, 24, 32):
            raise ValueError("KEK must be 16, 24, or 32 bytes (AES-128/192/256)")
        self._kek = kek

    @classmethod
    def from_key_b64(cls, key_b64: str) -> FieldEncryptor:
        """Build from a base64-encoded key."""
        return cls(base64.b64decode(key_b64))

    @classmethod
    def generate(cls) -> FieldEncryptor:
        """Generate a fresh AES-256 KEK (for bootstrapping / tests)."""
        return cls(os.urandom(32))

    def key_b64(self) -> str:
        """The KEK as base64 (to persist in a secret store)."""
        return base64.b64encode(self._kek).decode("ascii")

    @staticmethod
    def is_encrypted(value: Any) -> bool:
        """Whether ``value`` is a himmy ciphertext token."""
        return isinstance(value, str) and value.startswith(ENC_PREFIX)

    def encrypt(self, plaintext: str, *, aad: bytes = b"") -> str:
        """Encrypt ``plaintext`` (envelope); return a prefixed URL-safe token."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        dek = AESGCM.generate_key(bit_length=256)
        data_nonce = os.urandom(12)
        ciphertext = AESGCM(dek).encrypt(data_nonce, plaintext.encode("utf-8"), aad)
        wrap_nonce = os.urandom(12)
        wrapped_dek = AESGCM(self._kek).encrypt(wrap_nonce, dek, b"")
        blob = b"".join(
            [
                bytes([len(wrapped_dek)]),
                wrap_nonce,
                wrapped_dek,
                data_nonce,
                ciphertext,
            ]
        )
        return ENC_PREFIX + base64.urlsafe_b64encode(blob).decode("ascii")

    def decrypt(self, token: str, *, aad: bytes = b"") -> str:
        """Decrypt a token produced by :meth:`encrypt` (raises on tamper/wrong key)."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if not token.startswith(ENC_PREFIX):
            raise ValueError("not a himmy ciphertext token")
        blob = base64.urlsafe_b64decode(token[len(ENC_PREFIX) :])
        pos = 0
        edek_len = blob[pos]
        pos += 1
        wrap_nonce = blob[pos : pos + 12]
        pos += 12
        wrapped_dek = blob[pos : pos + edek_len]
        pos += edek_len
        data_nonce = blob[pos : pos + 12]
        pos += 12
        ciphertext = blob[pos:]
        dek = AESGCM(self._kek).decrypt(wrap_nonce, wrapped_dek, b"")
        return AESGCM(dek).decrypt(data_nonce, ciphertext, aad).decode("utf-8")


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


def build_field_encryptor() -> FieldEncryptor | None:
    """Build a :class:`FieldEncryptor` from ``HIMMY_ENCRYPTION_KEY``, or ``None`` (off)."""
    from himmy.config.secrets import get_secret

    key_b64 = get_secret("HIMMY_ENCRYPTION_KEY")
    return FieldEncryptor.from_key_b64(key_b64) if key_b64 else None


__all__ = ["FieldEncryptor", "RecordCipher", "build_field_encryptor", "ENC_PREFIX"]
