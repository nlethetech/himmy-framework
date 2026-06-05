"""WS4.4 — field-level encryption at rest (envelope AES-GCM)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("cryptography")

from cryptography.exceptions import InvalidTag  # noqa: E402

from himmy.config.secrets import configure_secrets  # noqa: E402
from himmy.services.storage.encryption import (  # noqa: E402
    FieldEncryptor,
    RecordCipher,
    build_field_encryptor,
)


def test_roundtrip() -> None:
    enc = FieldEncryptor.generate()
    token = enc.encrypt("super secret value")
    assert token.startswith("himmy:enc:v1:")
    assert "super secret value" not in token
    assert enc.decrypt(token) == "super secret value"


def test_each_encryption_is_unique() -> None:
    enc = FieldEncryptor.generate()
    assert enc.encrypt("x") != enc.encrypt("x")  # random DEK + nonce


def test_aad_binds_context() -> None:
    enc = FieldEncryptor.generate()
    token = enc.encrypt("v", aad=b"record-1")
    assert enc.decrypt(token, aad=b"record-1") == "v"
    with pytest.raises(InvalidTag):
        enc.decrypt(token, aad=b"record-2")  # wrong context → auth failure


def test_tamper_is_detected() -> None:
    enc = FieldEncryptor.generate()
    token = enc.encrypt("v")
    tampered = token[:-2] + ("AA" if not token.endswith("AA") else "BB")
    with pytest.raises((InvalidTag, ValueError)):
        enc.decrypt(tampered)


def test_wrong_key_cannot_decrypt() -> None:
    token = FieldEncryptor.generate().encrypt("v")
    with pytest.raises(InvalidTag):
        FieldEncryptor.generate().decrypt(token)


def test_is_encrypted() -> None:
    enc = FieldEncryptor.generate()
    assert FieldEncryptor.is_encrypted(enc.encrypt("v"))
    assert not FieldEncryptor.is_encrypted("plain")
    assert not FieldEncryptor.is_encrypted(None)


def test_record_cipher_fields_roundtrip() -> None:
    enc = FieldEncryptor.generate()
    cipher = RecordCipher(enc)
    record = {"id": "r1", "output_text": "secret", "count": 3}
    sealed = cipher.encrypt_fields(record, ["output_text"], aad=b"r1")
    assert FieldEncryptor.is_encrypted(sealed["output_text"])
    assert sealed["count"] == 3  # non-targeted fields untouched
    # Idempotent: encrypting again doesn't double-encrypt.
    assert cipher.encrypt_fields(sealed, ["output_text"]) == sealed
    opened = cipher.decrypt_fields(sealed, ["output_text"], aad=b"r1")
    assert opened["output_text"] == "secret"


def test_build_field_encryptor_from_key() -> None:
    key = FieldEncryptor.generate().key_b64()

    class _Secrets:
        def get(self, name: str) -> str | None:
            return key if name == "HIMMY_ENCRYPTION_KEY" else None

    configure_secrets(_Secrets())
    try:
        enc = build_field_encryptor()
        assert enc is not None
        assert enc.decrypt(enc.encrypt("v")) == "v"
    finally:
        configure_secrets(None)


def test_build_field_encryptor_off_by_default(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_ENCRYPTION_KEY", raising=False)
    configure_secrets(None)
    assert build_field_encryptor() is None
