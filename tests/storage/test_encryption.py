"""WS4.4 — field-level encryption at rest (envelope AES-GCM)."""

from __future__ import annotations

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
    RecordCipher,
    build_field_encryptor,
)


def test_roundtrip() -> None:
    enc = FieldEncryptor.generate()
    token = enc.encrypt("super secret value")
    assert token.startswith("himmy:enc:v1:")
    assert "super secret value" not in token
    assert enc.decrypt(token) == "super secret value"


def test_token_carries_local_version_segment() -> None:
    enc = FieldEncryptor.generate()
    token = enc.encrypt("v")
    # New token format: himmy:enc:v1:<VERSION>:<blob>
    assert token.startswith(f"{ENC_PREFIX}{LOCAL_VERSION}:")
    version, _blob = FieldEncryptor._parse_token(token)
    assert version == LOCAL_VERSION


# A GENUINE pre-change ciphertext, produced by the historical (version-less) encrypt()
# from git HEAD before the KekProvider/versioned-token work — NOT by today's encrypt().
# It was generated with the original framing
#   blob = [edek_len=48] || wrap_nonce(12) || wrapped_dek(48) || data_nonce(12) || ct
# under the fixed, known KEK below, and verified to decrypt with the ORIGINAL decrypt()
# logic. This guards against the framing regression that broke real legacy data: today's
# decrypt() MUST recover the original plaintext from these exact bytes.
_LEGACY_KEK_B64 = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="  # bytes(range(32))
_LEGACY_TOKEN = (
    "himmy:enc:v1:MBAREhMUFRYXGBkaGzy821IMj337gz9DUUI3JgOWEg1KXoQQ-a6zqSsTGhuLo7nm"
    "Ha2hPHvkDJ8e3TRhJ6ChoqOkpaanqKmqq066inNVhmfYUXxG75WV6703i4IHORJrhXrWXau03g=="
)
_LEGACY_PLAINTEXT = "legacy payload"


def test_legacy_versionless_token_still_decrypts() -> None:
    """A GENUINE pre-change version-less token still decrypts (no data loss).

    This embeds a ciphertext written by the *previous* himmy version's encrypt() (original
    blob framing, no VERSION segment). It must round-trip through today's
    ``FieldEncryptor(kek).decrypt()`` with the same KEK — proving on-disk ciphertext from
    before the KekProvider change is still readable. The token is fixed-byte, so it
    exercises the real historical wire format, not a re-derivation of the new one.
    """
    # Sanity: this really is the legacy 4-segment form (no version segment).
    assert _LEGACY_TOKEN.startswith(ENC_PREFIX)
    assert ":" not in _LEGACY_TOKEN[len(ENC_PREFIX) :]
    version, _blob = FieldEncryptor._parse_token(_LEGACY_TOKEN)
    assert version == LOCAL_VERSION

    enc = FieldEncryptor.from_key_b64(_LEGACY_KEK_B64)
    assert enc.decrypt(_LEGACY_TOKEN) == _LEGACY_PLAINTEXT

    # The legacy ciphertext can also be rotated forward to a versioned token (re-wrap the
    # DEK under a fresh KEK) without touching the protected plaintext.
    new_provider = LocalKekProvider.generate()
    rotated = enc.rotate_kek(_LEGACY_TOKEN, new_provider)
    assert ":" in rotated[len(ENC_PREFIX) :]  # now carries a VERSION segment
    assert (
        FieldEncryptor.from_provider(new_provider).decrypt(rotated) == _LEGACY_PLAINTEXT
    )


def test_legacy_token_wrong_kek_raises_invalid_tag() -> None:
    """A genuine legacy token under the wrong KEK fails loudly (no silent corruption)."""
    with pytest.raises(InvalidTag):
        FieldEncryptor.generate().decrypt(_LEGACY_TOKEN)


def test_from_provider_factory() -> None:
    provider = LocalKekProvider.generate()
    enc = FieldEncryptor.from_provider(provider)
    assert enc.provider is provider
    assert enc.decrypt(enc.encrypt("x")) == "x"


def test_raw_kek_constructor_still_works() -> None:
    """The historical FieldEncryptor(kek_bytes) signature is preserved."""
    import os

    enc = FieldEncryptor(os.urandom(32))
    assert enc.decrypt(enc.encrypt("v")) == "v"


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
