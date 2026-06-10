"""WS4.4 — field-level encryption at rest (envelope AES-GCM)."""

from __future__ import annotations

import base64
import random
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


# ---------------------------------------------------------------------------
# Seeded fuzz: legacy-token layout equivalence (crypto-critical claim in
# FieldEncryptor._split_blob's docstring — "the two AES-GCM wrap layouts are
# byte-identical"). Deterministic (fixed seed), stdlib-only.
# ---------------------------------------------------------------------------

#: AES-GCM wrap of a 32-byte DEK: 32 bytes + 16-byte tag (excludes the wrap nonce).
_GCM_WRAPPED_DEK_LEN = 48
#: 96-bit AES-GCM nonce, used for both the wrap and the data layer.
_NONCE_LEN = 12


def _reframe_as_legacy(token: str) -> str:
    """Re-frame a current versioned token into the historical version-less layout.

    Current blob framing is ``[edek_len] || (wrap_nonce(12) || gcm_wrapped) ||
    data_nonce(12) || ct`` where the length byte covers the folded-in wrap nonce. The
    historical framing kept the wrap nonce as a separate field — ``[edek_len] ||
    wrap_nonce(12) || gcm_wrapped(edek_len) || data_nonce(12) || ct`` — with the length
    byte covering only the GCM-wrapped DEK. The payload bytes are identical; only the
    length byte differs (by the 12-byte nonce) and the version segment is dropped.
    """
    version, blob = FieldEncryptor._parse_token(token)
    assert version == LOCAL_VERSION
    edek_len = blob[0]
    assert edek_len == _NONCE_LEN + _GCM_WRAPPED_DEK_LEN  # local provider invariant
    legacy_blob = bytes([edek_len - _NONCE_LEN]) + blob[1:]
    # Same shape as the genuine pre-change artifact above (edek_len byte == 48).
    assert legacy_blob[0] == _GCM_WRAPPED_DEK_LEN
    b64 = base64.urlsafe_b64encode(legacy_blob).decode("ascii")
    return f"{ENC_PREFIX}{b64}"


def _fuzz_plaintext(rng: random.Random, i: int) -> str:
    """Deterministic varied plaintexts: empty / 1-byte / ascii / unicode / large."""
    kind = i % 5
    if kind == 0:
        return ""
    if kind == 1:
        return chr(rng.randrange(0x20, 0x7F))
    if kind == 2:  # short ascii
        return "".join(
            chr(rng.randrange(0x20, 0x7F)) for _ in range(rng.randrange(1, 65))
        )
    if kind == 3:  # unicode incl. multi-byte codepoints (no surrogates)
        chars = []
        for _ in range(rng.randrange(1, 33)):
            cp = rng.randrange(0x20, 0x110000)
            while 0xD800 <= cp <= 0xDFFF:
                cp = rng.randrange(0x20, 0x110000)
            chars.append(chr(cp))
        return "".join(chars)
    # large (up to ~8 KiB)
    return "".join(
        chr(rng.randrange(0x20, 0x7F)) for _ in range(rng.randrange(1024, 8193))
    )


def test_fuzz_legacy_layout_roundtrips_byte_for_byte() -> None:
    """encrypt → reparse as legacy layout → decrypt round-trips, 300 seeded cases.

    Proves the docstring claim that the legacy and current wrap layouts carry identical
    bytes: a ciphertext produced by today's ``encrypt()`` decrypts identically when
    re-framed into the historical version-less token (separate wrap-nonce field, no
    VERSION segment). Covers empty/1-byte/ascii/unicode/large plaintexts and random AAD.
    """
    rng = random.Random(0x48494D4D)  # fixed seed → fully deterministic
    enc = FieldEncryptor(rng.randbytes(32))
    for i in range(300):
        plaintext = _fuzz_plaintext(rng, i)
        aad = rng.randbytes(rng.randrange(0, 17)) if i % 3 == 0 else b""
        token = enc.encrypt(plaintext, aad=aad)
        # Current-format round-trip, byte-for-byte.
        current = enc.decrypt(token, aad=aad)
        assert current == plaintext
        assert current.encode("utf-8") == plaintext.encode("utf-8")
        # Legacy-layout round-trip of the SAME ciphertext bytes.
        legacy_token = _reframe_as_legacy(token)
        assert FieldEncryptor._is_legacy_token(legacy_token)
        legacy = enc.decrypt(legacy_token, aad=aad)
        assert legacy == plaintext
        assert legacy.encode("utf-8") == plaintext.encode("utf-8")
        # And the legacy form can be rotated forward without touching the plaintext.
        if i % 50 == 0:
            new_provider = LocalKekProvider(rng.randbytes(32))
            rotated = enc.rotate_kek(legacy_token, new_provider)
            assert (
                FieldEncryptor.from_provider(new_provider).decrypt(rotated, aad=aad)
                == plaintext
            )


def _tampered(blob: bytes, pos: int, rng: random.Random) -> bytes:
    """Flip a random bit of ``blob[pos]`` (guaranteed to differ)."""
    return blob[:pos] + bytes([blob[pos] ^ (1 << rng.randrange(8))]) + blob[pos + 1 :]


def _legacy_and_current_tokens(blob: bytes) -> tuple[str, str]:
    """Encode ``blob`` as (current LOCAL-versioned token, legacy version-less token)."""
    current_b64 = base64.urlsafe_b64encode(blob).decode("ascii")
    legacy_blob = bytes([blob[0] - _NONCE_LEN]) + blob[1:]
    legacy_b64 = base64.urlsafe_b64encode(legacy_blob).decode("ascii")
    return f"{ENC_PREFIX}{LOCAL_VERSION}:{current_b64}", f"{ENC_PREFIX}{legacy_b64}"


def test_fuzz_tampered_ciphertext_nonce_tag_fail_authentication() -> None:
    """Any flipped bit in nonce/ciphertext/tag/wrap regions → InvalidTag, both layouts."""
    rng = random.Random(0x54414D50)  # fixed seed
    enc = FieldEncryptor(rng.randbytes(32))
    for i in range(100):
        plaintext = _fuzz_plaintext(rng, i)
        _version, blob = FieldEncryptor._parse_token(enc.encrypt(plaintext))
        edek_len = blob[0]
        ct_start = 1 + edek_len + _NONCE_LEN
        # One byte from each authenticated region: wrap nonce, wrapped DEK (incl. its
        # GCM tag), data nonce, ciphertext body, and the trailing data GCM tag.
        positions = [
            rng.randrange(1, 1 + _NONCE_LEN),  # wrap nonce
            rng.randrange(1 + _NONCE_LEN, 1 + edek_len),  # wrapped DEK + wrap tag
            rng.randrange(1 + edek_len, ct_start),  # data nonce
            rng.randrange(ct_start, len(blob)),  # ciphertext or data tag
            rng.randrange(len(blob) - 16, len(blob)),  # data GCM tag specifically
        ]
        for pos in positions:
            current_token, legacy_token = _legacy_and_current_tokens(
                _tampered(blob, pos, rng)
            )
            with pytest.raises(InvalidTag):
                enc.decrypt(current_token)
            with pytest.raises(InvalidTag):
                enc.decrypt(legacy_token)
        # The length byte isn't GCM-authenticated; corrupting it must still fail
        # loudly (misframed slices → InvalidTag, or a structural ValueError).
        bad_len = _tampered(blob, 0, rng)
        current_token, legacy_token = (
            f"{ENC_PREFIX}{LOCAL_VERSION}:"
            f"{base64.urlsafe_b64encode(bad_len).decode('ascii')}",
            f"{ENC_PREFIX}{base64.urlsafe_b64encode(bad_len).decode('ascii')}",
        )
        for token in (current_token, legacy_token):
            with pytest.raises((InvalidTag, ValueError)):
                enc.decrypt(token)


def test_fuzz_wrong_kek_fails_cleanly() -> None:
    """Decrypting with a different KEK raises InvalidTag for both token layouts."""
    rng = random.Random(0x4B454B)  # fixed seed
    enc = FieldEncryptor(rng.randbytes(32))
    wrong = FieldEncryptor(rng.randbytes(32))
    for i in range(60):
        plaintext = _fuzz_plaintext(rng, i)
        token = enc.encrypt(plaintext)
        legacy_token = _reframe_as_legacy(token)
        with pytest.raises(InvalidTag):
            wrong.decrypt(token)
        with pytest.raises(InvalidTag):
            wrong.decrypt(legacy_token)
