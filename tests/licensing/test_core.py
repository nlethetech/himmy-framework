"""Enterprise Edition licensing seam — offline verification + fail-closed gating.

Asserts the monetization invariants:

* a valid EE license -> enterprise + entitlements (intersected with the registry);
* no license -> community with EMPTY entitlements (core stays free);
* an EXPIRED license -> community (fail-closed), even though it parses;
* a TAMPERED signature / garbage token -> community, never a crash;
* a gated feature without entitlement raises a clear ``EnterpriseFeatureError``;
* verification is fully OFFLINE (no socket is ever opened).

Tests are self-contained: they mint with an EPHEMERAL keypair and monkeypatch the baked-in
public key, so they never depend on the vendor's admin private key.
"""

from __future__ import annotations

import json
import socket
import time

import pytest

import himmy.licensing.core as lic
from himmy.licensing import (
    FEATURE_AUDIT_EXPORT,
    FEATURE_OIDC_SSO,
    Edition,
    EnterpriseFeatureError,
    current_edition,
    current_entitlements,
    has_entitlement,
    require_entitlement,
    reset_license_cache,
    verify_license,
)


def _keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    pub_hex = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    return priv, pub_hex


def _mint(priv, *, edition="enterprise", features, customer="Acme", expires_at=0):
    payload = {
        "license_id": "lic-test",
        "edition": edition,
        "features": list(features),
        "customer": customer,
        "issued_at": int(time.time()),
        "expires_at": expires_at,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = priv.sign(raw)
    return f"{lic._b64url_encode(raw)}.{lic._b64url_encode(sig)}"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Isolate every test from the real env/file license and the process cache."""
    monkeypatch.delenv(lic.HIMMY_LICENSE_KEY_ENV, raising=False)
    # Point the license file at a nonexistent home so no on-disk license leaks in.
    monkeypatch.setattr(lic, "license_file_path", lambda: "/nonexistent/.himmy/license")
    # And make the secrets fallback return nothing.
    monkeypatch.setattr("himmy.config.secrets.get_secret", lambda *a, **k: None)
    reset_license_cache()
    yield
    reset_license_cache()


def test_valid_ee_license_grants_enterprise(monkeypatch):
    priv, pub_hex = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    token = _mint(priv, features=[FEATURE_OIDC_SSO, FEATURE_AUDIT_EXPORT])
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, token)
    reset_license_cache()

    assert current_edition() is Edition.ENTERPRISE
    assert current_entitlements() == frozenset(
        {FEATURE_OIDC_SSO, FEATURE_AUDIT_EXPORT}
    )
    assert has_entitlement(FEATURE_OIDC_SSO)
    # entitled -> require passes quietly
    require_entitlement(FEATURE_OIDC_SSO)


def test_no_license_is_community_and_core_unaffected():
    assert current_edition() is Edition.COMMUNITY
    assert current_entitlements() == frozenset()
    assert not has_entitlement(FEATURE_OIDC_SSO)


def test_expired_license_fails_closed(monkeypatch):
    priv, pub_hex = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    token = _mint(
        priv, features=[FEATURE_OIDC_SSO], expires_at=int(time.time()) - 3600
    )
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, token)
    reset_license_cache()

    assert current_edition() is Edition.COMMUNITY
    assert current_entitlements() == frozenset()
    assert lic.resolution_reason() == "expired"
    # the payload still parsed (so `verify` can explain WHY), but grants nothing
    assert lic.current_license() is not None


def test_tampered_signature_is_community(monkeypatch):
    priv, pub_hex = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    token = _mint(priv, features=[FEATURE_OIDC_SSO])
    payload_b64, _sig = token.split(".", 1)
    forged = payload_b64 + "." + lic._b64url_encode(b"\x00" * 64)
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, forged)
    reset_license_cache()

    assert current_edition() is Edition.COMMUNITY
    assert current_entitlements() == frozenset()


def test_wrong_public_key_is_community(monkeypatch):
    # A license signed by SOME key but verified against a DIFFERENT baked-in key -> deny.
    priv, _pub = _keypair()
    _other_priv, other_pub = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", other_pub)
    token = _mint(priv, features=[FEATURE_OIDC_SSO])
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, token)
    reset_license_cache()
    assert current_edition() is Edition.COMMUNITY


@pytest.mark.parametrize(
    "garbage", ["", "not-a-token", "a.b.c", "!!!.???", "onlyonepart"]
)
def test_garbage_tokens_never_crash(monkeypatch, garbage):
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, garbage)
    reset_license_cache()
    assert current_edition() is Edition.COMMUNITY
    assert verify_license(garbage) is None


def test_community_edition_edition_is_not_granted(monkeypatch):
    # Even a validly-signed COMMUNITY-edition license grants no entitlements.
    priv, pub_hex = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    token = _mint(priv, edition="community", features=[FEATURE_OIDC_SSO])
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, token)
    reset_license_cache()
    assert current_edition() is Edition.COMMUNITY
    assert current_entitlements() == frozenset()


def test_forged_feature_name_cannot_leak(monkeypatch):
    # A feature not in the registry is intersected out even on a valid EE license.
    priv, pub_hex = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    token = _mint(priv, features=[FEATURE_OIDC_SSO, "not_a_real_feature"])
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, token)
    reset_license_cache()
    assert current_entitlements() == frozenset({FEATURE_OIDC_SSO})


def test_require_entitlement_raises_clear_error():
    with pytest.raises(EnterpriseFeatureError) as excinfo:
        require_entitlement(FEATURE_OIDC_SSO)
    msg = str(excinfo.value)
    assert FEATURE_OIDC_SSO in msg
    assert "Enterprise Edition feature" in msg
    assert "himmy license install" in msg
    assert excinfo.value.feature == FEATURE_OIDC_SSO


def test_verification_is_offline(monkeypatch):
    # Assert no socket is ever opened during resolution/verification.
    def _boom(*a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("license verification must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)

    priv, pub_hex = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    token = _mint(priv, features=[FEATURE_OIDC_SSO])
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, token)
    reset_license_cache()
    assert current_edition() is Edition.ENTERPRISE
