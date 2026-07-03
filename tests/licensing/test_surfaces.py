"""The licensing seam wired into REAL Enterprise surfaces + the ``himmy license`` CLI.

Asserts community gets the clear upgrade error and entitled enterprise gets the feature:

* ``OidcAuthenticator.from_env`` is gated on ``oidc_sso``;
* the ``himmy audit privacy --export-bundle`` path is gated on ``audit_export``;
* ``himmy license`` shows/installs/verifies and NEVER grants enterprise without a key.
"""

from __future__ import annotations

import json
import time

import pytest

import himmy.licensing.core as lic
from himmy.licensing import EnterpriseFeatureError, reset_license_cache


def _keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    pub_hex = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    return priv, pub_hex


def _mint(priv, *, features):
    payload = {
        "license_id": "lic-test",
        "edition": "enterprise",
        "features": list(features),
        "customer": "Acme",
        "issued_at": int(time.time()),
        "expires_at": 0,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"{lic._b64url_encode(raw)}.{lic._b64url_encode(priv.sign(raw))}"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(lic.HIMMY_LICENSE_KEY_ENV, raising=False)
    monkeypatch.setattr(lic, "license_file_path", lambda: "/nonexistent/.himmy/license")
    monkeypatch.setattr("himmy.config.secrets.get_secret", lambda *a, **k: None)
    reset_license_cache()
    yield
    reset_license_cache()


def test_oidc_from_env_gated_on_community(monkeypatch):
    from himmy.api.auth.oidc import OidcAuthenticator

    monkeypatch.setenv("HIMMY_OIDC_ISSUER", "https://idp.example")
    monkeypatch.setenv("HIMMY_OIDC_AUDIENCE", "himmy")
    with pytest.raises(EnterpriseFeatureError) as excinfo:
        OidcAuthenticator.from_env()
    assert "oidc_sso" in str(excinfo.value)


def test_oidc_from_env_allowed_on_enterprise(monkeypatch):
    from himmy.api.auth.oidc import OidcAuthenticator

    priv, pub_hex = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    monkeypatch.setenv(lic.HIMMY_LICENSE_KEY_ENV, _mint(priv, features=["oidc_sso"]))
    reset_license_cache()

    monkeypatch.setenv("HIMMY_OIDC_ISSUER", "https://idp.example")
    monkeypatch.setenv("HIMMY_OIDC_AUDIENCE", "himmy")
    auth = OidcAuthenticator.from_env()  # must not raise on entitled enterprise
    assert isinstance(auth, OidcAuthenticator)


def test_audit_export_bundle_gated_on_community(monkeypatch, capsys):
    import himmy.cli.audit as cli_audit

    # community -> the export path returns 1 with the upgrade message, no crash
    rc = cli_audit._export_bundle(object(), object(), "/tmp/should-not-write.json")
    assert rc == 1
    err = capsys.readouterr().err
    assert "audit_export is an Enterprise Edition feature" in err


def test_license_cli_show_and_verify_community(monkeypatch, capsys):
    from himmy.cli.license_cmd import cmd_license

    args = type("A", (), {"action": None})()
    assert cmd_license(args) == 0
    out = capsys.readouterr().out
    assert "edition:      community" in out

    verify_args = type("A", (), {"action": "verify"})()
    assert cmd_license(verify_args) == 1


def test_license_cli_install_and_show_enterprise(monkeypatch, tmp_path, capsys):
    from himmy.cli import license_cmd

    priv, pub_hex = _keypair()
    monkeypatch.setattr(lic, "LICENSE_PUBLIC_KEY_HEX", pub_hex)
    token = _mint(priv, features=["oidc_sso", "audit_export"])

    dest = tmp_path / ".himmy" / "license"
    monkeypatch.setattr(license_cmd.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(lic, "license_file_path", lambda: str(dest))

    install_args = type("A", (), {"action": "install", "value": token})()
    assert license_cmd.cmd_license(install_args) == 0
    assert dest.read_text().strip() == token

    show_args = type("A", (), {"action": None})()
    license_cmd.cmd_license(show_args)
    out = capsys.readouterr().out
    assert "edition:      enterprise" in out
    assert "audit_export" in out


def test_license_cli_install_rejects_tampered(monkeypatch, capsys):
    from himmy.cli.license_cmd import cmd_license

    args = type("A", (), {"action": "install", "value": "garbage.token"})()
    assert cmd_license(args) == 1
    assert "invalid or tampered" in capsys.readouterr().out
