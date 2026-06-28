"""Tests for the ``himmy apikey`` CLI: mint/rotate/revoke/list round-trips."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path

from himmy.api.auth.apikey import (
    ApiKeyAuthenticator,
    _fingerprint,
    load_key_records,
)
from himmy.api.auth.base import AuthError
from himmy.cli.apikey_cmd import (
    cmd_apikey_list,
    cmd_apikey_mint,
    cmd_apikey_revoke,
    cmd_apikey_rotate,
)
from tests.conftest import run_async


def _ns(**kw: object) -> argparse.Namespace:
    base = {
        "file": None,
        "revocation_file": None,
        "subject": None,
        "tenant": None,
        "role": None,
        "all_tenants": False,
        "ttl_days": None,
        "key_id": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _secret_from_file(keys_file: Path) -> str:
    keys = json.loads(keys_file.read_text())
    return next(iter(keys))


def test_mint_writes_0600_and_loads(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    keys_file = tmp_path / "keys.json"
    rc = cmd_apikey_mint(
        _ns(file=str(keys_file), subject="svc", tenant=["t1"], role=["operator"])
    )
    assert rc == 0
    out = capsys.readouterr().out
    secret = _secret_from_file(keys_file)
    assert secret in out  # printed exactly once on mint

    # File is owner-only (0600).
    mode = stat.S_IMODE(keys_file.stat().st_mode)
    assert mode == 0o600

    # The minted key authenticates and binds to its tenant.
    auth = ApiKeyAuthenticator(records=load_key_records(keys_file))

    class _R:
        headers = {"x-himmy-internal-key": secret}
        client = type("C", (), {"host": "1.2.3.4"})()

    p = run_async(auth.authenticate(_R()))
    assert p.tenant_ids == frozenset({"t1"})


def test_rotate_disables_old_and_mints_new(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    keys_file = tmp_path / "keys.json"
    cmd_apikey_mint(_ns(file=str(keys_file), subject="svc", tenant=["t1"]))
    capsys.readouterr()
    old_secret = _secret_from_file(keys_file)
    old_id = _fingerprint(old_secret)

    cmd_apikey_rotate(_ns(file=str(keys_file), key_id=old_id))
    capsys.readouterr()

    auth = ApiKeyAuthenticator(records=load_key_records(keys_file))

    class _Old:
        headers = {"x-himmy-internal-key": old_secret}
        client = type("C", (), {"host": "1.2.3.4"})()

    # The old key is now disabled → 401.
    try:
        run_async(auth.authenticate(_Old()))
        raise AssertionError("old key should be disabled")
    except AuthError as exc:
        assert "disabled" in str(exc)


def test_revoke_writes_revocation_file(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    keys_file = tmp_path / "keys.json"
    revfile = tmp_path / "rev.json"
    cmd_apikey_mint(_ns(file=str(keys_file), subject="svc", tenant=["t1"]))
    capsys.readouterr()
    secret = _secret_from_file(keys_file)
    key_id = _fingerprint(secret)

    rc = cmd_apikey_revoke(
        _ns(file=str(keys_file), revocation_file=str(revfile), key_id=key_id)
    )
    assert rc == 0
    assert key_id in json.loads(revfile.read_text())

    auth = ApiKeyAuthenticator(
        records=load_key_records(keys_file), revocation_path=revfile
    )

    class _R:
        headers = {"x-himmy-internal-key": secret}
        client = type("C", (), {"host": "1.2.3.4"})()

    try:
        run_async(auth.authenticate(_R()))
        raise AssertionError("revoked key should 401")
    except AuthError as exc:
        assert "revoked" in str(exc)


def test_list_shows_metadata_not_secret(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    keys_file = tmp_path / "keys.json"
    cmd_apikey_mint(_ns(file=str(keys_file), subject="svc", tenant=["t1"]))
    capsys.readouterr()
    secret = _secret_from_file(keys_file)

    cmd_apikey_list(_ns(file=str(keys_file)))
    out = capsys.readouterr().out
    assert _fingerprint(secret) in out  # key_id shown
    assert secret not in out  # secret NEVER shown by list
    assert "svc" in out
