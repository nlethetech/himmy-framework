"""Lifecycle tests for the hardened API-key authenticator (P1).

Covers: hashed (non-recoverable) in-memory storage, constant-time hash compare,
per-key expiry → 401, disabled → 401, live revocation file → 401 WITHOUT a restart,
and that the existing shared + mapped key paths still authenticate unchanged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from himmy.api.auth.apikey import (
    REVOCATION_FAIL_CLOSED_ENV,
    REVOCATION_FILE_ENV,
    ApiKeyAuthenticator,
    KeyRecord,
    _fingerprint,
    _RevocationList,
    hash_key,
    load_key_records,
)
from himmy.api.auth.base import AuthError
from himmy.api.auth.principal import Principal
from tests.conftest import run_async


class _Req:
    """Minimal Starlette Request stand-in (headers + client host)."""

    def __init__(self, headers: dict[str, str], host: str = "1.2.3.4") -> None:
        self.headers = headers
        self.client = type("C", (), {"host": host})()


def _hdr(key: str) -> _Req:
    return _Req({"x-himmy-internal-key": key})


# ----------------------------------------------------------- back-compat paths
def test_shared_key_still_authenticates() -> None:
    auth = ApiKeyAuthenticator(shared_keys={"sek"})
    p = run_async(auth.authenticate(_hdr("sek")))
    assert p.all_tenants and "admin" in p.roles


def test_mapped_key_still_authenticates_and_binds() -> None:
    bound = Principal.build("svc", tenant_ids=["t1"], roles=["operator"])
    auth = ApiKeyAuthenticator(key_principals={"ka": bound})
    p = run_async(auth.authenticate(_hdr("ka")))
    assert p.tenant_ids == frozenset({"t1"})
    assert not p.all_tenants


def test_mapped_wins_over_shared_when_both_match_order() -> None:
    bound = Principal.build("svc", tenant_ids=["t1"])
    auth = ApiKeyAuthenticator(shared_keys={"shared"}, key_principals={"mapped": bound})
    assert run_async(auth.authenticate(_hdr("mapped"))).tenant_ids == frozenset({"t1"})
    assert run_async(auth.authenticate(_hdr("shared"))).all_tenants


def test_invalid_and_missing_still_401() -> None:
    auth = ApiKeyAuthenticator(shared_keys={"sek"})
    with pytest.raises(AuthError):
        run_async(auth.authenticate(_hdr("nope")))
    with pytest.raises(AuthError):
        run_async(auth.authenticate(_Req({})))


# ------------------------------------------------- storage retains no plaintext
def test_storage_retains_no_recoverable_plaintext() -> None:
    secret = "super-secret-key-value"
    auth = ApiKeyAuthenticator(shared_keys={secret})
    blob = json.dumps(
        [
            {"key_id": r.key_id, "key_hash": r.key_hash, "expires_at": r.expires_at}
            for r in auth._records
        ]
    )
    # The raw secret must not appear anywhere in the in-memory records.
    assert secret not in blob
    for record in auth._records:
        assert secret not in record.key_hash
        assert secret not in repr(record)
        # The hash actually verifies the secret (so it IS the right hash).
        assert record.key_hash == hash_key(secret)


def test_hash_is_not_the_plaintext_and_differs_per_secret() -> None:
    assert hash_key("a") != "a"
    assert hash_key("a") != hash_key("b")
    assert hash_key("a") == hash_key("a")  # deterministic


# ------------------------------------------------------ constant-time compare
def test_constant_time_compare_path_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    import himmy.api.auth.apikey as mod

    calls: list[tuple[str, str]] = []
    real = mod.hmac.compare_digest

    def spy(a, b):  # type: ignore[no-untyped-def]
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(mod.hmac, "compare_digest", spy)
    auth = ApiKeyAuthenticator(shared_keys={"sek"})
    run_async(auth.authenticate(_hdr("sek")))
    # The compare is over HASHES, not the raw secret (so timing leaks nothing useful).
    assert calls, "compare_digest must be on the auth path"
    a, b = calls[0]
    assert a == hash_key("sek") and b == hash_key("sek")


# ---------------------------------------------------------------- expiry → 401
def test_expired_key_is_rejected() -> None:
    rec = KeyRecord.from_secret(
        "ek", Principal.build("svc", tenant_ids=["t1"]), expires_at=time.time() - 1
    )
    auth = ApiKeyAuthenticator(records=[rec])
    with pytest.raises(AuthError, match="expired"):
        run_async(auth.authenticate(_hdr("ek")))


def test_unexpired_key_passes() -> None:
    rec = KeyRecord.from_secret(
        "ek", Principal.build("svc", tenant_ids=["t1"]), expires_at=time.time() + 60
    )
    auth = ApiKeyAuthenticator(records=[rec])
    assert run_async(auth.authenticate(_hdr("ek"))).subject == "svc"


def test_expiry_loaded_from_keys_file(tmp_path: Path) -> None:
    f = tmp_path / "keys.json"
    f.write_text(
        json.dumps(
            {
                "ek": {
                    "subject": "svc",
                    "tenant_ids": ["t1"],
                    "expires_at": "2000-01-01T00:00:00Z",
                }
            }
        )
    )
    auth = ApiKeyAuthenticator(records=load_key_records(f))
    with pytest.raises(AuthError, match="expired"):
        run_async(auth.authenticate(_hdr("ek")))


# -------------------------------------------------------------- disabled → 401
def test_disabled_key_is_rejected() -> None:
    rec = KeyRecord.from_secret(
        "dk", Principal.build("svc", tenant_ids=["t1"]), disabled=True
    )
    auth = ApiKeyAuthenticator(records=[rec])
    with pytest.raises(AuthError, match="disabled"):
        run_async(auth.authenticate(_hdr("dk")))


def test_disabled_loaded_from_keys_file(tmp_path: Path) -> None:
    f = tmp_path / "keys.json"
    f.write_text(json.dumps({"dk": {"subject": "svc", "disabled": True}}))
    auth = ApiKeyAuthenticator(records=load_key_records(f))
    with pytest.raises(AuthError, match="disabled"):
        run_async(auth.authenticate(_hdr("dk")))


# ---------------------------------------- live revocation (no restart) → 401
def test_revocation_kills_key_without_restart(tmp_path: Path) -> None:
    revfile = tmp_path / "revoked.json"
    revfile.write_text("[]")
    rec = KeyRecord.from_secret("rk", Principal.build("svc", tenant_ids=["t1"]))
    auth = ApiKeyAuthenticator(records=[rec], revocation_path=revfile)

    # Initially valid.
    assert run_async(auth.authenticate(_hdr("rk"))).subject == "svc"

    # Revoke it live by appending its key_id — same long-lived authenticator instance.
    revfile.write_text(json.dumps([_fingerprint("rk")]))
    # Bump mtime so the live re-read picks it up even on coarse filesystem clocks.
    import os

    future = time.time() + 10
    os.utime(revfile, (future, future))
    with pytest.raises(AuthError, match="revoked"):
        run_async(auth.authenticate(_hdr("rk")))


def test_revocation_file_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revfile = tmp_path / "revoked.json"
    revfile.write_text(json.dumps([_fingerprint("rk")]))
    monkeypatch.setenv(REVOCATION_FILE_ENV, str(revfile))
    rec = KeyRecord.from_secret("rk", Principal.build("svc", tenant_ids=["t1"]))
    auth = ApiKeyAuthenticator(records=[rec])
    with pytest.raises(AuthError, match="revoked"):
        run_async(auth.authenticate(_hdr("rk")))


def test_missing_revocation_file_is_inert(tmp_path: Path) -> None:
    rec = KeyRecord.from_secret("rk", Principal.build("svc", tenant_ids=["t1"]))
    auth = ApiKeyAuthenticator(records=[rec], revocation_path=tmp_path / "nope.json")
    assert run_async(auth.authenticate(_hdr("rk"))).subject == "svc"


# ----------------- revocation fail-static (does NOT silently disarm) ---------
def _bump_mtime(path: Path) -> None:
    """Force a future mtime so the live re-read picks the change up immediately."""
    import os

    future = time.time() + 10
    os.utime(path, (future, future))


def test_revocation_retained_when_file_deleted(tmp_path: Path) -> None:
    """A configured file going away must NOT un-revoke an already-revoked key."""
    revfile = tmp_path / "revoked.json"
    revfile.write_text(json.dumps([_fingerprint("rk")]))
    rec = KeyRecord.from_secret("rk", Principal.build("svc", tenant_ids=["t1"]))
    auth = ApiKeyAuthenticator(records=[rec], revocation_path=revfile)
    # Revoked while the file is present.
    with pytest.raises(AuthError, match="revoked"):
        run_async(auth.authenticate(_hdr("rk")))
    # Delete it (simulate a transient blip / cross-user unreadability).
    revfile.unlink()
    # Still revoked — the last-known set is retained, not cleared (no fail-open).
    with pytest.raises(AuthError, match="revoked"):
        run_async(auth.authenticate(_hdr("rk")))


def test_revocation_retained_when_file_corrupted(tmp_path: Path) -> None:
    revfile = tmp_path / "revoked.json"
    revfile.write_text(json.dumps([_fingerprint("rk")]))
    rec = KeyRecord.from_secret("rk", Principal.build("svc", tenant_ids=["t1"]))
    auth = ApiKeyAuthenticator(records=[rec], revocation_path=revfile)
    with pytest.raises(AuthError, match="revoked"):
        run_async(auth.authenticate(_hdr("rk")))
    # Corrupt the JSON and bump mtime so the refresh actually re-reads it.
    revfile.write_text("{ this is not json")
    _bump_mtime(revfile)
    with pytest.raises(AuthError, match="revoked"):
        run_async(auth.authenticate(_hdr("rk")))


def test_revocation_failure_logged_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    revfile = tmp_path / "revoked.json"
    revfile.write_text(json.dumps([_fingerprint("rk")]))
    rl = _RevocationList(revfile)
    assert rl.is_revoked(_fingerprint("rk"))
    revfile.unlink()
    with caplog.at_level("WARNING", logger="himmy.api.auth.apikey"):
        # Many consults of an unreadable file...
        for _ in range(5):
            assert rl.is_revoked(_fingerprint("rk"))  # still revoked (retained)
    # ...emit exactly ONE loud warning (deduplicated), not one per request.
    warnings = [r for r in caplog.records if "revocation list unavailable" in r.message]
    assert len(warnings) == 1


def test_revocation_recovery_clears_error_and_relogs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    revfile = tmp_path / "revoked.json"
    revfile.write_text(json.dumps([_fingerprint("rk")]))
    rl = _RevocationList(revfile)
    assert rl.is_revoked(_fingerprint("rk"))
    revfile.unlink()
    assert rl.is_revoked(_fingerprint("rk"))  # degraded
    # Restore the file (now revoking nothing); a fresh failure later should re-log.
    revfile.write_text(json.dumps([]))
    _bump_mtime(revfile)
    with caplog.at_level("WARNING", logger="himmy.api.auth.apikey"):
        assert not rl.is_revoked(_fingerprint("rk"))  # recovered, empty list
    assert any("recovered" in r.message for r in caplog.records)


def test_revocation_same_mtime_rewrite_is_observed(tmp_path: Path) -> None:
    """A content rewrite that PRESERVES the file mtime must still be picked up.

    Red-team r1: the refresh short-circuited on an unchanged ``st_mtime`` alone, so an
    mtime-preserving sync/restore (rsync --times / cp -p / git checkout) or two edits
    within the filesystem's mtime granularity left an already-revoked key authenticating
    forever. The freshness key is now a composite ``(st_mtime_ns, st_size, st_ino)``, so
    the size/inode delta of the rewrite is observed even with an identical mtime.
    """
    import os

    revfile = tmp_path / "revoked.json"
    # Prime the cache with an EMPTY revoked set (key "rk" is NOT yet revoked).
    revfile.write_text(json.dumps([]))
    primed_ns = revfile.stat().st_mtime_ns

    rl = _RevocationList(revfile)
    assert rl.is_revoked(_fingerprint("rk")) is False  # loads + caches the empty set

    # Revoke the key by rewriting the file, then PIN the mtime back to the primed value
    # (simulating an mtime-preserving restore). Only st_size/st_ino changed.
    revfile.write_text(json.dumps([_fingerprint("rk")]))
    os.utime(revfile, ns=(primed_ns, primed_ns))
    assert revfile.stat().st_mtime_ns == primed_ns  # mtime truly unchanged

    # Before the fix this returned False (stale empty set); now the composite fingerprint
    # caught the rewrite and the key is seen as revoked.
    assert rl.is_revoked(_fingerprint("rk")) is True


def test_configured_absent_file_stays_inert_not_failclosed(tmp_path: Path) -> None:
    """A file that NEVER loaded has an empty last-known set, so it stays inert."""
    rec = KeyRecord.from_secret("rk", Principal.build("svc", tenant_ids=["t1"]))
    auth = ApiKeyAuthenticator(records=[rec], revocation_path=tmp_path / "never.json")
    # No fail-closed env → an absent-from-the-start file must not lock the key out.
    assert run_async(auth.authenticate(_hdr("rk"))).subject == "svc"


# ------------------------------------ revocation FAIL-CLOSED opt-in -----------
def test_fail_closed_401s_when_file_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revfile = tmp_path / "revoked.json"
    revfile.write_text(json.dumps([]))  # revokes nothing while present
    rec = KeyRecord.from_secret("rk", Principal.build("svc", tenant_ids=["t1"]))
    auth = ApiKeyAuthenticator(records=[rec], revocation_path=revfile)
    # Healthy: authenticates.
    assert run_async(auth.authenticate(_hdr("rk"))).subject == "svc"
    monkeypatch.setenv(REVOCATION_FAIL_CLOSED_ENV, "1")
    revfile.unlink()
    # Configured file now unreadable + fail-closed opted in → 401 even though the
    # key is otherwise valid and was not on the (last-known empty) revoked list.
    with pytest.raises(AuthError, match="unavailable"):
        run_async(auth.authenticate(_hdr("rk")))


def test_fail_closed_inert_when_no_file_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed only engages for a CONFIGURED file — no file ⇒ never raises."""
    monkeypatch.setenv(REVOCATION_FAIL_CLOSED_ENV, "1")
    rec = KeyRecord.from_secret("rk", Principal.build("svc", tenant_ids=["t1"]))
    auth = ApiKeyAuthenticator(records=[rec])  # no revocation_path, env unset
    monkeypatch.delenv(REVOCATION_FILE_ENV, raising=False)
    assert run_async(auth.authenticate(_hdr("rk"))).subject == "svc"


# -------------------------------------------------------------- pepper hashing
def test_pepper_changes_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    import himmy.api.auth.apikey as mod

    monkeypatch.setattr(mod, "_pepper", lambda: b"")
    unkeyed = hash_key("x")
    monkeypatch.setattr(mod, "_pepper", lambda: b"server-pepper")
    keyed = hash_key("x")
    assert unkeyed != keyed


# ------------------------------------------------------------- binds_tenants
def test_binds_tenants_semantics() -> None:
    assert not ApiKeyAuthenticator(shared_keys={"s"}).binds_tenants
    bound = Principal.build("svc", tenant_ids=["t1"])
    assert ApiKeyAuthenticator(key_principals={"m": bound}).binds_tenants
    rec = KeyRecord.from_secret("r", bound)
    assert ApiKeyAuthenticator(records=[rec]).binds_tenants
