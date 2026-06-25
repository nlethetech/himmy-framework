"""Security regression tests for himmy.config.secrets hardening.

Two findings:

* #9 — ``KeychainSecrets.set`` used to shell out to ``security add-generic-password
  -w <value>``, exposing the secret in the process table (argv). The default path
  must now hand the value to the OS keychain via the ``keyring`` library so it never
  touches argv.
* #10 — ``FileSecrets`` created its secrets directory under the default umask
  (0o755 / world-traversable), leaking the *names* of stored secrets. The dir must
  be 0o700 after a write.

These tests FAIL against the old behavior and PASS with the fix. They are hermetic:
the keychain test injects a fake keyring backend and asserts the ``security`` CLI is
never invoked with the raw secret value.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from himmy.config.secrets import FileSecrets, KeychainSecrets


class _FakeKeyring:
    """In-process stand-in for the ``keyring`` module (no argv, no subprocess)."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.store.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.store[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        try:
            del self.store[(service, name)]
        except KeyError as exc:  # mimic keyring.errors.PasswordDeleteError
            raise RuntimeError("not found") from exc


def test_keychain_set_uses_keyring_not_argv() -> None:
    """Finding #9: the value goes through keyring, the security CLI is never run."""
    fake = _FakeKeyring()

    invoked: list[list[str]] = []

    def exploding_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        # If the fix regresses to the CLI path, this records the argv (which would
        # contain the raw secret) and lets the assertions below catch it.
        invoked.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    # The runner is supplied ONLY as a tripwire; with a keyring backend injected the
    # keychain path must never reach it.
    kc = KeychainSecrets(keyring_backend=fake, runner=exploding_runner)

    secret = "super-secret-token-value"
    kc.set("HIMMY_SMTP_PASSWORD", secret)

    # Stored in-process via keyring, round-trips, and removable.
    assert fake.store[("himmy", "HIMMY_SMTP_PASSWORD")] == secret
    assert kc.get("HIMMY_SMTP_PASSWORD") == secret
    kc.delete("HIMMY_SMTP_PASSWORD")
    kc.delete("HIMMY_SMTP_PASSWORD")  # idempotent even when absent
    assert kc.get("HIMMY_SMTP_PASSWORD") is None

    # The security CLI was never invoked, so the secret never reached argv.
    assert invoked == []
    for argv in invoked:  # defensive: belt-and-suspenders
        assert secret not in argv


def test_keychain_default_backend_prefers_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no runner injected, the default backend resolves the keyring library."""
    fake = _FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)

    kc = KeychainSecrets()  # no runner, no explicit backend → loads `keyring`
    kc.set("HIMMY_TELEGRAM_BOT_TOKEN", "123:abc")
    assert fake.store[("himmy", "HIMMY_TELEGRAM_BOT_TOKEN")] == "123:abc"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-mode semantics")
def test_file_secrets_dir_is_owner_only(tmp_path: Path) -> None:
    """Finding #10: the secrets dir is 0o700, not world-traversable."""
    base = tmp_path / ".himmy" / "secrets"
    fs = FileSecrets(base)
    fs.set("HIMMY_SMTP_PASSWORD", "pw")

    dir_mode = base.stat().st_mode & 0o777
    assert dir_mode == 0o700, f"secrets dir should be 0o700, got {oct(dir_mode)}"

    # Parent (.himmy) is also tightened to avoid leaking the secrets dir's existence.
    parent_mode = base.parent.stat().st_mode & 0o777
    assert parent_mode == 0o700, f".himmy should be 0o700, got {oct(parent_mode)}"

    # The secret file itself remains 0o600 (unchanged behavior).
    file_mode = (base / "HIMMY_SMTP_PASSWORD").stat().st_mode & 0o777
    assert file_mode == 0o600
