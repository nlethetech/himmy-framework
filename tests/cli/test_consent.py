"""WS4.6 A3 — ``himmy consent`` CLI (grant → status → history → revoke, offline).

The subcommand tree thinly delegates to the already-tested
:class:`~himmy.services.governance.consent_ledger.ConsentLedger` over a durable on-disk
:class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry` (``.himmy/consent.db``).
These tests drive the real ``main(["consent", ...])`` dispatch (so the parser wiring is
covered too), assert the JSON output shape on each command, prove that ``revoke`` writes a
signed ``erasure_tombstone`` to the same durable registry, and that an unknown ``purpose``
exits non-zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from himmy.cli.__main__ import main
from himmy.entities.registry import EntityRegistry
from himmy.entities.sqlite_registry import SqliteEntityRegistry


@pytest.fixture(autouse=True)
def _isolated_consent_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test in a fresh cwd so ``.himmy/consent.db`` is per-test isolated."""
    monkeypatch.chdir(tmp_path)


def _last_json(capsys: pytest.CaptureFixture[str]) -> object:
    """Parse the last (machine-readable) stdout line as JSON."""
    out = capsys.readouterr().out.strip().splitlines()
    return json.loads(out[-1])


def test_grant_status_history_revoke_roundtrip(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """grant → status → history → revoke all return the documented JSON shapes."""
    # grant: one consent record (GRANTED) for the pair.
    assert main(["consent", "grant", "teacher_a", "retain", "--json"]) == 0
    granted = cast(dict, _last_json(capsys))
    assert granted["subject_id"] == "teacher_a"
    assert granted["purpose"] == "retain"
    assert granted["state"] == "granted"

    # status: the PDP decision is ALLOW for a granted RETAIN.
    assert main(["consent", "status", "teacher_a", "retain", "--json"]) == 0
    status = cast(dict, _last_json(capsys))
    assert status["state"] == "granted"
    assert status["effect"] == "allow"
    assert status["allowed"] is True

    # history: the append-only chain (one version so far).
    assert main(["consent", "history", "teacher_a", "retain", "--json"]) == 0
    history = cast(list, _last_json(capsys))
    assert isinstance(history, list) and len(history) == 1
    assert history[0]["state"] == "granted"

    # revoke: withdraws + returns the WITHDRAWN record(s).
    assert main(["consent", "revoke", "teacher_a", "--json"]) == 0
    revoked = cast(list, _last_json(capsys))
    assert isinstance(revoked, list) and revoked
    assert all(r["state"] == "withdrawn" for r in revoked)

    # The durable registry now holds a signed erasure tombstone for the subject.
    registry = cast(
        EntityRegistry, SqliteEntityRegistry(str(tmp_path / ".himmy" / "consent.db"))
    )
    tombstones = registry.list_by_kind("erasure_tombstone")
    assert tombstones, "revoke must write an erasure tombstone"
    # And a fresh status decision is now a denial (consent withdrawn).
    capsys.readouterr()  # drain
    assert main(["consent", "status", "teacher_a", "retain", "--json"]) == 0
    after = cast(dict, _last_json(capsys))
    assert after["allowed"] is False


def test_history_persists_across_invocations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second grant (deny) appends a version — the chain survives separate CLI runs."""
    assert main(["consent", "grant", "teacher_a", "train", "--json"]) == 0
    capsys.readouterr()
    assert main(["consent", "deny", "teacher_a", "train", "--json"]) == 0
    capsys.readouterr()

    assert main(["consent", "history", "teacher_a", "train", "--json"]) == 0
    history = cast(list, _last_json(capsys))
    assert [r["state"] for r in history] == ["granted", "denied"]


def test_unknown_purpose_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown ``purpose`` token exits non-zero (no silent success)."""
    with pytest.raises(SystemExit) as exc:
        main(["consent", "grant", "teacher_a", "not_a_purpose", "--json"])
    # ``SystemExit`` with a message string ⇒ a non-zero (truthy) exit code.
    assert exc.value.code not in (0, None)
    err = capsys.readouterr().err
    assert "unknown purpose" in err or "not_a_purpose" in str(exc.value.code)


def test_revoke_without_consent_is_noop_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Revoking a subject with nothing on file exits 0 and emits an empty JSON list."""
    assert main(["consent", "revoke", "nobody", "--json"]) == 0
    revoked = cast(list, _last_json(capsys))
    assert revoked == []
