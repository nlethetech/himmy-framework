"""Tests for ``himmy doctor --storage`` (the storage diagnostics section).

Drive the real CLI dispatcher (``himmy.cli.__main__.main``) with a chdir'd
``tmp_path`` so the SQLite-store scan and Postgres-DSN branches are exercised
end-to-end without touching the developer's real ``.himmy/`` directory.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from himmy.cli.__main__ import main
from himmy.cli.commands import _redact_dsn


def _make_wal_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO t (id) VALUES (1)")
        conn.commit()
    finally:
        conn.close()


def test_doctor_storage_sqlite_reports_dbs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_STORE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    _make_wal_db(tmp_path / ".himmy" / "storage.db")

    rc = main(["doctor", "--storage"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "backend: sqlite" in out
    assert "storage.db" in out
    assert "wal" in out.lower()


def test_doctor_storage_sqlite_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_STORE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)  # no .himmy/ created

    rc = main(["doctor", "--storage"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "backend: sqlite" in out
    assert "no .himmy/ stores yet" in out


def test_doctor_storage_postgres_redacts_and_degrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Point at an unreachable Postgres on a closed port so the report degrades to
    # a single warning line (and never crashes) regardless of asyncpg presence.
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgresql://u:secretpw@127.0.0.1:1/db")
    monkeypatch.chdir(tmp_path)

    rc = main(["doctor", "--storage"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "backend: postgresql" in out
    assert "***" in out
    assert "secretpw" not in out
    assert "warning:" in out


def test_redact_dsn_masks_password() -> None:
    assert _redact_dsn("postgresql://u:secretpw@db.host:5432/app") == (
        "postgresql://u:***@db.host:5432/app"
    )


def test_redact_dsn_no_password_is_passthrough() -> None:
    assert _redact_dsn("postgresql://db.host:5432/app") == (
        "postgresql://db.host:5432/app"
    )


def test_redact_dsn_preserves_ipv6_brackets() -> None:
    # urlsplit strips the [] from an IPv6 host; the redactor must re-bracket so the
    # rebuilt DSN stays valid (and the :port isn't ambiguous with the address colons).
    redacted = _redact_dsn("postgresql://u:pw@[::1]:5432/db")
    assert redacted == "postgresql://u:***@[::1]:5432/db"
    assert "pw" not in redacted


def test_redact_dsn_ipv6_without_port() -> None:
    assert _redact_dsn("postgresql://u:pw@[2001:db8::1]/db") == (
        "postgresql://u:***@[2001:db8::1]/db"
    )


def test_doctor_without_storage_flag_omits_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    rc = main(["doctor"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "\nstorage:" not in out
