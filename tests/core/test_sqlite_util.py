"""Tests for hardened SQLite connections (WAL + busy timeout)."""

from __future__ import annotations

from pathlib import Path

from himmy.core.sqlite_util import connect_hardened


def test_file_db_uses_wal_and_busy_timeout(tmp_path: Path) -> None:
    conn = connect_hardened(str(tmp_path / "x.db"))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_memory_db_skips_wal(tmp_path: Path) -> None:
    conn = connect_hardened(":memory:")
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        assert mode in ("memory", "delete")
    finally:
        conn.close()
