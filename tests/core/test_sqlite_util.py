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


def test_synchronous_defaults_normal_and_full_is_honored(tmp_path: Path) -> None:
    """Default is NORMAL (=1); durability-critical stores opt into FULL (=2)."""
    normal = connect_hardened(str(tmp_path / "n.db"))
    full = connect_hardened(str(tmp_path / "f.db"), synchronous="FULL")
    try:
        assert normal.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert full.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    finally:
        normal.close()
        full.close()
