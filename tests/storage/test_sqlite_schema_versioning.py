"""Tests for SqliteStorageService schema versioning + forward migrations.

The base schema is ``CREATE TABLE IF NOT EXISTS`` (a no-op on an existing file), so a
future column would never reach a legacy ``.himmy/storage.db`` without a migration
runner. These tests pin the ``PRAGMA user_version`` stamp and the stepwise forward
migration that closes that gap.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from himmy.core.events import EventType, RunEvent
from himmy.services.storage import sqlite as sqlite_mod
from himmy.services.storage.sqlite import SQLITE_SCHEMA_VERSION, SqliteStorageService
from tests.conftest import run_async


def test_fresh_db_is_stamped_at_current_version(tmp_path: Path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    assert store.schema_version == SQLITE_SCHEMA_VERSION


def test_reopening_db_keeps_version_and_data(tmp_path: Path) -> None:
    path = str(tmp_path / "storage.db")
    first = SqliteStorageService(path)
    run_async(first.append_event(RunEvent(event_type=EventType.AGENT_RUN_STARTED)))
    run_async(first.close())
    # Reopening an existing file must not reset the version nor wipe data.
    second = SqliteStorageService(path)
    assert second.schema_version == SQLITE_SCHEMA_VERSION
    assert len(run_async(second.list_events())) == 1


def test_legacy_unversioned_db_is_migrated_forward(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A pre-versioning file (user_version 0) steps through every pending migration."""
    path = str(tmp_path / "legacy.db")

    # Simulate an old database: base tables present, but user_version never stamped
    # AND missing a column that a later release adds.
    raw = sqlite3.connect(path)
    raw.executescript(sqlite_mod._SCHEMA)
    raw.execute("ALTER TABLE runs DROP COLUMN updated_at")  # pretend it's an old shape
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    raw.commit()
    raw.close()

    # Inject a forward migration (version 2) that re-adds the column + bumps the target.
    migrations = (
        (2, ("ALTER TABLE runs ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''",)),
    )
    monkeypatch.setattr(sqlite_mod, "_MIGRATIONS", migrations)
    monkeypatch.setattr(sqlite_mod, "SQLITE_SCHEMA_VERSION", 2)

    store = SqliteStorageService(path)
    assert store.schema_version == 2
    cols = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(runs)").fetchall()
    }
    assert "updated_at" in cols  # the migration ran against the legacy file


def test_migration_runs_once_not_reapplied(tmp_path: Path, monkeypatch: Any) -> None:
    """Re-opening a db already at the target version is idempotent (no double-apply)."""
    path = str(tmp_path / "storage.db")
    monkeypatch.setattr(
        sqlite_mod,
        "_MIGRATIONS",
        ((2, ("CREATE TABLE IF NOT EXISTS _probe (x TEXT)",)),),
    )
    monkeypatch.setattr(sqlite_mod, "SQLITE_SCHEMA_VERSION", 2)

    run_async(SqliteStorageService(path).close())  # build + stamp at v2, then close
    reopened = SqliteStorageService(path)  # re-open: already at v2, migration skipped
    assert reopened.schema_version == 2
    # _probe exists and reopening did not error / double-create.
    assert (
        reopened._conn.execute(
            "SELECT name FROM sqlite_master WHERE name='_probe'"
        ).fetchone()
        is not None
    )
