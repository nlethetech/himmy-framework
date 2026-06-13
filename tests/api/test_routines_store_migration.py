"""T3c — the routines store forward-migrates a legacy (pre-T3c) database safely.

A database created before T3c has neither ``workspace_id``/``agent_id``/
``idempotency_key`` nor a nullable ``agent_path`` (it was ``NOT NULL``). Opening it with
the upgraded :class:`RoutinesStore` must: preserve every legacy row (stamping the
``__local__`` workspace), RELAX the ``agent_path`` NOT NULL (so an ``agent_id``-bound /v1
routine with a NULL ``agent_path`` can be inserted into the migrated file), and be
idempotent across re-opens. This pins the real-data-safety contract.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from himmy.api.routines import Routine, RoutinesStore, Schedule

_LEGACY_DDL = """
CREATE TABLE routines (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, agent_path TEXT NOT NULL, prompt TEXT NOT NULL,
  schedule_kind TEXT NOT NULL, schedule_at TEXT, schedule_hours INTEGER, provider TEXT,
  model TEXT, deliver TEXT NOT NULL DEFAULT 'none', enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_run_at TEXT, last_status TEXT,
  last_preview TEXT NOT NULL DEFAULT '', last_error TEXT, last_delivery TEXT);
"""


def _write_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_LEGACY_DDL)
    conn.execute(
        "INSERT INTO routines (id, name, agent_path, prompt, schedule_kind, "
        "schedule_hours, created_at, updated_at, enabled, last_status, last_preview) "
        "VALUES ('r1', 'old', 'agent.yaml', 'hi', 'every', 6, "
        "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1, 'ok', 'prev')"
    )
    conn.commit()
    conn.close()


def test_legacy_db_is_rebuilt_preserving_rows(tmp_path: Path) -> None:
    db = tmp_path / "routines.db"
    _write_legacy_db(db)

    store = RoutinesStore(str(db))
    rows = store.list()
    assert len(rows) == 1
    r = rows[0]
    # Legacy data preserved; the new columns are stamped/defaulted.
    assert r.id == "r1"
    assert r.workspace_id == "__local__"
    assert r.agent_path == "agent.yaml"
    assert r.agent_id is None
    assert r.last_status == "ok"
    assert r.last_preview == "prev"

    # agent_path NOT NULL was relaxed (column 'notnull' flag is 0 now).
    info = {
        row[1]: row[3]
        for row in store._conn.execute("PRAGMA table_info(routines)").fetchall()
    }
    assert info["agent_path"] == 0
    store.close()


def test_migration_is_idempotent_and_accepts_agent_id_routines(tmp_path: Path) -> None:
    db = tmp_path / "routines.db"
    _write_legacy_db(db)

    RoutinesStore(str(db)).close()  # first migrate
    store = RoutinesStore(str(db))  # re-open: must not double-migrate / lose rows
    assert len(store.list()) == 1

    # An agent_id-bound (/v1) routine with a NULL agent_path inserts cleanly.
    v1 = Routine(
        name="v1",
        workspace_id="acme",
        agent_id="a1",
        prompt="go",
        schedule=Schedule(kind="every", hours=2),
    )
    store.upsert(v1)
    acme = store.list(workspace_id="acme")
    assert [r.id for r in acme] == [v1.id]
    assert store.find_by_agent_id("a1", workspace_id="acme") == [v1.id]
    store.close()
