"""P0-B: run_events promotes event_type + tool_name to indexed columns (v2).

New writes populate the columns directly (so a learning miner answers "which
tools fail most" by index seek); a pre-P0-B v1 database upgrades in place and
backfills the columns from the stored JSON.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from himmy.core.events import EventType, RunEvent
from himmy.services.storage.sqlite import (
    _SCHEMA,
    SQLITE_SCHEMA_VERSION,
    SqliteStorageService,
)
from tests.conftest import run_async


def _cols(db: Path, table: str) -> set[str]:
    raw = sqlite3.connect(db)
    try:
        return {r[1] for r in raw.execute(f"PRAGMA table_info({table})")}
    finally:
        raw.close()


def _indexes(db: Path, table: str) -> set[str]:
    raw = sqlite3.connect(db)
    try:
        return {r[1] for r in raw.execute(f"PRAGMA index_list({table})")}
    finally:
        raw.close()


def test_fresh_db_is_v2_with_indexed_columns(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    svc = SqliteStorageService(str(db))
    assert svc.schema_version == SQLITE_SCHEMA_VERSION == 2

    assert {"event_type", "tool_name"} <= _cols(db, "run_events")
    idx = _indexes(db, "run_events")
    assert "run_events_event_type_idx" in idx
    assert "run_events_tool_name_idx" in idx


def test_append_event_populates_indexed_columns(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    svc = SqliteStorageService(str(db))
    run_async(
        svc.append_event(
            RunEvent(
                event_type=EventType.TOOL_FAILED,
                trace_id="t:1",
                thread_id="t",
                tool_call_id="c1",
                payload={"tool_name": "wire_money", "error_code": "BOOM"},
            )
        )
    )
    run_async(
        svc.append_event(
            RunEvent(
                event_type=EventType.AGENT_RUN_STARTED, trace_id="t:1", thread_id="t"
            )
        )
    )

    raw = sqlite3.connect(db)
    try:
        failed = raw.execute(
            "SELECT event_type, tool_name FROM run_events WHERE event_type=?",
            ("TOOL_FAILED",),
        ).fetchone()
        started = raw.execute(
            "SELECT tool_name FROM run_events WHERE event_type=?",
            ("AGENT_RUN_STARTED",),
        ).fetchone()
    finally:
        raw.close()
    assert failed == ("TOOL_FAILED", "wire_money")
    assert started[0] is None  # a non-tool event has no tool_name


def test_legacy_v1_db_upgrades_and_backfills(tmp_path: Path) -> None:
    """A pre-P0-B database (run_events without the columns) upgrades + backfills."""
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript(_SCHEMA)  # the frozen v1 shape (run_events lacks the columns)
    raw.execute("PRAGMA user_version = 1")
    raw.execute(
        "INSERT INTO run_events (event_id, thread_id, trace_id, payload) "
        "VALUES (?,?,?,?)",
        (
            "e1",
            "t",
            "t:1",
            json.dumps(
                {"event_type": "TOOL_FAILED", "payload": {"tool_name": "wire_money"}}
            ),
        ),
    )
    raw.commit()
    raw.close()
    assert "event_type" not in _cols(db, "run_events")  # truly a v1 shape

    # Opening with the service applies migration v2 (ALTER + index + backfill).
    svc = SqliteStorageService(str(db))
    assert svc.schema_version == 2

    raw = sqlite3.connect(db)
    try:
        row = raw.execute(
            "SELECT event_type, tool_name FROM run_events WHERE event_id=?", ("e1",)
        ).fetchone()
    finally:
        raw.close()
    assert row == ("TOOL_FAILED", "wire_money")  # backfilled from the stored JSON
