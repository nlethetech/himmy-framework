"""Studio Calendar: a durable event store for the GUI calendar.

A small SQLite store (``.himmy/calendar.db``, cwd-keyed singleton like the run store)
holding dated events (optional time, notes). Surfaced as a month/agenda view in the
GUI; an agent could later read it via a tool, but the store is self-contained.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_events (
    id         TEXT PRIMARY KEY,
    date       TEXT NOT NULL,       -- YYYY-MM-DD
    time       TEXT,                -- HH:MM or NULL (all-day)
    title      TEXT NOT NULL,
    notes      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS calendar_events_date_idx ON calendar_events (date);
"""


class CalendarEvent(BaseModel):
    id: str = Field(default_factory=new_uuid)
    date: str  # YYYY-MM-DD
    time: str | None = None
    title: str
    notes: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class CalendarStore:
    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, ev: CalendarEvent) -> CalendarEvent:
        self._conn.execute(
            "INSERT OR REPLACE INTO calendar_events "
            "(id, date, time, title, notes, created_at) VALUES (?,?,?,?,?,?)",
            (ev.id, ev.date, ev.time, ev.title, ev.notes, ev.created_at),
        )
        self._conn.commit()
        return ev

    def list(self, *, month: str | None = None) -> list[CalendarEvent]:
        """Events, optionally filtered to a ``YYYY-MM`` month; ordered by date+time."""
        if month:
            rows = self._conn.execute(
                "SELECT * FROM calendar_events WHERE date LIKE ? "
                "ORDER BY date, time IS NULL, time",
                (f"{month}-%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM calendar_events ORDER BY date, time IS NULL, time"
            ).fetchall()
        return [CalendarEvent(**dict(r)) for r in rows]

    def delete(self, event_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM calendar_events WHERE id = ?", (event_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()


_STORE: CalendarStore | None = None
_PATH: str | None = None


def _db_path() -> str:
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "calendar.db")


def get_calendar_store() -> CalendarStore:
    global _STORE, _PATH
    path = _db_path()
    if _STORE is None or _PATH != path:
        if _STORE is not None:
            _STORE.close()
        # K2: route the backend choice through the one aux-store selector. The Postgres
        # mirror lands in K5; until then the Postgres builder is None, so this resolves to
        # the durable SQLite store byte-for-byte as before while the choke point is wired.
        from himmy.services.storage.aux_store_factory import select_aux_store

        _STORE = select_aux_store(lambda: CalendarStore(path))
        _PATH = path
    return _STORE


def reset_calendar_store() -> None:
    global _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _PATH = None


__all__ = [
    "CalendarEvent",
    "CalendarStore",
    "get_calendar_store",
    "reset_calendar_store",
]
