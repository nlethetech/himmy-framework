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
        from himmy.api.studio_tenant_scope import ensure_workspace_column
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Additive + nullable tenant-isolation column; old single-box DBs open unchanged.
        ensure_workspace_column(self._conn, "calendar_events")

    def _row(self, r: sqlite3.Row) -> CalendarEvent:
        data = {k: r[k] for k in r.keys() if k != "workspace_id"}
        return CalendarEvent(**data)

    def add(self, ev: CalendarEvent, *, workspace_id: str | None = None) -> CalendarEvent:
        """Add an event, stamped with ``workspace_id`` (``None`` for the single local tenant)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO calendar_events "
            "(id, date, time, title, notes, created_at, workspace_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (ev.id, ev.date, ev.time, ev.title, ev.notes, ev.created_at, workspace_id),
        )
        self._conn.commit()
        return ev

    def list(
        self,
        *,
        month: str | None = None,
        workspace_id: str | frozenset[str] | None = None,
    ) -> list[CalendarEvent]:
        """Events, optionally filtered to a ``YYYY-MM`` month + scoped to ``workspace_id``.

        ``workspace_id is None`` (offline / ``all_tenants``) lists every event exactly as
        before; a tenant-bound value restricts the rows to that workspace (plus legacy NULL).
        """
        from himmy.api.studio_tenant_scope import scope_clause

        preds: list[str] = []
        params: list[object] = []
        if month:
            preds.append("date LIKE ?")
            params.append(f"{month}-%")
        clause, scope_params = scope_clause(workspace_id)
        if clause:
            preds.append(clause)
            params.extend(scope_params)
        where = f"WHERE {' AND '.join(preds)} " if preds else ""
        rows = self._conn.execute(
            f"SELECT * FROM calendar_events {where}ORDER BY date, time IS NULL, time",
            params,
        ).fetchall()
        return [self._row(r) for r in rows]

    def delete(
        self, event_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        """Delete an event; a ``workspace_id``-scoped caller cannot delete a foreign event."""
        from himmy.api.studio_tenant_scope import scope_clause

        clause, params = scope_clause(workspace_id)
        extra = f" AND {clause}" if clause else ""
        cur = self._conn.execute(
            f"DELETE FROM calendar_events WHERE id = ?{extra}", [event_id, *params]
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
        # K2 + K5: route the backend choice through the one aux-store selector. With a
        # Postgres DSN this resolves the K5 Postgres mirror (no .himmy/calendar.db sidecar);
        # otherwise the durable SQLite store byte-for-byte as before.
        from himmy.services.storage.aux_store_factory import select_aux_store

        def _pg() -> CalendarStore:
            from himmy.services.storage.postgres_aux import PostgresCalendarStore

            return PostgresCalendarStore(tenant="local")  # type: ignore[return-value]

        _STORE = select_aux_store(lambda: CalendarStore(path), _pg)
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
