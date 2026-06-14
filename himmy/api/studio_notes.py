"""Studio Notes: durable markdown notes, shared with agents.

A SQLite store at ``.himmy/notes.db`` (cwd-keyed singleton). The same store backs the
``notes`` tool pack (:mod:`himmy.toolkit.notes`), so a note you write in the GUI is one
an agent can read, and vice versa — the "system in the framework".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_updated_idx ON notes (updated_at);
"""


class Note(BaseModel):
    id: str = Field(default_factory=new_uuid)
    title: str = ""
    body: str = ""
    updated_at: str = Field(default_factory=utc_now_iso)


class NotesStore:
    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def list(self) -> list[Note]:
        rows = self._conn.execute(
            "SELECT * FROM notes ORDER BY updated_at DESC"
        ).fetchall()
        return [Note(**dict(r)) for r in rows]

    def get(self, note_id: str) -> Note | None:
        row = self._conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return Note(**dict(row)) if row else None

    def find_by_title(self, title: str) -> Note | None:
        row = self._conn.execute(
            "SELECT * FROM notes WHERE title = ? ORDER BY updated_at DESC LIMIT 1",
            (title,),
        ).fetchone()
        return Note(**dict(row)) if row else None

    def upsert(self, note: Note) -> Note:
        note.updated_at = utc_now_iso()
        self._conn.execute(
            "INSERT OR REPLACE INTO notes (id, title, body, updated_at) VALUES (?,?,?,?)",
            (note.id, note.title, note.body, note.updated_at),
        )
        self._conn.commit()
        return note

    def delete(self, note_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()


_STORE: NotesStore | None = None
_PATH: str | None = None


def notes_db_path() -> str:
    """Path to the notes DB (env override ``HIMMY_NOTES_PATH``, else ``.himmy``)."""
    import os

    env = os.environ.get("HIMMY_NOTES_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "notes.db")


def get_notes_store() -> NotesStore:
    global _STORE, _PATH
    path = notes_db_path()
    if _STORE is None or _PATH != path:
        if _STORE is not None:
            _STORE.close()
        # K2 + K5: route through the one aux-store selector. Postgres DSN -> the K5 mirror
        # (no .himmy/notes.db sidecar); else the durable SQLite store as before. The mirror
        # backs the ``notes`` tool pack identically (find_by_title/upsert/get/list/delete).
        from himmy.services.storage.aux_store_factory import select_aux_store

        def _pg() -> NotesStore:
            from himmy.services.storage.postgres_aux import PostgresNotesStore

            return PostgresNotesStore(tenant="local")  # type: ignore[return-value]

        _STORE = select_aux_store(lambda: NotesStore(path), _pg)
        _PATH = path
    return _STORE


def reset_notes_store() -> None:
    global _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _PATH = None


__all__ = [
    "Note",
    "NotesStore",
    "notes_db_path",
    "get_notes_store",
    "reset_notes_store",
]
