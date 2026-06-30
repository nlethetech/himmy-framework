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
        from himmy.api.studio_tenant_scope import ensure_workspace_column
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Additive + nullable tenant-isolation column; old single-box DBs open unchanged.
        ensure_workspace_column(self._conn, "notes")

    def _row(self, r: sqlite3.Row) -> Note:
        # workspace_id is not a Note field — strip it before constructing the model so a
        # migrated row (which carries the extra column) still validates.
        data = {k: r[k] for k in r.keys() if k != "workspace_id"}
        return Note(**data)

    def list(self, *, workspace_id: str | frozenset[str] | None = None) -> list[Note]:
        """Notes, optionally scoped to ``workspace_id`` (``None`` = ALL, byte-unchanged)."""
        from himmy.api.studio_tenant_scope import scope_clause

        clause, params = scope_clause(workspace_id)
        where = f"WHERE {clause} " if clause else ""
        rows = self._conn.execute(
            f"SELECT * FROM notes {where}ORDER BY updated_at DESC", params
        ).fetchall()
        return [self._row(r) for r in rows]

    def get(
        self, note_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> Note | None:
        """A note by id; a ``workspace_id``-scoped caller cannot read a foreign note (None)."""
        from himmy.api.studio_tenant_scope import scope_clause

        clause, params = scope_clause(workspace_id)
        extra = f" AND {clause}" if clause else ""
        row = self._conn.execute(
            f"SELECT * FROM notes WHERE id = ?{extra}", [note_id, *params]
        ).fetchone()
        return self._row(row) if row else None

    def find_by_title(
        self,
        title: str,
        *,
        workspace_id: str | frozenset[str] | None = None,
        for_write: bool = False,
    ) -> Note | None:
        """Find the newest note with ``title`` visible to ``workspace_id``.

        ``for_write=True`` switches to the STRICT write clause (no ``OR IS NULL``) so a
        title-keyed upsert by a bound tenant only ever matches its OWN note — never a legacy
        NULL-owned note. Without this a bound tenant's ``write_note(title=...)`` would reuse a
        shared/legacy note's id and overwrite its body (a cross-tenant write); ``for_write``
        makes it create a fresh tenant-owned note instead.
        """
        from himmy.api.studio_tenant_scope import scope_clause, scope_clause_write

        clause, params = (
            scope_clause_write(workspace_id) if for_write else scope_clause(workspace_id)
        )
        extra = f" AND {clause}" if clause else ""
        row = self._conn.execute(
            f"SELECT * FROM notes WHERE title = ?{extra} "
            "ORDER BY updated_at DESC LIMIT 1",
            [title, *params],
        ).fetchone()
        return self._row(row) if row else None

    def upsert(self, note: Note, *, workspace_id: str | None = None) -> Note:
        """Upsert a note, stamping ``workspace_id`` (``None`` for the single local tenant).

        Re-stamp / content-tamper guard: when a bound tenant (``workspace_id`` not None)
        upserts onto an id that already exists but is NOT writable by that tenant (a legacy
        NULL row or a foreign tenant's row), the upsert is ABORTED — the existing row is
        returned UNCHANGED. ``INSERT OR REPLACE`` keyed only on id would otherwise let a bound
        tenant not only re-stamp/steal but also clobber the *content* (title/body) of a
        shared/legacy note it must not be able to mutate (read-visible but IMMUTABLE).
        ``workspace_id is None`` (offline / shared key) is unchanged: every upsert stamps NULL
        and writes content exactly as before.
        """
        note.updated_at = utc_now_iso()
        stamp = workspace_id
        if workspace_id is not None:
            from himmy.api.studio_tenant_scope import row_writable_in_scope

            existing = self._conn.execute(
                "SELECT * FROM notes WHERE id = ?", (note.id,)
            ).fetchone()
            if existing is not None and not row_writable_in_scope(
                existing["workspace_id"], workspace_id
            ):
                # A bound tenant may not mutate a legacy/foreign-owned row at all: abort the
                # content write and return the row as it stands (immutable to this caller).
                return self._row(existing)
        self._conn.execute(
            "INSERT OR REPLACE INTO notes (id, title, body, updated_at, workspace_id) "
            "VALUES (?,?,?,?,?)",
            (note.id, note.title, note.body, note.updated_at, stamp),
        )
        self._conn.commit()
        return note

    def delete(
        self, note_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        """Delete a note; a ``workspace_id``-scoped caller cannot delete a foreign note.

        Strict WRITE clause (no ``OR IS NULL``): a legacy NULL-owned note is read-visible but
        undeletable by a bound tenant.
        """
        from himmy.api.studio_tenant_scope import scope_clause_write

        clause, params = scope_clause_write(workspace_id)
        extra = f" AND {clause}" if clause else ""
        cur = self._conn.execute(
            f"DELETE FROM notes WHERE id = ?{extra}", [note_id, *params]
        )
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
