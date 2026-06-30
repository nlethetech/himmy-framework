"""Studio Tasks: a durable task list shared with agents.

SQLite store at ``.himmy/tasks.db`` (cwd-keyed singleton). The ``tasks`` tool pack
(:mod:`himmy.toolkit.tasks_pack`) reads + writes the same store, so an agent can add
or complete tasks the user sees in the GUI, and vice versa.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    due        TEXT,
    created_at TEXT NOT NULL
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive, idempotent column adds so old DBs (and other consumers) keep working.

    SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we guard each ALTER on ``pragma
    table_info`` — adding a column only when it's absent. Every column is nullable / has a
    default, so the shared ``tasks`` tool pack and any existing row keep reading + writing
    unchanged. New planner columns land here so the store never has to be forked.
    """
    have = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    # priority: higher = more urgent (0 = none); drives the app's "what to do next" sort.
    if "priority" not in have:
        conn.execute("ALTER TABLE tasks ADD COLUMN priority INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    # workspace_id: the tenant-isolation column (additive + nullable). A NULL value means
    # "single local tenant / predates tenant binding"; see himmy.api.studio_tenant_scope.
    from himmy.api.studio_tenant_scope import ensure_workspace_column

    ensure_workspace_column(conn, "tasks")


class Task(BaseModel):
    id: str = Field(default_factory=new_uuid)
    title: str
    done: bool = False
    due: str | None = None
    priority: int = 0
    created_at: str = Field(default_factory=utc_now_iso)


class TasksStore:
    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        _migrate(self._conn)

    def _row(self, r: sqlite3.Row) -> Task:
        # ``priority`` is read via .keys() membership so a row from a not-yet-migrated
        # connection (defensive) still yields a valid Task with the default.
        keys = r.keys()
        return Task(
            id=r["id"],
            title=r["title"],
            done=bool(r["done"]),
            due=r["due"],
            priority=int(r["priority"]) if "priority" in keys and r["priority"] is not None else 0,
            created_at=r["created_at"],
        )

    def list(self, *, workspace_id: str | frozenset[str] | None = None) -> list[Task]:
        """Tasks, optionally scoped to ``workspace_id`` (``None`` = ALL, byte-unchanged).

        A tenant-bound caller passes its authorized workspace and sees only its own rows
        (plus legacy ``NULL`` rows); the offline / ``all_tenants`` path passes ``None`` and
        the WHERE clause collapses to nothing — identical to the pre-migration query.
        """
        from himmy.api.studio_tenant_scope import scope_clause

        clause, params = scope_clause(workspace_id)
        where = f"WHERE {clause} " if clause else ""
        rows = self._conn.execute(
            f"SELECT * FROM tasks {where}ORDER BY done, created_at DESC", params
        ).fetchall()
        return [self._row(r) for r in rows]

    def add(
        self,
        title: str,
        *,
        due: str | None = None,
        priority: int = 0,
        workspace_id: str | None = None,
    ) -> Task:
        """Add a task, stamped with ``workspace_id`` (``None`` for the single local tenant)."""
        t = Task(title=title, due=due, priority=priority)
        self._conn.execute(
            "INSERT INTO tasks (id, title, done, due, priority, created_at, workspace_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (t.id, t.title, int(t.done), t.due, t.priority, t.created_at, workspace_id),
        )
        self._conn.commit()
        return t

    def set_done(
        self,
        task_id: str,
        done: bool,
        *,
        workspace_id: str | frozenset[str] | None = None,
    ) -> bool:
        """Mark a task done/undone; a ``workspace_id``-scoped caller cannot touch a foreign row.

        Uses the strict WRITE clause (no ``OR IS NULL``) so a bound tenant can mutate only its
        OWN rows — a legacy NULL-owned row is read-visible but immutable to a bound tenant.
        """
        from himmy.api.studio_tenant_scope import scope_clause_write

        clause, params = scope_clause_write(workspace_id)
        extra = f" AND {clause}" if clause else ""
        cur = self._conn.execute(
            f"UPDATE tasks SET done = ? WHERE id = ?{extra}",
            [int(done), task_id, *params],
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update(
        self,
        task_id: str,
        *,
        due: str | None = None,
        priority: int | None = None,
        done: bool | None = None,
        workspace_id: str | frozenset[str] | None = None,
    ) -> Task | None:
        """Edit a task's due / priority / done in place; returns the updated row or None.

        Only the fields explicitly passed (non-None) are written, so a caller can patch one
        field without clobbering the others. ``due`` cannot be cleared through this method
        (None means "leave as is") — that's deliberate parity with the add() signature.
        """
        from himmy.api.studio_tenant_scope import scope_clause, scope_clause_write

        # WRITE uses the strict clause (no OR IS NULL): a bound tenant may edit only its OWN
        # rows, so a legacy NULL row is never clobbered/re-stamped through this path.
        write_clause, write_params = scope_clause_write(workspace_id)
        write_extra = f" AND {write_clause}" if write_clause else ""
        # READ-back uses the visibility clause so the (already-written) row is returned.
        read_clause, read_params = scope_clause(workspace_id)
        read_extra = f" AND {read_clause}" if read_clause else ""
        sets: list[str] = []
        vals: list[object] = []
        if due is not None:
            sets.append("due = ?")
            vals.append(due)
        if priority is not None:
            sets.append("priority = ?")
            vals.append(int(priority))
        if done is not None:
            sets.append("done = ?")
            vals.append(int(done))
        if sets:
            vals.append(task_id)
            vals.extend(write_params)
            cur = self._conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?{write_extra}", vals
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        row = self._conn.execute(
            f"SELECT * FROM tasks WHERE id = ?{read_extra}", [task_id, *read_params]
        ).fetchone()
        return self._row(row) if row is not None else None

    def complete_by_title(
        self, title: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        from himmy.api.studio_tenant_scope import scope_clause_write

        clause, params = scope_clause_write(workspace_id)
        extra = f" AND {clause}" if clause else ""
        cur = self._conn.execute(
            f"UPDATE tasks SET done = 1 WHERE title = ? AND done = 0{extra}",
            [title, *params],
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(
        self, task_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        """Delete a task; a ``workspace_id``-scoped caller cannot delete a foreign row.

        Strict WRITE clause (no ``OR IS NULL``): a bound tenant can delete only its OWN rows;
        a legacy NULL-owned row is read-visible but undeletable by a bound tenant.
        """
        from himmy.api.studio_tenant_scope import scope_clause_write

        clause, params = scope_clause_write(workspace_id)
        extra = f" AND {clause}" if clause else ""
        cur = self._conn.execute(
            f"DELETE FROM tasks WHERE id = ?{extra}", [task_id, *params]
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()


_STORE: TasksStore | None = None
_PATH: str | None = None


def tasks_db_path() -> str:
    import os

    env = os.environ.get("HIMMY_TASKS_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "tasks.db")


def get_tasks_store() -> TasksStore:
    global _STORE, _PATH
    path = tasks_db_path()
    if _STORE is None or _PATH != path:
        if _STORE is not None:
            _STORE.close()
        # K2 + K5: route through the one aux-store selector. Postgres DSN -> the K5 mirror
        # (no .himmy/tasks.db sidecar); else the durable SQLite store as before. The mirror
        # backs the ``tasks`` tool pack identically (add/complete_by_title/list/...).
        from himmy.services.storage.aux_store_factory import select_aux_store

        def _pg() -> TasksStore:
            from himmy.services.storage.postgres_aux import PostgresTasksStore

            return PostgresTasksStore(tenant="local")  # type: ignore[return-value]

        _STORE = select_aux_store(lambda: TasksStore(path), _pg)
        _PATH = path
    return _STORE


def reset_tasks_store() -> None:
    global _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _PATH = None


__all__ = [
    "Task",
    "TasksStore",
    "tasks_db_path",
    "get_tasks_store",
    "reset_tasks_store",
]
