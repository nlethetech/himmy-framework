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


class Task(BaseModel):
    id: str = Field(default_factory=new_uuid)
    title: str
    done: bool = False
    due: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class TasksStore:
    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def list(self) -> list[Task]:
        rows = self._conn.execute(
            "SELECT * FROM tasks ORDER BY done, created_at DESC"
        ).fetchall()
        return [
            Task(
                id=r["id"],
                title=r["title"],
                done=bool(r["done"]),
                due=r["due"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def add(self, title: str, *, due: str | None = None) -> Task:
        t = Task(title=title, due=due)
        self._conn.execute(
            "INSERT INTO tasks (id, title, done, due, created_at) VALUES (?,?,?,?,?)",
            (t.id, t.title, int(t.done), t.due, t.created_at),
        )
        self._conn.commit()
        return t

    def set_done(self, task_id: str, done: bool) -> bool:
        cur = self._conn.execute(
            "UPDATE tasks SET done = ? WHERE id = ?", (int(done), task_id)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def complete_by_title(self, title: str) -> bool:
        cur = self._conn.execute(
            "UPDATE tasks SET done = 1 WHERE title = ? AND done = 0", (title,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, task_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
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
