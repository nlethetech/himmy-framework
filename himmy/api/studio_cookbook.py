"""Studio Cookbook: saved agent + prompt "recipes".

A recipe pairs an agent with a ready-to-run prompt (and notes), so a useful task is
one click away. Durable SQLite store at ``.himmy/cookbook.db`` (cwd-keyed singleton);
running a recipe opens Chat with its agent + prompt prefilled.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recipes (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    agent_path TEXT NOT NULL DEFAULT '',
    prompt     TEXT NOT NULL DEFAULT '',
    notes      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


class Recipe(BaseModel):
    id: str = Field(default_factory=new_uuid)
    name: str
    agent_path: str = ""
    prompt: str = ""
    notes: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class CookbookStore:
    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def list(self) -> list[Recipe]:
        rows = self._conn.execute(
            "SELECT * FROM recipes ORDER BY created_at DESC"
        ).fetchall()
        return [Recipe(**dict(r)) for r in rows]

    def upsert(self, r: Recipe) -> Recipe:
        self._conn.execute(
            "INSERT OR REPLACE INTO recipes "
            "(id, name, agent_path, prompt, notes, created_at) VALUES (?,?,?,?,?,?)",
            (r.id, r.name, r.agent_path, r.prompt, r.notes, r.created_at),
        )
        self._conn.commit()
        return r

    def delete(self, recipe_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()


_STORE: CookbookStore | None = None
_PATH: str | None = None


def _db_path() -> str:
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "cookbook.db")


def get_cookbook_store() -> CookbookStore:
    global _STORE, _PATH
    path = _db_path()
    if _STORE is None or _PATH != path:
        if _STORE is not None:
            _STORE.close()
        # K2 + K5: route through the one aux-store selector. Postgres DSN -> the K5 mirror
        # (no .himmy/cookbook.db sidecar); else the durable SQLite store as before.
        from himmy.services.storage.aux_store_factory import select_aux_store

        def _pg() -> CookbookStore:
            from himmy.services.storage.postgres_aux import PostgresCookbookStore

            return PostgresCookbookStore(tenant="local")  # type: ignore[return-value]

        _STORE = select_aux_store(lambda: CookbookStore(path), _pg)
        _PATH = path
    return _STORE


def reset_cookbook_store() -> None:
    global _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _PATH = None


__all__ = [
    "Recipe",
    "CookbookStore",
    "get_cookbook_store",
    "reset_cookbook_store",
]
