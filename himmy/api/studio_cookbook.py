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
        from himmy.api.studio_tenant_scope import ensure_workspace_column
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Additive + nullable tenant-isolation column; old single-box DBs open unchanged.
        ensure_workspace_column(self._conn, "recipes")

    def _row(self, r: sqlite3.Row) -> Recipe:
        data = {k: r[k] for k in r.keys() if k != "workspace_id"}
        return Recipe(**data)

    def list(self, *, workspace_id: str | frozenset[str] | None = None) -> list[Recipe]:
        """Recipes, optionally scoped to ``workspace_id`` (``None`` = ALL, byte-unchanged)."""
        from himmy.api.studio_tenant_scope import scope_clause

        clause, params = scope_clause(workspace_id)
        where = f"WHERE {clause} " if clause else ""
        rows = self._conn.execute(
            f"SELECT * FROM recipes {where}ORDER BY created_at DESC", params
        ).fetchall()
        return [self._row(r) for r in rows]

    def get(
        self, recipe_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> Recipe | None:
        """A recipe by id; a ``workspace_id``-scoped caller cannot read a foreign one (None)."""
        from himmy.api.studio_tenant_scope import scope_clause

        clause, params = scope_clause(workspace_id)
        extra = f" AND {clause}" if clause else ""
        row = self._conn.execute(
            f"SELECT * FROM recipes WHERE id = ?{extra}", [recipe_id, *params]
        ).fetchone()
        return self._row(row) if row else None

    def upsert(self, r: Recipe, *, workspace_id: str | None = None) -> Recipe:
        """Upsert a recipe, stamping ``workspace_id`` (``None`` for the single local tenant).

        Re-stamp guard: a bound tenant upserting onto an id owned by another tenant (or a
        legacy NULL row) PRESERVES the existing owner rather than capturing/clobbering it —
        the ``INSERT OR REPLACE`` keyed only on id would otherwise let a bound tenant steal a
        shared/legacy recipe by re-stamping it. ``workspace_id is None`` (offline) is unchanged.
        """
        stamp = workspace_id
        if workspace_id is not None:
            from himmy.api.studio_tenant_scope import row_writable_in_scope

            existing = self._conn.execute(
                "SELECT workspace_id FROM recipes WHERE id = ?", (r.id,)
            ).fetchone()
            if existing is not None and not row_writable_in_scope(
                existing["workspace_id"], workspace_id
            ):
                stamp = existing["workspace_id"]
        self._conn.execute(
            "INSERT OR REPLACE INTO recipes "
            "(id, name, agent_path, prompt, notes, created_at, workspace_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (r.id, r.name, r.agent_path, r.prompt, r.notes, r.created_at, stamp),
        )
        self._conn.commit()
        return r

    def delete(
        self, recipe_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        """Delete a recipe; a ``workspace_id``-scoped caller cannot delete a foreign one.

        Strict WRITE clause (no ``OR IS NULL``): a legacy NULL-owned recipe is read-visible
        but undeletable by a bound tenant.
        """
        from himmy.api.studio_tenant_scope import scope_clause_write

        clause, params = scope_clause_write(workspace_id)
        extra = f" AND {clause}" if clause else ""
        cur = self._conn.execute(
            f"DELETE FROM recipes WHERE id = ?{extra}", [recipe_id, *params]
        )
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
