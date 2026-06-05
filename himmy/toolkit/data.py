"""Data pack: ``sql_query`` — run a read-only SQL query against SQLite or Postgres.

Read-only is enforced in depth, not by trusting the SQL string:

* a single statement only (no ``SELECT 1; DROP TABLE`` stacking);
* for SQLite, a connection **authorizer** denies every non-read action at the engine
  level (INSERT/UPDATE/DELETE/DDL/PRAGMA-writes are refused even if they slip past the
  string check), and file DBs are opened in ``mode=ro``;
* for Postgres, the query runs inside a ``READ ONLY`` transaction.

The database is chosen from the toolkit config: ``sql_dsn`` (Postgres, needs the
``postgres`` extra) takes precedence, else ``sqlite_path``; with neither set the tool
returns a clear "no database configured" error rather than guessing.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig

# Authorizer actions that are safe for a read-only query.
_ALLOWED_SQLITE_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):  # WITH RECURSIVE (Py 3.11+)
    _ALLOWED_SQLITE_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


def _single_statement(sql: str) -> str:
    """Return the lone statement in ``sql`` or raise if there is more than one."""
    statements = [s for s in sql.split(";") if s.strip()]
    if len(statements) != 1:
        raise ToolSecurityError("exactly one SQL statement is allowed")
    return statements[0].strip()


def _readonly_authorizer(action: int, *_args: Any) -> int:
    """SQLite authorizer: allow read actions, deny everything else."""
    return (
        sqlite3.SQLITE_OK if action in _ALLOWED_SQLITE_ACTIONS else sqlite3.SQLITE_DENY
    )


def _query_sqlite(path: str, sql: str, params: list[Any], limit: int) -> dict[str, Any]:
    """Run a read-only query against a SQLite database file (or ``:memory:``)."""
    if path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.set_authorizer(_readonly_authorizer)
        cursor = conn.execute(sql, params)
        rows = [dict(r) for r in cursor.fetchmany(limit)]
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    finally:
        conn.close()


async def _query_postgres(
    dsn: str, sql: str, params: list[Any], limit: int
) -> dict[str, Any]:
    """Run a read-only query against Postgres inside a READ ONLY transaction."""
    try:
        import asyncpg  # type: ignore
    except Exception as exc:  # pragma: no cover - optional extra missing
        raise ToolSecurityError(
            "sql_dsn (Postgres) needs the 'postgres' extra: pip install 'himmy[postgres]'"
        ) from exc
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction(readonly=True):
            records = await conn.fetch(sql, *params)
        rows = [dict(r) for r in records[:limit]]
        columns = list(rows[0].keys()) if rows else []
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    finally:
        await conn.close()


_SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string", "description": "A single read-only SQL statement."},
        "params": {"type": "array", "description": "Positional query parameters."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
    },
    "required": ["sql"],
    "additionalProperties": False,
}


def register_data_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register the ``sql_query`` tool bound to the configured database."""

    async def sql_query(args: dict[str, Any]) -> dict[str, Any]:
        statement = _single_statement(str(args["sql"]))
        bound = list(args.get("params") or [])
        capped = max(1, min(int(args.get("limit", 100)), 1000))
        if config.sql_dsn:
            return await _query_postgres(config.sql_dsn, statement, bound, capped)
        if config.sqlite_path:
            return await asyncio.to_thread(
                _query_sqlite, config.sqlite_path, statement, bound, capped
            )
        raise ToolSecurityError(
            "no database configured (set HIMMY_SQLITE_PATH or HIMMY_SQL_DSN)"
        )

    register_local_tool(
        registry,
        name="sql_query",
        handler=sql_query,
        description="Run a single read-only SQL query; returns {columns, rows}.",
        args_json_schema=_SQL_SCHEMA,
        metadata={"pack": "data"},
    )


__all__ = ["register_data_pack"]
