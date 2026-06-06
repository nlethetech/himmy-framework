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
import re
import sqlite3
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig

#: How many distinct values to surface per text column in an empty-result hint.
_HINT_VALUES_PER_COLUMN = 15
#: SQLite declared types treated as text (so we hint their distinct values).
_TEXT_AFFINITY = ("CHAR", "CLOB", "TEXT")

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


def _referenced_tables(sql: str) -> list[str]:
    """Best-effort: the table names a query reads from (``FROM`` / ``JOIN``)."""
    return list(
        dict.fromkeys(
            re.findall(r"\b(?:from|join)\s+([A-Za-z_][\w]*)", sql, re.IGNORECASE)
        )
    )


def _sqlite_introspect(path: str) -> dict[str, list[tuple[str, str]]]:
    """Return ``{table: [(column, type), ...]}`` for a SQLite file (read-only)."""
    if path == ":memory:":  # a fresh :memory: connection can't see the other one
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        schema: dict[str, list[tuple[str, str]]] = {}
        for table in tables:
            # PRAGMA is read-only here (no authorizer on this internal connection).
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            schema[table] = [(c[1], str(c[2] or "")) for c in cols]
        return schema
    finally:
        conn.close()


def _where_literals(sql: str) -> list[tuple[str, str]]:
    """Extract ``(column, literal)`` pairs from ``col = 'x'`` / ``col LIKE 'x'``."""
    return re.findall(
        r"([A-Za-z_][\w]*)\s*(?:=|like)\s*'%?([^%']*)%?'", sql, re.IGNORECASE
    )


def _value_hint(path: str, sql: str) -> str | None:
    """For a query that matched nothing, surface the real values + a "did you mean".

    This is what lets a model self-correct ``WHERE name='chicken'`` against a stored
    ``'chickens'``: it sees the actual values, plus an explicit corrected value for any
    near-miss filter literal, instead of guessing again.
    """
    import difflib

    schema = _sqlite_introspect(path)
    if not schema:
        return None
    tables = _referenced_tables(sql) or list(schema)
    literals = {col.lower(): val for col, val in _where_literals(sql)}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        parts: list[str] = []
        suggestions: list[str] = []
        for table in tables:
            for col, decl in schema.get(table, []):
                if not any(t in decl.upper() for t in _TEXT_AFFINITY) and decl:
                    continue  # only hint text-ish columns
                try:
                    rows = conn.execute(
                        f"SELECT DISTINCT {col} FROM {table} "
                        f"LIMIT {_HINT_VALUES_PER_COLUMN + 1}"
                    ).fetchall()
                except sqlite3.Error:
                    continue
                values = [str(r[0]) for r in rows if r[0] is not None]
                if not values:
                    continue
                shown = values[:_HINT_VALUES_PER_COLUMN]
                more = " …" if len(values) > _HINT_VALUES_PER_COLUMN else ""
                parts.append(f"{table}.{col} ∈ {{{', '.join(shown)}}}{more}")
                # If the query filtered this column with a near-miss literal, name the fix.
                guess = literals.get(col.lower())
                if guess and guess not in values:
                    close = difflib.get_close_matches(guess, values, n=1, cutoff=0.5)
                    if close:
                        suggestions.append(
                            f"{col} = '{close[0]}' (you wrote '{guess}')"
                        )
    finally:
        conn.close()
    if not parts:
        return None
    hint = (
        "0 rows — your filter did not match the stored values. "
        "Actual distinct values: " + "; ".join(parts) + "."
    )
    if suggestions:
        hint += (
            " Did you mean: " + "; ".join(suggestions) + "? "
            "Immediately re-run sql_query with the corrected value — do not answer yet."
        )
    else:
        hint += " Re-run sql_query using an exact value from these — do not answer yet."
    return hint


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
        result = {"columns": columns, "rows": rows, "row_count": len(rows)}
    finally:
        conn.close()
    # An empty result on a SELECT is the classic "guessed the wrong literal" failure:
    # surface the real values so the model self-corrects (no extra config needed).
    if not result["rows"] and re.match(r"\s*select\b", sql, re.IGNORECASE):
        hint = _value_hint(path, sql)
        if hint:
            result["hint"] = hint
    return result


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


async def _postgres_schema(dsn: str) -> dict[str, list[tuple[str, str]]]:
    """Introspect a Postgres database's public schema (read-only)."""
    import asyncpg  # type: ignore

    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction(readonly=True):
            records = await conn.fetch(
                "SELECT table_name, column_name, data_type "
                "FROM information_schema.columns WHERE table_schema = 'public' "
                "ORDER BY table_name, ordinal_position"
            )
    finally:
        await conn.close()
    schema: dict[str, list[tuple[str, str]]] = {}
    for r in records:
        schema.setdefault(r["table_name"], []).append(
            (r["column_name"], r["data_type"])
        )
    return schema


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

_SCHEMA_TOOL_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def register_data_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``sql_query`` + ``sql_schema`` bound to the configured database."""

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

    async def sql_schema(_args: dict[str, Any]) -> dict[str, Any]:
        if config.sql_dsn:
            schema = await _postgres_schema(config.sql_dsn)
        elif config.sqlite_path:
            schema = await asyncio.to_thread(_sqlite_introspect, config.sqlite_path)
        else:
            raise ToolSecurityError(
                "no database configured (set HIMMY_SQLITE_PATH or HIMMY_SQL_DSN)"
            )
        return {
            "tables": {
                table: [{"column": c, "type": t} for c, t in cols]
                for table, cols in schema.items()
            }
        }

    register_local_tool(
        registry,
        name="sql_query",
        read_only=True,
        handler=sql_query,
        description=(
            "Run a single read-only SQL query; returns {columns, rows}. If a filter "
            "matches no rows, the result includes a `hint` with the real distinct "
            "values so you can re-run with a correct one. Call sql_schema first if "
            "you don't know the tables/columns."
        ),
        args_json_schema=_SQL_SCHEMA,
        metadata={"pack": "data"},
    )
    register_local_tool(
        registry,
        name="sql_schema",
        read_only=True,
        handler=sql_schema,
        description="Inspect the database: list tables and their columns + types.",
        args_json_schema=_SCHEMA_TOOL_SCHEMA,
        metadata={"pack": "data"},
    )


__all__ = ["register_data_pack"]
