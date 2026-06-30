"""Shared tenant/subject stamping for the cwd-keyed singleton Studio SQLite stores.

The Studio task/note/calendar/cookbook (and the notify ring) stores were born
single-user-local: one cwd-keyed ``.himmy/<name>.db`` with NO tenant column, so once an
authenticator binds a caller to a workspace a tenant-bound Studio key could read + write
ACROSS tenants (the IDOR/BOLA recurrence class the RBAC pass closes everywhere else). The
fix is a DATA-MODEL change — a ``workspace_id`` column on each store — but it must land
WITHOUT disturbing the offline / zero-config / single-box path:

* the migration is **additive + nullable** (``ALTER TABLE ADD COLUMN`` guarded on
  ``PRAGMA table_info``), so an existing single-box DB with no ``workspace_id`` column
  opens, reads, and writes byte-unchanged — pre-existing rows simply carry ``NULL``;
* every read filter and every write stamp is **opt-in**: the scope is ``None`` for an
  unrestricted principal (offline / ``all_tenants`` / a trusted shared key), and a ``None``
  scope means "no filtering, show + stamp everything exactly as before". So the shared tool
  packs (``tasks``/``notes`` packs) that call ``store.list()`` / ``store.add(...)`` with NO
  workspace argument keep compiling and behaving identically;
* the SELECT predicate for a bound principal additionally matches legacy ``NULL`` rows
  (``workspace_id IS NULL``) so a pre-migration single-box row stays visible after the
  column lands — the migration never hides data that predates tenant binding.

This is the singleton-SQLite analogue of the canonical run store's tenant axis
(:func:`himmy.api.auth.context.studio_tenant_filter`); the routers derive the scope from
the verified principal (never the request body) and thread it through these helpers, so the
scope is enforced by construction and the coverage gate
(:mod:`tests.api.test_tenant_scope_coverage`) can DRAIN each store from the time-boxed
``_STUDIO_TENANT_PENDING`` allow-list.
"""

from __future__ import annotations

import sqlite3


def ensure_workspace_column(conn: sqlite3.Connection, table: str) -> None:
    """Idempotently add a nullable ``workspace_id`` column to ``table`` (additive migration).

    SQLite has no ``ADD COLUMN IF NOT EXISTS``, so the add is guarded on ``PRAGMA
    table_info``: the column is created only when absent, and it is nullable with no default
    so every pre-existing row keeps its value (``NULL`` = "belongs to the single local
    tenant / predates tenant binding"). Safe to call on every store open — an already
    migrated DB is a no-op, and an old single-box DB gains the column without rewriting a
    single row. A companion index keeps the scoped ``WHERE workspace_id ...`` reads cheap.
    """
    have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if "workspace_id" not in have:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN workspace_id TEXT")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_workspace_idx "
            f"ON {table} (workspace_id)"
        )
        conn.commit()


def scope_clause(
    workspace_id: str | frozenset[str] | None, *, column: str = "workspace_id"
) -> tuple[str, list[str]]:
    """Return ``(sql_fragment, params)`` constraining a query to ``workspace_id`` (or ALL).

    * ``workspace_id is None`` (unrestricted / offline / ``all_tenants`` principal) yields
      an EMPTY fragment + no params — "no filtering", so the single-box read is
      byte-unchanged; callers splice it as a no-op.
    * a concrete ``workspace_id`` (a tenant-bound principal's authorized workspace) yields
      ``(column = ? OR column IS NULL)`` so the caller sees its OWN rows AND any legacy
      pre-migration row (``NULL`` predates tenant binding and belongs to no tenant, mirroring
      :func:`himmy.api.auth.context.authorize_studio_object`).
    * a ``frozenset`` of workspaces (a principal bound to several tenants — the LIST path of
      :func:`himmy.api.auth.context.studio_tenant_filter`) yields
      ``(column IN (?, ...) OR column IS NULL)``. An EMPTY set is treated as "nothing
      visible except legacy NULL rows" (the principal binds no tenant), never as "ALL".

    The fragment carries no leading ``AND``/``WHERE`` — the caller composes it (it is always
    spliced as the sole or an ANDed predicate by the helpers below).
    """
    if workspace_id is None:
        return "", []
    if isinstance(workspace_id, str):
        return f"({column} = ? OR {column} IS NULL)", [workspace_id]
    ids = sorted(workspace_id)
    if not ids:
        return f"{column} IS NULL", []
    placeholders = ", ".join("?" for _ in ids)
    return f"({column} IN ({placeholders}) OR {column} IS NULL)", list(ids)


def scope_clause_write(
    workspace_id: str | frozenset[str] | None, *, column: str = "workspace_id"
) -> tuple[str, list[str]]:
    """Like :func:`scope_clause` but for MUTATIONS — does NOT match legacy ``NULL`` rows.

    The read-side :func:`scope_clause` deliberately matches ``workspace_id IS NULL`` so a
    bound tenant can still SEE pre-migration / offline-written rows that predate tenant
    binding. Applying that same ``OR col IS NULL`` predicate to ``UPDATE``/``DELETE``/upsert
    paths, however, is a cross-tenant WRITE primitive: a tenant-bound principal could
    complete/edit/delete (and, via ``INSERT OR REPLACE`` upsert, re-stamp/steal) any
    NULL-owned legacy row it does not own. Those shared rows must be READ-visible but
    IMMUTABLE to a bound tenant, so the write predicate is the strict subset:

    * ``workspace_id is None`` (unrestricted / offline / ``all_tenants``) yields an EMPTY
      fragment — "no filtering", so the single-box write is byte-unchanged and the shared
      tool packs (which pass no workspace) keep writing exactly as before;
    * a concrete ``workspace_id`` yields ``column = ?`` (NO ``OR IS NULL``) so a bound
      tenant can mutate ONLY its OWN stamped rows — never a legacy NULL row, never a foreign
      tenant's row;
    * a ``frozenset`` yields ``column IN (?, ...)`` (NO ``OR IS NULL``); an EMPTY set yields
      ``1 = 0`` (a bound principal with no tenant may mutate nothing — fail-closed).

    The fragment carries no leading ``AND``/``WHERE`` — the caller composes it.
    """
    if workspace_id is None:
        return "", []
    if isinstance(workspace_id, str):
        return f"{column} = ?", [workspace_id]
    ids = sorted(workspace_id)
    if not ids:
        return "1 = 0", []
    placeholders = ", ".join("?" for _ in ids)
    return f"{column} IN ({placeholders})", list(ids)


def row_writable_in_scope(row_workspace: str | None, workspace_id: str | None) -> bool:
    """Whether a row stamped ``row_workspace`` may be MUTATED by a ``workspace_id`` writer.

    The write-side companion to :func:`row_in_scope`. ``workspace_id is None`` (unrestricted
    / offline) is always True (byte-unchanged). A bound writer may mutate ONLY rows stamped
    with its EXACT workspace — never a legacy ``NULL`` row (read-visible but immutable) and
    never a foreign tenant's. Used by by-id upsert guards to fold a re-stamp/steal attempt to
    404 (or to preserve an existing row's owner instead of capturing it).
    """
    if workspace_id is None:
        return True
    return row_workspace == workspace_id


def row_in_scope(row_workspace: str | None, workspace_id: str | None) -> bool:
    """Whether a row stamped ``row_workspace`` is visible to a ``workspace_id``-scoped reader.

    The Python-side companion to :func:`scope_clause` for by-id reads / mutations that fetch
    a row then decide visibility. ``workspace_id is None`` (unrestricted) is always True
    (offline byte-unchanged); a bound reader sees its own workspace's rows AND legacy
    ``NULL`` rows. A False verdict is folded to 404 by callers so existence never leaks.
    """
    if workspace_id is None:
        return True
    return row_workspace is None or row_workspace == workspace_id


__all__ = [
    "ensure_workspace_column",
    "scope_clause",
    "scope_clause_write",
    "row_in_scope",
    "row_writable_in_scope",
]
