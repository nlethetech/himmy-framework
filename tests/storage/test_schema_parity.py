"""K2: SQLite <-> Postgres schema-parity guard (offline, presence-scoped).

The hand-mirrored migration chains —
:data:`himmy.services.storage.sqlite._MIGRATIONS` (gated by ``PRAGMA user_version``) and
:data:`himmy.services.storage.postgres.STORAGE_MIGRATIONS` (tracked in
``schema_migrations``) — used to be checked only by import-time assertions on DDL string
*contents* (``"ai_call_log" in STORAGE_DDL``), which would not notice a table or column
added to one chain but not the other. This module replaces that with a REAL parity guard:
it builds a fresh live SQLite storage schema and parses the committed Postgres DDL +
migrations, then asserts the two are in parity at the **table/object** level — modulo a
single, frozen, documented allowlist of deliberate divergences — so the next time someone
adds a table to one chain without the other, this test reds.

OFFLINE-LANE SCOPE (reviewer must_fix on honesty): there is NO live Postgres in the unit
lane, so the Postgres side is reconstructed by PARSING the committed DDL/migration strings,
and the comparison is scoped to **table/object PRESENCE** (which tables/views exist on each
side), NOT a full column-by-column structural diff of every table. A full structural diff
would require running both DDLs against a throwaway PostgreSQL + SQLite — that belongs in
the Postgres CI lane (``test_postgres_migrate_advisory_lock`` runs there against a real DB).
The core ``runs`` table is DELIBERATELY divergent (SQLite keeps a ``payload`` JSON blob;
Postgres explodes it into typed columns), so a naive column-equality check would be wrong by
design; the presence-level guard is the honest offline contract and is enumerated below.

The aux-store registry (:data:`_AUX_SQLITE_TABLES`) freezes the SQLite tables every aux
store lays down. Adding a table to an aux store without updating this registry reds the
enumeration test, which is the K3/K4/K5 trip-wire: those items add the matching Postgres
mirror, and this registry is where the new table is declared on the SQLite side.
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path

from himmy.services.storage.postgres import STORAGE_DDL, STORAGE_MIGRATIONS
from himmy.services.storage.sqlite import SqliteStorageService

# --------------------------------------------------------------------------- helpers
_CREATE_TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", re.IGNORECASE)
_CREATE_VIEW_RE = re.compile(
    r"CREATE (?:OR REPLACE )?VIEW\s+(\w+)", re.IGNORECASE
)


def _live_sqlite_core_tables() -> set[str]:
    """Build a fresh durable SQLite storage DB and return its table names."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "storage.db")
        SqliteStorageService(path)  # runs base DDL + the full _MIGRATIONS chain
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        finally:
            conn.close()
    return {r[0] for r in rows}


def _postgres_core_tables() -> set[str]:
    """Parse the committed Postgres base DDL + migrations for declared table names."""
    tables: set[str] = set(_CREATE_TABLE_RE.findall(STORAGE_DDL))
    for _version, _name, statements in STORAGE_MIGRATIONS:
        for stmt in statements:
            tables.update(_CREATE_TABLE_RE.findall(stmt))
    return tables


def _postgres_views() -> set[str]:
    """Parse the committed Postgres DDL for declared view names."""
    return set(_CREATE_VIEW_RE.findall(STORAGE_DDL))


#: The K3/K4 auxiliary-store Postgres mirror tables. They live in the ONE Postgres
#: ``schema_migrations`` ledger (migration v4) but are NAMESPACED (``aux_*``) and so are
#: deliberately absent from the CORE SQLite storage schema — each aux store keeps its OWN
#: per-file SQLite table under a DIFFERENT name (e.g. ``agent_checkpoints`` vs
#: ``aux_agent_checkpoints``). The per-store SQLite<->Postgres mapping is asserted by
#: :func:`test_aux_postgres_mirror_tables_present` below, not by core-table name equality.
_AUX_POSTGRES_TABLES = frozenset(
    {
        "aux_agent_checkpoints",
        "aux_graph_checkpoints",
        "aux_conversations",
        "aux_conversation_messages",
        "aux_projects",
        "aux_routines",
        "aux_teams",
        "aux_workflows",
        # K5: the Studio CRUD sidecars (calendar/cookbook/notes/tasks/notify) + memory.
        "aux_calendar_events",
        "aux_recipes",
        "aux_notes",
        "aux_tasks",
        "aux_notifications",
        "aux_notify_settings",
        "aux_memories",
        "aux_memory_links",
    }
)

#: Tables the Postgres core surface declares that the SQLite core surface deliberately
#: does NOT. Each is an explicit, documented divergence — adding to this set is a
#: deliberate act that requires updating this constant (the whole point of the guard).
#:
#: * ``schema_migrations`` — the Postgres migration ledger. SQLite tracks its migration
#:   version in ``PRAGMA user_version`` instead, so there is no table.
#: * ``evaluation_suites`` — a Postgres-only analytics table; the SQLite store does not
#:   persist evaluation suites (only ``evaluation_runs``).
#: * the ``aux_*`` mirrors — namespaced K3/K4 aux-store tables (mapped per-store below).
_POSTGRES_ONLY_TABLES = (
    frozenset({"schema_migrations", "evaluation_suites"}) | _AUX_POSTGRES_TABLES
)

#: Views the Postgres surface declares that SQLite lacks. ``ai_call_log`` flattens the
#: request/response run-event pair into one analytics row — a Postgres-only convenience.
_POSTGRES_ONLY_VIEWS = frozenset({"ai_call_log"})

#: Tables the SQLite core surface declares that the Postgres core surface deliberately
#: does NOT. None today — the SQLite core is a strict subset of the Postgres core.
_SQLITE_ONLY_TABLES: frozenset[str] = frozenset()


def test_core_table_parity_modulo_documented_divergences() -> None:
    """Core SQLite tables == core Postgres tables, modulo the frozen allowlist.

    Fails when a table is added to one chain but not the other (and not declared as a
    documented divergence) — the enforced version of the hand-mirrored invariant.
    """
    sqlite_tables = _live_sqlite_core_tables()
    postgres_tables = _postgres_core_tables()

    # Postgres has every SQLite core table (SQLite is a subset of Postgres).
    missing_in_pg = sqlite_tables - postgres_tables - _SQLITE_ONLY_TABLES
    assert not missing_in_pg, (
        f"SQLite core tables absent from the Postgres schema: {sorted(missing_in_pg)} "
        "— add the matching Postgres DDL/migration or declare a divergence."
    )

    # SQLite has every Postgres core table EXCEPT the documented Postgres-only set.
    missing_in_sqlite = postgres_tables - sqlite_tables - _POSTGRES_ONLY_TABLES
    assert not missing_in_sqlite, (
        f"Postgres core tables absent from the SQLite schema: {sorted(missing_in_sqlite)} "
        "— add the matching SQLite DDL/migration or extend _POSTGRES_ONLY_TABLES."
    )


def test_documented_divergences_are_exact_not_stale() -> None:
    """The frozen divergence allowlist must match reality EXACTLY (no stale entries).

    If a Postgres-only table is later mirrored into SQLite (so it is no longer a
    divergence), this reds — forcing the allowlist to be trimmed rather than silently
    masking a now-real parity. Symmetric for any future SQLite-only table.
    """
    sqlite_tables = _live_sqlite_core_tables()
    postgres_tables = _postgres_core_tables()

    actual_pg_only = postgres_tables - sqlite_tables
    assert actual_pg_only == _POSTGRES_ONLY_TABLES, (
        "Postgres-only tables drifted from the documented allowlist: "
        f"actual={sorted(actual_pg_only)} documented={sorted(_POSTGRES_ONLY_TABLES)}"
    )

    actual_sqlite_only = sqlite_tables - postgres_tables
    assert actual_sqlite_only == _SQLITE_ONLY_TABLES, (
        "SQLite-only tables drifted from the documented allowlist: "
        f"actual={sorted(actual_sqlite_only)} documented={sorted(_SQLITE_ONLY_TABLES)}"
    )


def test_postgres_only_views_present() -> None:
    """The documented Postgres-only views exist in the committed DDL (no silent drop)."""
    assert _postgres_views() == _POSTGRES_ONLY_VIEWS


# ----------------------------------------------------------------- aux-store registry
#: The SQLite tables each aux store lays down. This is the enumerated per-store parity
#: surface the reviewer required: K3/K4/K5 add the matching Postgres mirror for each, and
#: a store gaining a NEW table without updating this registry reds
#: :func:`test_aux_store_tables_match_registry`. Keyed by the store's logical name.
_AUX_SQLITE_TABLES: dict[str, set[str]] = {
    "approvals": {"agent_checkpoints"},
    "graph_checkpoints": {"graph_checkpoints"},
    "teams": {"teams"},
    "workflows": {"workflows"},
    "routines": {"routines"},
    "conversations": {"conversations", "conversation_messages", "projects"},
    "calendar": {"calendar_events"},
    "cookbook": {"recipes"},
    "notes": {"notes"},
    "tasks": {"tasks"},
    "memory": {"memories", "memory_links"},
}


def _build_aux_store(name: str) -> object:
    """Construct an aux store at ``:memory:`` for schema inspection."""
    if name == "approvals":
        from himmy.api.studio_approvals import SqliteCheckpointStore

        return SqliteCheckpointStore(":memory:")
    if name == "graph_checkpoints":
        from himmy.runtime.checkpoint import SqliteGraphCheckpointStore

        return SqliteGraphCheckpointStore(":memory:")
    if name == "teams":
        from himmy.api.teams_store import TeamsStore

        return TeamsStore(":memory:")
    if name == "workflows":
        from himmy.api.teams_store import WorkflowsStore

        return WorkflowsStore(":memory:")
    if name == "routines":
        from himmy.api.routines import RoutinesStore

        return RoutinesStore(":memory:")
    if name == "conversations":
        from himmy.services.storage.conversations import ConversationStore

        return ConversationStore(":memory:")
    if name == "calendar":
        from himmy.api.studio_calendar import CalendarStore

        return CalendarStore(":memory:")
    if name == "cookbook":
        from himmy.api.studio_cookbook import CookbookStore

        return CookbookStore(":memory:")
    if name == "notes":
        from himmy.api.studio_notes import NotesStore

        return NotesStore(":memory:")
    if name == "tasks":
        from himmy.api.studio_tasks import TasksStore

        return TasksStore(":memory:")
    if name == "memory":
        from himmy.services.memory.store import SqliteMemoryStore

        return SqliteMemoryStore(":memory:")
    raise AssertionError(f"unknown aux store {name!r}")


def _aux_store_tables(store: object) -> set[str]:
    """Return the table names of an aux store via its hardened sqlite3 connection."""
    conn = getattr(store, "_conn", None)
    assert isinstance(conn, sqlite3.Connection), (
        "aux store does not expose a sqlite3 connection on ``_conn``"
    )
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def test_aux_store_tables_match_registry() -> None:
    """Every aux store's live SQLite tables match the frozen per-store registry.

    A store gaining a table (or losing one) without updating ``_AUX_SQLITE_TABLES`` reds
    here — the trip-wire that forces the K3/K4/K5 Postgres mirror author to declare the
    new table on the SQLite side too.
    """
    for name, expected in _AUX_SQLITE_TABLES.items():
        store = _build_aux_store(name)
        try:
            actual = _aux_store_tables(store)
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
        assert actual == expected, (
            f"aux store {name!r} tables drifted: actual={sorted(actual)} "
            f"registry={sorted(expected)} — update _AUX_SQLITE_TABLES (and the Postgres "
            "mirror) in lock-step."
        )


def test_aux_registry_enumerates_every_routed_store() -> None:
    """The registry enumerates every aux store routed through the K2 selector.

    Guards against silently forgetting a store: the eleven stores wired through
    ``select_aux_store`` are exactly the keys of ``_AUX_SQLITE_TABLES``.
    """
    routed = {
        "approvals",
        "graph_checkpoints",
        "teams",
        "workflows",
        "routines",
        "conversations",
        "calendar",
        "cookbook",
        "notes",
        "tasks",
        "memory",
    }
    assert set(_AUX_SQLITE_TABLES) == routed


# ----------------------------------------------------------------- K3/K4 aux PG mirrors
#: Per aux store, the Postgres mirror table(s) the K3/K4 items add to STORAGE_MIGRATIONS,
#: mapped to the SQLite table(s) they mirror. ``None`` = no Postgres mirror in K3/K4 (the
#: K5 stores: calendar/cookbook/notes/tasks/memory). A K3/K4 store with a SQLite table but
#: no listed mirror — or a mirror name absent from the committed Postgres DDL — reds the
#: test below, so a mirror can never silently drift from its SQLite source.
_AUX_PG_MIRROR: dict[str, set[str] | None] = {
    "approvals": {"aux_agent_checkpoints"},
    "graph_checkpoints": {"aux_graph_checkpoints"},
    "conversations": {
        "aux_conversations",
        "aux_conversation_messages",
        "aux_projects",
    },
    "routines": {"aux_routines"},
    "teams": {"aux_teams"},
    "workflows": {"aux_workflows"},
    # K5: the Studio CRUD sidecars + memory now have their Postgres mirrors.
    "calendar": {"aux_calendar_events"},
    "cookbook": {"aux_recipes"},
    "notes": {"aux_notes"},
    "tasks": {"aux_tasks"},
    "memory": {"aux_memories", "aux_memory_links"},
}

#: The K5 notify mirror is NOT a ``select_aux_store``-routed ``*Store`` class — the notify
#: sink (:mod:`himmy.api.routers.studio_notify`) is a module-level deque + best-effort SQL
#: mirror, routed by ``aux_postgres_enabled()`` directly. Its Postgres mirror tables are
#: therefore declared here (not in the per-store ``_AUX_PG_MIRROR`` map) and folded into the
#: documented Postgres-only set below.
_NOTIFY_PG_MIRROR = frozenset({"aux_notifications", "aux_notify_settings"})


def test_aux_pg_mirror_map_enumerates_every_aux_store() -> None:
    """The K3/K4 mirror map covers exactly the aux stores the SQLite registry enumerates."""
    assert set(_AUX_PG_MIRROR) == set(_AUX_SQLITE_TABLES)


def test_aux_postgres_mirror_tables_present() -> None:
    """Every K3/K4 Postgres mirror table is declared in the committed Postgres migrations.

    The K3/K4 trip-wire's Postgres half: a store whose SQLite side is routed into Postgres
    must have its mirror table(s) in ``STORAGE_MIGRATIONS``. Adding a SQLite aux table for a
    K3/K4 store without the matching ``aux_*`` Postgres mirror (or vice-versa) reds here.
    """
    postgres_tables = _postgres_core_tables()
    for name, mirror in _AUX_PG_MIRROR.items():
        if mirror is None:
            continue
        missing = mirror - postgres_tables
        assert not missing, (
            f"aux store {name!r} Postgres mirror tables absent from STORAGE_MIGRATIONS: "
            f"{sorted(missing)}"
        )


def test_aux_pg_mirror_tables_match_documented_pg_only_set() -> None:
    """The union of all aux mirror tables equals the documented ``aux_*`` Postgres-only set.

    Keeps :data:`_AUX_POSTGRES_TABLES` honest: if a mirror table is added to the map but not
    the documented set (or vice-versa) this reds, so the core-table divergence allowlist and
    the per-store mirror map can never drift apart. The notify mirror (not a per-store
    ``select_aux_store`` class) is folded in via :data:`_NOTIFY_PG_MIRROR`.
    """
    declared: set[str] = set(_NOTIFY_PG_MIRROR)
    for mirror in _AUX_PG_MIRROR.values():
        if mirror is not None:
            declared |= mirror
    assert declared == set(_AUX_POSTGRES_TABLES)


def test_notify_pg_mirror_tables_present() -> None:
    """The K5 notify Postgres mirror tables are declared in the committed migrations."""
    postgres_tables = _postgres_core_tables()
    missing = set(_NOTIFY_PG_MIRROR) - postgres_tables
    assert not missing, (
        f"notify Postgres mirror tables absent from STORAGE_MIGRATIONS: {sorted(missing)}"
    )


def test_s3_conversation_subject_id_column_in_lockstep() -> None:
    """S3: the conversation ``subject_id`` linkage column exists on BOTH backends.

    The presence guard above is table-scoped; this asserts the specific S3 column was added
    to the SQLite ConversationStore AND mirrored into the Postgres ``aux_conversations`` table
    via a STORAGE_MIGRATIONS entry (the lockstep the K2 guard requires for a new column).
    """
    from himmy.services.storage.conversations import ConversationStore

    store = ConversationStore(":memory:")
    try:
        cols = {
            r["name"]
            for r in store._conn.execute(
                "PRAGMA table_info(conversations)"
            ).fetchall()
        }
    finally:
        store.close()
    assert "subject_id" in cols

    # The Postgres mirror adds the same column via a migration ALTER (offline string check).
    alters = [
        stmt
        for _v, _n, stmts in STORAGE_MIGRATIONS
        for stmt in stmts
        if "aux_conversations" in stmt and "subject_id" in stmt
    ]
    assert alters, "no STORAGE_MIGRATIONS entry adds aux_conversations.subject_id"
