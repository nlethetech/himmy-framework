"""Legacy-db migration tests for the graph-memory link table.

A pre-graph ``.himmy/memory.db`` has the ``memories`` table but no ``memory_links``
table. Reopening it through :class:`SqliteMemoryStore` must additively create the link
table in place (no data loss), make link persistence + traversal work, and be
idempotent across repeated opens. Uses ``tmp_path`` for every sqlite artifact.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from himmy.services.memory import (
    MemoryLink,
    MemoryService,
    SqliteMemoryStore,
)
from himmy.services.memory.store import CAUSED_BY
from tests.conftest import run_async


def _make_legacy_db(path: str) -> None:
    """Create a pre-graph db: the original 6-column ``memories`` table, one row, no links."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memories (
            memory_id  TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            kind       TEXT NOT NULL,
            text       TEXT NOT NULL,
            metadata   TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO memories (memory_id, subject_id, kind, text, metadata, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("legacy1", "u", "semantic", "a legacy fact", "{}", "2026-01-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()


def _has_link_table(path: str) -> bool:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_links'"
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def test_legacy_db_gains_link_table_without_data_loss(tmp_path: Path) -> None:
    """Opening a pre-graph db creates memory_links and preserves the existing row."""
    db = str(tmp_path / "memory.db")
    _make_legacy_db(db)
    assert not _has_link_table(db)  # precondition: legacy db has no link table

    store = SqliteMemoryStore(db)
    assert _has_link_table(db)  # migration created it in place
    # The pre-existing row is intact and reads with bi-temporal/tier defaults.
    legacy = store.get("legacy1")
    assert legacy is not None
    assert legacy.text == "a legacy fact"
    assert legacy.tier == "recall"
    assert legacy.valid_from == "2026-01-01T00:00:00Z"
    # Link persistence now works on the migrated db.
    store.save_link(
        MemoryLink(from_memory_id="legacy1", to_memory_id="m2", relation=CAUSED_BY)
    )
    assert store.links_from("legacy1")[0].relation == CAUSED_BY
    store.close()


def test_migration_is_idempotent_across_reopens(tmp_path: Path) -> None:
    """Reopening repeatedly neither errors nor drops links (CREATE IF NOT EXISTS)."""
    db = str(tmp_path / "memory.db")
    _make_legacy_db(db)

    first = SqliteMemoryStore(db)
    saved = first.save_link(MemoryLink(from_memory_id="legacy1", to_memory_id="m2"))
    first.close()

    # Reopen twice more; the link table and its data must survive each migration.
    second = SqliteMemoryStore(db)
    second.close()
    third = SqliteMemoryStore(db)
    out = third.links_from("legacy1")
    assert len(out) == 1
    assert out[0].link_id == saved.link_id
    third.close()


def test_traversal_works_on_migrated_legacy_db(tmp_path: Path) -> None:
    """End-to-end: remember + link + traverse over a migrated legacy db file."""
    db = str(tmp_path / "memory.db")
    _make_legacy_db(db)
    svc = MemoryService(SqliteMemoryStore(db))
    b = svc.remember("a new fact", subject_id="u")
    c = svc.remember("another new fact", subject_id="u")
    svc.link("legacy1", b.memory_id, relation=CAUSED_BY)
    svc.link(b.memory_id, c.memory_id, relation=CAUSED_BY)
    graph = run_async(svc.traverse_graph("legacy1", max_depth=2))
    assert graph.memory_ids() == {"legacy1", b.memory_id, c.memory_id}
    assert graph.edge_count == 2
    assert graph.truncated is False
