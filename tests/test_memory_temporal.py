"""Bi-temporal memory tests: validity windows, as-of recall, active-only, tiers.

These exercise the Graphiti-style "invalidate, don't delete" semantics layered onto
the EntityRecord-backed memory store, plus the tiered (Letta) recall scoping.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from himmy.core.ids import utc_now_iso
from himmy.services.memory import (
    InMemoryMemoryStore,
    MemoryContextAdapter,
    MemoryRecord,
    MemoryService,
    SqliteMemoryStore,
)
from himmy.services.memory.temporal import is_valid_at
from tests.conftest import run_async


def test_invalidate_sets_valid_to_and_drops_from_active_recall() -> None:
    """A fact, once invalidated, leaves active recall but stays in the full list."""
    svc = MemoryService(InMemoryMemoryStore())
    rec = svc.remember("the user lives in Mumbai", subject_id="u")

    # Default recall (no active filter) still sees the fact.
    before = run_async(svc.recall("user lives", subject_id="u"))
    assert len(before) == 1

    svc.invalidate(rec.memory_id, valid_to=utc_now_iso())

    # active_only recall now excludes it...
    active = run_async(svc.recall("user lives", subject_id="u", active_only=True))
    assert active == []
    # ...but the historical (default) recall still returns it (not deleted).
    still = run_async(svc.recall("user lives", subject_id="u"))
    assert len(still) == 1
    assert still[0].record.valid_to is not None


def test_as_of_point_in_time_recall_returns_value_true_then() -> None:
    """as_of recall answers 'what was true at instant T' across a supersession."""
    svc = MemoryService(InMemoryMemoryStore())
    old = svc.remember("home city is Mumbai", subject_id="u", stable_key="u/home")
    moment = utc_now_iso()
    time.sleep(0.01)

    # Supersede the fact: invalidate the old, add the new (same stable_key).
    svc.invalidate(old.memory_id, valid_to=utc_now_iso())
    svc.remember("home city is Bangalore", subject_id="u", stable_key="u/home")
    now = utc_now_iso()

    past = run_async(svc.recall("home city", subject_id="u", as_of=moment))
    assert [h.record.text for h in past] == ["home city is Mumbai"]

    present = run_async(
        svc.recall("home city", subject_id="u", as_of=now, active_only=True)
    )
    assert [h.record.text for h in present] == ["home city is Bangalore"]


def test_is_valid_at_half_open_interval() -> None:
    """Validity is the half-open interval [valid_from, valid_to)."""
    rec = MemoryRecord(
        text="x",
        valid_from="2024-01-01T00:00:00Z",
        valid_to="2024-06-01T00:00:00Z",
    )
    assert is_valid_at(rec, "2024-01-01T00:00:00Z")  # inclusive start
    assert is_valid_at(rec, "2024-03-01T00:00:00Z")  # inside
    assert not is_valid_at(rec, "2024-06-01T00:00:00Z")  # exclusive end
    assert not is_valid_at(rec, "2023-12-31T23:59:59Z")  # before start
    open_rec = MemoryRecord(text="y", valid_from="2024-01-01T00:00:00Z")
    assert is_valid_at(open_rec, "2099-01-01T00:00:00Z")  # None valid_to = forever


def test_tier_scoped_recall() -> None:
    """recall(tier=...) only returns facts in that Letta tier."""
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("core profile fact", subject_id="u", tier="core")
    svc.remember("a recall-tier fact", subject_id="u", tier="recall")
    svc.remember("cold archival note", subject_id="u", tier="archival")

    core = run_async(svc.recall("fact", subject_id="u", tier="core"))
    assert [h.record.text for h in core] == ["core profile fact"]
    archival = run_async(svc.recall("fact", subject_id="u", tier="archival"))
    assert [h.record.text for h in archival] == ["cold archival note"]


def test_promote_moves_tier_and_is_idempotent() -> None:
    """promote() moves a fact to a new tier; promoting to the same tier is a no-op."""
    svc = MemoryService(InMemoryMemoryStore())
    rec = svc.remember("fact to promote", subject_id="u", tier="recall")
    moved = svc.promote(rec.memory_id, "core")
    assert moved is not None
    assert moved.tier == "core"
    # The store now reflects the new tier.
    assert svc.get(rec.memory_id).tier == "core"  # type: ignore[union-attr]
    # Promoting to the same tier returns the record unchanged.
    same = svc.promote(rec.memory_id, "core")
    assert same is not None and same.tier == "core"


def test_sqlite_bitemporal_columns_persist(tmp_path: Path) -> None:
    """Bi-temporal fields round-trip through the durable SQLite store."""
    db = str(tmp_path / "mem.db")
    store = SqliteMemoryStore(db)
    svc = MemoryService(store)
    rec = svc.remember(
        "durable fact",
        subject_id="u",
        tier="archival",
        source="imported",
        stable_key="u/durable",
    )
    svc.invalidate(rec.memory_id, valid_to="2025-01-01T00:00:00Z")
    store.close()

    reopened = SqliteMemoryStore(db)
    got = reopened.get(rec.memory_id)
    reopened.close()
    assert got is not None
    assert got.tier == "archival"
    assert got.source == "imported"
    assert got.stable_key == "u/durable"
    assert got.valid_to == "2025-01-01T00:00:00Z"


def test_context_adapter_core_always_injected_archival_excluded() -> None:
    """The adapter injects core (always) + recall (thresholded), never archival.

    Archival is cold storage reached only via an explicit recall tool — so a core
    fact is injected even on an off-topic query, while an archival fact is not, yet
    the archival fact IS returned by an explicit tier-scoped recall.
    """
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("user is allergic to peanuts", subject_id="u", tier="core")
    svc.remember("cold archived trivia about comets", subject_id="u", tier="archival")
    adapter = MemoryContextAdapter(
        svc, tiers=("core", "recall"), similarity_threshold=0.5
    )

    # An off-topic query: the core fact is still injected (always-on working set),
    # the archival fact is not in the adapter's tier set at all.
    field = run_async(
        adapter.fetch("memory", {"subject_id": "u", "query": "quantum chromodynamics"})
    )
    assert field is not None
    texts = [m["text"] for m in field.value["memories"]]
    assert "user is allergic to peanuts" in texts
    assert "cold archived trivia about comets" not in texts

    # But the archival fact IS reachable via an explicit tier-scoped recall.
    archival = run_async(svc.recall("comets", subject_id="u", tier="archival"))
    assert [h.record.text for h in archival] == ["cold archived trivia about comets"]


def test_legacy_db_migrates_in_place(tmp_path: Path) -> None:
    """A pre-bitemporal 6-column memory.db upgrades in place with defaulted columns."""
    db = str(tmp_path / "legacy.db")
    raw = sqlite3.connect(db)
    raw.executescript(
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
    raw.execute(
        "INSERT INTO memories VALUES (?,?,?,?,?,?)",
        ("m1", "u", "semantic", "old fact", "{}", "2024-01-01T00:00:00Z"),
    )
    raw.commit()
    raw.close()

    store = SqliteMemoryStore(db)
    columns = {
        row["name"]
        for row in store._conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    # The migration added every new bi-temporal/tier/provenance column.
    assert {"tier", "valid_from", "valid_to", "superseded_by", "source"} <= columns

    # The legacy row reads back with sensible defaults and a backfilled valid_from.
    row = store.get("m1")
    store.close()
    assert row is not None
    assert row.text == "old fact"
    assert row.tier == "recall"
    assert row.source == "user"
    assert row.valid_from == "2024-01-01T00:00:00Z"  # backfilled from created_at
    assert row.valid_to is None  # legacy rows are active
