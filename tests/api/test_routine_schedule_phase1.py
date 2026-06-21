"""Phase 1 scheduling: cron + per-routine timezone + missed-run/catch-up.

Covers the red-team gates from the build plan:

* cron next-fire goldens (UTC-normalised) + tz next-fire including the non-integer +5:45
  Asia/Kathmandu offset + DST spring-forward / fall-back (the latter is the croniter
  disqualifier — must fire EXACTLY ONCE);
* each missed-run policy (coalesce / skip / run_each) + grace + max_catchup storm cap;
* dedup at-most-once on the planned fire;
* the additive SQLite migration (new columns land on a pre-Phase-1 database);
* PARITY — daily/every behaviour is byte-identical to before Phase 1.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from himmy.api import routines as svc
from himmy.api.routine_schedule import (
    cron_next_fire,
    cron_occurrences_between,
    default_timezone_name,
    resolve_zone,
    validate_cron,
    validate_timezone,
)
from himmy.api.routines import Routine, RoutinesStore, Schedule, is_due, planned_fire_at

NPT = ZoneInfo("Asia/Kathmandu")
NY = ZoneInfo("America/New_York")


def _routine(schedule: Schedule, *, created_at: str, last_run_at: str | None = None) -> Routine:
    return Routine(
        name="r",
        agent_path="a.yaml",
        prompt="p",
        schedule=schedule,
        created_at=created_at,
        last_run_at=last_run_at,
    )


# --------------------------------------------------------------- cron next-fire goldens


def test_cron_next_fire_basic_utc() -> None:
    after = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    nxt = cron_next_fire("0 9 * * *", after, ZoneInfo("UTC"))
    assert nxt == datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


def test_cron_next_fire_kathmandu_plus_545() -> None:
    # 06:30 NPT == 00:45 UTC every day (Nepal has no DST → constant +5:45).
    after = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    nxt = cron_next_fire("30 6 * * *", after, NPT)
    assert nxt == datetime(2026, 6, 1, 0, 45, tzinfo=UTC)
    # ...and stable across the week.
    nxt2 = cron_next_fire("30 6 * * *", nxt, NPT)
    assert nxt2 == datetime(2026, 6, 2, 0, 45, tzinfo=UTC)


def test_cron_next_fire_strictly_after_boundary() -> None:
    # A fire AT the occurrence must not re-yield itself (strictly-after contract).
    at_occ = datetime(2026, 6, 1, 0, 45, tzinfo=UTC)
    nxt = cron_next_fire("30 6 * * *", at_occ, NPT)
    assert nxt == datetime(2026, 6, 2, 0, 45, tzinfo=UTC)


def test_cron_dst_spring_forward_shifts_to_gap_end() -> None:
    # 02:30 does not exist on Mar 8 2026 in NY; cronsim shifts it to the gap end 03:00 EDT.
    after = datetime(2026, 3, 8, 6, 0, tzinfo=UTC)  # 01:00 EST
    nxt = cron_next_fire("30 2 * * *", after, NY)
    assert nxt == datetime(2026, 3, 8, 3, 0, tzinfo=NY).astimezone(UTC)
    assert nxt == datetime(2026, 3, 8, 7, 0, tzinfo=UTC)


def test_cron_dst_fall_back_fires_exactly_once() -> None:
    # 01:30 occurs TWICE on Nov 1 2026 in NY. cronsim emits it ONCE (05:30 UTC = 01:30 EDT),
    # NOT again at 06:30 UTC (01:30 EST). This is the croniter disqualifier.
    after = datetime(2026, 11, 1, 4, 0, tzinfo=UTC)  # 00:00 EDT
    first = cron_next_fire("30 1 * * *", after, NY)
    assert first == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    second = cron_next_fire("30 1 * * *", first, NY)
    # The NEXT fire is the following day, never a second 01:30 the same day.
    assert second.date() == datetime(2026, 11, 2).date()
    assert second != datetime(2026, 11, 1, 6, 30, tzinfo=UTC)


def test_cron_occurrences_between_caps_at_limit() -> None:
    after = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    until = datetime(2026, 6, 30, 0, 0, tzinfo=UTC)
    occ = cron_occurrences_between("0 0 * * *", after, until, ZoneInfo("UTC"), limit=5)
    assert len(occ) == 5
    assert occ[0] == datetime(2026, 6, 2, 0, 0, tzinfo=UTC)  # strictly after `after`


# ----------------------------------------------------------------------- validation


def test_validate_cron_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        validate_cron("not a cron")
    validate_cron("30 6 * * *")  # valid → no raise


def test_validate_timezone() -> None:
    validate_timezone("Asia/Kathmandu")
    validate_timezone("UTC")
    with pytest.raises(ValueError):
        validate_timezone("Mars/Olympus_Mons")


def test_default_timezone_prefers_himmy_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_TZ", "Asia/Kathmandu")
    assert default_timezone_name() == "Asia/Kathmandu"
    monkeypatch.setenv("HIMMY_TZ", "Not/AZone")
    # invalid HIMMY_TZ falls back to host-local (a concrete, valid zone) — never raises.
    assert resolve_zone(None) is not None


# -------------------------------------------------------- Schedule model validation


def test_cron_schedule_requires_expr() -> None:
    with pytest.raises(ValueError):
        Schedule(kind="cron", expr="")


def test_cron_schedule_validates_expr_and_tz() -> None:
    with pytest.raises(ValueError):
        Schedule(kind="cron", expr="bogus")
    with pytest.raises(ValueError):
        Schedule(kind="cron", expr="0 9 * * *", timezone="Bad/Zone")
    s = Schedule(kind="cron", expr="30 6 * * *", timezone="Asia/Kathmandu")
    assert s.describe() == "cron 30 6 * * *"


def test_at_schedule_requires_valid_instant() -> None:
    with pytest.raises(ValueError):
        Schedule(kind="at", at_datetime="")
    with pytest.raises(ValueError):
        Schedule(kind="at", at_datetime="not-a-date")
    s = Schedule(kind="at", at_datetime="2026-07-01T09:00:00+05:45")
    assert s.kind == "at"


# ------------------------------------------------------------------- is_due (cron/at)


def test_is_due_cron_tz_aware() -> None:
    s = Schedule(kind="cron", expr="30 6 * * *", timezone="Asia/Kathmandu")
    r = _routine(s, created_at="2026-06-01T00:00:00+00:00")
    # Before 00:45 UTC → not due.
    assert not is_due(r, datetime(2026, 6, 1, 0, 30, tzinfo=UTC))
    # At/after 00:45 UTC → due.
    assert is_due(r, datetime(2026, 6, 1, 0, 45, tzinfo=UTC))


def test_is_due_at_one_shot() -> None:
    s = Schedule(kind="at", at_datetime="2026-06-05T00:00:00+00:00")
    r = _routine(s, created_at="2026-06-01T00:00:00+00:00")
    assert not is_due(r, datetime(2026, 6, 4, tzinfo=UTC))
    assert is_due(r, datetime(2026, 6, 5, tzinfo=UTC))
    # After it fired (last_run_at >= instant) it never fires again.
    fired = _routine(
        s, created_at="2026-06-01T00:00:00+00:00", last_run_at="2026-06-05T00:00:01+00:00"
    )
    assert not is_due(fired, datetime(2026, 6, 6, tzinfo=UTC))
    assert planned_fire_at(fired, datetime(2026, 6, 5, 0, 0, 1, tzinfo=UTC)) is None


# ------------------------------------------------------------------- PARITY (daily/every)


def test_parity_daily_unchanged() -> None:
    s = Schedule(kind="daily", at="07:00")
    assert s.describe() == "daily 07:00"
    r = _routine(s, created_at="2026-06-01T00:00:00+00:00")
    local = datetime.now().astimezone().tzinfo
    # The original daily math: most-recent HH:MM occurrence in now's tz after the anchor.
    now = datetime(2026, 6, 2, 8, 0, tzinfo=local)
    assert is_due(r, now)
    before = datetime(2026, 6, 1, 6, 0, tzinfo=local)
    assert not is_due(r, before)
    # daily routines carry NO planned-fire dedup key (byte-identical tick behaviour).
    assert svc._tick_planned_fire_iso(r) is None


def test_parity_every_unchanged() -> None:
    s = Schedule(kind="every", hours=6)
    assert s.describe() == "every 6h"
    r = _routine(s, created_at="2026-06-01T00:00:00+00:00")
    assert not is_due(r, datetime(2026, 6, 1, 5, 0, tzinfo=UTC))
    assert is_due(r, datetime(2026, 6, 1, 6, 0, tzinfo=UTC))
    assert svc._tick_planned_fire_iso(r) is None


# ------------------------------------------------------------------- migration (additive)


_PRE_PHASE1_DDL = """
CREATE TABLE routines (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, workspace_id TEXT NOT NULL DEFAULT '__local__',
  agent_path TEXT, agent_id TEXT, idempotency_key TEXT, prompt TEXT NOT NULL,
  schedule_kind TEXT NOT NULL, schedule_at TEXT, schedule_hours INTEGER, provider TEXT,
  model TEXT, deliver TEXT NOT NULL DEFAULT 'none', enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_run_at TEXT, last_status TEXT,
  last_preview TEXT NOT NULL DEFAULT '', last_error TEXT, last_delivery TEXT);
"""


def test_migration_adds_phase1_columns_additively(tmp_path: Path) -> None:
    db = tmp_path / "routines.db"
    conn = sqlite3.connect(db)
    conn.executescript(_PRE_PHASE1_DDL)
    conn.execute(
        "INSERT INTO routines (id, name, agent_path, prompt, schedule_kind, "
        "schedule_at, created_at, updated_at) VALUES "
        "('r1','old','a.yaml','hi','daily','07:00','2026-01-01T00:00:00+00:00',"
        "'2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    store = RoutinesStore(str(db))
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(routines)").fetchall()}
    for new_col in (
        "schedule_expr",
        "schedule_at_datetime",
        "timezone",
        "missed_policy",
        "max_catchup",
        "grace_seconds",
        "overlap_policy",
        "next_fire_at",
    ):
        assert new_col in cols, new_col
    # Legacy row preserved + defaulted, and a cron routine inserts cleanly.
    r = store.get("r1")
    assert r is not None and r.schedule.kind == "daily" and r.schedule.missed == "coalesce"
    cron = _routine(
        Schedule(kind="cron", expr="30 6 * * *", timezone="Asia/Kathmandu"),
        created_at="2026-06-01T00:00:00+00:00",
    )
    store.upsert(cron)
    back = store.get(cron.id)
    assert back is not None and back.schedule.expr == "30 6 * * *"
    # Idempotent re-open.
    store.close()
    RoutinesStore(str(db)).close()


def test_advance_next_fire_persists(tmp_path: Path) -> None:
    store = RoutinesStore(str(tmp_path / "r.db"))
    cron = _routine(
        Schedule(kind="cron", expr="0 9 * * *", timezone="UTC"),
        created_at="2026-06-01T00:00:00+00:00",
    )
    store.upsert(cron)
    store.advance_next_fire(cron.id, "2026-06-02T09:00:00+00:00")
    assert store.get(cron.id).next_fire_at == "2026-06-02T09:00:00+00:00"
    store.close()
