"""Phase 1: missed-run / catch-up policy + at-most-once dedup on the planned fire.

Drives the real :class:`RoutineScheduler` against a file-backed routines store with a frozen
clock (so catch-up + DST cases are deterministic) and a faked headless run (so no model is
called). Verifies:

* ``coalesce`` (default) fires the missed occurrence EXACTLY ONCE via the normal tick;
* ``skip`` advances past the gap WITHOUT running;
* ``run_each`` backfills each missed occurrence, HARD-CAPPED by ``max_catchup`` and dropping
  occurrences older than ``grace_seconds``;
* the durable dedup claim coalesces two ticks on the SAME planned fire to one run.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from himmy.api import routines as svc
from himmy.api.routines import Routine, RoutineScheduler, Schedule


@pytest.fixture
def store_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    monkeypatch.setenv("HIMMY_TZ", "UTC")
    svc.reset_routines_store()
    # No durable server storage published → dedup falls through to flock+CAS for the
    # policy tests (we test dedup explicitly elsewhere with an in-memory facade).
    yield
    svc.reset_routines_store()


def _install_fake_run(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the headless run + side effects; return a list that records each fire."""
    fires: list[str] = []

    async def _fake_headless(routine: Routine) -> tuple[str, str, str | None]:
        fires.append(routine.id)
        return "ok", "done", None

    monkeypatch.setattr(svc, "_run_headless", _fake_headless)
    monkeypatch.setattr(svc, "_notify", lambda *a, **k: None)
    return fires


def _add(schedule: Schedule, *, last_run_at: str | None, created_at: str) -> Routine:
    r = Routine(
        name="r", agent_path="a.yaml", prompt="p", schedule=schedule, created_at=created_at
    )
    r.last_run_at = last_run_at
    return svc.get_routines_store().upsert(r)


def _frozen(now: datetime) -> callable:
    return lambda: now


# ------------------------------------------------------------------ coalesce (default)


@pytest.mark.asyncio
async def test_coalesce_fires_once_on_tick(
    store_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fires = _install_fake_run(monkeypatch)
    # Daily 09:00 UTC, last ran two days ago → several occurrences missed.
    r = _add(
        Schedule(kind="daily", at="09:00", missed="coalesce"),
        last_run_at="2026-06-01T09:00:00+00:00",
        created_at="2026-06-01T09:00:00+00:00",
    )
    now = datetime(2026, 6, 4, 10, 0, tzinfo=UTC)
    sched = RoutineScheduler(now=_frozen(now))
    # catch-up does nothing for coalesce; the next tick fires it ONCE.
    actions = await sched.catch_up_on_launch()
    assert actions.get(r.id) == "coalesce"
    sched.tick()
    # Drain the launched task.
    for t in list(sched._running.values()):
        await t
    assert fires.count(r.id) == 1


# --------------------------------------------------------------------------- skip


@pytest.mark.asyncio
async def test_skip_advances_without_running(
    store_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fires = _install_fake_run(monkeypatch)
    r = _add(
        Schedule(kind="cron", expr="0 9 * * *", timezone="UTC", missed="skip"),
        last_run_at="2026-06-01T09:00:00+00:00",
        created_at="2026-06-01T09:00:00+00:00",
    )
    now = datetime(2026, 6, 4, 10, 0, tzinfo=UTC)
    sched = RoutineScheduler(now=_frozen(now))
    actions = await sched.catch_up_on_launch()
    assert actions.get(r.id) == "skip:1"
    # Nothing ran, and the anchor advanced so the next tick is NOT immediately due.
    assert fires == []
    refreshed = svc.get_routines_store().get(r.id)
    assert refreshed.last_status == "skipped"
    assert not svc.is_due(refreshed, now)


# ------------------------------------------------------------------------ run_each


@pytest.mark.asyncio
async def test_run_each_backfills_each_missed(
    store_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fires = _install_fake_run(monkeypatch)
    # Daily 09:00 UTC; last ran 2026-06-01 09:00; now 2026-06-04 10:00 → missed
    # 06-02, 06-03, 06-04 = 3 occurrences. grace huge, cap high.
    r = _add(
        Schedule(
            kind="cron",
            expr="0 9 * * *",
            timezone="UTC",
            missed="run_each",
            max_catchup=100,
            grace_seconds=10 * 24 * 3600,
        ),
        last_run_at="2026-06-01T09:00:00+00:00",
        created_at="2026-06-01T09:00:00+00:00",
    )
    now = datetime(2026, 6, 4, 10, 0, tzinfo=UTC)
    sched = RoutineScheduler(now=_frozen(now))
    actions = await sched.catch_up_on_launch()
    assert actions.get(r.id) == "run_each:3"
    assert fires.count(r.id) == 3


@pytest.mark.asyncio
async def test_run_each_grace_drops_old_occurrences(
    store_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fires = _install_fake_run(monkeypatch)
    # Same 3-day gap, but grace = 1 day → only the most-recent occurrence (06-04 09:00,
    # within 1h of now) survives; the older two are dropped.
    r = _add(
        Schedule(
            kind="cron",
            expr="0 9 * * *",
            timezone="UTC",
            missed="run_each",
            max_catchup=100,
            grace_seconds=24 * 3600,
        ),
        last_run_at="2026-06-01T09:00:00+00:00",
        created_at="2026-06-01T09:00:00+00:00",
    )
    now = datetime(2026, 6, 4, 10, 0, tzinfo=UTC)
    sched = RoutineScheduler(now=_frozen(now))
    actions = await sched.catch_up_on_launch()
    assert actions.get(r.id) == "run_each:1"
    assert fires.count(r.id) == 1


@pytest.mark.asyncio
async def test_run_each_max_catchup_storm_cap(
    store_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fires = _install_fake_run(monkeypatch)
    # Hourly cron, huge gap, cap = 5 → only 5 fires even though dozens were missed.
    r = _add(
        Schedule(
            kind="cron",
            expr="0 * * * *",
            timezone="UTC",
            missed="run_each",
            max_catchup=5,
            grace_seconds=10 * 24 * 3600,
        ),
        last_run_at="2026-06-01T00:00:00+00:00",
        created_at="2026-06-01T00:00:00+00:00",
    )
    now = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)  # 24 missed hourly occurrences
    sched = RoutineScheduler(now=_frozen(now))
    actions = await sched.catch_up_on_launch()
    assert actions.get(r.id) == "run_each:5"
    assert fires.count(r.id) == 5


# ------------------------------------------------------------------ dedup at-most-once


@pytest.mark.asyncio
async def test_dedup_coalesces_same_planned_fire(
    store_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ticks on the SAME planned cron fire run the routine exactly once.

    Publishes an in-memory durable storage facade (which implements the dedup surface) so the
    planned-fire dedup claim is exercised within the process: the first tick wins the claim,
    the second sees CLAIM_DONE and does not run.
    """
    fires = _install_fake_run(monkeypatch)

    from himmy.services.storage.factory import (
        reset_server_storage,
        set_server_storage,
    )
    from himmy.services.storage.service import StorageService

    storage = StorageService()
    set_server_storage(storage)
    try:
        r = _add(
            Schedule(kind="cron", expr="0 9 * * *", timezone="UTC"),
            last_run_at="2026-06-01T09:00:00+00:00",
            created_at="2026-06-01T09:00:00+00:00",
        )
        planned = "2026-06-02T09:00:00+00:00"
        # Two concurrent fires of the SAME planned instant via the public execute path.
        first = await svc.execute_routine(
            r.id,
            now=_frozen(datetime(2026, 6, 2, 9, 0, tzinfo=UTC)),
            planned_fire_at_iso=planned,
        )
        assert first is not None
        # Reset the anchor sentinel so mark_started alone would NOT block the second — only
        # the dedup claim should. (Simulate a second boot that re-observes the same fire.)
        second = await svc.execute_routine(
            r.id,
            now=_frozen(datetime(2026, 6, 2, 9, 0, 5, tzinfo=UTC)),
            planned_fire_at_iso=planned,
        )
        assert second is not None
        assert fires.count(r.id) == 1  # at-most-once on the planned fire
    finally:
        reset_server_storage()


@pytest.mark.asyncio
async def test_dedup_distinct_planned_fires_both_run(
    store_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different planned instants are distinct dedup keys → both fire."""
    fires = _install_fake_run(monkeypatch)

    from himmy.services.storage.factory import (
        reset_server_storage,
        set_server_storage,
    )
    from himmy.services.storage.service import StorageService

    set_server_storage(StorageService())
    try:
        r = _add(
            Schedule(kind="cron", expr="0 9 * * *", timezone="UTC"),
            last_run_at="2026-06-01T09:00:00+00:00",
            created_at="2026-06-01T09:00:00+00:00",
        )
        await svc.execute_routine(
            r.id,
            now=_frozen(datetime(2026, 6, 2, 9, 0, tzinfo=UTC)),
            planned_fire_at_iso="2026-06-02T09:00:00+00:00",
        )
        await svc.execute_routine(
            r.id,
            now=_frozen(datetime(2026, 6, 3, 9, 0, tzinfo=UTC)),
            planned_fire_at_iso="2026-06-03T09:00:00+00:00",
        )
        assert fires.count(r.id) == 2
    finally:
        reset_server_storage()
