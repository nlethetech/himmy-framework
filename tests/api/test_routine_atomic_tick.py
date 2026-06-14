"""K4: the atomic cluster-wide routine-tick claim (the conditional ``mark_started``).

The cluster-wide double-execution fix threads the ``last_run_at`` captured at
due-evaluation through ``tick`` -> ``execute_routine`` -> ``_execute_locked`` ->
``mark_started``, which issues a CONDITIONAL UPDATE (``WHERE last_run_at IS ?``) so a
routine due on every tick is claimed exactly once even when two writers race. The Postgres
mirror uses the IDENTICAL ``IS NOT DISTINCT FROM`` predicate; this module exercises the
dialect-SHARED contract live on SQLite (the offline default), plus the NULL-first-run case
the predicate must handle.
"""

from __future__ import annotations

from himmy.api.routines import (
    UNCONDITIONAL,
    RoutinesStore,
    Schedule,
)
from himmy.api.routines import (
    Routine as RoutineModel,
)


def _routine(last_run_at: str | None = None) -> RoutineModel:
    r = RoutineModel(
        name="nightly",
        agent_path="a.yaml",
        prompt="do",
        schedule=Schedule(kind="every", hours=1),
    )
    r.last_run_at = last_run_at
    return r


def test_conditional_claim_first_run_null_anchor() -> None:
    """A fresh routine (NULL ``last_run_at``) is claimable with the NULL sentinel."""
    store = RoutinesStore(":memory:")
    r = _routine(last_run_at=None)
    store.upsert(r)
    # The tick captured last_run_at = None at due-time.
    won = store.mark_started(r.id, "2026-06-13T01:00:00+00:00", expected_last_run_at=None)
    assert won is True
    assert store.get(r.id).last_run_at == "2026-06-13T01:00:00+00:00"


def test_two_writers_same_tick_claim_exactly_once() -> None:
    """Two writers racing on the SAME captured sentinel: exactly one wins."""
    store = RoutinesStore(":memory:")
    r = _routine(last_run_at="2026-06-13T00:00:00+00:00")
    store.upsert(r)
    sentinel = r.last_run_at  # both ticks observed this value at due-time

    first = store.mark_started(
        r.id, "2026-06-13T01:00:00+00:00", expected_last_run_at=sentinel
    )
    second = store.mark_started(
        r.id, "2026-06-13T01:00:05+00:00", expected_last_run_at=sentinel
    )
    assert first is True
    assert second is False  # the anchor moved; the second claim matches 0 rows
    # The winner's start time is the one persisted.
    assert store.get(r.id).last_run_at == "2026-06-13T01:00:00+00:00"


def test_stale_sentinel_loses_after_a_prior_run() -> None:
    """A tick whose captured anchor is stale (a run already advanced it) cannot claim."""
    store = RoutinesStore(":memory:")
    r = _routine(last_run_at="2026-06-13T00:00:00+00:00")
    store.upsert(r)
    # A run advances the anchor.
    assert store.mark_started(
        r.id, "2026-06-13T01:00:00+00:00", expected_last_run_at="2026-06-13T00:00:00+00:00"
    )
    # A laggard tick still holding the OLD anchor must not re-fire.
    stale = store.mark_started(
        r.id, "2026-06-13T01:05:00+00:00", expected_last_run_at="2026-06-13T00:00:00+00:00"
    )
    assert stale is False


def test_unconditional_run_now_always_claims() -> None:
    """``run_now`` (UNCONDITIONAL) fires regardless of ``last_run_at`` (flock guards overlap)."""
    store = RoutinesStore(":memory:")
    r = _routine(last_run_at="2026-06-13T00:00:00+00:00")
    store.upsert(r)
    # Even though last_run_at is set, the unconditional stamp lands.
    won = store.mark_started(
        r.id, "2026-06-13T02:00:00+00:00", expected_last_run_at=UNCONDITIONAL
    )
    assert won is True
    assert store.get(r.id).last_run_at == "2026-06-13T02:00:00+00:00"


def test_mark_started_default_is_unconditional() -> None:
    """Omitting the sentinel keeps the legacy ``run_now`` semantics (unconditional)."""
    store = RoutinesStore(":memory:")
    r = _routine(last_run_at="2026-06-13T00:00:00+00:00")
    store.upsert(r)
    won = store.mark_started(r.id, "2026-06-13T03:00:00+00:00")
    assert won is True
