"""Retention/compaction for the spine tables: runs, recommendations, memory objects.

These tables (like ``run_events``) are never auto-deleted; the operator-invoked
``prune_runs`` / ``prune_recommendations`` / ``prune_memory`` calls bound their growth.
The critical invariant pinned here is that run pruning is scoped to TERMINAL statuses
only — a live/leased/queued run the queue reaper depends on is ALWAYS preserved — and
that pruning a run cascades to its recommendations so no advisory row is left dangling.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from himmy.core.ids import new_uuid
from himmy.services.storage.inmemory import (
    InMemoryOrchestrationStore,
    InMemoryRecommendationStore,
    InMemoryRunStore,
)
from himmy.services.storage.models import (
    EpisodicMemoryObject,
    MemoryObject,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _iso(days: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _run(status: RunStatus, days: float) -> RunRecord:
    return RunRecord(
        run_id=new_uuid(),
        workspace_id="w",
        subject_id="sub",
        status=status,
        created_at=_iso(days),
    )


def _rec(run_id: str, days: float) -> RecommendationItem:
    return RecommendationItem(
        recommendation_id=new_uuid(),
        run_id=run_id,
        workspace_id="w",
        subject_id="sub",
        kind="k",
        title="t",
        status=RecommendationStatus.PROPOSED,
        created_at=_iso(days),
    )


# --------------------------------------------------------------------- sqlite runs


def test_prune_runs_age_keeps_live_and_recent(tmp_path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    old_done = _run(RunStatus.SUCCEEDED, 100)
    new_done = _run(RunStatus.FAILED, 1)
    old_live = _run(RunStatus.RUNNING, 100)
    old_queued = _run(RunStatus.QUEUED, 100)
    for r in (old_done, new_done, old_live, old_queued):
        run_async(store.save_run(r))

    removed = run_async(store.prune_runs(older_than_days=90))
    assert removed == 1
    surviving = {r.run_id for r in run_async(store.list_runs())}
    assert old_done.run_id not in surviving
    # live/leased/queued runs are preserved regardless of age (queue reaper depends on them)
    assert old_live.run_id in surviving
    assert old_queued.run_id in surviving
    assert new_done.run_id in surviving


def test_prune_runs_cascades_recommendations(tmp_path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    old_done = _run(RunStatus.SUCCEEDED, 100)
    run_async(store.save_run(old_done))
    rec = _rec(old_done.run_id, 100)
    run_async(store.save_recommendation(rec))

    run_async(store.prune_runs(older_than_days=90))
    # the recommendation referencing the pruned run is gone (no dangling row)
    assert run_async(store.get_recommendation(rec.recommendation_id)) is None


def test_prune_runs_keep_last_only_terminal(tmp_path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    live = _run(RunStatus.RUNNING, 100)
    run_async(store.save_run(live))
    for _ in range(5):
        run_async(store.save_run(_run(RunStatus.SUCCEEDED, 1)))

    run_async(store.prune_runs(keep_last=2))
    surviving = run_async(store.list_runs())
    terminal = [
        r
        for r in surviving
        if r.status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.PARKED)
    ]
    assert len(terminal) == 2
    # the live run is never counted against keep_last, and never pruned
    assert live.run_id in {r.run_id for r in surviving}


def test_prune_runs_requires_a_bound(tmp_path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    with pytest.raises(ValueError):
        run_async(store.prune_runs())


# --------------------------------------------------- sqlite recommendations + memory


def test_prune_recommendations_age_and_keep_last(tmp_path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    for _ in range(3):
        run_async(store.save_recommendation(_rec("x", 100)))
    for _ in range(2):
        run_async(store.save_recommendation(_rec("x", 1)))

    assert run_async(store.prune_recommendations(older_than_days=90)) == 3
    remaining = run_async(store.list_recommendations())
    assert len(remaining) == 2
    assert run_async(store.prune_recommendations(keep_last=1)) == 1
    assert len(run_async(store.list_recommendations())) == 1


def test_prune_memory_both_tables(tmp_path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    run_async(
        store.save_memory(MemoryObject(memory_id=new_uuid(), created_at=_iso(100)))
    )
    run_async(store.save_memory(MemoryObject(memory_id=new_uuid(), created_at=_iso(1))))
    run_async(
        store.save_episodic_memory(
            EpisodicMemoryObject(episode_id=new_uuid(), created_at=_iso(100))
        )
    )

    removed = run_async(store.prune_memory(older_than_days=90))
    assert removed == 2  # one cognitive + one episodic
    assert len(run_async(store.list_memory())) == 1
    assert len(run_async(store.list_episodic_memory())) == 0


def test_prune_memory_requires_a_bound(tmp_path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    with pytest.raises(ValueError):
        run_async(store.prune_memory())


# ------------------------------------------------------------------ inmemory parity


def test_inmemory_prune_runs_keeps_live() -> None:
    rs = InMemoryRunStore()
    live = _run(RunStatus.RUNNING, 100)
    done = _run(RunStatus.SUCCEEDED, 100)
    run_async(rs.save_run(live))
    run_async(rs.save_run(done))

    assert run_async(rs.prune_runs(older_than_days=90)) == 1
    remaining = {r.run_id for r in run_async(rs.list_runs())}
    assert live.run_id in remaining
    assert done.run_id not in remaining


def test_inmemory_prune_recommendations_and_memory() -> None:
    recs = InMemoryRecommendationStore()
    run_async(recs.save_recommendation(_rec("x", 100)))
    run_async(recs.save_recommendation(_rec("x", 1)))
    assert run_async(recs.prune_recommendations(older_than_days=90)) == 1
    assert len(run_async(recs.list_recommendations())) == 1

    orch = InMemoryOrchestrationStore()
    run_async(orch.save_memory(MemoryObject(memory_id=new_uuid(), created_at=_iso(100))))
    run_async(
        orch.save_episodic_memory(
            EpisodicMemoryObject(episode_id=new_uuid(), created_at=_iso(100))
        )
    )
    assert run_async(orch.prune_memory(older_than_days=90)) == 2


def test_inmemory_prune_requires_a_bound() -> None:
    with pytest.raises(ValueError):
        run_async(InMemoryRunStore().prune_runs())
