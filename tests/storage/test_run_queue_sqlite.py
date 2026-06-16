"""Q1+Q2: the leased run work-queue on the durable SQLite backend.

Covers the migration (schema v5 columns + ``trigger_dedup`` table), the
``claim_next_queued_run`` CAS (at-most-once, oldest-first, lane-keyed), lease renewal,
the expired-lease reaper (never touches a live peer or a HITL-paused run), and redrive.
The Q0 recoverable-input encryption at rest is covered in ``test_run_input_recoverability``.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

from himmy.core.ids import iso_plus_seconds, utc_now_iso
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.run_lane import lane_for_model_key
from himmy.services.storage.sqlite import SQLITE_SCHEMA_VERSION, SqliteStorageService
from tests.conftest import run_async


def _queued(workspace: str = "w", *, lane: str | None = None) -> RunRecord:
    return RunRecord(
        workspace_id=workspace,
        subject_id="s",
        status=RunStatus.QUEUED,
        lane_key=lane,
    )


def test_migration_adds_queue_columns_and_dedup_table(tmp_path: Path) -> None:
    """Schema v5 promotes the eight queue columns + creates ``trigger_dedup``."""
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    assert store.schema_version == SQLITE_SCHEMA_VERSION >= 5
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(runs)").fetchall()}
    for col in (
        "owner_id",
        "lease_expires_at",
        "heartbeat_at",
        "attempt",
        "max_attempts",
        "next_attempt_at",
        "last_error",
        "lane_key",
    ):
        assert col in cols, col
    tables = {
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "trigger_dedup" in tables
    run_async(store.close())


def test_legacy_db_backfills_attempt_and_next_attempt(tmp_path: Path) -> None:
    """A v4 ``runs`` row upgraded to v5 backfills attempt=0 and next_attempt_at=created_at."""
    path = str(tmp_path / "legacy.db")
    # Build a v4-shaped runs table with one QUEUED row, user_version stamped at 4.
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, "
        "subject_id TEXT NOT NULL, idempotency_key TEXT, status TEXT NOT NULL, "
        "payload TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    # A real v4 db also carries run_events with the v2-promoted event_type/tool_name
    # columns; include it so this fixture is a faithful v4 shape (later migrations index
    # run_events, which a real v4 db always has).
    raw.execute(
        "CREATE TABLE run_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
        "event_id TEXT NOT NULL UNIQUE, thread_id TEXT, trace_id TEXT, "
        "event_type TEXT, tool_name TEXT, payload TEXT NOT NULL DEFAULT '{}')"
    )
    raw.execute(
        "INSERT INTO runs (run_id, workspace_id, subject_id, status, payload, "
        "created_at, updated_at) VALUES "
        "('r1','w','s','QUEUED','{\"run_id\":\"r1\",\"workspace_id\":\"w\","
        "\"subject_id\":\"s\",\"status\":\"QUEUED\",\"created_at\":\"2020-01-01T00:00:00+00:00\","
        "\"updated_at\":\"2020-01-01T00:00:00+00:00\"}',"
        "'2020-01-01T00:00:00+00:00','2020-01-01T00:00:00+00:00')"
    )
    raw.execute("PRAGMA user_version = 4")
    raw.commit()
    raw.close()

    store = SqliteStorageService(path)
    assert store.schema_version >= 5
    row = store._conn.execute(
        "SELECT attempt, next_attempt_at FROM runs WHERE run_id='r1'"
    ).fetchone()
    assert row[0] == 0
    assert row[1] == "2020-01-01T00:00:00+00:00"  # backfilled to created_at
    run_async(store.close())


def test_claim_is_at_most_once_and_oldest_first(tmp_path: Path) -> None:
    """A QUEUED run produces exactly one RUNNING claim; oldest is claimed first."""
    store = SqliteStorageService(str(tmp_path / "q.db"))
    r1 = _queued()
    r1.created_at = "2020-01-01T00:00:00+00:00"
    r1.next_attempt_at = r1.created_at
    r2 = _queued()
    r2.created_at = "2020-01-02T00:00:00+00:00"
    r2.next_attempt_at = r2.created_at
    run_async(store.save_run(r2))
    run_async(store.save_run(r1))

    first = run_async(store.claim_next_queued_run("A", 60))
    assert first is not None and first.run_id == r1.run_id  # oldest
    assert first.status == RunStatus.RUNNING
    assert first.owner_id == "A"
    assert first.attempt == 1
    assert first.lease_expires_at is not None

    second = run_async(store.claim_next_queued_run("B", 60))
    assert second is not None and second.run_id == r2.run_id
    # Nothing left.
    assert run_async(store.claim_next_queued_run("C", 60)) is None
    run_async(store.close())


def test_claim_respects_next_attempt_at_gate(tmp_path: Path) -> None:
    """A run whose ``next_attempt_at`` is in the future is not yet claimable."""
    store = SqliteStorageService(str(tmp_path / "q.db"))
    r = _queued()
    r.next_attempt_at = iso_plus_seconds(utc_now_iso(), 3600)  # 1h out
    run_async(store.save_run(r))
    assert run_async(store.claim_next_queued_run("A", 60)) is None
    # With an explicit later ``now`` it becomes claimable.
    later = iso_plus_seconds(utc_now_iso(), 7200)
    got = run_async(store.claim_next_queued_run("A", 60, now=later))
    assert got is not None and got.run_id == r.run_id
    run_async(store.close())


def test_lane_keying_gates_local_not_cloud(tmp_path: Path) -> None:
    """A claim restricted to cloud/default lanes never claims a local-lane run (Q2)."""
    store = SqliteStorageService(str(tmp_path / "q.db"))
    local = _queued(lane=lane_for_model_key("ollama:llama3"))
    cloud = _queued(lane=lane_for_model_key("anthropic:claude-3-5"))
    run_async(store.save_run(local))
    run_async(store.save_run(cloud))

    # Local probe down: dispatcher offers only cloud/default lanes.
    got = run_async(store.claim_next_queued_run("A", 60, lanes=["cloud", "default"]))
    assert got is not None and got.run_id == cloud.run_id
    # No more cloud runs; the local one is still QUEUED (NOT starved-claimed).
    assert (
        run_async(store.claim_next_queued_run("A", 60, lanes=["cloud", "default"]))
        is None
    )
    still = run_async(store.get_run(local.run_id))
    assert still.status == RunStatus.QUEUED
    # When local recovers, it is claimable.
    got_local = run_async(store.claim_next_queued_run("A", 60, lanes=["local"]))
    assert got_local is not None and got_local.run_id == local.run_id
    run_async(store.close())


def test_empty_lane_allowlist_claims_nothing(tmp_path: Path) -> None:
    """An empty ``lanes`` allowlist means 'no claimable lane' — claim nothing."""
    store = SqliteStorageService(str(tmp_path / "q.db"))
    run_async(store.save_run(_queued(lane="cloud")))
    assert run_async(store.claim_next_queued_run("A", 60, lanes=[])) is None
    run_async(store.close())


def test_renew_lease_requires_current_owner(tmp_path: Path) -> None:
    """Only the current owner of a RUNNING run can renew its lease."""
    store = SqliteStorageService(str(tmp_path / "q.db"))
    run_async(store.save_run(_queued()))
    claimed = run_async(store.claim_next_queued_run("A", 60))
    assert claimed is not None
    assert run_async(store.renew_lease(claimed.run_id, "A", 120)) is True
    assert run_async(store.renew_lease(claimed.run_id, "B", 120)) is False
    run_async(store.close())


def test_reaper_requeues_only_expired_running_leases(tmp_path: Path) -> None:
    """The reaper re-queues an expired RUNNING lease but never a live peer."""
    store = SqliteStorageService(str(tmp_path / "q.db"))
    dead = _queued()
    live = _queued()
    run_async(store.save_run(dead))
    run_async(store.save_run(live))
    # Claim both; ``dead`` gets an already-expired lease (negative seconds).
    claimed_dead = run_async(store.claim_next_queued_run("A", -5))
    claimed_live = run_async(store.claim_next_queued_run("B", 3600))
    assert {claimed_dead.run_id, claimed_live.run_id} == {dead.run_id, live.run_id}

    requeued = run_async(store.requeue_expired_leases())
    assert claimed_dead.run_id in requeued
    assert claimed_live.run_id not in requeued

    back_dead = run_async(store.get_run(claimed_dead.run_id))
    assert back_dead.status == RunStatus.QUEUED
    assert back_dead.owner_id is None
    assert back_dead.lease_expires_at is None
    back_live = run_async(store.get_run(claimed_live.run_id))
    assert back_live.status == RunStatus.RUNNING
    assert back_live.owner_id == "B"
    run_async(store.close())


def test_reaper_excludes_hitl_paused_runs(tmp_path: Path) -> None:
    """AWAITING_APPROVAL / RESOLVING runs are never reaped (preserve HITL exclusions)."""
    store = SqliteStorageService(str(tmp_path / "q.db"))
    paused = _queued()
    paused.status = RunStatus.AWAITING_APPROVAL
    paused.lease_expires_at = iso_plus_seconds(utc_now_iso(), -100)  # "expired", but not RUNNING
    run_async(store.save_run(paused))
    requeued = run_async(store.requeue_expired_leases())
    assert paused.run_id not in requeued
    assert run_async(store.get_run(paused.run_id)).status == RunStatus.AWAITING_APPROVAL
    run_async(store.close())


def test_two_writers_race_one_claim_each(tmp_path: Path) -> None:
    """Two SqliteStorageService instances on ONE file, racing many claims, never double-claim.

    Each of N QUEUED runs must end up RUNNING under exactly one owner — the cross-instance
    (cross-process proxy) at-most-once guarantee of the ``BEGIN IMMEDIATE`` CAS. Two OS threads
    each run their own event loop and a writer hammering ``claim_next_queued_run`` against the
    SAME on-disk database.
    """
    path = str(tmp_path / "race.db")
    seed = SqliteStorageService(path)
    n = 30
    for _ in range(n):
        run_async(seed.save_run(_queued()))
    run_async(seed.close())

    claimed_by: dict[str, list[str]] = {"A": [], "B": []}
    lock = threading.Lock()

    def worker(owner: str) -> None:
        store = SqliteStorageService(path)

        async def drain() -> None:
            while True:
                got = await store.claim_next_queued_run(owner, 3600)
                if got is None:
                    # Another worker may still be mid-claim; retry briefly, then stop.
                    remaining = await store.list_runs(status=RunStatus.QUEUED)
                    if not remaining:
                        return
                    await asyncio.sleep(0)
                    continue
                with lock:
                    claimed_by[owner].append(got.run_id)

        asyncio.run(drain())
        asyncio.run(store.close())

    threads = [
        threading.Thread(target=worker, args=("A",)),
        threading.Thread(target=worker, args=("B",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_claimed = claimed_by["A"] + claimed_by["B"]
    # Exactly once each: no run claimed twice, all runs claimed.
    assert len(all_claimed) == len(set(all_claimed)) == n

    verify = SqliteStorageService(path)
    running = run_async(verify.list_runs(status=RunStatus.RUNNING))
    assert len(running) == n
    assert all(r.owner_id in ("A", "B") and r.attempt == 1 for r in running)
    run_async(verify.close())


def test_redrive_resets_parked_and_failed_only(tmp_path: Path) -> None:
    """Redrive resets a PARKED/FAILED run to QUEUED with attempt zeroed; ignores live runs."""
    store = SqliteStorageService(str(tmp_path / "q.db"))
    parked = _queued()
    parked.status = RunStatus.PARKED
    parked.attempt = 3
    parked.last_error = "exhausted"
    parked.error = "exhausted"
    run_async(store.save_run(parked))

    assert run_async(store.redrive_run(parked.run_id, workspace_id="w")) is True
    back = run_async(store.get_run(parked.run_id))
    assert back.status == RunStatus.QUEUED
    assert back.attempt == 0
    assert back.last_error is None
    assert back.error is None
    assert back.next_attempt_at is not None
    # Re-driving a now-QUEUED run is a no-op (only PARKED/FAILED are redrivable).
    assert run_async(store.redrive_run(parked.run_id, workspace_id="w")) is False
    # Wrong workspace cannot redrive (tenant scope).
    failed = _queued()
    failed.status = RunStatus.FAILED
    run_async(store.save_run(failed))
    assert run_async(store.redrive_run(failed.run_id, workspace_id="other")) is False
    assert run_async(store.redrive_run(failed.run_id, workspace_id="w")) is True
    run_async(store.close())
