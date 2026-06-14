"""Q2: leased-queue CAS against a LIVE Postgres (SKIP LOCKED claim).

SKIPPED unless ``HIMMY_TEST_POSTGRES_DSN`` is set AND ``asyncpg`` is importable — the same gate
as the rest of the Postgres suite. This is the test that proves the ``SELECT … FOR UPDATE SKIP
LOCKED`` claim is at-most-once across concurrent claimers (which the offline string/param tests
cannot exercise), plus lease renewal / reaper / redrive / lane keying on the real backend.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.postgres import PostgresStorageService
from tests.conftest import run_async

_DSN = os.environ.get("HIMMY_TEST_POSTGRES_DSN")

try:  # pragma: no cover - import probe
    import asyncpg  # type: ignore  # noqa: F401

    _HAVE_ASYNCPG = True
except ImportError:  # pragma: no cover
    _HAVE_ASYNCPG = False

pytestmark = pytest.mark.skipif(
    not _DSN or not _HAVE_ASYNCPG,
    reason="requires HIMMY_TEST_POSTGRES_DSN + the [postgres] extra",
)


async def _fresh_storage() -> PostgresStorageService:
    storage = await PostgresStorageService.connect(_DSN)
    await storage.migrate()
    return storage


async def _drain_queue(storage: PostgresStorageService) -> None:
    """Claim away any PRE-EXISTING queued runs so the shared test DB is clean.

    ``claim_next_queued_run`` is GLOBAL (a worker claims the next queued run across every
    workspace — correct for a shared work queue), so these tests' assertions about claiming
    EXACTLY their own runs only hold on an empty queue. Other live suites sharing the one
    ``himmy_test`` database leave queued runs behind; drain them (claiming flips them to
    RUNNING, out of the queue) before enqueuing this test's runs. The properties under test
    — at-most-once SKIP LOCKED, lease renewal, reaping, redrive — are unaffected.
    """
    while await storage.claim_next_queued_run("drain", 300) is not None:
        pass


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def test_concurrent_claims_are_at_most_once() -> None:
    """N concurrent claimers over M queued runs each get a DISTINCT run (SKIP LOCKED)."""

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            await _drain_queue(storage)
            ws = _uid("ws")
            ids = [_uid("run") for _ in range(12)]
            for rid in ids:
                await storage.save_run(
                    RunRecord(
                        run_id=rid,
                        workspace_id=ws,
                        subject_id="s",
                        status=RunStatus.QUEUED,
                    )
                )
            results = await asyncio.gather(
                *(storage.claim_next_queued_run(f"w{i}", 300) for i in range(12))
            )
            claimed = [r.run_id for r in results if r is not None]
            assert sorted(claimed) == sorted(ids)  # each claimed exactly once
            assert len({r.owner_id for r in results if r}) >= 1
            # All of THIS workspace's runs are RUNNING now.
            running = await storage.list_runs(workspace_id=ws, status=RunStatus.RUNNING)
            assert len(running) == 12
        finally:
            await storage.close()

    run_async(scenario())


def test_renew_reap_redrive_live() -> None:
    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            await _drain_queue(storage)
            ws = _uid("ws")
            rid = _uid("run")
            await storage.save_run(
                RunRecord(
                    run_id=rid, workspace_id=ws, subject_id="s", status=RunStatus.QUEUED
                )
            )
            claimed = await storage.claim_next_queued_run("A", -1)  # expired lease
            assert claimed is not None and claimed.run_id == rid
            # Wrong owner cannot renew.
            assert await storage.renew_lease(rid, "B", 300) is False
            # Reaper re-queues the expired lease.
            requeued = await storage.requeue_expired_leases()
            assert rid in requeued
            back = await storage.get_run(rid)
            assert back.status == RunStatus.QUEUED and back.owner_id is None
            # Park + redrive.
            back.status = RunStatus.PARKED
            back.attempt = 9
            await storage.save_run(back)
            assert await storage.redrive_run(rid, workspace_id=ws) is True
            rd = await storage.get_run(rid)
            assert rd.status == RunStatus.QUEUED and rd.attempt == 0
        finally:
            await storage.close()

    run_async(scenario())


def test_lane_keying_live() -> None:
    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            await _drain_queue(storage)
            ws = _uid("ws")
            local_id, cloud_id = _uid("run"), _uid("run")
            await storage.save_run(
                RunRecord(
                    run_id=local_id,
                    workspace_id=ws,
                    subject_id="s",
                    status=RunStatus.QUEUED,
                    lane_key="local",
                )
            )
            await storage.save_run(
                RunRecord(
                    run_id=cloud_id,
                    workspace_id=ws,
                    subject_id="s",
                    status=RunStatus.QUEUED,
                    lane_key="cloud",
                )
            )
            got = await storage.claim_next_queued_run(
                "A", 300, lanes=["cloud", "default"]
            )
            assert got is not None and got.run_id == cloud_id
            # No more cloud runs; the local one is not starved-claimed.
            assert (
                await storage.claim_next_queued_run("A", 300, lanes=["cloud", "default"])
                is None
            )
            still = await storage.get_run(local_id)
            assert still.status == RunStatus.QUEUED
        finally:
            await storage.close()

    run_async(scenario())


def test_trigger_dedup_ttl_cas_live() -> None:
    """Q4: the durable dedup TTL-CAS on a real Postgres (ON CONFLICT take-over + completion)."""
    from himmy.services.storage.trigger_dedup import (
        CLAIM_DONE,
        CLAIM_IN_FLIGHT,
        CLAIM_WON,
    )

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            key = _uid("evt")
            won = await storage.dedup_try_claim("webhook", key, lease_seconds=300)
            assert won.outcome is CLAIM_WON
            # Concurrent duplicate while in-flight is refused.
            again = await storage.dedup_try_claim("webhook", key, lease_seconds=300)
            assert again.outcome is CLAIM_IN_FLIGHT
            # Complete -> a later claim returns the stored result.
            await storage.dedup_complete(
                "webhook", key, result="reply", ttl_seconds=86400
            )
            done = await storage.dedup_try_claim("webhook", key, lease_seconds=300)
            assert done.outcome is CLAIM_DONE and done.result == "reply"
            # An expired in-flight lease is reclaimable (crash recovery): far-future now.
            key2 = _uid("evt")
            await storage.dedup_try_claim(
                "webhook", key2, lease_seconds=1, now="2020-01-01T00:00:00+00:00"
            )
            reclaim = await storage.dedup_try_claim(
                "webhook", key2, lease_seconds=300, now="2099-01-01T00:00:00+00:00"
            )
            assert reclaim.outcome is CLAIM_WON
            swept = await storage.dedup_sweep(now="2099-06-01T00:00:00+00:00")
            assert swept >= 1
        finally:
            await storage.close()

    run_async(scenario())


def test_concurrent_dedup_claims_are_at_most_once_live() -> None:
    """N concurrent claimers on ONE delivery id: exactly one WINS, the rest are duplicates."""
    from himmy.services.storage.trigger_dedup import CLAIM_WON

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            key = _uid("evt")
            results = await asyncio.gather(
                *(
                    storage.dedup_try_claim("webhook", key, lease_seconds=300)
                    for _ in range(8)
                )
            )
            wins = [r for r in results if r.outcome is CLAIM_WON]
            assert len(wins) == 1  # at-most-once claim across concurrent webhook replicas
        finally:
            await storage.close()

    run_async(scenario())
