"""Q2: leased-queue parity on the in-memory backend (the zero-config default store).

The in-memory ``StorageService`` is the offline default; its queue methods are the
single-event-loop analog of the SQLite/Postgres CAS (no ``await`` between scan and mutate).
These assert the same claim/renew/reap/redrive semantics so a test or CLI on the in-memory
store behaves like the durable backends.
"""

from __future__ import annotations

import asyncio

from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.run_lane import lane_for_model_key
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _queued(lane: str | None = None) -> RunRecord:
    return RunRecord(
        workspace_id="w", subject_id="s", status=RunStatus.QUEUED, lane_key=lane
    )


def test_claim_at_most_once_and_increments_attempt() -> None:
    async def scenario() -> None:
        s = StorageService()
        r1, r2 = _queued(), _queued()
        await s.save_run(r1)
        await s.save_run(r2)
        first = await s.claim_next_queued_run("A", 60)
        assert first is not None and first.status == RunStatus.RUNNING
        assert first.owner_id == "A" and first.attempt == 1
        second = await s.claim_next_queued_run("B", 60)
        assert second is not None and second.run_id != first.run_id
        assert await s.claim_next_queued_run("C", 60) is None

    run_async(scenario())


def test_concurrent_claims_collapse_to_distinct_runs() -> None:
    """Many concurrent claims on one loop hand out DISTINCT runs (no double-claim)."""

    async def scenario() -> None:
        s = StorageService()
        for _ in range(10):
            await s.save_run(_queued())
        results = await asyncio.gather(
            *(s.claim_next_queued_run(f"w{i}", 60) for i in range(10))
        )
        claimed = [r.run_id for r in results if r is not None]
        assert len(claimed) == len(set(claimed)) == 10

    run_async(scenario())


def test_lane_keying() -> None:
    async def scenario() -> None:
        s = StorageService()
        local = _queued(lane=lane_for_model_key("claude-cli:haiku"))
        cloud = _queued(lane=lane_for_model_key("openai:gpt-4o"))
        await s.save_run(local)
        await s.save_run(cloud)
        got = await s.claim_next_queued_run("A", 60, lanes=["cloud", "default"])
        assert got is not None and got.run_id == cloud.run_id
        assert (
            await s.claim_next_queued_run("A", 60, lanes=["cloud", "default"]) is None
        )
        assert (await s.get_run(local.run_id)).status == RunStatus.QUEUED

    run_async(scenario())


def test_renew_reap_redrive() -> None:
    async def scenario() -> None:
        s = StorageService()
        await s.save_run(_queued())
        claimed = await s.claim_next_queued_run("A", -1)  # already-expired lease
        assert await s.renew_lease(claimed.run_id, "A", 120) is True
        # but it was claimed with an expired lease, so reap re-queues it
        await s.claim_next_queued_run("A", -1)  # no-op (nothing queued)
        # Re-expire: renew with a negative lease, then reap.
        await s.renew_lease(claimed.run_id, "A", -1)
        requeued = await s.requeue_expired_leases()
        assert claimed.run_id in requeued
        back = await s.get_run(claimed.run_id)
        assert back.status == RunStatus.QUEUED and back.owner_id is None
        # park + redrive
        back.status = RunStatus.PARKED
        back.attempt = 5
        await s.save_run(back)
        assert await s.redrive_run(back.run_id, workspace_id="w") is True
        rd = await s.get_run(back.run_id)
        assert rd.status == RunStatus.QUEUED and rd.attempt == 0

    run_async(scenario())
