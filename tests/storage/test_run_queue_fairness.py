"""T3: per-tenant fairness + cross-node concurrency cap on the leased run-queue claim.

These exercise the ``fairness`` / ``workspace_concurrency`` parameters added to
``claim_next_queued_run`` on the in-memory + SQLite backends (the Postgres path has its own
live suite). They prove: (1) the claim interleaves tenants fairly (a flooding tenant cannot
monopolise free worker slots), (2) the per-tenant concurrency cap holds across claims, (3)
``count_active_runs_for_workspace`` counts only non-terminal runs, and (4) single-tenant /
default (fairness OFF) behaviour is byte-identical global FIFO.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _queued(ws: str) -> RunRecord:
    return RunRecord(workspace_id=ws, subject_id="s", status=RunStatus.QUEUED)


def _inmem() -> StorageService:
    return StorageService()


def _sqlite(tmp_path) -> SqliteStorageService:  # type: ignore[no-untyped-def]
    return SqliteStorageService(str(tmp_path / "runs.db"))


_BACKENDS: list[tuple[str, Callable[..., object]]] = [
    ("inmemory", _inmem),
    ("sqlite", _sqlite),
]


@pytest.mark.parametrize("name,make", _BACKENDS)
def test_fairness_interleaves_tenants(name, make, tmp_path) -> None:
    """A tenant flooding the queue does NOT starve a quieter tenant under fairness.

    Tenant A enqueues 5 runs FIRST (so pure FIFO would drain all of A before B); B then
    enqueues 1. With fairness the FIRST claim should still go to A (both at 0 running), but
    after A has one running, the next claim must prefer B (least-loaded), not A's backlog.
    """

    async def scenario() -> None:
        s = make() if name == "inmemory" else make(tmp_path)
        a_ids = [_queued("A") for _ in range(5)]
        for r in a_ids:
            await s.save_run(r)
        b = _queued("B")
        await s.save_run(b)
        # First claim: both tenants at 0 running -> FIFO tie-break picks the oldest (A's).
        c1 = await s.claim_next_queued_run("w1", 600, fairness=True)
        assert c1 is not None and c1.workspace_id == "A"
        # Second claim: A now has 1 running, B has 0 -> least-loaded picks B.
        c2 = await s.claim_next_queued_run("w2", 600, fairness=True)
        assert c2 is not None and c2.workspace_id == "B"
        # Third claim: only A's backlog remains -> A again.
        c3 = await s.claim_next_queued_run("w3", 600, fairness=True)
        assert c3 is not None and c3.workspace_id == "A"

    run_async(scenario())


@pytest.mark.parametrize("name,make", _BACKENDS)
def test_per_tenant_concurrency_cap(name, make, tmp_path) -> None:
    """A workspace already at its cap of LIVE-leased running runs is skipped."""

    async def scenario() -> None:
        s = make() if name == "inmemory" else make(tmp_path)
        for _ in range(4):
            await s.save_run(_queued("A"))
        await s.save_run(_queued("B"))
        # Cap A at 2 concurrent. Claim twice -> both A (B not preferred while A==0/1<2,
        # but B is least-loaded once A has 1). Force the cap by repeatedly claiming.
        claimed: list[str] = []
        for _ in range(3):
            r = await s.claim_next_queued_run(
                "w", 600, fairness=True, workspace_concurrency=2
            )
            if r is None:
                break
            claimed.append(r.workspace_id)
        # A can appear at most twice (its cap); B once. So A's backlog (2 leftover) is NOT
        # all drained — the cap protected the shared slots.
        assert claimed.count("A") <= 2
        assert "B" in claimed
        # A 4th claim cannot exceed A's cap: with B drained and A at 2, A is excluded.
        leftover = await s.claim_next_queued_run(
            "w", 600, fairness=True, workspace_concurrency=2
        )
        assert leftover is None

    run_async(scenario())


@pytest.mark.parametrize("name,make", _BACKENDS)
def test_expired_lease_does_not_count_toward_cap(name, make, tmp_path) -> None:
    """A RUNNING run whose lease expired is NOT counted as live load (it will be reaped)."""

    async def scenario() -> None:
        s = make() if name == "inmemory" else make(tmp_path)
        await s.save_run(_queued("A"))
        await s.save_run(_queued("A"))
        # Claim one with an already-expired lease -> not live load.
        expired = await s.claim_next_queued_run("w", -1, fairness=True)
        assert expired is not None and expired.workspace_id == "A"
        # Even with cap=1, A's other run is claimable because the expired one doesn't count.
        nxt = await s.claim_next_queued_run(
            "w2", 600, fairness=True, workspace_concurrency=1
        )
        assert nxt is not None and nxt.workspace_id == "A"

    run_async(scenario())


@pytest.mark.parametrize("name,make", _BACKENDS)
def test_count_active_runs_for_workspace(name, make, tmp_path) -> None:
    """Only non-terminal runs count toward a tenant's outstanding budget."""

    async def scenario() -> None:
        s = make() if name == "inmemory" else make(tmp_path)
        q = _queued("A")
        await s.save_run(q)
        running = _queued("A")
        running.status = RunStatus.RUNNING
        await s.save_run(running)
        done = _queued("A")
        done.status = RunStatus.SUCCEEDED
        await s.save_run(done)
        parked = _queued("A")
        parked.status = RunStatus.PARKED
        await s.save_run(parked)
        other = _queued("B")
        await s.save_run(other)
        assert await s.count_active_runs_for_workspace("A") == 2  # QUEUED + RUNNING
        assert await s.count_active_runs_for_workspace("B") == 1
        assert await s.count_active_runs_for_workspace("Z") == 0

    run_async(scenario())


@pytest.mark.parametrize("name,make", _BACKENDS)
def test_single_tenant_fifo_parity_fairness_off(name, make, tmp_path) -> None:
    """Fairness OFF (the default) is byte-identical global FIFO — single-tenant parity."""

    async def scenario() -> None:
        s = make() if name == "inmemory" else make(tmp_path)
        ids = [_queued("solo") for _ in range(4)]
        for r in ids:
            await s.save_run(r)
        order = []
        while True:
            r = await s.claim_next_queued_run("w", 600)
            if r is None:
                break
            order.append(r.run_id)
        # FIFO: claimed in insertion order.
        assert order == [r.run_id for r in ids]

    run_async(scenario())


@pytest.mark.parametrize("name,make", _BACKENDS)
def test_lone_tenant_not_starved_under_fairness(name, make, tmp_path) -> None:
    """A SINGLE tenant under fairness drains fully (no deadlock / self-starvation)."""

    async def scenario() -> None:
        s = make() if name == "inmemory" else make(tmp_path)
        for _ in range(5):
            await s.save_run(_queued("solo"))
        n = 0
        while True:
            r = await s.claim_next_queued_run(
                "w", 600, fairness=True, workspace_concurrency=8
            )
            if r is None:
                break
            n += 1
        assert n == 5

    run_async(scenario())
