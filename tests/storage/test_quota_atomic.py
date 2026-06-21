"""T3 — the HARD atomic per-tenant quotas (routine-count + outstanding-run).

Both caps were SOFT: an unlocked count-then-create/enqueue TOCTOU that overshoots the cap
under concurrent same-workspace creates (the live adversarial proved 8 concurrent creates vs
cap=2 stored 3). These pin the FIX — the count and the create are fused into ONE atomic store
op (advisory-locked transaction on Postgres, ``BEGIN IMMEDIATE`` on SQLite, no-await-interleave
in-memory) — so concurrent creates SERIALISE and land EXACTLY at the cap:

* the routine-count quota (``RoutinesStore.create_routine_if_under_quota``);
* the outstanding-run quota (``StorageService.save_run_if_under_quota``).

Each is exercised both serially (boundary correctness) and under real concurrency (no
overshoot, no spurious reject), across the in-memory + SQLite backends. The live-Postgres
multi-process variants live in ``tests/storage/test_quota_atomic_postgres_live.py`` and
``scripts/scheduler_*_e2e.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from himmy.api.routines import Routine, RoutinesStore, Schedule
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async

# --------------------------------------------------------- helpers


def _routine(workspace: str, name: str, *, enabled: bool = True) -> Routine:
    return Routine(
        workspace_id=workspace,
        agent_id="agent-1",
        name=name,
        prompt="hi",
        schedule=Schedule(kind="daily", at="07:00"),
        enabled=enabled,
    )


def _run(workspace: str, *, idem: str | None = None) -> RunRecord:
    return RunRecord(
        workspace_id=workspace,
        subject_id="s",
        status=RunStatus.QUEUED,
        idempotency_key=idem,
    )


# =========================================================== (A) routine-count quota


def test_routine_quota_serial_lands_exactly_at_cap() -> None:
    """Serial creates: the (cap+1)-th is rejected; exactly cap rows are stored."""
    store = RoutinesStore(":memory:")
    cap = 3
    admitted = 0
    for i in range(6):
        _r, ok = store.create_routine_if_under_quota(
            _routine("acme", f"r{i}"), cap=cap
        )
        admitted += int(ok)
    assert admitted == cap
    assert store.count(workspace_id="acme", enabled_only=True) == cap


def test_routine_quota_concurrent_no_overshoot(tmp_path: Path) -> None:
    """Many concurrent creates against a small cap land EXACTLY at cap (no TOCTOU overshoot).

    Drives the file-backed SQLite store from many threads so the ``BEGIN IMMEDIATE`` write
    lock is genuinely contended (the in-memory ``:memory:`` store would serialise on the GIL
    anyway; the file store exercises the real write-lock path).
    """
    import threading

    db = str(tmp_path / "routines.db")
    store = RoutinesStore(db)
    cap = 2
    n = 16
    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(n)

    def _create(i: int) -> None:
        barrier.wait()  # release all threads at once → maximal contention
        _r, ok = store.create_routine_if_under_quota(
            _routine("acme", f"r{i}"), cap=cap
        )
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=_create, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == cap, f"admitted {sum(results)} != cap {cap}"
    assert store.count(workspace_id="acme", enabled_only=True) == cap


def test_routine_quota_is_per_tenant() -> None:
    """Two workspaces each reach their FULL cap independently (no cross-tenant serialise)."""
    store = RoutinesStore(":memory:")
    cap = 2
    for ws in ("acme", "globex"):
        admitted = sum(
            int(store.create_routine_if_under_quota(_routine(ws, f"r{i}"), cap=cap)[1])
            for i in range(5)
        )
        assert admitted == cap
    assert store.count(workspace_id="acme", enabled_only=True) == cap
    assert store.count(workspace_id="globex", enabled_only=True) == cap


def test_routine_quota_disabled_routines_do_not_count() -> None:
    """A disabled routine does not consume an enabled-cap slot."""
    store = RoutinesStore(":memory:")
    _r, ok = store.create_routine_if_under_quota(
        _routine("acme", "off", enabled=False), cap=1
    )
    assert ok
    # The disabled one does not count, so an enabled create still fits under cap=1.
    _r2, ok2 = store.create_routine_if_under_quota(_routine("acme", "on"), cap=1)
    assert ok2


def test_routine_quota_zero_cap_admits_unconditionally() -> None:
    """cap<=0 disables the quota: every create is admitted (plain upsert)."""
    store = RoutinesStore(":memory:")
    for i in range(5):
        _r, ok = store.create_routine_if_under_quota(_routine("acme", f"r{i}"), cap=0)
        assert ok


# =========================================================== (B) outstanding-run quota


@pytest.mark.parametrize("backend", ["inmemory", "sqlite"])
def test_run_quota_serial_lands_exactly_at_cap(backend: str, tmp_path: Path) -> None:
    """Serial creates: the (cap+1)-th is rejected; exactly cap active runs remain."""
    if backend == "inmemory":
        storage: StorageService | SqliteStorageService = StorageService()
    else:
        storage = SqliteStorageService(str(tmp_path / "store.db"))
    cap = 3

    async def _scenario() -> None:
        admitted = 0
        for _ in range(6):
            _r, ok = await storage.save_run_if_under_quota(_run("acme"), cap=cap)
            admitted += int(ok)
        assert admitted == cap
        assert await storage.count_active_runs_for_workspace("acme") == cap

    run_async(_scenario())


def test_run_quota_concurrent_no_overshoot_inmemory() -> None:
    """Concurrent in-memory creates on one loop land EXACTLY at cap (no interleave overshoot)."""
    storage = StorageService()
    cap = 2
    n = 16

    async def _scenario() -> None:
        results = await asyncio.gather(
            *(storage.save_run_if_under_quota(_run("acme"), cap=cap) for _ in range(n))
        )
        admitted = sum(1 for _r, ok in results if ok)
        assert admitted == cap
        assert await storage.count_active_runs_for_workspace("acme") == cap

    run_async(_scenario())


def test_run_quota_concurrent_no_overshoot_sqlite(tmp_path: Path) -> None:
    """Concurrent SQLite creates (real ``to_thread`` write-lock contention) land at cap."""
    storage = SqliteStorageService(str(tmp_path / "store.db"))
    cap = 2
    n = 16

    async def _scenario() -> None:
        results = await asyncio.gather(
            *(storage.save_run_if_under_quota(_run("acme"), cap=cap) for _ in range(n))
        )
        admitted = sum(1 for _r, ok in results if ok)
        assert admitted == cap
        assert await storage.count_active_runs_for_workspace("acme") == cap

    run_async(_scenario())


def test_run_quota_idempotent_resubmit_does_not_consume_slot() -> None:
    """A re-submit of an existing idempotency key returns the prior run, not a new slot."""
    storage = StorageService()

    async def _scenario() -> None:
        r1, ok1 = await storage.save_run_if_under_quota(
            _run("acme", idem="k1"), cap=1
        )
        assert ok1
        # Re-submit the SAME key under cap=1 (already full): must return the prior run,
        # admitted=True, WITHOUT a spurious 429 or a second slot.
        r2, ok2 = await storage.save_run_if_under_quota(
            _run("acme", idem="k1"), cap=1
        )
        assert ok2 and r2.run_id == r1.run_id
        assert await storage.count_active_runs_for_workspace("acme") == 1

    run_async(_scenario())


def test_run_quota_is_per_tenant() -> None:
    """One tenant at its cap does not block another tenant's create."""
    storage = StorageService()

    async def _scenario() -> None:
        _r, ok = await storage.save_run_if_under_quota(_run("acme"), cap=1)
        assert ok
        _r2, ok2 = await storage.save_run_if_under_quota(_run("acme"), cap=1)
        assert not ok2  # acme at cap
        _r3, ok3 = await storage.save_run_if_under_quota(_run("globex"), cap=1)
        assert ok3  # globex independent

    run_async(_scenario())


def test_run_quota_zero_cap_admits_unconditionally() -> None:
    """cap<=0 disables the quota: every create is admitted."""
    storage = StorageService()

    async def _scenario() -> None:
        for _ in range(10):
            _r, ok = await storage.save_run_if_under_quota(_run("acme"), cap=0)
            assert ok

    run_async(_scenario())
