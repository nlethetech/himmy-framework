"""Live-PG-gated tests for the HARD per-tenant quotas (routine-count + outstanding-run).

SKIPPED unless ``HIMMY_TEST_POSTGRES_DSN`` is set AND ``asyncpg`` is importable (the same gate
as the other live aux tests). They run the ATOMIC quota ops against a REAL PostgreSQL 16,
asserting the property the offline lane cannot fully prove: under genuine concurrency the
advisory-locked count-then-create lands EXACTLY at the cap (no TOCTOU overshoot), different
workspaces never serialise against each other, and the xact-scoped advisory lock leaks
nothing (``pg_locks`` empty after).

The cross-PROCESS proof (8 OS processes vs cap=2) lives in
``scripts/scheduler_quota_e2e.py``; these are the in-suite regression guards.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from himmy.services.storage.aux_store_factory import (
    aux_loop,
    reset_aux_store_factory,
)

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


@pytest.fixture(autouse=True)
def _schema() -> object:  # pragma: no cover - only runs with a live DB
    from himmy.services.storage.postgres import PostgresStorageService

    reset_aux_store_factory()

    async def _migrate() -> None:
        storage = await PostgresStorageService.connect(_DSN)
        try:
            await storage.migrate()
        finally:
            await storage.close()

    aux_loop().run(_migrate())
    yield
    reset_aux_store_factory()


def _ws(prefix: str) -> str:  # pragma: no cover - live only
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _advisory_locks_held() -> int:  # pragma: no cover - live only
    conn = await asyncpg.connect(_DSN)
    try:
        return int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM pg_locks WHERE locktype = 'advisory'"
            )
            or 0
        )
    finally:
        await conn.close()


# =========================================================== (A) routine-count quota


def test_routine_quota_concurrent_lands_at_cap_pg() -> None:  # pragma: no cover
    """Many concurrent same-workspace routine creates on real PG land EXACTLY at cap."""
    from himmy.api.routines import Routine, Schedule
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    store = PostgresRoutinesStore(tenant="local", dsn=_DSN)
    ws = _ws("acme")
    cap, n = 2, 12

    def _attempt(i: int) -> bool:
        routine = Routine(
            workspace_id=ws,
            agent_id="agent-1",
            name=f"r{i}",
            prompt="hi",
            schedule=Schedule(kind="daily", at="07:00"),
            enabled=True,
        )
        _r, ok = store.create_routine_if_under_quota(routine, cap=cap)
        return bool(ok)

    # Drive the sync facade from a thread pool so the aux-loop transactions genuinely
    # overlap (each create blocks on the per-workspace advisory lock).
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        admitted = sum(ex.map(_attempt, range(n)))

    assert admitted == cap, f"admitted {admitted} != cap {cap}"
    assert store.count(workspace_id=ws, enabled_only=True) == cap
    assert aux_loop().run(_advisory_locks_held()) == 0


def test_routine_quota_two_workspaces_independent_pg() -> None:  # pragma: no cover
    """Two workspaces hammered concurrently each reach their FULL cap (no cross-serialise)."""
    from himmy.api.routines import Routine, Schedule
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    store = PostgresRoutinesStore(tenant="local", dsn=_DSN)
    ws1, ws2 = _ws("t1"), _ws("t2")
    cap, n = 2, 8

    def _attempt(arg: tuple[str, int]) -> None:
        ws, i = arg
        store.create_routine_if_under_quota(
            Routine(
                workspace_id=ws,
                agent_id="agent-1",
                name=f"r{i}",
                prompt="hi",
                schedule=Schedule(kind="daily", at="07:00"),
                enabled=True,
            ),
            cap=cap,
        )

    import concurrent.futures

    work = [(ws, i) for ws in (ws1, ws2) for i in range(n)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(work)) as ex:
        list(ex.map(_attempt, work))

    assert store.count(workspace_id=ws1, enabled_only=True) == cap
    assert store.count(workspace_id=ws2, enabled_only=True) == cap


# =========================================================== (B) outstanding-run quota


def test_run_quota_concurrent_lands_at_cap_pg() -> None:  # pragma: no cover
    """Many concurrent same-workspace run creates on real PG land EXACTLY at cap."""
    from himmy.services.storage.models import RunRecord, RunStatus
    from himmy.services.storage.postgres import PostgresStorageService

    ws = _ws("acme")
    cap, n = 2, 12

    async def _scenario() -> tuple[int, int]:
        storage = await PostgresStorageService.connect(_DSN, min_size=2, max_size=10)
        try:
            results = await asyncio.gather(
                *(
                    storage.save_run_if_under_quota(
                        RunRecord(
                            workspace_id=ws, subject_id="s", status=RunStatus.QUEUED
                        ),
                        cap=cap,
                    )
                    for _ in range(n)
                )
            )
            admitted = sum(1 for _r, ok in results if ok)
            active = await storage.count_active_runs_for_workspace(ws)
            return admitted, active
        finally:
            await storage.close()

    admitted, active = asyncio.run(_scenario())
    assert admitted == cap, f"admitted {admitted} != cap {cap}"
    assert active == cap
    # Advisory locks are xact-scoped → released on commit; verify none leaked.
    assert asyncio.run(_advisory_locks_held()) == 0
