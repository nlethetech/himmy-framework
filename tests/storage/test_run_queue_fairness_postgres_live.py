"""T3: per-tenant fairness + HARD cross-node concurrency cap on LIVE Postgres.

SKIPPED unless ``HIMMY_TEST_POSTGRES_DSN`` is set AND ``asyncpg`` is importable. Proves the
fairness claim against the REAL backend (the offline in-RAM/SQLite tests cannot exercise the
``pg_try_advisory_xact_lock`` per-workspace serialisation that makes the cap a HARD cap under
concurrent claimers). The cross-process proof lives in ``scripts/scheduler_fairness_e2e.py``;
this asserts the SQL-level guarantees on one connection pool.
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


async def _drain(storage: PostgresStorageService) -> None:
    while await storage.claim_next_queued_run("drain", 300, fairness=True) is not None:
        pass


def _uid(p: str) -> str:
    return f"{p}-{uuid.uuid4()}"


def test_fairness_prefers_least_loaded_workspace() -> None:
    """With one tenant already running, the next claim prefers the quieter tenant."""

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            await _drain(storage)
            wa, wb = _uid("wa"), _uid("wb")
            for _ in range(3):
                await storage.save_run(
                    RunRecord(
                        run_id=_uid("a"), workspace_id=wa, subject_id="s",
                        status=RunStatus.QUEUED,
                    )
                )
            await storage.save_run(
                RunRecord(
                    run_id=_uid("b"), workspace_id=wb, subject_id="s",
                    status=RunStatus.QUEUED,
                )
            )
            # First claim (both at 0 running) -> the oldest, which is wa's.
            c1 = await storage.claim_next_queued_run("o1", 600, fairness=True)
            assert c1 is not None and c1.workspace_id == wa
            # Second claim: wa now has 1 live run, wb has 0 -> wb is least-loaded.
            c2 = await storage.claim_next_queued_run("o2", 600, fairness=True)
            assert c2 is not None and c2.workspace_id == wb
        finally:
            await storage.close()

    run_async(scenario())


def test_hard_cap_holds_under_concurrent_claims() -> None:
    """Concurrent claimers NEVER push a tenant above its cap; the cap fills over polls.

    The per-workspace ``pg_try_advisory_xact_lock`` serialises claims for one workspace, so a
    single concurrent burst may under-fill (a contended workspace is skipped, not waited on) —
    that is SAFE (never over-cap). Like the real dispatcher, we poll in rounds: assert the live
    RUNNING count NEVER exceeds the cap and eventually REACHES it.
    """

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            await _drain(storage)
            ws = _uid("ws")
            for _ in range(10):
                await storage.save_run(
                    RunRecord(
                        run_id=_uid("r"), workspace_id=ws, subject_id="s",
                        status=RunStatus.QUEUED,
                    )
                )
            cap = 3
            won: set[str] = set()
            # Poll rounds of concurrent claims (the dispatcher loop). Long lease so nothing
            # frees mid-test: the cap is a HARD ceiling on concurrent live runs.
            for _ in range(6):
                results = await asyncio.gather(
                    *(
                        storage.claim_next_queued_run(
                            f"o{i}", 600, fairness=True, workspace_concurrency=cap
                        )
                        for i in range(8)
                    )
                )
                for r in results:
                    if r is not None:
                        won.add(r.run_id)
                # INVARIANT: never above the cap.
                assert len(won) <= cap, f"cap breached: {len(won)} > {cap}"
                if len(won) == cap:
                    break
            assert len(won) == cap, f"cap never filled: {len(won)} of {cap}"
        finally:
            await storage.close()

    run_async(scenario())


def test_advisory_keys_distinct_namespace() -> None:
    """The fair-claim advisory namespace differs from the migration/leader keys (no collision)."""
    from himmy.services.storage.postgres import (
        FAIR_CLAIM_ADVISORY_NAMESPACE,
        MIGRATION_ADVISORY_LOCK_KEY,
    )

    # The migration key is a 64-bit single-int lock; the fair-claim cap uses the TWO-int form
    # under a small fixed namespace — different lock spaces in Postgres, never contend.
    assert FAIR_CLAIM_ADVISORY_NAMESPACE != MIGRATION_ADVISORY_LOCK_KEY
