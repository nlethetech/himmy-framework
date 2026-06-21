"""LIVE Postgres proof: scheduler-leader advisory lease is exactly-one + clean failover.

SKIPPED unless ``HIMMY_TEST_POSTGRES_DSN`` is set AND ``asyncpg`` is importable (the same gate
as the rest of the Postgres suite). This is the test that proves — on a REAL Postgres, across
SEPARATE connections (the cross-node case the offline fakes cannot exercise) — that:

* exactly ONE node holds the scheduler-leader lease at a time (``pg_try_advisory_lock``);
* a leader's connection DROP (process death) auto-releases the lease (clean failover) so the
  next contender wins, with no permanently-held lock;
* the lease lives in a DIFFERENT namespace from the migration lock (they never block each
  other).

The cross-process exactly-ONCE *fire* proof under multiple OS worker processes is driven by
``scripts/scheduler_multinode_e2e.py`` (not a unit test — it spawns real processes).
"""

from __future__ import annotations

import os

import pytest

from himmy.services.storage.postgres import (
    MIGRATION_ADVISORY_LOCK_KEY,
    SCHEDULER_LEADER_LOCK_KEY,
    PostgresStorageService,
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


def test_lease_keys_are_distinct_namespaces() -> None:
    """The leader lease key must differ from the migration lock key (no cross-block)."""
    assert SCHEDULER_LEADER_LOCK_KEY != MIGRATION_ADVISORY_LOCK_KEY


def test_only_one_node_holds_the_lease() -> None:
    """Two independent storage 'nodes' contend: exactly one acquires the lease."""

    async def scenario() -> None:
        node_a = await PostgresStorageService.connect(_DSN)
        node_b = await PostgresStorageService.connect(_DSN)
        try:
            ha = await node_a.try_acquire_scheduler_leader()
            hb = await node_b.try_acquire_scheduler_leader()
            # Exactly one wins.
            assert (ha is None) != (hb is None)
            winner = node_a if ha is not None else node_b
            handle = ha if ha is not None else hb
            assert await winner.scheduler_leader_is_held(handle) is True
            # The other genuinely could not acquire.
            loser = node_b if ha is not None else node_a
            assert (await loser.try_acquire_scheduler_leader()) is None
            # Release frees it for the loser.
            await winner.release_scheduler_leader(handle)
            h2 = await loser.try_acquire_scheduler_leader()
            assert h2 is not None
            await loser.release_scheduler_leader(h2)
        finally:
            await node_a.close()
            await node_b.close()

    from tests.conftest import run_async

    run_async(scenario())


def test_dropped_session_auto_releases_lease() -> None:
    """Killing the leader's connection (process-death proxy) auto-releases the lease."""

    async def scenario() -> None:
        leader = await PostgresStorageService.connect(_DSN)
        contender = await PostgresStorageService.connect(_DSN)
        try:
            handle = await leader.try_acquire_scheduler_leader()
            assert handle is not None
            # Contender is locked out while the lease is live.
            assert (await contender.try_acquire_scheduler_leader()) is None

            # Simulate process death: FORCE-terminate the leader's pool WITHOUT a graceful
            # advisory_unlock (graceful pool.close() would block on the held lease connection —
            # which is itself the proof the lease holds a live connection). pool.terminate()
            # is synchronous and drops every backend connection, so Postgres releases the
            # session's advisory lock automatically — exactly the clean-failover guarantee.
            leader._pool.terminate()

            # The contender can now win (poll briefly: backend cleanup is near-instant but the
            # server may take a beat to notice the dropped socket).
            import asyncio

            won = None
            for _ in range(50):
                won = await contender.try_acquire_scheduler_leader()
                if won is not None:
                    break
                await asyncio.sleep(0.1)
            assert won is not None, "failover did not free the lease after leader death"
            await contender.release_scheduler_leader(won)
        finally:
            import contextlib

            # leader's pool was already terminated; terminate() is idempotent + non-blocking.
            with contextlib.suppress(Exception):
                leader._pool.terminate()
            await contender.close()

    from tests.conftest import run_async

    run_async(scenario())


def test_leadership_object_over_live_postgres() -> None:
    """The SchedulerLeadership wrapper acquires + releases against real Postgres."""

    async def scenario() -> None:
        from himmy.api.scheduler_leader import acquire_scheduler_leadership

        class _C:
            pass

        node_a = await PostgresStorageService.connect(_DSN)
        node_b = await PostgresStorageService.connect(_DSN)
        try:
            ca, cb = _C(), _C()
            ca.storage = node_a  # type: ignore[attr-defined]
            cb.storage = node_b  # type: ignore[attr-defined]
            la = await acquire_scheduler_leadership(ca)
            lb = await acquire_scheduler_leadership(cb)
            assert la.is_leader is True
            assert lb.is_leader is False
            # Leader releases; follower refresh promotes.
            await la.release()
            assert (await lb.refresh()) is True
            await lb.release()
        finally:
            await node_a.close()
            await node_b.close()

    from tests.conftest import run_async

    run_async(scenario())
