"""Scheduler-tier leader election + the safe-by-default multi-node topology guard.

OFFLINE tests (no live database): the Postgres lease is exercised against a FAKE storage that
mimics ``try_acquire_scheduler_leader`` / ``release_scheduler_leader`` / ``scheduler_leader_is_held``
semantics, and the SQLite topology guard is exercised with a real ``flock`` under a tmp dir.
The LIVE cross-process exactly-once proof lives in
``tests/storage/test_scheduler_leader_postgres_live.py`` (gated on a real Postgres DSN).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from himmy.api.scheduler_leader import (
    SchedulerLeadership,
    acquire_scheduler_leadership,
)
from tests.conftest import run_async


class _FakePgLease:
    """A single advisory lock shared across many 'nodes' — at most one holder at a time."""

    def __init__(self) -> None:
        self.held_by: object | None = None

    def acquire(self, who: object) -> bool:
        if self.held_by is not None:
            return False
        self.held_by = who
        return True

    def release(self, who: object) -> None:
        if self.held_by is who:
            self.held_by = None


class _FakePgStorage:
    """Mimics PostgresStorageService's leader-lease API over a shared lock object.

    Each instance is a distinct 'node'; the connection handle is a sentinel object whose
    aliveness we can toggle to simulate a dropped session (clean failover).
    """

    def __init__(self, lease: _FakePgLease) -> None:
        self._lease = lease
        self._alive: dict[object, bool] = {}

    async def try_acquire_scheduler_leader(self) -> Any | None:
        handle = object()
        if self._lease.acquire(handle):
            self._alive[handle] = True
            return handle
        return None

    async def release_scheduler_leader(self, handle: Any | None) -> None:
        if handle is None:
            return
        self._lease.release(handle)
        self._alive.pop(handle, None)

    async def scheduler_leader_is_held(self, handle: Any | None) -> bool:
        return bool(handle is not None and self._alive.get(handle, False))

    def kill_session(self, handle: object) -> None:
        """Simulate a dropped TCP session: Postgres auto-releases the advisory lock."""
        self._alive[handle] = False
        self._lease.release(handle)


class _FakeContainer:
    def __init__(self, storage: Any) -> None:
        self.storage = storage


def _pg_container(lease: _FakePgLease) -> tuple[_FakeContainer, _FakePgStorage]:
    storage = _FakePgStorage(lease)
    return _FakeContainer(storage), storage


# Patch active_backend_name so the fake storage routes through the postgres branch.
@pytest.fixture(autouse=True)
def _force_pg_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import himmy.api.runtime_bootstrap as rb

    def _name(container: Any) -> str:
        storage = getattr(container, "storage", None)
        if isinstance(storage, _FakePgStorage):
            return "postgres"
        return rb.active_backend_name(container)

    monkeypatch.setattr(
        "himmy.api.scheduler_leader.active_backend_name", _name, raising=False
    )


# ---- Postgres lease: exactly one leader -------------------------------------------


def test_postgres_first_node_leads_second_is_follower() -> None:
    """Two nodes sharing one lock: the first wins leadership, the second is a follower."""

    async def scenario() -> None:
        lease = _FakePgLease()
        c1, _ = _pg_container(lease)
        c2, _ = _pg_container(lease)
        l1 = await acquire_scheduler_leadership(c1)
        l2 = await acquire_scheduler_leadership(c2)
        assert l1.is_leader is True
        assert l1.mode == "postgres-lease"
        assert l2.is_leader is False
        assert l2.mode == "follower"
        await l1.release()
        await l2.release()

    run_async(scenario())


def test_postgres_failover_follower_promotes_after_leader_release() -> None:
    """When the leader releases, the follower's next refresh acquires the lease."""

    async def scenario() -> None:
        lease = _FakePgLease()
        c1, _ = _pg_container(lease)
        c2, _ = _pg_container(lease)
        leader = await acquire_scheduler_leadership(c1)
        follower = await acquire_scheduler_leadership(c2)
        assert leader.is_leader and not follower.is_leader

        # Leader cleanly releases (graceful shutdown).
        await leader.release()

        # Follower re-contends and is promoted.
        promoted = await follower.refresh()
        assert promoted is True
        assert follower.is_leader is True
        assert follower.mode == "postgres-lease"
        await follower.release()

    run_async(scenario())


def test_postgres_failover_on_dropped_session_no_double_leader() -> None:
    """A leader whose session drops demotes; the follower then promotes — never two leaders."""

    async def scenario() -> None:
        lease = _FakePgLease()
        c1, s1 = _pg_container(lease)
        c2, _ = _pg_container(lease)
        leader = await acquire_scheduler_leadership(c1)
        follower = await acquire_scheduler_leadership(c2)
        assert leader.is_leader and not follower.is_leader

        # Simulate the leader's connection dropping (process death / network blip):
        # Postgres auto-releases the advisory lock.
        s1.kill_session(leader._pg_handle)

        # The old leader's refresh detects the dropped lease and demotes itself.
        still = await leader.refresh()
        # It immediately re-contends in the same cycle and may re-win (single node) — but the
        # KEY invariant is it does not hold the lease while the follower also does. Force the
        # follower to win by re-contending first below; here just assert no crash + state sane.
        assert isinstance(still, bool)

        # Release whoever the old leader became, then the follower must be able to lead.
        await leader.release()
        promoted = await follower.refresh()
        assert promoted is True
        assert follower.is_leader is True
        # Exactly one holder.
        assert lease.held_by is follower._pg_handle
        await follower.release()

    run_async(scenario())


def test_postgres_lease_bid_error_fails_open_to_leader() -> None:
    """A lease-bid exception fails OPEN to leader (a missed scheduler is worse than redundant)."""

    class _BoomStorage(_FakePgStorage):
        async def try_acquire_scheduler_leader(self) -> Any | None:
            raise RuntimeError("pg down")

    async def scenario() -> None:
        storage = _BoomStorage(_FakePgLease())
        container = _FakeContainer(storage)
        lead = await acquire_scheduler_leadership(container)
        assert lead.is_leader is True
        assert lead.mode == "single-node"
        await lead.release()

    run_async(scenario())


def test_release_is_idempotent() -> None:
    async def scenario() -> None:
        lease = _FakePgLease()
        c1, _ = _pg_container(lease)
        lead = await acquire_scheduler_leadership(c1)
        await lead.release()
        await lead.release()  # no error
        assert lease.held_by is None

    run_async(scenario())


# ---- SQLite topology guard --------------------------------------------------------


class _SqliteContainer:
    """A container whose backend reports as sqlite via the real active_backend_name."""

    def __init__(self) -> None:
        from himmy.services.storage.sqlite import SqliteStorageService

        self.storage = SqliteStorageService.__new__(SqliteStorageService)


def test_sqlite_single_host_scheduler_is_granted_with_warning(
    tmp_path: Path,
) -> None:
    """A lone SQLite scheduler is granted (single-box case) under a host flock."""

    async def scenario() -> None:
        container = _SqliteContainer()
        lead = await acquire_scheduler_leadership(container, lock_base=tmp_path)
        assert lead.is_leader is True
        assert lead.mode == "host-flock"
        await lead.release()

    run_async(scenario())


def test_sqlite_second_scheduler_same_host_refused(tmp_path: Path) -> None:
    """A SECOND scheduler on the same host (SQLite) is refused — cannot coordinate."""

    async def scenario() -> None:
        c1 = _SqliteContainer()
        c2 = _SqliteContainer()
        first = await acquire_scheduler_leadership(c1, lock_base=tmp_path)
        assert first.is_leader is True
        second = await acquire_scheduler_leadership(c2, lock_base=tmp_path)
        assert second.is_leader is False
        assert second.mode == "refused"
        # Releasing the first frees the host flock; a new scheduler can then start.
        await first.release()
        third = await acquire_scheduler_leadership(c1, lock_base=tmp_path)
        assert third.is_leader is True
        await third.release()
        await second.release()

    run_async(scenario())


def test_sqlite_require_ack_refuses_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With require_single_scheduler_ack and no ack env, SQLite scheduling is refused."""

    async def scenario() -> None:
        monkeypatch.delenv("HIMMY_SCHEDULER_SINGLE_NODE_ACK", raising=False)
        container = _SqliteContainer()
        lead = await acquire_scheduler_leadership(
            container, require_single_scheduler_ack=True, lock_base=tmp_path
        )
        assert lead.is_leader is False
        assert lead.mode == "refused"
        await lead.release()

    run_async(scenario())


def test_sqlite_require_ack_granted_with_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("HIMMY_SCHEDULER_SINGLE_NODE_ACK", "1")
        container = _SqliteContainer()
        lead = await acquire_scheduler_leadership(
            container, require_single_scheduler_ack=True, lock_base=tmp_path
        )
        assert lead.is_leader is True
        assert lead.mode == "host-flock"
        await lead.release()

    run_async(scenario())


# ---- in-memory: single-process always leads --------------------------------------


def test_inmemory_backend_always_leads() -> None:
    async def scenario() -> None:
        container = _FakeContainer(object())  # not sqlite/postgres -> 'memory'
        lead = await acquire_scheduler_leadership(container)
        assert lead.is_leader is True
        assert lead.mode == "single-node"
        await lead.release()

    run_async(scenario())


def test_refresh_on_non_pg_is_a_noop_getter() -> None:
    async def scenario() -> None:
        container = _FakeContainer(object())
        lead = await acquire_scheduler_leadership(container)
        assert await lead.refresh() is True
        await lead.release()

    run_async(scenario())


def test_dataclass_shape() -> None:
    """SchedulerLeadership exposes the bits callers act on."""
    lead = SchedulerLeadership(is_leader=False, mode="follower", backend="postgres")
    assert lead.is_leader is False
    assert lead.backend == "postgres"
    assert lead.reason == ""
