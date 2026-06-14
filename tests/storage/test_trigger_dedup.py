"""Durable inbound-trigger dedup: TTL-CAS storage + DurableIdempotencyStore (Q4).

Exercises the ``trigger_dedup`` TTL-CAS surface on the in-memory + SQLite backends and the
:class:`DurableIdempotencyStore` mark-after-success protocol:

* a fresh claim WINS; a concurrent duplicate is IN_FLIGHT; a completed one is DONE;
* an expired in-flight lease (a crashed worker) is RECLAIMABLE — the work is not lost;
* a TTL-expired completed key fires AGAIN;
* a delivery id COMPLETED before a "restart" is still deduped after it (cross-instance);
* a FAILED handler RELEASES the claim so a redelivery re-runs (at-least-once, the safe side).

The Postgres path is unit-tested for its DDL/CAS SQL shape in
``test_run_queue_postgres_offline.py``-style guards and live-skipped where no DB is present.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest

from himmy.connectors.sdk import ConnectorError
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from himmy.services.storage.trigger_dedup import (
    CLAIM_DONE,
    CLAIM_IN_FLIGHT,
    CLAIM_WON,
    DurableIdempotencyStore,
)


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _backends() -> list[tuple[str, Any]]:
    """An in-memory + a file-backed SQLite store (the SQLite one durable across instances)."""
    tmp = tempfile.mkdtemp()
    return [
        ("inmemory", StorageService()),
        ("sqlite", SqliteStorageService(str(Path(tmp) / "dedup.db"))),
    ]


@pytest.mark.parametrize("label,store", _backends())
def test_ttl_cas_claim_lifecycle(label: str, store: Any) -> None:
    async def go() -> None:
        # 1. fresh claim wins
        c1 = await store.dedup_try_claim("webhook", "evt-1", lease_seconds=300)
        assert c1.outcome is CLAIM_WON, label
        # 2. a concurrent duplicate while in-flight is refused (no second run)
        c2 = await store.dedup_try_claim("webhook", "evt-1", lease_seconds=300)
        assert c2.outcome is CLAIM_IN_FLIGHT, label
        # 3. complete it -> a later claim is DONE and returns the stored result
        await store.dedup_complete(
            "webhook", "evt-1", result="reply", ttl_seconds=86400
        )
        c3 = await store.dedup_try_claim("webhook", "evt-1", lease_seconds=300)
        assert c3.outcome is CLAIM_DONE and c3.result == "reply", label

    _run(go())


@pytest.mark.parametrize("label,store", _backends())
def test_expired_inflight_lease_is_reclaimable(label: str, store: Any) -> None:
    """A worker that crashed mid-run leaves an expiring lease that the next claim reclaims."""

    async def go() -> None:
        # Claim with a lease that is already in the past relative to a future ``now``.
        won = await store.dedup_try_claim(
            "webhook", "evt-x", lease_seconds=1, now="2020-01-01T00:00:00+00:00"
        )
        assert won.outcome is CLAIM_WON, label
        # Far-future ``now``: the in-flight lease has expired -> reclaimable as WON, not lost.
        again = await store.dedup_try_claim(
            "webhook", "evt-x", lease_seconds=300, now="2099-01-01T00:00:00+00:00"
        )
        assert again.outcome is CLAIM_WON, label

    _run(go())


@pytest.mark.parametrize("label,store", _backends())
def test_completed_ttl_expiry_allows_a_later_redelivery(label: str, store: Any) -> None:
    async def go() -> None:
        await store.dedup_try_claim(
            "webhook", "evt-ttl", lease_seconds=300, now="2020-01-01T00:00:00+00:00"
        )
        await store.dedup_complete(
            "webhook",
            "evt-ttl",
            result="r",
            ttl_seconds=60,
            now="2020-01-01T00:00:00+00:00",
        )
        # Within TTL -> still DONE.
        within = await store.dedup_try_claim(
            "webhook", "evt-ttl", lease_seconds=300, now="2020-01-01T00:00:30+00:00"
        )
        assert within.outcome is CLAIM_DONE, label
        # After TTL -> a redelivery is allowed to fire again (reclaimed).
        after = await store.dedup_try_claim(
            "webhook", "evt-ttl", lease_seconds=300, now="2099-01-01T00:00:00+00:00"
        )
        assert after.outcome is CLAIM_WON, label

    _run(go())


@pytest.mark.parametrize("label,store", _backends())
def test_release_drops_only_inflight_never_completed(label: str, store: Any) -> None:
    async def go() -> None:
        await store.dedup_try_claim("webhook", "ev", lease_seconds=300)
        # Release an in-flight row -> a redelivery wins a fresh claim.
        await store.dedup_release("webhook", "ev")
        assert (
            await store.dedup_try_claim("webhook", "ev", lease_seconds=300)
        ).outcome is CLAIM_WON, label
        # Complete it, then release: a COMPLETED row must NOT be dropped.
        await store.dedup_complete("webhook", "ev", result="done", ttl_seconds=86400)
        await store.dedup_release("webhook", "ev")
        assert (
            await store.dedup_try_claim("webhook", "ev", lease_seconds=300)
        ).outcome is CLAIM_DONE, label

    _run(go())


def test_durable_store_dedup_after_success_and_failure_release() -> None:
    """run_once_async runs once on success; a failure releases so a redelivery re-runs."""
    tmp = Path(tempfile.mkdtemp()) / "d.db"
    store = DurableIdempotencyStore(SqliteStorageService(str(tmp)), scope="webhook")

    async def go() -> None:
        runs: list[str] = []

        async def ok() -> str:
            runs.append("ok")
            return "reply-ok"

        # First delivery runs the handler; a duplicate returns the cached result, no re-run.
        assert await store.run_once_async("d1", ok) == "reply-ok"
        assert await store.run_once_async("d1", ok) == "reply-ok"
        assert runs == ["ok"]

        attempts: list[int] = []

        async def boom() -> str:
            attempts.append(1)
            raise RuntimeError("handler failed")

        with pytest.raises(RuntimeError):
            await store.run_once_async("d2", boom)

        # The failure released the in-flight claim -> the redelivery actually re-runs.
        async def recover() -> str:
            attempts.append(2)
            return "recovered"

        assert await store.run_once_async("d2", recover) == "recovered"
        assert attempts == [1, 2]

    _run(go())


def test_durable_store_dedup_survives_a_restart() -> None:
    """A delivery completed before a restart is still deduped after it (new store, same file)."""
    tmp = Path(tempfile.mkdtemp()) / "d.db"

    async def go() -> None:
        store1 = DurableIdempotencyStore(SqliteStorageService(str(tmp)), scope="webhook")
        ran1: list[int] = []

        async def call1() -> str:
            ran1.append(1)
            return "first"

        assert await store1.run_once_async("evt-r", call1) == "first"
        assert ran1 == [1]

        # Simulate a restart: a brand-new store instance on the SAME database file.
        store2 = DurableIdempotencyStore(SqliteStorageService(str(tmp)), scope="webhook")
        ran2: list[int] = []

        async def call2() -> str:
            ran2.append(1)
            return "should-not-run"

        assert await store2.run_once_async("evt-r", call2) == "first"
        assert ran2 == []  # the agent did NOT fire again after restart

    _run(go())


def test_durable_store_concurrent_duplicate_raises_connector_error() -> None:
    """A live in-flight lease makes a concurrent duplicate raise (the two cannot both fire)."""
    tmp = Path(tempfile.mkdtemp()) / "d.db"
    store = DurableIdempotencyStore(SqliteStorageService(str(tmp)), scope="webhook")

    async def go() -> None:
        gate = asyncio.Event()
        release = asyncio.Event()

        async def slow() -> str:
            gate.set()
            await release.wait()
            return "slow-reply"

        first = asyncio.create_task(store.run_once_async("c1", slow))
        await gate.wait()  # the first call now holds the in-flight lease

        async def dup() -> str:  # pragma: no cover - must not run
            return "dup"

        with pytest.raises(ConnectorError):
            await store.run_once_async("c1", dup)

        release.set()
        assert await first == "slow-reply"

    _run(go())
