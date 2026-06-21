#!/usr/bin/env python3
"""Real multi-PROCESS proof of the cross-node scheduling guarantees on LIVE Postgres.

This is NOT a unit test — it spawns real OS processes (each a distinct "node" with its own
asyncpg pool / Postgres session) against a throwaway database, to prove on the real backend
exactly what an in-process fake cannot:

  PHASE A — LEADER ELECTION is exactly-one + clean FAILOVER.
    N processes each bid for the scheduler-leader advisory lease. Exactly ONE wins. The
    winner is then KILLED (SIGKILL — a process death, no graceful unlock); Postgres releases
    the dropped session's advisory lock automatically, and a fresh bid by another process
    then succeeds. No permanently-held lock, no double-leader.

  PHASE B — the per-fire dedup-CAS is at-most-once across PROCESSES (the leader-free backstop).
    N processes simultaneously race ``dedup_try_claim`` for the SAME schedule key
    ``(routine_id, planned_fire_at)`` against the shared Postgres. EXACTLY ONE gets
    ``CLAIM_WON``; every other gets a duplicate verdict. This is the guarantee that makes a
    brief double-leader (or leaderless) window safe — the fire still happens once.

Usage::

    HIMMY_DATABASE_URL=postgresql://user@host:5432/himmy_sched_e2e \\
        python scripts/scheduler_multinode_e2e.py [--nodes 8]

Exits 0 on success, non-zero on any violated invariant. The caller owns DB lifecycle (create
the throwaway DB + schema before, drop it after) — this script only reads/writes the dedup +
advisory-lock state and cleans up its own dedup rows.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from multiprocessing import Process, Queue


def _dsn() -> str:
    dsn = os.environ.get("HIMMY_DATABASE_URL") or os.environ.get(
        "HIMMY_TEST_POSTGRES_DSN"
    )
    if not dsn:
        print(
            "error: set HIMMY_DATABASE_URL (or HIMMY_TEST_POSTGRES_DSN) to the throwaway DB",
            file=sys.stderr,
        )
        sys.exit(2)
    return dsn


# --------------------------------------------------------------------- PHASE A


def _leader_bid_proc(dsn: str, hold_event_path: str, result_q: Queue) -> None:
    """Child: bid for the lease; if won, signal + hold it until the file is removed."""

    async def run() -> None:
        from himmy.services.storage.postgres import PostgresStorageService

        storage = await PostgresStorageService.connect(dsn)
        try:
            handle = await storage.try_acquire_scheduler_leader()
            won = handle is not None
            result_q.put((os.getpid(), won))
            if won:
                # Hold the lease (and thus the session) until told to stop, OR forever (this
                # process is the one the parent SIGKILLs to prove failover).
                import time

                while os.path.exists(hold_event_path):
                    time.sleep(0.05)
                await storage.release_scheduler_leader(handle)
        finally:
            # NB: on SIGKILL this never runs — that is the POINT (Postgres auto-releases).
            await storage.close()

    asyncio.run(run())


def phase_a_leader_election(dsn: str, nodes: int) -> bool:
    """N processes bid; assert exactly one wins; kill it; assert another can then win."""
    import tempfile
    import time

    print(f"[PHASE A] leader election across {nodes} processes")
    hold = os.path.join(tempfile.gettempdir(), f"himmy-e2e-hold-{uuid.uuid4().hex}")
    open(hold, "w").close()  # presence == "keep holding the lease"
    q: Queue = Queue()
    procs = [
        Process(target=_leader_bid_proc, args=(dsn, hold, q), name=f"node-{i}")
        for i in range(nodes)
    ]
    for p in procs:
        p.start()

    results: dict[int, bool] = {}
    deadline = time.time() + 30
    while len(results) < nodes and time.time() < deadline:
        try:
            pid, won = q.get(timeout=5)
            results[pid] = won
        except Exception:  # noqa: BLE001 - queue timeout
            break

    winners = [pid for pid, won in results.items() if won]
    ok = len(winners) == 1
    print(f"  bids={len(results)} winners={len(winners)} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        os.remove(hold)
        for p in procs:
            p.terminate()
        return False

    # FAILOVER: SIGKILL the winner (process death, no graceful unlock). Postgres must release
    # the advisory lock on the dropped session so a fresh bid wins.
    winner_pid = winners[0]
    winner_proc = next(p for p in procs if p.pid == winner_pid)
    print(f"  SIGKILL leader pid={winner_pid} (proving auto-release on death)")
    winner_proc.kill()
    winner_proc.join(timeout=5)

    async def _rebid() -> bool:
        from himmy.services.storage.postgres import PostgresStorageService

        storage = await PostgresStorageService.connect(dsn)
        try:
            handle = None
            for _ in range(100):  # ~10s: backend cleanup of the killed session is near-instant
                handle = await storage.try_acquire_scheduler_leader()
                if handle is not None:
                    break
                await asyncio.sleep(0.1)
            if handle is None:
                return False
            await storage.release_scheduler_leader(handle)
            return True
        finally:
            await storage.close()

    failover_ok = asyncio.run(_rebid())
    print(f"  re-bid after leader death -> {'OK' if failover_ok else 'FAIL'}")

    # Release the remaining holders + clean up.
    os.remove(hold)
    for p in procs:
        if p.is_alive():
            p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    return ok and failover_ok


# --------------------------------------------------------------------- PHASE B


def _dedup_claim_proc(dsn: str, scope: str, key: str, barrier_q: Queue, result_q: Queue) -> None:
    """Child: wait for the go signal, then race a single dedup claim for (scope, key)."""

    async def run() -> None:
        from himmy.services.storage.postgres import PostgresStorageService
        from himmy.services.storage.trigger_dedup import CLAIM_WON

        storage = await PostgresStorageService.connect(dsn)
        try:
            barrier_q.put(os.getpid())  # "I'm ready"
            # Spin until the parent flips the go flag (broadcast via a shared file).
            go_path = key + ".go"
            import time

            while not os.path.exists(go_path):
                time.sleep(0.005)
            claim = await storage.dedup_try_claim(scope, key, lease_seconds=300)
            result_q.put((os.getpid(), claim.outcome == CLAIM_WON))
        finally:
            await storage.close()

    asyncio.run(run())


def phase_b_dedup_cas(dsn: str, nodes: int) -> bool:
    """N processes race the SAME schedule-key claim; assert exactly one wins."""
    import tempfile
    import time

    print(f"[PHASE B] per-fire dedup-CAS across {nodes} processes (one schedule key)")
    routine_id = f"routine-{uuid.uuid4().hex}"
    planned = "2026-06-21T06:30:00+00:00"
    scope = "schedule"
    base = os.path.join(tempfile.gettempdir(), f"himmy-e2e-dedup-{uuid.uuid4().hex}")
    key = f"{base}|{routine_id}:{planned}"  # base path doubles as the go-flag prefix

    barrier: Queue = Queue()
    results: Queue = Queue()
    procs = [
        Process(
            target=_dedup_claim_proc,
            args=(dsn, scope, key, barrier, results),
            name=f"claimer-{i}",
        )
        for i in range(nodes)
    ]
    for p in procs:
        p.start()

    # Wait for every child to report ready, then fire the go flag so they race together.
    ready = 0
    deadline = time.time() + 30
    while ready < nodes and time.time() < deadline:
        try:
            barrier.get(timeout=5)
            ready += 1
        except Exception:  # noqa: BLE001
            break
    open(key + ".go", "w").close()

    outcomes: dict[int, bool] = {}
    deadline = time.time() + 30
    while len(outcomes) < nodes and time.time() < deadline:
        try:
            pid, won = results.get(timeout=5)
            outcomes[pid] = won
        except Exception:  # noqa: BLE001
            break
    for p in procs:
        p.join(timeout=5)

    winners = [pid for pid, won in outcomes.items() if won]
    ok = len(outcomes) == nodes and len(winners) == 1
    print(
        f"  claims={len(outcomes)} winners={len(winners)} -> {'OK' if ok else 'FAIL'}"
    )

    # Cleanup: remove the go flag + the dedup row we created.
    try:
        os.remove(key + ".go")
    except OSError:
        pass

    async def _cleanup() -> None:
        from himmy.services.storage.postgres import PostgresStorageService

        storage = await PostgresStorageService.connect(dsn)
        try:
            await storage.dedup_complete(scope, key, result="fired", ttl_seconds=0)
            await storage.dedup_sweep()
        finally:
            await storage.close()

    try:
        asyncio.run(_cleanup())
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        pass
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=8, help="processes per phase")
    args = parser.parse_args()
    dsn = _dsn()

    a = phase_a_leader_election(dsn, args.nodes)
    b = phase_b_dedup_cas(dsn, args.nodes)
    ok = a and b
    print(f"\nRESULT: leader-election={'PASS' if a else 'FAIL'} "
          f"dedup-CAS={'PASS' if b else 'FAIL'} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
