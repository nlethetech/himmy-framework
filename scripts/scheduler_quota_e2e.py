#!/usr/bin/env python
"""T3 multi-node e2e: the HARD per-tenant QUOTAS across REAL OS processes (live PostgreSQL 16).

Proves on a LIVE PostgreSQL 16 (not simulated) that the two per-tenant quotas are HARD — a
clean check-then-act under genuine multi-PROCESS concurrency lands EXACTLY at the cap, never
the old unlocked count-then-create/enqueue overshoot:

* ROUTINE-COUNT quota: N processes each fire one ``create_routine_if_under_quota`` for ONE
  workspace at cap=K → exactly K admitted, exactly K rows stored, the rest a clean reject.
* OUTSTANDING-RUN quota: N processes each fire one ``save_run_if_under_quota`` for ONE
  workspace at cap=K → exactly K admitted, exactly K active runs, the rest a clean reject.
* PER-TENANT independence: two workspaces hammered concurrently each reach their FULL cap (the
  per-workspace advisory key means they never serialise against each other).
* NO LEAKED ADVISORY LOCKS: after the storm, ``pg_locks`` holds zero advisory locks (the
  xact-scoped lock auto-released on every commit/rollback).

Usage (driver):
    python scripts/scheduler_quota_e2e.py --nodes 8 --cap 2

It creates a THROWAWAY database (default himmy_quota_e2e), inits the schema, runs the proof,
and DROPS the database. The existing bholi / nagarik / postgres databases are never touched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid

ADMIN_DSN = os.environ.get(
    "HIMMY_QUOTA_E2E_ADMIN_DSN", "postgresql://samriddhagc@localhost:5432/postgres"
)
TEST_DB = os.environ.get("HIMMY_QUOTA_E2E_DB", "himmy_quota_e2e")
TEST_DSN = f"postgresql://samriddhagc@localhost:5432/{TEST_DB}"

# A protected list: the driver NEVER drops any of these even by misconfig.
_PROTECTED = {"bholi", "nagarik", "postgres", "template0", "template1"}


# --------------------------------------------------------------------------- workers


async def _routine_worker(workspace: str, name: str, cap: int) -> None:
    """One PROCESS: attempt a single quota-gated routine create; emit the outcome as JSON."""
    import asyncpg  # noqa: F401

    from himmy.api.routines import Routine, Schedule
    from himmy.services.storage.aux_store_factory import reset_aux_store_factory
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    os.environ["HIMMY_DATABASE_URL"] = TEST_DSN
    reset_aux_store_factory()
    store = PostgresRoutinesStore(tenant="local", dsn=TEST_DSN)
    routine = Routine(
        workspace_id=workspace,
        agent_id="agent-1",
        name=name,
        prompt="hi",
        schedule=Schedule(kind="daily", at="07:00"),
        enabled=True,
    )
    _r, admitted = store.create_routine_if_under_quota(routine, cap=cap)
    print(json.dumps({"workspace": workspace, "admitted": bool(admitted)}))


async def _run_worker(workspace: str, cap: int) -> None:
    """One PROCESS: attempt a single quota-gated run create; emit the outcome as JSON."""
    import asyncpg  # noqa: F401

    from himmy.services.storage.models import RunRecord, RunStatus
    from himmy.services.storage.postgres import PostgresStorageService

    storage = await PostgresStorageService.connect(TEST_DSN, min_size=1, max_size=4)
    try:
        run = RunRecord(workspace_id=workspace, subject_id="s", status=RunStatus.QUEUED)
        _r, admitted = await storage.save_run_if_under_quota(run, cap=cap)
        print(json.dumps({"workspace": workspace, "admitted": bool(admitted)}))
    finally:
        await storage.close()


# --------------------------------------------------------------------------- driver


async def _spawn(args: list[str]) -> dict:
    """Run a child PROCESS of this script with ``args``; parse its single JSON line."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        __file__,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "HIMMY_DATABASE_URL": TEST_DSN},
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"child failed ({proc.returncode}): {stderr.decode()[-2000:]}"
        )
    line = stdout.decode().strip().splitlines()[-1]
    return json.loads(line)


async def _create_db() -> None:
    import asyncpg

    if TEST_DB in _PROTECTED:
        raise SystemExit(f"refusing to use protected db {TEST_DB!r}")
    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        exists = await admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname=$1", TEST_DB
        )
        if exists:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname=$1 AND pid<>pg_backend_pid()",
                TEST_DB,
            )
            await admin.execute(f'DROP DATABASE "{TEST_DB}"')
        await admin.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await admin.close()
    from himmy.services.storage.postgres import PostgresStorageService

    storage = await PostgresStorageService.connect(TEST_DSN)
    try:
        await storage.migrate()
    finally:
        await storage.close()


async def _drop_db() -> None:
    import asyncpg

    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname=$1 AND pid<>pg_backend_pid()",
            TEST_DB,
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}"')
    finally:
        await admin.close()


async def _advisory_locks_held() -> int:
    import asyncpg

    conn = await asyncpg.connect(TEST_DSN)
    try:
        return int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM pg_locks WHERE locktype = 'advisory'"
            )
            or 0
        )
    finally:
        await conn.close()


async def _count_routines(workspace: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(TEST_DSN)
    try:
        return int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM aux_routines WHERE workspace_id=$1", workspace
            )
            or 0
        )
    finally:
        await conn.close()


async def _count_active_runs(workspace: str) -> int:
    from himmy.services.storage.postgres import PostgresStorageService

    storage = await PostgresStorageService.connect(TEST_DSN)
    try:
        return await storage.count_active_runs_for_workspace(workspace)
    finally:
        await storage.close()


async def _drive(nodes: int, cap: int) -> int:
    await _create_db()
    failures: list[str] = []
    try:
        # --- (A) routine quota: N processes, one workspace, cap=K ---------------------
        ws_a = f"acme-{uuid.uuid4().hex[:8]}"
        results = await asyncio.gather(
            *(
                _spawn(["--role", "routine", "--workspace", ws_a, "--name",
                        f"r{i}", "--cap", str(cap)])
                for i in range(nodes)
            )
        )
        admitted = sum(1 for r in results if r["admitted"])
        stored = await _count_routines(ws_a)
        print(f"[routine] admitted={admitted} stored={stored} cap={cap} nodes={nodes}")
        if admitted != cap:
            failures.append(f"routine admitted {admitted} != cap {cap}")
        if stored != cap:
            failures.append(f"routine stored {stored} != cap {cap}")

        # --- (B) run quota: N processes, one workspace, cap=K -------------------------
        ws_b = f"globex-{uuid.uuid4().hex[:8]}"
        results = await asyncio.gather(
            *(
                _spawn(["--role", "run", "--workspace", ws_b, "--cap", str(cap)])
                for i in range(nodes)
            )
        )
        admitted = sum(1 for r in results if r["admitted"])
        active = await _count_active_runs(ws_b)
        print(f"[run] admitted={admitted} active={active} cap={cap} nodes={nodes}")
        if admitted != cap:
            failures.append(f"run admitted {admitted} != cap {cap}")
        if active != cap:
            failures.append(f"run active {active} != cap {cap}")

        # --- (C) per-tenant independence: two workspaces, run quota ------------------
        ws_c1 = f"t1-{uuid.uuid4().hex[:8]}"
        ws_c2 = f"t2-{uuid.uuid4().hex[:8]}"
        await asyncio.gather(
            *(
                _spawn(["--role", "run", "--workspace", ws, "--cap", str(cap)])
                for ws in (ws_c1, ws_c2)
                for _ in range(nodes)
            )
        )
        a1 = await _count_active_runs(ws_c1)
        a2 = await _count_active_runs(ws_c2)
        print(f"[per-tenant] ws1={a1} ws2={a2} cap={cap}")
        if a1 != cap or a2 != cap:
            failures.append(f"per-tenant: ws1={a1} ws2={a2} (each should be {cap})")

        # --- (D) no leaked advisory locks -------------------------------------------
        held = await _advisory_locks_held()
        print(f"[locks] advisory held after storm = {held}")
        if held != 0:
            failures.append(f"advisory locks leaked: {held}")
    finally:
        await _drop_db()

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS: both quotas HARD under multi-process concurrency; no lock leak")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["routine", "run"], default=None)
    p.add_argument("--workspace", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--cap", type=int, default=2)
    p.add_argument("--nodes", type=int, default=8)
    args = p.parse_args()

    if args.role == "routine":
        asyncio.run(_routine_worker(args.workspace, args.name, args.cap))
        return 0
    if args.role == "run":
        asyncio.run(_run_worker(args.workspace, args.cap))
        return 0
    return asyncio.run(_drive(args.nodes, args.cap))


if __name__ == "__main__":
    raise SystemExit(main())
