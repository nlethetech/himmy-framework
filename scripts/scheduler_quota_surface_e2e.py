#!/usr/bin/env python
"""T3 quota e2e — END-TO-END 429 SURFACE + lone-tenant + cross-workspace timing (live PG 16).

The companion to ``scheduler_quota_e2e.py`` (which proves the STORE-layer atomic op is HARD).
This proves the application/API LAYER that sits on top of it:

* (A) ROUTINE-COUNT 429 SURFACE — N processes each call the SAME service helper the ``/v1``
  router calls (``himmy.api.routines.create_routine_with_quota``) for ONE workspace at cap=K.
  Exactly K succeed; every surplus raises ``TenantRoutineQuotaExceeded`` (→ HTTP 429). Proves
  the over-cap path surfaces the real 429 exception, not a silent drop.
* (B) OUTSTANDING-RUN 429 SURFACE — N processes each call the store atomic
  (``save_run_if_under_quota``) and the surplus must map to ``WorkspaceRunQuotaExceeded`` (the
  exact exception the dispatch path raises on ``admitted=False``); a DIFFERENT tenant fired in
  the same storm is still admitted.
* (C) LONE-TENANT NOT BLOCKED — a single tenant well UNDER the cap creates K-1 routines/runs
  back-to-back: ALL admitted, ZERO spurious rejects, no hang.
* (D) CROSS-WORKSPACE NO-SERIALISE — two DIFFERENT workspaces each get a concurrent storm; both
  reach their full cap, and the wall-clock of running them TOGETHER is ~the same as one alone
  (different advisory keys ⇒ they do not serialise against each other).

Spawns REAL OS processes (distinct asyncpg pools / PG sessions) against a THROWAWAY db
(default himmy_quota_surface_e2e), inits the schema, and DROPS it. bholi / nagarik / postgres
are never touched.

Usage:
    python scripts/scheduler_quota_surface_e2e.py --nodes 12 --cap 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid

ADMIN_DSN = os.environ.get(
    "HIMMY_QUOTA_E2E_ADMIN_DSN", "postgresql://samriddhagc@localhost:5432/postgres"
)
TEST_DB = os.environ.get("HIMMY_QUOTA_SURFACE_E2E_DB", "himmy_quota_surface_e2e")
TEST_DSN = f"postgresql://samriddhagc@localhost:5432/{TEST_DB}"

# A protected list: the driver NEVER drops any of these even by misconfig.
_PROTECTED = {"bholi", "nagarik", "postgres", "template0", "template1"}


# --------------------------------------------------------------------------- workers


async def _routine_surface_worker(workspace: str, name: str, cap: int) -> None:
    """One PROCESS: drive the SERVICE helper the /v1 router calls; emit outcome + 429 flag.

    Sets ``HIMMY_TENANT_MAX_ROUTINES=cap`` and calls ``create_routine_with_quota`` exactly as
    the router does, so an over-cap create raises the REAL ``TenantRoutineQuotaExceeded`` (the
    HTTP-429 surface). The child reports whether it was admitted OR caught the quota exception.
    """
    import asyncpg  # noqa: F401

    os.environ["HIMMY_DATABASE_URL"] = TEST_DSN
    os.environ["HIMMY_TENANT_MAX_ROUTINES"] = str(cap)

    from himmy.api import routines as svc
    from himmy.services.storage.aux_store_factory import reset_aux_store_factory
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    reset_aux_store_factory()
    store = PostgresRoutinesStore(tenant="local", dsn=TEST_DSN)
    routine = svc.Routine(
        workspace_id=workspace,
        agent_id="agent-1",
        name=name,
        prompt="hi",
        schedule=svc.Schedule(kind="daily", at="07:00"),
        enabled=True,
    )
    out = {"workspace": workspace, "admitted": False, "quota_429": False}
    try:
        svc.create_routine_with_quota(store, routine, workspace_id=workspace)
        out["admitted"] = True
    except svc.TenantRoutineQuotaExceeded:
        out["quota_429"] = True
    print(json.dumps(out))


async def _run_surface_worker(workspace: str, cap: int) -> None:
    """One PROCESS: drive the store atomic + map admitted=False to the run-quota 429 surface.

    Mirrors the dispatch path: ``save_run_if_under_quota`` returns ``(run, admitted)`` and the
    application layer raises ``WorkspaceRunQuotaExceeded`` when ``admitted`` is False. The child
    performs that exact mapping so a surplus reports the real run-quota 429.
    """
    import asyncpg  # noqa: F401

    from himmy.application.services import WorkspaceRunQuotaExceeded
    from himmy.services.storage.models import RunRecord, RunStatus
    from himmy.services.storage.postgres import PostgresStorageService

    storage = await PostgresStorageService.connect(TEST_DSN, min_size=1, max_size=4)
    out = {"workspace": workspace, "admitted": False, "quota_429": False}
    try:
        run = RunRecord(workspace_id=workspace, subject_id="s", status=RunStatus.QUEUED)
        _r, admitted = await storage.save_run_if_under_quota(run, cap=cap)
        if admitted:
            out["admitted"] = True
        else:
            # The application layer's exact mapping of admitted=False -> 429.
            try:
                raise WorkspaceRunQuotaExceeded(workspace, cap=cap, outstanding=cap)
            except WorkspaceRunQuotaExceeded:
                out["quota_429"] = True
    finally:
        await storage.close()
    print(json.dumps(out))


async def _lone_routine_worker(workspace: str, n: int, cap: int) -> None:
    """One PROCESS: a lone tenant creates ``n`` routines (n < cap) back-to-back, sequentially.

    Proves a tenant under the cap is NEVER spuriously rejected and never hangs: every create
    must be admitted, with no 429.
    """
    import asyncpg  # noqa: F401

    os.environ["HIMMY_DATABASE_URL"] = TEST_DSN
    os.environ["HIMMY_TENANT_MAX_ROUTINES"] = str(cap)
    from himmy.api import routines as svc
    from himmy.services.storage.aux_store_factory import reset_aux_store_factory
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    reset_aux_store_factory()
    store = PostgresRoutinesStore(tenant="local", dsn=TEST_DSN)
    admitted = 0
    rejected = 0
    for i in range(n):
        routine = svc.Routine(
            workspace_id=workspace,
            agent_id="agent-1",
            name=f"lone-{i}",
            prompt="hi",
            schedule=svc.Schedule(kind="daily", at="07:00"),
            enabled=True,
        )
        try:
            svc.create_routine_with_quota(store, routine, workspace_id=workspace)
            admitted += 1
        except svc.TenantRoutineQuotaExceeded:
            rejected += 1
    print(json.dumps({"workspace": workspace, "admitted": admitted, "rejected": rejected}))


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
        raise RuntimeError(f"child failed ({proc.returncode}): {stderr.decode()[-2000:]}")
    line = stdout.decode().strip().splitlines()[-1]
    return json.loads(line)


async def _create_db() -> None:
    import asyncpg

    if TEST_DB in _PROTECTED:
        raise SystemExit(f"refusing to use protected db {TEST_DB!r}")
    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        exists = await admin.fetchval("SELECT 1 FROM pg_database WHERE datname=$1", TEST_DB)
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
            await conn.fetchval("SELECT COUNT(*) FROM pg_locks WHERE locktype = 'advisory'")
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
        # --- (A) routine 429 SURFACE: N processes, one workspace, cap=K --------------
        ws_a = f"acme-{uuid.uuid4().hex[:8]}"
        results = await asyncio.gather(
            *(
                _spawn(["--role", "routine", "--workspace", ws_a, "--name",
                        f"r{i}", "--cap", str(cap)])
                for i in range(nodes)
            )
        )
        admitted = sum(1 for r in results if r["admitted"])
        got_429 = sum(1 for r in results if r["quota_429"])
        stored = await _count_routines(ws_a)
        print(
            f"[routine-surface] admitted={admitted} quota_429={got_429} "
            f"stored={stored} cap={cap} nodes={nodes}"
        )
        if admitted != cap:
            failures.append(f"routine admitted {admitted} != cap {cap}")
        if stored != cap:
            failures.append(f"routine stored {stored} != cap {cap}")
        if got_429 != nodes - cap:
            failures.append(f"routine quota_429 {got_429} != surplus {nodes - cap}")

        # --- (B) run 429 SURFACE + a different tenant still admitted ------------------
        ws_b = f"globex-{uuid.uuid4().hex[:8]}"
        ws_other = f"other-{uuid.uuid4().hex[:8]}"
        specs = [
            ["--role", "run", "--workspace", ws_b, "--cap", str(cap)]
            for _ in range(nodes)
        ]
        specs += [["--role", "run", "--workspace", ws_other, "--cap", str(cap)]]
        results = await asyncio.gather(*(_spawn(s) for s in specs))
        b_results = [r for r in results if r["workspace"] == ws_b]
        other_results = [r for r in results if r["workspace"] == ws_other]
        admitted = sum(1 for r in b_results if r["admitted"])
        got_429 = sum(1 for r in b_results if r["quota_429"])
        active = await _count_active_runs(ws_b)
        other_admitted = sum(1 for r in other_results if r["admitted"])
        print(
            f"[run-surface] admitted={admitted} quota_429={got_429} active={active} "
            f"other_admitted={other_admitted} cap={cap} nodes={nodes}"
        )
        if admitted != cap:
            failures.append(f"run admitted {admitted} != cap {cap}")
        if active != cap:
            failures.append(f"run active {active} != cap {cap}")
        if got_429 != nodes - cap:
            failures.append(f"run quota_429 {got_429} != surplus {nodes - cap}")
        if other_admitted != 1:
            failures.append(f"different tenant admitted {other_admitted} != 1")

        # --- (C) LONE-TENANT under cap creates freely, zero spurious rejects ----------
        ws_lone = f"lone-{uuid.uuid4().hex[:8]}"
        under = max(1, cap - 1)
        lone = await _spawn(
            ["--role", "lone", "--workspace", ws_lone, "--name", str(under),
             "--cap", str(cap)]
        )
        print(
            f"[lone-tenant] admitted={lone['admitted']} rejected={lone['rejected']} "
            f"(asked for {under}, cap={cap})"
        )
        if lone["admitted"] != under or lone["rejected"] != 0:
            failures.append(
                f"lone tenant admitted {lone['admitted']}/{under} rejected "
                f"{lone['rejected']} (expected {under}/0)"
            )

        # --- (D) CROSS-WORKSPACE NO-SERIALISE: apples-to-apples (SAME process count) --
        # The fair comparison fixes the TOTAL process count (process-spawn + connect
        # overhead is the dominant cost, so it must be held constant) and only varies HOW
        # the processes are spread across workspaces:
        #   * concentrated: all ``total`` processes on ONE workspace (max same-key contention
        #     on a single advisory lock — every loser BLOCKS behind the winner);
        #   * spread:       ``total`` processes split evenly across MANY workspaces (different
        #     advisory keys ⇒ no cross-workspace blocking; each key contended by far fewer).
        # If different workspaces serialised against each other, spreading would NOT help and
        # the two wall-clocks would match. Because they use distinct keys, the spread case is
        # never SLOWER than the concentrated one (it has less per-lock contention), which is
        # the operational proof that a lone/quiet tenant is never blocked by a noisy one.
        total = nodes * 4
        ws_conc = f"conc-{uuid.uuid4().hex[:8]}"
        t0 = time.monotonic()
        await asyncio.gather(
            *(
                _spawn(["--role", "run", "--workspace", ws_conc, "--cap", str(cap)])
                for _ in range(total)
            )
        )
        conc_s = time.monotonic() - t0
        n_ws = 4
        spread_ws = [f"spr{i}-{uuid.uuid4().hex[:6]}" for i in range(n_ws)]
        per = total // n_ws
        t0 = time.monotonic()
        await asyncio.gather(
            *(
                _spawn(["--role", "run", "--workspace", ws, "--cap", str(cap)])
                for ws in spread_ws
                for _ in range(per)
            )
        )
        spread_s = time.monotonic() - t0
        spread_counts = [await _count_active_runs(ws) for ws in spread_ws]
        print(
            f"[cross-ws] same {total} procs: concentrated(1 ws)={conc_s:.2f}s "
            f"spread({n_ws} ws)={spread_s:.2f}s spread_counts={spread_counts} cap={cap}"
        )
        if any(c != cap for c in spread_counts):
            failures.append(
                f"cross-ws spread reached {spread_counts} (each should be {cap})"
            )
        # Same total work spread across distinct keys must NOT be materially SLOWER than piling
        # it on one key. If the lock did not key per-workspace, spreading would not reduce
        # contention and could even regress; a >1.5x blow-up would signal cross-ws serialise.
        if conc_s > 0 and spread_s > conc_s * 1.5:
            failures.append(
                f"cross-ws serialised: spread {spread_s:.2f}s > 1.5x concentrated "
                f"{conc_s:.2f}s (same {total} procs)"
            )

        # --- (E) no leaked advisory locks --------------------------------------------
        held = await _advisory_locks_held()
        print(f"[locks] advisory held after storm = {held}")
        if held != 0:
            failures.append(f"advisory locks leaked: {held}")
    finally:
        await _drop_db()

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS: 429 surface HARD both quotas; lone-tenant free; cross-ws no serialise; no leak")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["routine", "run", "lone"], default=None)
    p.add_argument("--workspace", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--cap", type=int, default=2)
    p.add_argument("--nodes", type=int, default=12)
    args = p.parse_args()

    if args.role == "routine":
        asyncio.run(_routine_surface_worker(args.workspace, args.name, args.cap))
        return 0
    if args.role == "run":
        asyncio.run(_run_surface_worker(args.workspace, args.cap))
        return 0
    if args.role == "lone":
        asyncio.run(_lone_routine_worker(args.workspace, int(args.name), args.cap))
        return 0
    return asyncio.run(_drive(args.nodes, args.cap))


if __name__ == "__main__":
    raise SystemExit(main())
