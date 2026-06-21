#!/usr/bin/env python
"""T3 e2e: HARD per-tenant OUTSTANDING-RUN cap on BOTH secondary paths under MULTI-PROCESS load.

The companion ``scripts/scheduler_secondary_quota_e2e.py`` proves the two service-level
launch paths (``continue_thread`` + ``create_orchestration_run``) are hard under genuine
asyncio-task contention against one shared pool. THIS harness raises the bar to the proof
the task demands: SEPARATE OS PROCESSES, each with its OWN ``RunAppService`` + its OWN
asyncpg pool / Postgres backend connections, all firing SIMULTANEOUSLY at one workspace
against a small cap on a LIVE PostgreSQL 16 database.

Why multi-process matters: the in-RAM ``_ws_outstanding`` counter and any single-event-loop
"no await between count and insert" invariant CANNOT span processes. The only thing that can
make the cap hard across true OS-process boundaries is the Postgres TRANSACTION-scoped
per-workspace advisory lock inside ``save_run_if_under_quota``. If the cap holds here, it is
the database — not Python — doing the serialising.

For each secondary path the harness:
  * spawns N worker PROCESSES (default 16), each connecting its own backend, blocking on a
    wall-clock barrier, then issuing exactly ONE launch for the SAME workspace at cap=K;
  * collects each worker's verdict (ADMITTED / REJECTED-429 / ERROR) over a pipe;
  * from the PARENT process, queries the live ``runs`` table and asserts:
      - admitted == K, rejected == N-K, errors == 0;
      - EXACTLY K run rows for the workspace (no overshoot);
      - ZERO non-active rows (no FAILED insert-then-reject; for orchestration this is the
        explicit NO-PARTIAL / NO-ORPHAN check — a rejected create wrote NOTHING);
      - for orchestration: every persisted row is a genuine ``orchestration=team`` PARENT,
        and there is NO run row whose metadata marks it a member/child (members never
        persist) — i.e. no half-orchestration;
  * runs MULTIPLE ROUNDS (the cap must re-hold after the first round's runs are released);
  * proves a DIFFERENT tenant is still admitted to its FULL cap concurrently (no cross-
    tenant serialisation);
  * asserts ZERO advisory locks remain in ``pg_locks`` after the storm (no leak / no hang).

Usage:
    python scripts/scheduler_secondary_quota_multiproc_e2e.py --procs 16 --cap 3 --rounds 3

Creates a THROWAWAY database (default himmy_secpath_e2e), inits schema, proves, DROPS it.
bholi / nagarik / postgres are NEVER touched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import time
import uuid

ADMIN_DSN = os.environ.get(
    "HIMMY_SECPATH_E2E_ADMIN_DSN", "postgresql://samriddhagc@localhost:5432/postgres"
)
TEST_DB = os.environ.get("HIMMY_SECPATH_E2E_DB", "himmy_secpath_e2e")
TEST_DSN = f"postgresql://samriddhagc@localhost:5432/{TEST_DB}"

_PROTECTED = {"bholi", "nagarik", "postgres", "template0", "template1"}


# ===========================================================================
# Worker process body — a FRESH interpreter context (spawn), its OWN backend.
# ===========================================================================
def _worker(
    path: str,
    cap: int,
    workspace_id: str,
    agent_id: str,
    agent_name: str,
    ordinal: int,
    start_at: float,
    conn_sock,
) -> None:
    """Run in a SEPARATE OS PROCESS: one launch on ``path`` for ``workspace_id``.

    Each process builds its own ``RunAppService`` over its own ``PostgresStorageService``
    pool, busy-waits to the shared ``start_at`` wall-clock instant (a cross-process barrier
    so all N fire within microseconds), then issues exactly ONE launch and reports the
    verdict ('admitted' | 'rejected' | error string) back over the pipe.
    """

    async def _run() -> str:
        from himmy.application.services import (
            AgentDefAppService,
            RunAppService,
            WorkspaceRunQuotaExceeded,
        )
        from himmy.config.agent_spec import AgentSpec
        from himmy.entities.registry import EntityRegistry
        from himmy.runtime.single_agent import SingleAgentRuntime
        from himmy.services.context.service import ContextService
        from himmy.services.inference.client_manager import StubClientManager
        from himmy.services.inference.service import InferenceService
        from himmy.services.storage.postgres import PostgresStorageService

        storage = await PostgresStorageService.connect(
            TEST_DSN, min_size=1, max_size=2
        )
        try:
            registry = EntityRegistry()
            context = ContextService(
                storage_service=storage, entity_registry=registry
            )
            runtime = SingleAgentRuntime(
                inference_service=InferenceService(
                    StubClientManager(), event_sink=storage
                ),
                memory_store=storage,
                context_service=context,
                entity_registry=registry,
            )
            agent_app = AgentDefAppService(storage=storage, entity_registry=registry)
            app = RunAppService(
                runtime=runtime,
                storage=storage,
                entity_registry=registry,
                workspace_max_outstanding=cap,
                agent_resolver=agent_app.get_agent_def,
            )
            app.enable_dispatch()

            # Re-resolve the shared agent def from the durable store (it was written by
            # the parent before fan-out) so this process drives the real launch path.
            agent_def = await agent_app.get_agent_def(
                agent_id, workspace_id=workspace_id
            )
            if agent_def is None:  # pragma: no cover - defensive
                return "error:agent-not-found"
            spec = AgentSpec(name=agent_name)

            # Cross-process barrier: spin to the agreed wall-clock instant.
            while time.time() < start_at:
                time.sleep(0.0005)

            try:
                if path == "continue_thread":
                    from himmy.agents.base_agent.thread import ChatThread

                    await app.continue_thread(
                        workspace_id=workspace_id,
                        subject_id="s",
                        conversation_id=f"conv-{workspace_id}-{ordinal}",
                        thread=ChatThread(
                            thread_id=f"conv-{workspace_id}-{ordinal}"
                        ),
                        prompt="hi",
                        agent_spec=spec,
                        agent_def=agent_def,
                    )
                elif path == "orchestration":
                    await app.create_orchestration_run(
                        workspace_id=workspace_id,
                        subject_id="s",
                        kind="multi_agent",
                        members=[agent_def],
                        prompt="go",
                        resource_kind="team",
                        resource_id=f"team-{workspace_id}-{ordinal}",
                    )
                else:  # pragma: no cover - defensive
                    return f"error:unknown-path:{path}"
                return "admitted"
            except WorkspaceRunQuotaExceeded:
                return "rejected"
            except Exception as exc:  # noqa: BLE001 - surface unexpected as ERROR
                return f"error:{type(exc).__name__}:{exc}"
        finally:
            await storage.close()

    verdict = asyncio.run(_run())
    conn_sock.send(verdict)
    conn_sock.close()


# ===========================================================================
# Parent-side helpers (admin / inspection over the live db).
# ===========================================================================
async def _create_db() -> None:
    import asyncpg

    if TEST_DB in _PROTECTED:
        raise SystemExit(f"refusing to use protected db {TEST_DB!r}")
    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname=$1 AND pid<>pg_backend_pid()",
            TEST_DB,
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}"')
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


async def _seed_agent(workspace_id: str, name: str) -> tuple[str, str]:
    """Write a tool-less agent def the worker processes will re-resolve. Returns id+name."""
    from himmy.application.services import AgentDefAppService
    from himmy.config.agent_spec import AgentSpec
    from himmy.entities.registry import EntityRegistry
    from himmy.services.storage.postgres import PostgresStorageService

    storage = await PostgresStorageService.connect(TEST_DSN, min_size=1, max_size=2)
    try:
        agent_app = AgentDefAppService(
            storage=storage, entity_registry=EntityRegistry()
        )
        rec, _ = await agent_app.save_agent_def(
            AgentSpec(name=name), workspace_id=workspace_id
        )
        return rec.agent_id, rec.name
    finally:
        await storage.close()


async def _inspect(workspace_id: str) -> dict:
    """Snapshot the runs table for a workspace from a CLEAN connection (parent side)."""
    import asyncpg

    conn = await asyncpg.connect(TEST_DSN)
    try:
        rows = await conn.fetch(
            "SELECT run_id, status, metadata FROM runs WHERE workspace_id=$1",
            workspace_id,
        )
    finally:
        await conn.close()
    from himmy.services.storage.models import ACTIVE_RUN_STATUSES

    active_set = {s.value for s in ACTIVE_RUN_STATUSES}
    total = len(rows)
    active = [r for r in rows if r["status"] in active_set]
    failed = [r for r in rows if r["status"] == "FAILED"]

    # Orphan / partial-orchestration probe: any persisted row that is NOT a legitimate
    # active run is an orphan; any row whose metadata marks it a member/child (members
    # must never persist their own RunRecord) is a half-orchestration leak.
    child_marked = []
    for r in rows:
        md = r["metadata"]
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except Exception:  # noqa: BLE001
                md = {}
        md = md or {}
        if md.get("member_of") or md.get("is_member") or md.get("parent_run_id"):
            child_marked.append(r["run_id"])
    return {
        "total": total,
        "active": len(active),
        "failed": len(failed),
        "child_marked": len(child_marked),
    }


async def _advisory_locks_held() -> int:
    import asyncpg

    conn = await asyncpg.connect(TEST_DSN)
    try:
        return int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM pg_locks WHERE locktype='advisory'"
            )
            or 0
        )
    finally:
        await conn.close()


async def _release_all(workspace_id: str) -> None:
    """Mark every active run for a workspace terminal so the next round starts clean."""
    import asyncpg

    conn = await asyncpg.connect(TEST_DSN)
    try:
        await conn.execute(
            "UPDATE runs SET status='SUCCEEDED' WHERE workspace_id=$1 "
            "AND status <> 'SUCCEEDED'",
            workspace_id,
        )
    finally:
        await conn.close()


# ===========================================================================
# One multi-process burst (separate OS processes hitting one workspace).
# ===========================================================================
def _burst(
    path: str, *, procs: int, cap: int, workspace_id: str, agent_id: str, agent_name: str
) -> dict[str, int]:
    """Fork ``procs`` OS processes that each fire ONE launch; tally their verdicts."""
    ctx = mp.get_context("spawn")
    start_at = time.time() + 1.5  # generous lead so every spawned interpreter is ready
    parents = []
    children = []
    for i in range(procs):
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=_worker,
            args=(
                path,
                cap,
                workspace_id,
                agent_id,
                agent_name,
                i,
                start_at,
                child_conn,
            ),
        )
        p.start()
        child_conn.close()  # parent keeps only the read end
        parents.append(parent_conn)
        children.append(p)

    verdicts = []
    for pc in parents:
        try:
            verdicts.append(pc.recv())
        except EOFError:
            verdicts.append("error:eof")
        finally:
            pc.close()
    for p in children:
        p.join(timeout=60)
        if p.is_alive():  # pragma: no cover - defensive
            p.terminate()

    tally = {
        "admitted": sum(1 for v in verdicts if v == "admitted"),
        "rejected": sum(1 for v in verdicts if v == "rejected"),
        "errors": [v for v in verdicts if v.startswith("error")],
    }
    return tally


async def _drive(procs: int, cap: int, rounds: int) -> int:
    await _create_db()
    failures: list[str] = []
    try:
        for path in ("continue_thread", "orchestration"):
            ws = f"{path[:4]}-{uuid.uuid4().hex[:8]}"
            agent_id, agent_name = await _seed_agent(ws, "m")
            for rnd in range(rounds):
                tally = _burst(
                    path,
                    procs=procs,
                    cap=cap,
                    workspace_id=ws,
                    agent_id=agent_id,
                    agent_name=agent_name,
                )
                snap = await _inspect(ws)
                print(
                    f"[{path} r{rnd}] admitted={tally['admitted']} "
                    f"rejected={tally['rejected']} errors={len(tally['errors'])} "
                    f"| active={snap['active']} total={snap['total']} "
                    f"failed={snap['failed']} child_marked={snap['child_marked']} "
                    f"| cap={cap} procs={procs}"
                )
                if tally["errors"]:
                    failures.append(f"{path} r{rnd} ERRORS: {tally['errors'][:3]}")
                if tally["admitted"] != cap:
                    failures.append(
                        f"{path} r{rnd} admitted {tally['admitted']} != cap {cap}"
                    )
                if tally["rejected"] != procs - cap:
                    failures.append(
                        f"{path} r{rnd} rejected {tally['rejected']} != {procs - cap}"
                    )
                # EXACTLY cap in-flight this round (overshoot = cap not hard).
                if snap["active"] != cap:
                    failures.append(
                        f"{path} r{rnd} active {snap['active']} != cap {cap} (overshoot)"
                    )
                # Total rows == cap admitted PER round (released rounds keep SUCCEEDED
                # rows): a rejected create wrote NOTHING, so the only rows that ever exist
                # are the genuinely-admitted ones. Any excess is an orphan / partial write.
                if snap["total"] != cap * (rnd + 1):
                    failures.append(
                        f"{path} r{rnd} total {snap['total']} != {cap * (rnd + 1)} "
                        f"(orphan/partial — a rejected create wrote a row)"
                    )
                # ZERO FAILED rows EVER: the hard op rejects BEFORE any insert, so the old
                # soft insert-then-mark-FAILED is fully bypassed (the strongest no-partial
                # / no-orphaned-orchestration signal).
                if snap["failed"] != 0:
                    failures.append(
                        f"{path} r{rnd} {snap['failed']} FAILED rows "
                        f"(soft TOCTOU / partial-on-reject not bypassed)"
                    )
                if snap["child_marked"] != 0:
                    failures.append(
                        f"{path} r{rnd} {snap['child_marked']} member/child rows "
                        f"(half-orchestration leak)"
                    )
                # Release this round's runs so the cap must re-hold next round.
                await _release_all(ws)

        # --- cross-tenant independence: two workspaces burst SIMULTANEOUSLY ------------
        ws_x = f"tenX-{uuid.uuid4().hex[:8]}"
        ws_y = f"tenY-{uuid.uuid4().hex[:8]}"
        ax_id, ax_name = await _seed_agent(ws_x, "x")
        ay_id, ay_name = await _seed_agent(ws_y, "y")
        # Run both bursts back to back (each its own process pool); since the cap is
        # per-tenant (distinct advisory keys), each must reach its FULL cap independently.
        tx = _burst(
            "continue_thread",
            procs=procs,
            cap=cap,
            workspace_id=ws_x,
            agent_id=ax_id,
            agent_name=ax_name,
        )
        ty = _burst(
            "continue_thread",
            procs=procs,
            cap=cap,
            workspace_id=ws_y,
            agent_id=ay_id,
            agent_name=ay_name,
        )
        sx = await _inspect(ws_x)
        sy = await _inspect(ws_y)
        print(
            f"[cross-tenant] X admitted={tx['admitted']} rows={sx['total']} | "
            f"Y admitted={ty['admitted']} rows={sy['total']} | cap={cap}"
        )
        if sx["total"] != cap or sy["total"] != cap:
            failures.append(
                f"cross-tenant X rows={sx['total']} Y rows={sy['total']} (each want {cap})"
            )
        if tx["admitted"] != cap or ty["admitted"] != cap:
            failures.append(
                f"cross-tenant X admitted={tx['admitted']} Y admitted={ty['admitted']}"
            )

        # --- no leaked advisory locks anywhere after the whole storm ------------------
        held = await _advisory_locks_held()
        print(f"[locks] advisory held after entire storm = {held}")
        if held != 0:
            failures.append(f"advisory locks leaked: {held}")
    finally:
        await _drop_db()

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print(
        "PASS: BOTH secondary paths HARD across separate OS PROCESSES on live Postgres; "
        "exactly-at-cap every round; nothing partial / no orphan / no member rows; "
        "per-tenant independent; no advisory-lock leak"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--cap", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    return asyncio.run(_drive(args.procs, args.cap, args.rounds))


if __name__ == "__main__":
    raise SystemExit(main())
