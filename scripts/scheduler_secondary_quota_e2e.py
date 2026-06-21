#!/usr/bin/env python
"""T3 e2e: the HARD per-tenant OUTSTANDING-RUN cap on BOTH secondary launch paths.

The store-level ``save_run_if_under_quota`` op is already proven hard under genuine
multi-process concurrency by ``scripts/scheduler_quota_e2e.py``. THIS harness proves the
two SERVICE-level launch paths that were just routed through that op behave correctly
end-to-end on a LIVE PostgreSQL 16 durable backend:

* ``RunAppService.continue_thread`` (a /v1 conversation turn) — previously a SOFT
  ``save_run_if_absent`` + count-then-mark-FAILED admit (TOCTOU);
* ``RunAppService.create_orchestration_run`` (DISPATCH mode) — previously enforced NO cap
  at create time at all.

For each path it fires N concurrent launches for ONE workspace at cap=K against a REAL
Postgres pool (genuine advisory-lock contention across asyncio tasks) and asserts:

* EXACTLY K admitted, the surplus a clean ``WorkspaceRunQuotaExceeded`` (the 429 surface);
* EXACTLY K active run rows in the shared store (no overshoot);
* ZERO FAILED-marked rows (the soft insert-then-mark-FAILED path is bypassed — proof the
  hard op rejects BEFORE any write, so nothing partial / no orphaned QUEUED orchestration);
* per-tenant independence (two workspaces each reach their FULL cap, no cross-serialise);
* an idempotent re-submit consumes NO extra slot;
* NO leaked advisory locks after the storm (``pg_locks`` advisory == 0).

Usage:
    python scripts/scheduler_secondary_quota_e2e.py --nodes 12 --cap 3

Creates a THROWAWAY database (default himmy_secpath_e2e), inits the schema, runs the
proof, and DROPS it. bholi / nagarik / postgres are never touched.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

ADMIN_DSN = os.environ.get(
    "HIMMY_SECPATH_E2E_ADMIN_DSN", "postgresql://samriddhagc@localhost:5432/postgres"
)
TEST_DB = os.environ.get("HIMMY_SECPATH_E2E_DB", "himmy_secpath_e2e")
TEST_DSN = f"postgresql://samriddhagc@localhost:5432/{TEST_DB}"

_PROTECTED = {"bholi", "nagarik", "postgres", "template0", "template1"}


# --------------------------------------------------------------------------- app wiring


def _build_app(storage):
    """A RunAppService in DISPATCH mode over the live Postgres backend."""
    from himmy.application.services import AgentDefAppService, RunAppService
    from himmy.entities.registry import EntityRegistry
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.context.service import ContextService
    from himmy.services.inference.client_manager import StubClientManager
    from himmy.services.inference.service import InferenceService

    registry = EntityRegistry()
    context = ContextService(storage_service=storage, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
    )
    agent_app = AgentDefAppService(storage=storage, entity_registry=registry)
    app = RunAppService(
        runtime=runtime,
        storage=storage,
        entity_registry=registry,
        workspace_max_outstanding=0,  # set per-scenario via the private knob below
        agent_resolver=agent_app.get_agent_def,
    )
    app.enable_dispatch()
    return app, agent_app


async def _store_agent(agent_app, name, *, workspace_id):
    from himmy.config.agent_spec import AgentSpec

    rec, _ = await agent_app.save_agent_def(AgentSpec(name=name), workspace_id=workspace_id)
    return rec


# --------------------------------------------------------------------------- drivers


async def _continue_once(app, agent_def, *, workspace_id, conversation_id, key=None):
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.application.services import WorkspaceRunQuotaExceeded
    from himmy.config.agent_spec import AgentSpec

    spec = AgentSpec(name=agent_def.name)
    try:
        await app.continue_thread(
            workspace_id=workspace_id,
            subject_id="s",
            conversation_id=conversation_id,
            thread=ChatThread(thread_id=conversation_id),
            prompt="hi",
            agent_spec=spec,
            agent_def=agent_def,
            idempotency_key=key,
        )
        return True
    except WorkspaceRunQuotaExceeded:
        return False


async def _orchestrate_once(app, member, *, workspace_id, resource_id, key=None):
    from himmy.application.services import WorkspaceRunQuotaExceeded

    try:
        await app.create_orchestration_run(
            workspace_id=workspace_id,
            subject_id="s",
            kind="multi_agent",
            members=[member],
            prompt="go",
            resource_kind="team",
            resource_id=resource_id,
            idempotency_key=key,
        )
        return True
    except WorkspaceRunQuotaExceeded:
        return False


# --------------------------------------------------------------------------- db helpers


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
        # migrate() applies create_schema + every leased-queue/Q3 column migration the
        # dispatch path needs (e.g. owner_id, lane_key); create_schema alone is older.
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
                "SELECT COUNT(*) FROM pg_locks WHERE locktype='advisory'"
            )
            or 0
        )
    finally:
        await conn.close()


async def _count_failed(workspace: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(TEST_DSN)
    try:
        return int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM runs WHERE workspace_id=$1 AND status='FAILED'",
                workspace,
            )
            or 0
        )
    finally:
        await conn.close()


async def _count_rows(workspace: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(TEST_DSN)
    try:
        return int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM runs WHERE workspace_id=$1", workspace
            )
            or 0
        )
    finally:
        await conn.close()


# --------------------------------------------------------------------------- driver


async def _drive(nodes: int, cap: int) -> int:
    from himmy.services.storage.postgres import PostgresStorageService

    await _create_db()
    failures: list[str] = []
    storage = await PostgresStorageService.connect(TEST_DSN, min_size=2, max_size=nodes + 4)
    try:
        app, agent_app = _build_app(storage)
        app._workspace_max_outstanding = cap  # type: ignore[attr-defined]

        # --- (A) continue_thread burst -------------------------------------------------
        ws_a = f"acme-{uuid.uuid4().hex[:8]}"
        a_def = await _store_agent(agent_app, "chat", workspace_id=ws_a)
        res = await asyncio.gather(
            *(
                _continue_once(app, a_def, workspace_id=ws_a, conversation_id=f"c{i}")
                for i in range(nodes)
            )
        )
        admitted = sum(1 for ok in res if ok)
        active = await storage.count_active_runs_for_workspace(ws_a)
        failed = await _count_failed(ws_a)
        rows = await _count_rows(ws_a)
        print(
            f"[continue_thread] admitted={admitted} active={active} rows={rows} "
            f"failed={failed} cap={cap} nodes={nodes}"
        )
        if admitted != cap:
            failures.append(f"continue admitted {admitted} != cap {cap}")
        if active != cap or rows != cap:
            failures.append(f"continue active={active} rows={rows} (want {cap})")
        if failed != 0:
            failures.append(f"continue marked {failed} FAILED (soft TOCTOU not bypassed)")

        # --- (B) create_orchestration_run burst ---------------------------------------
        ws_b = f"globex-{uuid.uuid4().hex[:8]}"
        b_member = await _store_agent(agent_app, "m", workspace_id=ws_b)
        res = await asyncio.gather(
            *(
                _orchestrate_once(app, b_member, workspace_id=ws_b, resource_id=f"t{i}")
                for i in range(nodes)
            )
        )
        admitted = sum(1 for ok in res if ok)
        active = await storage.count_active_runs_for_workspace(ws_b)
        failed = await _count_failed(ws_b)
        rows = await _count_rows(ws_b)
        print(
            f"[orchestration] admitted={admitted} active={active} rows={rows} "
            f"failed={failed} cap={cap} nodes={nodes}"
        )
        if admitted != cap:
            failures.append(f"orchestration admitted {admitted} != cap {cap}")
        if active != cap or rows != cap:
            failures.append(
                f"orchestration active={active} rows={rows} (want {cap}) — PARTIAL?"
            )
        if failed != 0:
            failures.append(f"orchestration marked {failed} FAILED (partial state)")

        # --- (C) per-tenant independence (continue path) ------------------------------
        ws_c1 = f"t1-{uuid.uuid4().hex[:8]}"
        ws_c2 = f"t2-{uuid.uuid4().hex[:8]}"
        d1 = await _store_agent(agent_app, "a", workspace_id=ws_c1)
        d2 = await _store_agent(agent_app, "b", workspace_id=ws_c2)
        await asyncio.gather(
            *(
                _continue_once(app, d1, workspace_id=ws_c1, conversation_id=f"x{i}")
                for i in range(nodes)
            ),
            *(
                _continue_once(app, d2, workspace_id=ws_c2, conversation_id=f"y{i}")
                for i in range(nodes)
            ),
        )
        a1 = await storage.count_active_runs_for_workspace(ws_c1)
        a2 = await storage.count_active_runs_for_workspace(ws_c2)
        print(f"[per-tenant] ws1={a1} ws2={a2} cap={cap}")
        if a1 != cap or a2 != cap:
            failures.append(f"per-tenant ws1={a1} ws2={a2} (each want {cap})")

        # --- (D) idempotent re-submit consumes no slot --------------------------------
        ws_d = f"idem-{uuid.uuid4().hex[:8]}"
        d_def = await _store_agent(agent_app, "i", workspace_id=ws_d)
        app._workspace_max_outstanding = 1  # type: ignore[attr-defined]
        ok1 = await _continue_once(
            app, d_def, workspace_id=ws_d, conversation_id="c1", key="dup"
        )
        ok2 = await _continue_once(
            app, d_def, workspace_id=ws_d, conversation_id="c1", key="dup"
        )
        rows_d = await _count_rows(ws_d)
        print(f"[idempotent] ok1={ok1} ok2={ok2} rows={rows_d}")
        if not (ok1 and ok2 and rows_d == 1):
            failures.append(
                f"idempotent re-submit ok1={ok1} ok2={ok2} rows={rows_d} (want True,True,1)"
            )
        app._workspace_max_outstanding = cap  # type: ignore[attr-defined]

        # --- (E) no leaked advisory locks ---------------------------------------------
        held = await _advisory_locks_held()
        print(f"[locks] advisory held after storm = {held}")
        if held != 0:
            failures.append(f"advisory locks leaked: {held}")
    finally:
        await storage.close()
        await _drop_db()

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS: both SECONDARY paths HARD on live Postgres; nothing partial; no lock leak")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=12)
    p.add_argument("--cap", type=int, default=3)
    args = p.parse_args()
    return asyncio.run(_drive(args.nodes, args.cap))


if __name__ == "__main__":
    raise SystemExit(main())
