"""T3 — the HARD per-tenant outstanding-run cap on the TWO SECONDARY launch paths.

The primary ``create_run`` dispatch path already routes through the atomic
``save_run_if_under_quota`` op (count+insert fused under a per-workspace advisory lock /
``BEGIN IMMEDIATE``). These tests pin that the two OTHER launch paths now go through the
SAME hard op so the per-tenant outstanding cap is byte-for-byte HARD on EVERY launch:

* ``continue_thread`` (a /v1 conversation turn) — previously a SOFT
  ``save_run_if_absent`` + count-then-mark-FAILED admit with a TOCTOU window;
* ``create_orchestration_run`` in DISPATCH mode — previously enforced NO cap at create
  time at all (it short-circuited to ``return stored`` and deferred to the dispatcher).

Each path is asserted: a concurrent burst beyond the cap lands EXACTLY at the cap, the
surplus raises :class:`WorkspaceRunQuotaExceeded` (the 429 surface) with NOTHING written
(no FAILED-marked rows — proving the soft TOCTOU path is bypassed; no orphaned QUEUED
parent for orchestration — proving no partial/half-created orchestration); the cap is
per-tenant (different advisory keys never serialise); an idempotent re-submit consumes no
slot; and the exempt paths (cap=0 / ``__local__``) keep their byte-identical legacy
behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

from himmy.agents.base_agent.thread import ChatThread
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
from himmy.services.storage.models import (
    ACTIVE_RUN_STATUSES,
    LOCAL_WORKSPACE,
    RunStatus,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async

# --------------------------------------------------------------------------- harness


def _dispatch_app(
    *, workspace_max_outstanding: int
) -> tuple[StorageService, RunAppService, AgentDefAppService]:
    """A RunAppService in DISPATCH mode (enqueue-only) over an in-memory store."""
    storage = StorageService()
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
        workspace_max_outstanding=workspace_max_outstanding,
        agent_resolver=agent_app.get_agent_def,
    )
    app.enable_dispatch()  # enqueue, do not execute
    return storage, app, agent_app


async def _store_agent(
    agent_app: AgentDefAppService, name: str, *, workspace_id: str
):
    """Store a tool-less agent def usable by continue_thread / orchestration members."""
    spec = AgentSpec(name=name)
    rec, _ = await agent_app.save_agent_def(spec, workspace_id=workspace_id)
    return rec, spec


async def _continue(
    app: RunAppService,
    agent_def,
    agent_spec: AgentSpec,
    *,
    workspace_id: str,
    conversation_id: str,
    idempotency_key: str | None = None,
):
    """Drive continue_thread with a fresh ChatThread (no prior turns needed)."""
    return await app.continue_thread(
        workspace_id=workspace_id,
        subject_id="s",
        conversation_id=conversation_id,
        thread=ChatThread(thread_id=conversation_id),
        prompt="hi",
        agent_spec=agent_spec,
        agent_def=agent_def,
        idempotency_key=idempotency_key,
    )


async def _orchestrate(
    app: RunAppService,
    members,
    *,
    workspace_id: str,
    resource_id: str,
    idempotency_key: str | None = None,
):
    """Drive create_orchestration_run (a 1-member team — writes ONE parent run row)."""
    return await app.create_orchestration_run(
        workspace_id=workspace_id,
        subject_id="s",
        kind="multi_agent",
        members=members,
        prompt="go",
        resource_kind="team",
        resource_id=resource_id,
        idempotency_key=idempotency_key,
    )


def _count_failed(storage: StorageService, workspace_id: str) -> int:
    """How many runs for the workspace landed in a FAILED terminal state."""
    return sum(
        1
        for r in storage._run_store._runs.values()  # type: ignore[attr-defined]
        if r.workspace_id == workspace_id and r.status == RunStatus.FAILED
    )


def _count_rows(storage: StorageService, workspace_id: str) -> int:
    """Total run rows persisted for the workspace (any status)."""
    return sum(
        1
        for r in storage._run_store._runs.values()  # type: ignore[attr-defined]
        if r.workspace_id == workspace_id
    )


# --------------------------------------------------------------- continue_thread


def test_continue_thread_concurrent_burst_lands_exactly_at_cap() -> None:
    """N concurrent continue_thread for ONE workspace vs cap=K → exactly K, surplus 429.

    Crucially asserts ZERO FAILED rows: the old soft path marked over-cap runs FAILED
    (insert-then-reject); the hard op rejects BEFORE any insert, so nothing partial.
    """
    storage, app, agent_app = _dispatch_app(workspace_max_outstanding=3)

    async def _scenario() -> None:
        agent_def, spec = await _store_agent(agent_app, "chat", workspace_id="acme")
        n = 12
        results = await asyncio.gather(
            *(
                _continue(
                    app,
                    agent_def,
                    spec,
                    workspace_id="acme",
                    conversation_id=f"conv-{i}",
                )
                for i in range(n)
            ),
            return_exceptions=True,
        )
        admitted = [r for r in results if not isinstance(r, Exception)]
        rejected = [r for r in results if isinstance(r, WorkspaceRunQuotaExceeded)]
        other = [
            r
            for r in results
            if isinstance(r, Exception)
            and not isinstance(r, WorkspaceRunQuotaExceeded)
        ]
        assert not other, other
        assert len(admitted) == 3, len(admitted)
        assert len(rejected) == 12 - 3
        assert all(r.status == RunStatus.QUEUED for r in admitted)
        # Active runs == cap; nothing marked FAILED (the soft TOCTOU path is bypassed).
        active = await app.storage.count_active_runs_for_workspace("acme")
        assert active == 3
        assert _count_failed(storage, "acme") == 0
        assert _count_rows(storage, "acme") == 3  # surplus wrote NO row

    run_async(_scenario())


def test_continue_thread_idempotent_resubmit_consumes_no_slot() -> None:
    """Re-submitting the SAME idempotency key returns the prior run, no extra slot used."""
    storage, app, agent_app = _dispatch_app(workspace_max_outstanding=1)

    async def _scenario() -> None:
        agent_def, spec = await _store_agent(agent_app, "chat", workspace_id="acme")
        first = await _continue(
            app, agent_def, spec, workspace_id="acme",
            conversation_id="c1", idempotency_key="dup",
        )
        # Same key again — must NOT relaunch and must NOT consume the only slot.
        again = await _continue(
            app, agent_def, spec, workspace_id="acme",
            conversation_id="c1", idempotency_key="dup",
        )
        assert again.run_id == first.run_id
        assert await app.storage.count_active_runs_for_workspace("acme") == 1
        assert _count_rows(storage, "acme") == 1

    run_async(_scenario())


def test_continue_thread_cap_is_per_tenant() -> None:
    """One tenant at its cap does not block another tenant's continuation."""
    _storage, app, agent_app = _dispatch_app(workspace_max_outstanding=1)

    async def _scenario() -> None:
        a_def, a_spec = await _store_agent(agent_app, "a", workspace_id="acme")
        g_def, g_spec = await _store_agent(agent_app, "g", workspace_id="globex")
        await _continue(app, a_def, a_spec, workspace_id="acme", conversation_id="c1")
        with pytest.raises(WorkspaceRunQuotaExceeded):
            await _continue(
                app, a_def, a_spec, workspace_id="acme", conversation_id="c2"
            )
        ok = await _continue(
            app, g_def, g_spec, workspace_id="globex", conversation_id="c3"
        )
        assert ok.status == RunStatus.QUEUED

    run_async(_scenario())


def test_continue_thread_local_workspace_is_exempt() -> None:
    """The reserved __local__ workspace is never capped on the continuation path."""
    _storage, app, agent_app = _dispatch_app(workspace_max_outstanding=1)

    async def _scenario() -> None:
        a_def, a_spec = await _store_agent(
            agent_app, "a", workspace_id=LOCAL_WORKSPACE
        )
        for i in range(5):
            r = await _continue(
                app, a_def, a_spec,
                workspace_id=LOCAL_WORKSPACE, conversation_id=f"c{i}",
            )
            assert r.status == RunStatus.QUEUED

    run_async(_scenario())


def test_continue_thread_zero_cap_disables_quota() -> None:
    """cap=0 keeps the byte-identical legacy soft path (unbounded)."""
    _storage, app, agent_app = _dispatch_app(workspace_max_outstanding=0)

    async def _scenario() -> None:
        a_def, a_spec = await _store_agent(agent_app, "a", workspace_id="acme")
        for i in range(8):
            r = await _continue(
                app, a_def, a_spec, workspace_id="acme", conversation_id=f"c{i}"
            )
            assert r.status == RunStatus.QUEUED

    run_async(_scenario())


# ----------------------------------------------------- create_orchestration_run


def test_orchestration_concurrent_burst_lands_exactly_at_cap_nothing_partial() -> None:
    """N concurrent orchestration creates vs cap=K → exactly K parents, NOTHING partial.

    The over-cap creates raise the 429 with no parent row written (and members never
    persist their own rows), so an over-cap orchestration is rejected wholesale — zero
    orphaned QUEUED parents, zero half-created member state.
    """
    storage, app, agent_app = _dispatch_app(workspace_max_outstanding=2)

    async def _scenario() -> None:
        member, _spec = await _store_agent(agent_app, "m", workspace_id="acme")
        n = 10
        results = await asyncio.gather(
            *(
                _orchestrate(
                    app, [member], workspace_id="acme", resource_id=f"team-{i}"
                )
                for i in range(n)
            ),
            return_exceptions=True,
        )
        admitted = [r for r in results if not isinstance(r, Exception)]
        rejected = [r for r in results if isinstance(r, WorkspaceRunQuotaExceeded)]
        other = [
            r
            for r in results
            if isinstance(r, Exception)
            and not isinstance(r, WorkspaceRunQuotaExceeded)
        ]
        assert not other, other
        assert len(admitted) == 2, len(admitted)
        assert len(rejected) == 10 - 2
        assert all(r.status == RunStatus.QUEUED for r in admitted)
        # EXACTLY cap parent rows; no orphaned/partial rows beyond the cap, none FAILED.
        active = await app.storage.count_active_runs_for_workspace("acme")
        assert active == 2
        assert _count_rows(storage, "acme") == 2
        assert _count_failed(storage, "acme") == 0
        # Every persisted parent is a genuine orchestration run (a logical unit).
        rows = [
            r
            for r in storage._run_store._runs.values()  # type: ignore[attr-defined]
            if r.workspace_id == "acme"
        ]
        assert all(r.status in set(ACTIVE_RUN_STATUSES) for r in rows)
        assert all(r.metadata.get("orchestration") == "team" for r in rows)

    run_async(_scenario())


def test_orchestration_idempotent_resubmit_consumes_no_slot() -> None:
    """Re-submitting the same orchestration idempotency key consumes no extra slot."""
    storage, app, agent_app = _dispatch_app(workspace_max_outstanding=1)

    async def _scenario() -> None:
        member, _spec = await _store_agent(agent_app, "m", workspace_id="acme")
        first = await _orchestrate(
            app, [member], workspace_id="acme", resource_id="t1",
            idempotency_key="dup",
        )
        again = await _orchestrate(
            app, [member], workspace_id="acme", resource_id="t1",
            idempotency_key="dup",
        )
        assert again.run_id == first.run_id
        assert await app.storage.count_active_runs_for_workspace("acme") == 1
        assert _count_rows(storage, "acme") == 1

    run_async(_scenario())


def test_orchestration_cap_is_per_tenant() -> None:
    """One tenant's orchestration cap does not block another tenant."""
    _storage, app, agent_app = _dispatch_app(workspace_max_outstanding=1)

    async def _scenario() -> None:
        a_member, _ = await _store_agent(agent_app, "a", workspace_id="acme")
        g_member, _ = await _store_agent(agent_app, "g", workspace_id="globex")
        await _orchestrate(app, [a_member], workspace_id="acme", resource_id="t1")
        with pytest.raises(WorkspaceRunQuotaExceeded):
            await _orchestrate(
                app, [a_member], workspace_id="acme", resource_id="t2"
            )
        ok = await _orchestrate(
            app, [g_member], workspace_id="globex", resource_id="t3"
        )
        assert ok.status == RunStatus.QUEUED

    run_async(_scenario())


def test_orchestration_local_workspace_is_exempt() -> None:
    """The reserved __local__ workspace is never capped on the orchestration path."""
    _storage, app, agent_app = _dispatch_app(workspace_max_outstanding=1)

    async def _scenario() -> None:
        member, _ = await _store_agent(agent_app, "m", workspace_id=LOCAL_WORKSPACE)
        for i in range(5):
            r = await _orchestrate(
                app, [member], workspace_id=LOCAL_WORKSPACE, resource_id=f"t{i}"
            )
            assert r.status == RunStatus.QUEUED

    run_async(_scenario())


def test_orchestration_zero_cap_disables_quota() -> None:
    """cap=0 keeps the byte-identical legacy dispatch short-circuit (unbounded)."""
    _storage, app, agent_app = _dispatch_app(workspace_max_outstanding=0)

    async def _scenario() -> None:
        member, _ = await _store_agent(agent_app, "m", workspace_id="acme")
        for i in range(8):
            r = await _orchestrate(
                app, [member], workspace_id="acme", resource_id=f"t{i}"
            )
            assert r.status == RunStatus.QUEUED

    run_async(_scenario())
