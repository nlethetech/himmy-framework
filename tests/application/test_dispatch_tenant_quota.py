"""T3 — per-tenant outstanding QUOTA enforced AT ENQUEUE on the durable/dispatch path.

The T0.4 outstanding-run cap was inline-only: in dispatch mode ``_launch_or_enqueue``
returned BEFORE admission, so a tenant could flood the shared queue with unbounded QUEUED
runs across worker processes (the in-RAM counter cannot span processes). T3 enforces the cap
at enqueue by COUNTING the tenant's non-terminal runs in the shared store. These tests assert
the cap bites in dispatch mode, is per-tenant, exempts the reserved ``__local__`` workspace,
and is disabled by a zero cap (single-tenant parity).
"""

from __future__ import annotations

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.application.services import RunAppService, WorkspaceRunQuotaExceeded
from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.models import LOCAL_WORKSPACE, RunStatus
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _dispatch_app(
    *, workspace_max_outstanding: int, agent_resolver=None
) -> tuple[StorageService, RunAppService]:
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
    app = RunAppService(
        runtime=runtime,
        storage=storage,
        entity_registry=registry,
        workspace_max_outstanding=workspace_max_outstanding,
        agent_resolver=agent_resolver,
    )
    app.enable_dispatch()  # enqueue, do not execute
    return storage, app


def _persona_task() -> tuple[Persona, Task]:
    return Persona(name="A"), Task(title="t", prompt="hi")


def test_durable_outstanding_cap_rejects_burst_in_dispatch_mode() -> None:
    """In dispatch mode the (cap+1)-th enqueue for a tenant is a quota breach (429)."""
    _storage, app = _dispatch_app(workspace_max_outstanding=2)
    persona, task = _persona_task()

    async def _scenario() -> None:
        r1 = await app.create_run(
            workspace_id="acme", subject_id="s", persona=persona, task=task
        )
        r2 = await app.create_run(
            workspace_id="acme", subject_id="s", persona=persona, task=task
        )
        # Both are left QUEUED (the dispatcher would claim them); none executed inline.
        assert r1.status == RunStatus.QUEUED and r2.status == RunStatus.QUEUED
        with pytest.raises(WorkspaceRunQuotaExceeded) as exc:
            await app.create_run(
                workspace_id="acme", subject_id="s", persona=persona, task=task
            )
        assert exc.value.cap == 2
        # The rejected run was marked FAILED, not left orphaned QUEUED.
        active = await app.storage.count_active_runs_for_workspace("acme")
        assert active == 2

    run_async(_scenario())


def test_durable_cap_is_per_tenant() -> None:
    """One tenant at its cap does not block another tenant's enqueue."""
    _storage, app = _dispatch_app(workspace_max_outstanding=1)
    persona, task = _persona_task()

    async def _scenario() -> None:
        await app.create_run(
            workspace_id="acme", subject_id="s", persona=persona, task=task
        )
        with pytest.raises(WorkspaceRunQuotaExceeded):
            await app.create_run(
                workspace_id="acme", subject_id="s", persona=persona, task=task
            )
        # globex is independent.
        ok = await app.create_run(
            workspace_id="globex", subject_id="s", persona=persona, task=task
        )
        assert ok.status == RunStatus.QUEUED

    run_async(_scenario())


def test_local_workspace_is_exempt_from_durable_cap() -> None:
    """The reserved __local__ single-user workspace is never capped (single-box parity)."""
    _storage, app = _dispatch_app(workspace_max_outstanding=1)
    persona, task = _persona_task()

    async def _scenario() -> None:
        for _ in range(5):
            r = await app.create_run(
                workspace_id=LOCAL_WORKSPACE,
                subject_id="s",
                persona=persona,
                task=task,
            )
            assert r.status == RunStatus.QUEUED

    run_async(_scenario())


def test_zero_cap_disables_durable_quota() -> None:
    """workspace_max_outstanding=0 disables the cap (unbounded enqueue)."""
    _storage, app = _dispatch_app(workspace_max_outstanding=0)
    persona, task = _persona_task()

    async def _scenario() -> None:
        for _ in range(10):
            r = await app.create_run(
                workspace_id="acme", subject_id="s", persona=persona, task=task
            )
            assert r.status == RunStatus.QUEUED

    run_async(_scenario())


def test_terminal_runs_free_the_durable_slot() -> None:
    """A SUCCEEDED/FAILED run no longer counts, freeing the tenant's outstanding slot."""
    _storage, app = _dispatch_app(workspace_max_outstanding=1)
    persona, task = _persona_task()

    async def _scenario() -> None:
        r1 = await app.create_run(
            workspace_id="acme", subject_id="s", persona=persona, task=task
        )
        # Simulate the dispatcher finishing it.
        r1.status = RunStatus.SUCCEEDED
        await app.storage.save_run(r1)
        # The freed slot lets the next enqueue through under cap=1.
        r2 = await app.create_run(
            workspace_id="acme", subject_id="s", persona=persona, task=task
        )
        assert r2.status == RunStatus.QUEUED

    run_async(_scenario())


# --------------------------------------------------------- execution isolation


def test_claimed_run_resolves_agent_def_with_its_own_tenant() -> None:
    """A claimed run resolves its agent_def with ITS OWN workspace_id (no cross-tenant leak).

    The dispatcher hands ``dispatch_claimed_run`` a run already flipped to RUNNING; it must
    re-resolve the stored agent for the run's OWN tenant, never a global/ambient one — so a
    run can only ever bind its tenant's spec/scope. We assert the resolver is invoked with the
    run's workspace_id.
    """
    seen: list[tuple[str, str]] = []

    async def _resolver(agent_id: str, *, workspace_id: str):  # type: ignore[no-untyped-def]
        seen.append((agent_id, workspace_id))
        return None  # no lineage edge needed for this assertion

    _storage, app = _dispatch_app(
        workspace_max_outstanding=0, agent_resolver=_resolver
    )
    persona, task = _persona_task()

    async def _scenario() -> None:
        from himmy.services.storage.models import RunRecord
        from himmy.services.storage.run_input import encode_run_input

        run = RunRecord(
            run_id="iso1",
            workspace_id="acme",
            subject_id="s",
            status=RunStatus.RUNNING,
            owner_id="w0",
            attempt=1,
            max_attempts=1,
            metadata={"agent_id": "agent-xyz"},
        )
        run.input_blob = encode_run_input(
            persona=persona,
            task=task,
            llm_config=None,
            agent_spec=None,
            hitl=False,
            plan=False,
            run_id=run.run_id,
        )
        await app.storage.save_run(run)
        await app.dispatch_claimed_run(run)

    run_async(_scenario())
    # The resolver was called for THIS run's tenant only.
    assert seen == [("agent-xyz", "acme")], seen
