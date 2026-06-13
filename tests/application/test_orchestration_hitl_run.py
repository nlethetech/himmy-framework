"""Orchestration HITL at the RunAppService layer: AWAITING_APPROVAL + approve/reject.

Pins the /v1 service choke point for a TEAM/WORKFLOW run whose member gates on an
approval-gated tool:

* ``create_orchestration_run`` (a graph team / workflow) reaches AWAITING_APPROVAL carrying
  the MEMBER checkpoint id (under ``metadata['checkpoint_id']``), the orchestration
  checkpoint id, the awaiting member, and ``hitl_kind == 'orchestration'`` — and STOPS;
* ``pending_approvals`` returns the gated tool with redacted args via the unchanged path;
* ``resume_run(approved=True)`` fires the gated tool EXACTLY ONCE and the run SUCCEEDS;
* ``resume_run(approved=False)`` records the rejection WITHOUT running the tool;
* a double-approve is a no-op (the tool never fires twice);
* ``sweep_stuck_runs`` SKIPS the paused orchestration run.
"""

from __future__ import annotations

import asyncio

import pytest

from himmy.application.services import (
    AgentDefAppService,
    RunAppService,
    RunNotApprovableError,
)
from himmy.config.agent_spec import AgentSpec
from himmy.entities.registry import EntityRegistry
from himmy.runtime.checkpoint import (
    InMemoryCheckpointStore,
    InMemoryGraphCheckpointStore,
)
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.models import RunStatus
from himmy.services.storage.service import StorageService
from tests.application import _per_run_tools
from tests.conftest import run_async

_TOOLS_MODULE = "tests.application._per_run_tools"


def setup_function() -> None:
    """Reset the gated-tool counter before each test."""
    _per_run_tools.WIRE_CALLS.clear()
    _per_run_tools.CALLS.clear()


@pytest.fixture(autouse=True)
def _allow_operator_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator specs keep ``tools_module`` only under this opt-in (T0.3)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")


def _app() -> tuple[StorageService, RunAppService, AgentDefAppService, object]:
    """A RunAppService wired with a HITL checkpoint store + a shared graph store."""
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
    graph_store = InMemoryGraphCheckpointStore()
    app = RunAppService(
        runtime=runtime,
        storage=storage,
        entity_registry=registry,
        checkpoint_store=InMemoryCheckpointStore(),
        agent_resolver=agent_app.get_agent_def,
        graph_checkpoint_store_provider=lambda: graph_store,
    )
    return storage, app, agent_app, graph_store


async def _store_member(
    agent_app: AgentDefAppService, name: str, *, tools: list[str] | None = None
):
    """Store an operator-provisioned member agent (gated tool optional)."""
    spec = AgentSpec(
        name=name,
        tools_module=_TOOLS_MODULE,
        tools=list(tools or []),
    )
    rec, _ = await agent_app.save_agent_def(
        spec, workspace_id="w1", operator_provisioned=True
    )
    return rec


async def _await_status(app: RunAppService, run_id: str, wanted: set, timeout=10.0):
    """Poll a run until it reaches one of ``wanted`` statuses (or timeout)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        run = await app.get_run(run_id)
        if run is not None and run.status in wanted:
            return run
        await asyncio.sleep(0.01)
    return await app.get_run(run_id)


_TERMINAL = {
    RunStatus.AWAITING_APPROVAL,
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
}


async def _launch_paused(app: RunAppService, agent_app: AgentDefAppService, graph_store):
    """Launch a 2-member workflow that pauses at the first (gated) member."""
    step1 = await _store_member(agent_app, "step1", tools=["wire_money"])
    step2 = await _store_member(agent_app, "step2")
    run = await app.create_orchestration_run(
        workspace_id="w1",
        subject_id="s1",
        kind="graph",
        members=[step1, step2],
        prompt="go",
        resource_kind="workflow",
        resource_id="wf1",
        operator_provisioned=True,
        graph_checkpoint_store=graph_store,
    )
    return await _await_status(app, run.run_id, _TERMINAL)


def test_workflow_run_pauses_to_awaiting_approval() -> None:
    """A workflow whose member gates reaches AWAITING_APPROVAL with the right metadata."""

    async def _scenario() -> None:
        _storage, app, agent_app, graph_store = _app()
        run = await _launch_paused(app, agent_app, graph_store)

        assert run.status == RunStatus.AWAITING_APPROVAL
        meta = run.metadata or {}
        assert meta.get("hitl_kind") == "orchestration"
        assert meta.get("awaiting_member") == "step1"
        assert meta.get("checkpoint_id")  # the MEMBER checkpoint (pending-approvals path)
        assert meta.get("orchestration_checkpoint_id")  # the GRAPH checkpoint
        assert not _per_run_tools.WIRE_CALLS  # the gated tool did NOT run

        # pending_approvals reads the member checkpoint via the unchanged path.
        pending = await app.pending_approvals(run.run_id, workspace_id="w1")
        assert pending is not None
        assert len(pending) == 1
        assert pending[0]["tool_name"] == "wire_money"

    run_async(_scenario())


def test_pending_approvals_redacts_secret_looking_args() -> None:
    """A paused member's gated tool surfaces with secret-looking arg values masked."""

    async def _store_secret_member(agent_app: AgentDefAppService):
        spec = AgentSpec(
            name="s",
            tools_module="tests.application._secret_tool",
            tools=["push_secret"],
        )
        rec, _ = await agent_app.save_agent_def(
            spec, workspace_id="w1", operator_provisioned=True
        )
        return rec

    async def _scenario() -> None:
        _storage, app, agent_app, graph_store = _app()
        member = await _store_secret_member(agent_app)
        run = await app.create_orchestration_run(
            workspace_id="w1",
            subject_id="s1",
            kind="graph",
            members=[member],
            prompt="push",
            resource_kind="workflow",
            resource_id="wf1",
            operator_provisioned=True,
            graph_checkpoint_store=graph_store,
        )
        run = await _await_status(app, run.run_id, _TERMINAL)
        assert run.status == RunStatus.AWAITING_APPROVAL

        pending = await app.pending_approvals(run.run_id, workspace_id="w1")
        assert pending is not None and len(pending) == 1
        assert pending[0]["tool_name"] == "push_secret"
        # The secret-looking arg value is masked (the unchanged redaction).
        assert pending[0]["args"].get("api_token") == "••••"

    run_async(_scenario())


def test_sweeper_skips_paused_orchestration_run() -> None:
    """The stuck-run sweeper never reaps an AWAITING_APPROVAL orchestration run."""

    async def _scenario() -> None:
        _storage, app, agent_app, graph_store = _app()
        run = await _launch_paused(app, agent_app, graph_store)
        assert run.status == RunStatus.AWAITING_APPROVAL

        swept = await app.sweep_stuck_runs(ttl_seconds=0.0)
        assert run.run_id not in swept
        after = await app.get_run(run.run_id)
        assert after.status == RunStatus.AWAITING_APPROVAL

    run_async(_scenario())


def test_approve_fires_tool_once_and_succeeds() -> None:
    """Approve fires the gated tool exactly once and the run SUCCEEDS."""

    async def _scenario() -> None:
        _storage, app, agent_app, graph_store = _app()
        run = await _launch_paused(app, agent_app, graph_store)
        assert run.status == RunStatus.AWAITING_APPROVAL

        await app.resume_run(run.run_id, approved=True, workspace_id="w1")
        done = await _await_status(
            app, run.run_id, {RunStatus.SUCCEEDED, RunStatus.FAILED}
        )
        assert done.status == RunStatus.SUCCEEDED, done.error
        assert len(_per_run_tools.WIRE_CALLS) == 1  # fired EXACTLY once
        assert (done.metadata or {}).get("route") == ["step1", "step2"]

    run_async(_scenario())


def test_reject_records_rejection_without_running_tool() -> None:
    """Reject records the rejection, never runs the tool, and the run terminates."""

    async def _scenario() -> None:
        _storage, app, agent_app, graph_store = _app()
        run = await _launch_paused(app, agent_app, graph_store)

        await app.resume_run(run.run_id, approved=False, workspace_id="w1")
        done = await _await_status(
            app, run.run_id, {RunStatus.SUCCEEDED, RunStatus.FAILED}
        )
        assert done.status == RunStatus.SUCCEEDED  # graph still completes after a reject
        assert not _per_run_tools.WIRE_CALLS  # the gated tool was NEVER executed

    run_async(_scenario())


def test_double_approve_is_a_no_op() -> None:
    """A second approve of an already-resumed run is a clean no-op (409 / no re-run)."""

    async def _scenario() -> None:
        _storage, app, agent_app, graph_store = _app()
        run = await _launch_paused(app, agent_app, graph_store)

        await app.resume_run(run.run_id, approved=True, workspace_id="w1")
        done = await _await_status(
            app, run.run_id, {RunStatus.SUCCEEDED, RunStatus.FAILED}
        )
        assert done.status == RunStatus.SUCCEEDED

        # The run is no longer AWAITING_APPROVAL, so a second approve is refused (409).
        with pytest.raises(RunNotApprovableError):
            await app.resume_run(run.run_id, approved=True, workspace_id="w1")
        # The gated tool still fired exactly once.
        assert len(_per_run_tools.WIRE_CALLS) == 1

    run_async(_scenario())
