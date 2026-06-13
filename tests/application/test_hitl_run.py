"""T2f — /v1 HITL approval/resume at the RunAppService layer.

These tests pin the headline /v1 gap (UNIT 5) at the service choke point — where the
tenant boundary and the exactly-once gate are real — independent of the HTTP layer:

* a ``hitl=True`` run resolved from a stored agent whose tool is approval-gated reaches
  AWAITING_APPROVAL carrying ``metadata['checkpoint_id']`` (and STOPS — never swept);
* ``resume_run(approved=True)`` fires the gated tool EXACTLY ONCE and completes;
* a double-approve is a no-op (the gated tool never fires twice, the winner's terminal
  state is preserved);
* ``resume_run(approved=False)`` records the rejection WITHOUT running the gated tool;
* the run rebuilds its runtime FROM THE STORED DB SPEC (not a filesystem path);
* ``sweep_stuck_runs`` SKIPS an AWAITING_APPROVAL run;
* hitl=True with an inline persona (no agent) / no checkpoint store is rejected.
"""

from __future__ import annotations

import asyncio

import pytest

from himmy.agents.personas.persona import Persona
from himmy.application.services import (
    AgentDefAppService,
    HitlNotSupportedError,
    HitlRequiresAgentError,
    RunAppService,
    RunNotApprovableError,
)
from himmy.config.agent_spec import AgentSpec
from himmy.entities.registry import EntityRegistry
from himmy.runtime.checkpoint import InMemoryCheckpointStore
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.models import RunStatus
from himmy.services.storage.service import StorageService
from tests.application import _per_run_tools
from tests.conftest import run_async

_TOOLS_MODULE = "tests.application._per_run_tools"


def _hitl_app() -> tuple[StorageService, RunAppService, AgentDefAppService]:
    """A RunAppService wired with a HITL checkpoint store + a stored-agent resolver."""
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
        checkpoint_store=InMemoryCheckpointStore(),
        agent_resolver=agent_app.get_agent_def,
    )
    return storage, app, agent_app


async def _store_gated_agent(
    agent_app: AgentDefAppService, *, workspace_id: str = "w1"
):
    """Store an operator-provisioned agent whose ONLY tool is approval-gated."""
    spec = AgentSpec(name="treasurer", tools_module=_TOOLS_MODULE, tools=["wire_money"])
    rec, _ = await agent_app.save_agent_def(
        spec, workspace_id=workspace_id, operator_provisioned=True
    )
    return rec


async def _create_hitl_run(app: RunAppService, agent_rec, *, workspace_id="w1"):
    """Create a hitl run by the stored agent and wait for it to pause/terminate."""
    spec = agent_rec.agent_spec()
    run = await app.create_run(
        workspace_id=workspace_id,
        subject_id="s1",
        persona=spec.to_persona(),
        task=spec.make_task("wire 1000 to the vendor"),
        agent_spec=spec,
        agent_def=agent_rec,
        operator_provisioned=True,
        hitl=True,
    )
    return await _await_status(
        app,
        run.run_id,
        {RunStatus.AWAITING_APPROVAL, RunStatus.SUCCEEDED, RunStatus.FAILED},
    )


async def _await_status(app: RunAppService, run_id: str, wanted: set, timeout=10.0):
    """Poll a run until it reaches one of ``wanted`` statuses (or timeout)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        run = await app.get_run(run_id)
        if run is not None and run.status in wanted:
            return run
        await asyncio.sleep(0.01)
    return await app.get_run(run_id)


def setup_function() -> None:
    """Reset the recording lists before each test (shared module-level state)."""
    _per_run_tools.CALLS.clear()
    _per_run_tools.WIRE_CALLS.clear()


# ------------------------------------------------------------------ pause


def test_hitl_run_reaches_awaiting_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hitl run on a gated tool reaches AWAITING_APPROVAL with a checkpoint id (T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        rec = await _store_gated_agent(agent_app)
        paused = await _create_hitl_run(app, rec)
        assert paused is not None
        assert paused.status == RunStatus.AWAITING_APPROVAL
        assert paused.metadata.get("checkpoint_id")
        assert paused.metadata.get("hitl") is True
        # The gated tool has NOT executed (it is awaiting approval).
        assert not _per_run_tools.WIRE_CALLS

    run_async(_scenario())


def test_pending_approvals_lists_the_gated_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paused run's pending-approvals lists the gated tool (T2f read side)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        rec = await _store_gated_agent(agent_app)
        paused = await _create_hitl_run(app, rec)
        pending = await app.pending_approvals(paused.run_id, workspace_id="w1")
        assert pending is not None
        assert [p["tool_name"] for p in pending] == ["wire_money"]

    run_async(_scenario())


def test_sweep_skips_awaiting_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    """sweep_stuck_runs must NOT reap an AWAITING_APPROVAL run to FAILED (T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        rec = await _store_gated_agent(agent_app)
        paused = await _create_hitl_run(app, rec)
        assert paused.status == RunStatus.AWAITING_APPROVAL
        # ttl=0 would sweep every non-terminal run — but the paused run is skipped.
        swept = await app.sweep_stuck_runs(ttl_seconds=0.0)
        assert paused.run_id not in swept
        still = await app.get_run(paused.run_id)
        assert still is not None and still.status == RunStatus.AWAITING_APPROVAL

    run_async(_scenario())


# ------------------------------------------------------------------ approve


def test_approve_fires_the_gated_tool_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """approve resumes to SUCCEEDED and the gated tool fired EXACTLY ONCE (T2f headline)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        rec = await _store_gated_agent(agent_app)
        paused = await _create_hitl_run(app, rec)
        assert paused.status == RunStatus.AWAITING_APPROVAL

        await app.resume_run(paused.run_id, approved=True, workspace_id="w1")
        done = await _await_status(
            app, paused.run_id, {RunStatus.SUCCEEDED, RunStatus.FAILED}
        )
        assert done is not None and done.status == RunStatus.SUCCEEDED
        # The gated tool fired EXACTLY ONCE.
        assert len(_per_run_tools.WIRE_CALLS) == 1

    run_async(_scenario())


def test_double_approve_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second approve neither re-runs the gated tool nor flips the terminal state (T2f).

    The second approve sees a non-AWAITING_APPROVAL run (it is already RUNNING/terminal)
    and is refused with :class:`RunNotApprovableError` (HTTP 409). The gated tool's
    exactly-once is enforced by the runtime ``claim()`` regardless.
    """
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        rec = await _store_gated_agent(agent_app)
        paused = await _create_hitl_run(app, rec)

        await app.resume_run(paused.run_id, approved=True, workspace_id="w1")
        # The run is no longer AWAITING_APPROVAL, so a second approve is refused.
        with pytest.raises(RunNotApprovableError):
            await app.resume_run(paused.run_id, approved=True, workspace_id="w1")
        done = await _await_status(
            app, paused.run_id, {RunStatus.SUCCEEDED, RunStatus.FAILED}
        )
        assert done is not None and done.status == RunStatus.SUCCEEDED
        # Still exactly one execution despite the second approve attempt.
        assert len(_per_run_tools.WIRE_CALLS) == 1

    run_async(_scenario())


def test_concurrent_approves_fire_the_tool_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent resume_in_background drives fire the gated tool exactly once (T2f).

    Drives the lower-level ``_resume_in_background`` directly with the SAME checkpoint so
    both pass the run-status gate and race on the runtime ``claim()`` — proving the
    exactly-once guarantee is the claim, not just the run-status check.
    """
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        rec = await _store_gated_agent(agent_app)
        paused = await _create_hitl_run(app, rec)
        checkpoint_id = paused.metadata["checkpoint_id"]

        await asyncio.gather(
            app._resume_in_background(
                paused.run_id,
                checkpoint_id=checkpoint_id,
                approved=True,
                actor="a",
                agent_def=rec,
                workspace_id="w1",
            ),
            app._resume_in_background(
                paused.run_id,
                checkpoint_id=checkpoint_id,
                approved=True,
                actor="b",
                agent_def=rec,
                workspace_id="w1",
            ),
        )
        # Exactly one execution despite two concurrent resumes (the runtime ``claim()``).
        assert len(_per_run_tools.WIRE_CALLS) == 1
        # And the winner drove the run to a terminal state (the loser was a clean no-op,
        # never leaving the run stranded in RUNNING).
        done = await app.get_run(paused.run_id)
        assert done is not None and done.status == RunStatus.SUCCEEDED

    run_async(_scenario())


# ------------------------------------------------------------------ reject


def test_reject_records_rejection_without_running_the_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reject records the rejection and NEVER executes the gated tool (T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        rec = await _store_gated_agent(agent_app)
        paused = await _create_hitl_run(app, rec)

        await app.resume_run(paused.run_id, approved=False, workspace_id="w1")
        done = await _await_status(
            app, paused.run_id, {RunStatus.SUCCEEDED, RunStatus.FAILED}
        )
        assert done is not None and done.status == RunStatus.SUCCEEDED
        # The gated tool was NEVER executed.
        assert not _per_run_tools.WIRE_CALLS

    run_async(_scenario())


# --------------------------------------------------------- tenant scoping


def test_resume_is_tenant_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume from the wrong workspace cannot see the paused run (404-shaped, T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        rec = await _store_gated_agent(agent_app)
        paused = await _create_hitl_run(app, rec)

        with pytest.raises(RunNotApprovableError) as exc:
            await app.resume_run(paused.run_id, approved=True, workspace_id="other")
        assert exc.value.status == "unknown"
        # The paused run is untouched + the gated tool never fired.
        still = await app.get_run(paused.run_id)
        assert still is not None and still.status == RunStatus.AWAITING_APPROVAL
        assert not _per_run_tools.WIRE_CALLS

    run_async(_scenario())


# ------------------------------------------------------------ rejection paths


def test_hitl_with_inline_persona_is_rejected() -> None:
    """hitl=True without a stored agent is rejected (an inline persona has no tools)."""
    _storage, app, _agent_app = _hitl_app()

    async def _scenario() -> None:
        with pytest.raises(HitlRequiresAgentError):
            await app.create_run(
                workspace_id="w1",
                subject_id="s1",
                persona=Persona(name="plain"),
                task=AgentSpec(name="plain").make_task("hi"),
                hitl=True,
            )

    run_async(_scenario())


def test_plan_with_tool_less_stored_agent_is_rejected() -> None:
    """plan=True on a stored-but-tool-LESS agent is rejected (reviewer must_fix, T2g).

    A bare persona spec builds no per-run tool registry, so the gated update_plan tool
    has nowhere to live and the run would silently complete without pausing at
    PLAN-READY. The service must reject it up front — like the inline-persona case — not
    let it slip through to a wrong outcome.
    """
    _storage, app, agent_app = _hitl_app()

    async def _scenario() -> None:
        # A stored agent declaring NO tool surface: builds_tool_registry() is False.
        spec = AgentSpec(name="plainstored")
        assert spec.builds_tool_registry() is False
        rec, _ = await agent_app.save_agent_def(spec, workspace_id="w1")
        with pytest.raises(HitlRequiresAgentError):
            await app.create_run(
                workspace_id="w1",
                subject_id="s1",
                persona=spec.to_persona(),
                task=spec.make_task("do the thing"),
                agent_spec=spec,
                agent_def=rec,
                plan=True,
            )

    run_async(_scenario())


def test_hitl_without_checkpoint_store_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hitl=True with no checkpoint store wired is rejected (no inbox to pause into)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    storage = StorageService()
    registry = EntityRegistry()
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        entity_registry=registry,
    )
    # No checkpoint_store / resolver: HITL is unavailable on this deployment.
    app = RunAppService(runtime=runtime, storage=storage, entity_registry=registry)
    spec = AgentSpec(name="t", tools_module=_TOOLS_MODULE, tools=["wire_money"])

    async def _scenario() -> None:
        agent_app = AgentDefAppService(storage=storage, entity_registry=registry)
        rec, _ = await agent_app.save_agent_def(
            spec, workspace_id="w1", operator_provisioned=True
        )
        with pytest.raises(HitlNotSupportedError):
            await app.create_run(
                workspace_id="w1",
                subject_id="s1",
                persona=spec.to_persona(),
                task=spec.make_task("go"),
                agent_spec=spec,
                agent_def=rec,
                operator_provisioned=True,
                hitl=True,
            )

    run_async(_scenario())
