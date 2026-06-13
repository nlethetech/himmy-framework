"""Orchestration HITL — a gated tool inside a workflow/graph member pauses + resumes.

These cover the headline gap: an approval-gated tool fired inside a TEAM/WORKFLOW member
run now pauses to AWAITING_APPROVAL (graph + workflow kinds) and resumes exactly-once via
approve/reject, while the unsupported kinds (multi_agent / group_chat / delegate) FAIL
CLOSED. They drive the runner BODY (``run_orchestration``) directly with the offline stub
and a ``requires_approval`` tool whose side-effect counter proves exactly-once.

Fixture: each member is an OPERATOR-provisioned spec naming the test ``tools_module`` that
registers ``wire_money`` (``requires_approval=True``, recording every execution in
``WIRE_CALLS``). The graph member binds it, the stub calls it once via AUTO_TOOLS, and the
gate denies it for lack of approval — pausing the member into a durable checkpoint.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from himmy.application.orchestration_runner import run_orchestration
from himmy.config.agent_spec import AgentSpec
from himmy.runtime.checkpoint import (
    InMemoryCheckpointStore,
    SqliteCheckpointStore,
    SqliteGraphCheckpointStore,
)
from himmy.services.storage.models import AgentDefRecord
from himmy.services.storage.service import StorageService
from tests.application import _per_run_tools
from tests.conftest import run_async

_TOOLS_MODULE = "tests.application._per_run_tools"


def _member(
    name: str,
    *,
    tools: list[str] | None = None,
    workspace: str = "acme",
) -> AgentDefRecord:
    """An OPERATOR-provisioned stored agent naming the gated-tool ``tools_module``."""
    spec = AgentSpec(
        name=name,
        description=f"{name} agent",
        tools_module=_TOOLS_MODULE,
        tools=list(tools or []),
    )
    return AgentDefRecord.from_spec(spec, workspace_id=workspace)


def setup_function() -> None:
    """Reset the gated-tool execution counter before each test."""
    _per_run_tools.WIRE_CALLS.clear()
    _per_run_tools.CALLS.clear()


@pytest.fixture(autouse=True)
def _allow_operator_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator specs may keep ``tools_module`` only under this opt-in (T0.3)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")


def _pause(graph_store: SqliteGraphCheckpointStore, cp_store):
    """Start a 2-member workflow that pauses at the first (gated) member; return outcome."""
    members = [_member("step1", tools=["wire_money"]), _member("step2")]
    return members, run_async(
        run_orchestration(
            kind="graph",
            members=members,
            prompt="go",
            resource_kind="workflow",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=True,
            graph_checkpoint_store=graph_store,
            checkpoint_store=cp_store,
        )
    )


def _resume(members, graph_store, cp_store, *, approve: bool):
    """Drive the approve/reject resume splice and return the outcome."""
    return run_async(
        run_orchestration(
            kind="graph",
            members=members,
            prompt="",
            resource_kind="workflow",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=True,
            graph_checkpoint_store=graph_store,
            checkpoint_store=cp_store,
            graph_resume_id=_orch_cp(graph_store),
            approve_member=approve,
        )
    )


def _orch_cp(graph_store: SqliteGraphCheckpointStore) -> str:
    """The single interrupted graph checkpoint id (the orchestration resume id)."""
    interrupted = graph_store.list_by_status("interrupted")
    assert interrupted, "expected one interrupted graph checkpoint"
    return interrupted[0].checkpoint_id


def test_workflow_member_pauses_to_awaiting_approval(tmp_path: Path) -> None:
    """A workflow member calling a gated tool pauses with member + orchestration ids.

    The graph stops ONE superstep early: the paused node is still in the frontier, NOT in
    completed_nodes, and the tool has NOT run.
    """
    graph_store = SqliteGraphCheckpointStore(str(tmp_path / "g.db"))
    cp_store = SqliteCheckpointStore(str(tmp_path / "cp.db"))
    _members, outcome = _pause(graph_store, cp_store)

    assert outcome.awaiting_approval is True
    assert outcome.awaiting_member == "step1"
    assert outcome.member_checkpoint_id  # the paused MEMBER checkpoint
    assert outcome.orchestration_checkpoint_id  # the durable GRAPH checkpoint
    assert not _per_run_tools.WIRE_CALLS  # the gated tool did NOT run

    # Graph stopped one superstep early: node still pending, not completed, not swept.
    chk = graph_store.load(outcome.orchestration_checkpoint_id)
    assert chk is not None
    assert chk.status == "interrupted"
    assert chk.paused_node == "step1"
    assert chk.frontier == ["step1"]
    assert chk.completed_nodes == []
    assert chk.superstep == 0
    assert chk.member_checkpoint_id == outcome.member_checkpoint_id

    # The MEMBER checkpoint is awaiting_approval and names the gated tool.
    member_chk = cp_store.load(outcome.member_checkpoint_id)
    assert member_chk is not None
    assert member_chk.status == "awaiting_approval"
    assert [p.tool_name for p in member_chk.pending_tool_calls] == ["wire_money"]


def test_approve_fires_tool_once_and_completes(tmp_path: Path) -> None:
    """Approve runs the gated tool EXACTLY ONCE, the graph continues, and completes."""
    graph_store = SqliteGraphCheckpointStore(str(tmp_path / "g.db"))
    cp_store = SqliteCheckpointStore(str(tmp_path / "cp.db"))
    members, _outcome = _pause(graph_store, cp_store)

    resumed = _resume(members, graph_store, cp_store, approve=True)

    assert resumed.failed is False
    assert resumed.awaiting_approval is False
    assert resumed.stopped_reason == "completed"
    assert len(_per_run_tools.WIRE_CALLS) == 1  # fired EXACTLY once
    assert resumed.route == ["step1", "step2"]  # paused node NOT re-run; step2 ran

    chk = graph_store.load(resumed.graph_checkpoint_id)
    assert chk is not None
    assert chk.status == "completed"
    assert chk.completed_nodes == ["step1", "step2"]
    assert chk.visit_counts == {"step1": 1, "step2": 1}  # each advanced exactly once
    assert chk.superstep == 2
    # The member fields are cleared so a later interrupt is never mistaken for approval.
    assert chk.member_checkpoint_id == ""
    assert chk.paused_node == ""


def test_reject_records_rejection_without_running_the_tool(tmp_path: Path) -> None:
    """Reject records the rejection, never runs the gated tool, and the graph continues."""
    graph_store = SqliteGraphCheckpointStore(str(tmp_path / "g.db"))
    cp_store = SqliteCheckpointStore(str(tmp_path / "cp.db"))
    members, outcome = _pause(graph_store, cp_store)

    resumed = _resume(members, graph_store, cp_store, approve=False)

    assert resumed.failed is False
    assert resumed.stopped_reason == "completed"
    assert not _per_run_tools.WIRE_CALLS  # the gated tool was NEVER executed
    assert resumed.route == ["step1", "step2"]
    # The member checkpoint is resolved as rejected.
    assert cp_store.load(outcome.member_checkpoint_id).status == "rejected"


def test_double_approve_is_a_no_op(tmp_path: Path) -> None:
    """A second (concurrent) approve drives the tool once; the loser is a clean no-op."""
    graph_store = SqliteGraphCheckpointStore(str(tmp_path / "g.db"))
    cp_store = SqliteCheckpointStore(str(tmp_path / "cp.db"))
    members, outcome = _pause(graph_store, cp_store)
    orch_cp = outcome.orchestration_checkpoint_id

    async def _both() -> list:
        return await asyncio.gather(
            run_orchestration(
                kind="graph",
                members=members,
                prompt="",
                resource_kind="workflow",
                storage=StorageService(),
                shared_inference=None,
                operator_provisioned=True,
                graph_checkpoint_store=graph_store,
                checkpoint_store=cp_store,
                graph_resume_id=orch_cp,
                approve_member=True,
            ),
            run_orchestration(
                kind="graph",
                members=members,
                prompt="",
                resource_kind="workflow",
                storage=StorageService(),
                shared_inference=None,
                operator_provisioned=True,
                graph_checkpoint_store=graph_store,
                checkpoint_store=cp_store,
                graph_resume_id=orch_cp,
                approve_member=True,
            ),
            return_exceptions=True,
        )

    run_async(_both())
    # Whatever the interleaving, the gated tool ran AT MOST once (the member claim gates
    # it). A claim-loser raises a HimmyError('already resolved') the service treats as a
    # no-op; the winner completes. Exactly-once is the load-bearing assertion.
    assert len(_per_run_tools.WIRE_CALLS) == 1


def test_two_member_pause_resume_pause_then_complete(tmp_path: Path) -> None:
    """A workflow gating at TWO members pauses, resumes, pauses again, then completes."""
    graph_store = SqliteGraphCheckpointStore(str(tmp_path / "g.db"))
    cp_store = SqliteCheckpointStore(str(tmp_path / "cp.db"))
    members = [
        _member("step1", tools=["wire_money"]),
        _member("step2", tools=["wire_money"]),
    ]

    def _run(*, resume_id: str | None, approve: bool | None):
        return run_async(
            run_orchestration(
                kind="graph",
                members=members,
                prompt="go",
                resource_kind="workflow",
                storage=StorageService(),
                shared_inference=None,
                operator_provisioned=True,
                graph_checkpoint_store=graph_store,
                checkpoint_store=cp_store,
                graph_resume_id=resume_id,
                approve_member=approve,
            )
        )

    first = _run(resume_id=None, approve=None)
    assert first.awaiting_approval is True
    assert first.awaiting_member == "step1"
    orch_cp = first.orchestration_checkpoint_id
    first_member_cp = first.member_checkpoint_id

    # Approve member 1: it fires once and the graph advances to member 2, which pauses.
    second = _run(resume_id=orch_cp, approve=True)
    assert second.awaiting_approval is True
    assert second.awaiting_member == "step2"
    assert second.member_checkpoint_id != first_member_cp  # restamped to member 2
    assert len(_per_run_tools.WIRE_CALLS) == 1

    # Approve member 2: it fires once and the whole workflow completes.
    third = _run(resume_id=orch_cp, approve=True)
    assert third.failed is False
    assert third.stopped_reason == "completed"
    assert len(_per_run_tools.WIRE_CALLS) == 2
    assert third.route == ["step1", "step2"]


def test_process_restart_durability(tmp_path: Path) -> None:
    """Killing + restarting over the same SQLite files then approving still completes."""
    g_db = str(tmp_path / "g.db")
    cp_db = str(tmp_path / "cp.db")
    members = [_member("step1", tools=["wire_money"]), _member("step2")]

    # Run 1 (process A): pause.
    outcome = run_async(
        run_orchestration(
            kind="graph",
            members=members,
            prompt="go",
            resource_kind="workflow",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=True,
            graph_checkpoint_store=SqliteGraphCheckpointStore(g_db),
            checkpoint_store=SqliteCheckpointStore(cp_db),
        )
    )
    assert outcome.awaiting_approval is True
    orch_cp = outcome.orchestration_checkpoint_id

    # Run 2 (process B): brand-new stores over the SAME files (no shared memory).
    resumed = run_async(
        run_orchestration(
            kind="graph",
            members=members,
            prompt="",
            resource_kind="workflow",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=True,
            graph_checkpoint_store=SqliteGraphCheckpointStore(g_db),
            checkpoint_store=SqliteCheckpointStore(cp_db),
            graph_resume_id=orch_cp,
            approve_member=True,
        )
    )
    assert resumed.failed is False
    assert resumed.stopped_reason == "completed"
    assert len(_per_run_tools.WIRE_CALLS) == 1
    assert resumed.route == ["step1", "step2"]


def test_crash_after_resolve_advances_not_stuck(tmp_path: Path) -> None:
    """A crash AFTER the member resolves but BEFORE the graph advances still completes.

    Simulate it by resolving the MEMBER checkpoint out-of-band (so the resume's claim is
    lost) while the GRAPH checkpoint is still interrupted at that node. The retry must
    recover and advance to completion rather than stranding the run.
    """
    graph_store = SqliteGraphCheckpointStore(str(tmp_path / "g.db"))
    cp_store = SqliteCheckpointStore(str(tmp_path / "cp.db"))
    members, outcome = _pause(graph_store, cp_store)

    # Simulate the crash window: the member checkpoint is already resolved (the winning
    # resume executed the tool + recorded it) but the graph never advanced.
    member_chk = cp_store.load(outcome.member_checkpoint_id)
    cp_store.save(member_chk.model_copy(update={"status": "approved"}))
    _per_run_tools.WIRE_CALLS.append({"recorded": True})  # the prior winner already ran it
    before = len(_per_run_tools.WIRE_CALLS)

    resumed = _resume(members, graph_store, cp_store, approve=True)

    assert resumed.failed is False
    assert resumed.stopped_reason == "completed"
    assert resumed.route == ["step1", "step2"]  # advanced past the resolved member
    # The gated tool did NOT run a second time (recovery re-threads text only).
    assert len(_per_run_tools.WIRE_CALLS) == before


@pytest.mark.parametrize("kind", ["multi_agent", "group_chat"])
def test_unsupported_kind_fails_closed(kind: str) -> None:
    """A gated tool in a multi_agent / group_chat run FAILS CLOSED naming the tool."""
    members = [_member("m1", tools=["wire_money"])]
    outcome = run_async(
        run_orchestration(
            kind=kind,
            members=members,
            prompt="wire money to the vendor",
            resource_kind="team",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=True,
            checkpoint_store=InMemoryCheckpointStore(),
        )
    )
    assert outcome.failed is True
    assert "wire_money" in (outcome.error or "")
    assert not _per_run_tools.WIRE_CALLS  # never executed un-approved


def test_tool_less_workflow_is_unchanged(tmp_path: Path) -> None:
    """A workflow with NO gated tool completes normally (byte-identical path)."""
    members = [_member("a"), _member("b")]
    outcome = run_async(
        run_orchestration(
            kind="graph",
            members=members,
            prompt="go",
            resource_kind="workflow",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=True,
            graph_checkpoint_store=SqliteGraphCheckpointStore(str(tmp_path / "g.db")),
            checkpoint_store=SqliteCheckpointStore(str(tmp_path / "cp.db")),
        )
    )
    assert outcome.failed is False
    assert outcome.awaiting_approval is False
    assert outcome.stopped_reason == "completed"
    assert outcome.route == ["a", "b"]
