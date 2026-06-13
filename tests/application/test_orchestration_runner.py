"""T3b — the orchestration runner: team/group-chat/graph engines + durable graph resume.

These tests cover the BODY a /v1 team/workflow run executes (``run_orchestration``):

* a ``multi_agent`` team produces an answer and a routing trail over the stub provider;
* a ``group_chat`` team produces an answer;
* a workflow (the durable linear graph pipeline) completes and threads each member's output
  to the next, recording a graph checkpoint id;
* a long graph run RESUMES after a simulated restart — interrupt it mid-pipeline (a tiny
  timeout), reopen a FRESH :class:`SqliteGraphCheckpointStore` over the SAME file (a new
  process), and resume from the persisted checkpoint to completion (the T3b acceptance).

They run fully offline (the stub provider via the in-memory storage), so no network and no
real model are needed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from himmy.application.orchestration_runner import run_orchestration
from himmy.config.agent_spec import AgentSpec
from himmy.runtime.checkpoint import SqliteGraphCheckpointStore
from himmy.services.storage.models import AgentDefRecord
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _member(name: str, workspace: str = "acme") -> AgentDefRecord:
    """A stored agent record over a tool-less persona spec (runs on the stub provider)."""
    spec = AgentSpec(name=name, description=f"{name} agent")
    return AgentDefRecord.from_spec(spec, workspace_id=workspace)


def test_multi_agent_run_produces_answer() -> None:
    async def _scenario() -> None:
        members = [_member("triage"), _member("writer")]
        outcome = await run_orchestration(
            kind="multi_agent",
            members=members,
            prompt="hello",
            resource_kind="team",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=False,
        )
        assert outcome.failed is False
        assert outcome.output_text
        # The entry member ran (its name leads the route).
        assert outcome.route and outcome.route[0] == "triage"

    run_async(_scenario())


def test_group_chat_run_produces_answer() -> None:
    async def _scenario() -> None:
        members = [_member("alice"), _member("bob")]
        outcome = await run_orchestration(
            kind="group_chat",
            members=members,
            prompt="discuss",
            resource_kind="team",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=False,
        )
        assert outcome.failed is False
        assert outcome.output_text
        assert set(outcome.route) <= {"alice", "bob"}

    run_async(_scenario())


def test_duplicate_member_names_are_disambiguated() -> None:
    """Two stored agents sharing a name run without a member-name collision."""

    async def _scenario() -> None:
        members = [_member("worker"), _member("worker")]
        outcome = await run_orchestration(
            kind="multi_agent",
            members=members,
            prompt="hi",
            resource_kind="team",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=False,
        )
        assert outcome.failed is False

    run_async(_scenario())


def test_workflow_pipeline_completes_with_checkpoint(tmp_path: Path) -> None:
    async def _scenario() -> None:
        members = [_member("step1"), _member("step2")]
        store = SqliteGraphCheckpointStore(str(tmp_path / "g.db"))
        outcome = await run_orchestration(
            kind="graph",
            members=members,
            prompt="go",
            resource_kind="workflow",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=False,
            graph_checkpoint_store=store,
        )
        assert outcome.failed is False
        assert outcome.graph_checkpoint_id
        assert outcome.stopped_reason == "completed"
        # The persisted checkpoint is loadable from the durable file.
        loaded = store.load(outcome.graph_checkpoint_id)
        assert loaded is not None
        assert loaded.status == "completed"

    run_async(_scenario())


def test_graph_run_resumes_after_simulated_restart(tmp_path: Path) -> None:
    """A workflow interrupted mid-pipeline resumes from its durable checkpoint (T3b).

    A tiny per-run timeout interrupts the pipeline after the first step persists a
    checkpoint; a FRESH store over the SAME file (a new process) then resumes it to
    completion using the recorded checkpoint id.
    """
    from himmy.orchestrators import END, StateGraph

    db = str(tmp_path / "graph.db")
    ran: list[str] = []

    def _make(name: str, *, sleep: float = 0.0) -> object:
        async def _node(state: dict[str, object]) -> dict[str, object]:
            if sleep:
                await asyncio.sleep(sleep)
            ran.append(name)
            return {"last": name, "count": int(state.get("count", 0)) + 1}

        return _node

    async def _scenario() -> None:
        store = SqliteGraphCheckpointStore(db)
        # A 3-node linear graph whose middle node sleeps long enough to trip a tiny
        # timeout AFTER the first node's checkpoint persists — forces INTERRUPTED.
        graph = StateGraph(name="wf")
        graph.add_node("a", _make("a"))
        graph.add_node("b", _make("b", sleep=5.0))
        graph.add_node("c", _make("c"))
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)
        compiled = graph.compile(checkpoint_store=store)

        interrupted = await compiled.invoke(
            {"count": 0}, timeout_seconds=0.2, checkpoint_id="cp-1"
        )
        assert interrupted.status == "interrupted"
        assert interrupted.checkpoint_id == "cp-1"
        assert ran == ["a"]  # only the first node completed before the timeout

        # Simulate a restart: a brand-new store over the SAME file (no shared memory) +
        # a freshly compiled graph (topology is code; state is durable).
        store2 = SqliteGraphCheckpointStore(db)
        graph2 = StateGraph(name="wf")
        graph2.add_node("a", _make("a"))
        graph2.add_node("b", _make("b"))  # no sleep now → completes
        graph2.add_node("c", _make("c"))
        graph2.set_entry_point("a")
        graph2.add_edge("a", "b")
        graph2.add_edge("b", "c")
        graph2.add_edge("c", END)
        compiled2 = graph2.compile(checkpoint_store=store2)

        resumed = await compiled2.invoke(resume="cp-1")
        assert resumed.status == "completed", resumed
        # Resumed from b onward (a was already done), so a did NOT re-run.
        assert ran == ["a", "b", "c"]
        assert resumed.final_state.get("last") == "c"

    run_async(_scenario())
