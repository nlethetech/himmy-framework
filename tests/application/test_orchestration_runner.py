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
from typing import Any

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


def test_graph_node_routes_to_pinned_provider_not_stub(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A workflow/graph member that pins a provider must execute on THAT provider's
    backend — via the ``<provider>:<model>`` dispatch key the team inference manager
    registers — not silently fall through to the stub default.

    Regression for the bug where ``_run_graph`` passed ``spec.to_llm_config()`` (whose
    model_key is the plain model name), missing the dispatch key so every graph/workflow
    run executed on the stub regardless of the pinned provider. multi_agent/group_chat
    threaded the dispatch key correctly; the graph node did not.
    """
    from himmy.services.inference.client_manager import StubClientManager
    from himmy.services.inference.models import InferenceResponse, InferenceStatus

    class _SpyManager:
        provider_name = "spy"

        def resolve(self, model_key: str) -> str:
            return f"spy:{model_key}"

        async def generate(self, request: Any) -> InferenceResponse:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="SPY_PROVIDER_RAN",
                input_tokens=1,
                output_tokens=1,
                provider_name="spy",
            )

    def _fake_build_manager_for(provider: str | None, model: Any = None) -> Any:
        return _SpyManager() if provider == "spy" else StubClientManager()

    # build_team_inference imports build_manager_for from himmy.cli.provider at call time.
    monkeypatch.setattr(
        "himmy.cli.provider.build_manager_for", _fake_build_manager_for
    )

    async def _scenario() -> None:
        spec = AgentSpec(
            name="step1", description="pinned", provider="spy", model="m1"
        )
        member = AgentDefRecord.from_spec(spec, workspace_id="acme")
        outcome = await run_orchestration(
            kind="graph",
            members=[member],
            prompt="go",
            resource_kind="workflow",
            storage=StorageService(),
            shared_inference=None,
            operator_provisioned=False,
            graph_checkpoint_store=SqliteGraphCheckpointStore(str(tmp_path / "g.db")),
        )
        assert outcome.failed is False
        # Routed to the pinned provider (NOT the stub default).
        assert "SPY_PROVIDER_RAN" in (outcome.output_text or "")
        assert "[stub:" not in (outcome.output_text or "")

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
