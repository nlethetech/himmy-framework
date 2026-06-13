"""End-to-end /v1: a WORKFLOW run whose member gates pauses + resumes over HTTP.

Drives the full surface — ``POST /v1/workflows/{id}/run`` -> AWAITING_APPROVAL ->
``GET /v1/runs/{id}/pending-approvals`` -> ``POST /v1/runs/{id}/approve`` — to prove the
orchestration HITL machinery is wired all the way through the routers and the durable
graph + member checkpoint stores (offline, no auth → operator).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api import teams_store as svc
from himmy.api.studio_runs import reset_run_store
from tests.application import _per_run_tools

_TOOLS_MODULE = "tests.application._per_run_tools"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A clean project root with isolated teams/workflows + durable checkpoint stores."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_TEAMS_PATH", str(tmp_path / "teams.db"))
    monkeypatch.setenv("HIMMY_WORKFLOWS_PATH", str(tmp_path / "workflows.db"))
    monkeypatch.setenv(
        "HIMMY_GRAPH_CHECKPOINTS_PATH", str(tmp_path / "graph_checkpoints.db")
    )
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    svc.reset_teams_store()
    svc.reset_workflows_store()
    svc.reset_graph_checkpoint_store()
    reset_run_store()
    yield tmp_path
    svc.reset_teams_store()
    svc.reset_workflows_store()
    svc.reset_graph_checkpoint_store()


@pytest.fixture
def client(project: Path) -> Iterator[TestClient]:
    """An offline /v1 client over a fresh in-RAM container (no auth → operator)."""
    with TestClient(create_app(ApiContainer.build_default())) as test_client:
        yield test_client


def setup_function() -> None:
    """Reset the gated-tool counter before each test."""
    _per_run_tools.WIRE_CALLS.clear()
    _per_run_tools.CALLS.clear()


def _store_agent(client: TestClient, name: str, *, tools: list[str] | None = None) -> str:
    """Store an operator-provisioned /v1 agent naming the gated-tool module."""
    spec: dict = {"name": name, "description": name, "tools_module": _TOOLS_MODULE}
    if tools:
        spec["tools"] = tools
    res = client.post("/v1/agents", json={"workspace_id": "acme", "spec": spec})
    assert res.status_code == 201, res.text
    return res.json()["agent_id"]


def _poll(client: TestClient, run_id: str, *, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    got: dict = {}
    while time.monotonic() < deadline:
        got = client.get(f"/v1/runs/{run_id}", params={"workspace_id": "acme"}).json()
        if got.get("status") in ("SUCCEEDED", "FAILED", "AWAITING_APPROVAL"):
            return got
        time.sleep(0.02)
    return got


def test_workflow_run_pauses_then_approve_completes(client: TestClient) -> None:
    """A workflow whose first member gates pauses, then approve fires the tool once."""
    step1 = _store_agent(client, "step1", tools=["wire_money"])
    step2 = _store_agent(client, "step2")

    wf = client.post(
        "/v1/workflows",
        json={"workspace_id": "acme", "name": "pipeline", "members": [step1, step2]},
    )
    assert wf.status_code == 201, wf.text
    wf_id = wf.json()["id"]

    run = client.post(
        f"/v1/workflows/{wf_id}/run",
        json={"workspace_id": "acme", "subject_id": "s1", "prompt": "go"},
    )
    assert run.status_code == 200, run.text
    run_id = run.json()["run_id"]

    paused = _poll(client, run_id)
    assert paused["status"] == "AWAITING_APPROVAL", paused
    assert (paused.get("metadata") or {}).get("hitl_kind") == "orchestration"
    assert (paused.get("metadata") or {}).get("awaiting_member") == "step1"
    assert not _per_run_tools.WIRE_CALLS  # the gated tool did NOT run

    # The pending-approvals read surfaces the gated tool.
    pend = client.get(
        f"/v1/runs/{run_id}/pending-approvals", params={"workspace_id": "acme"}
    )
    assert pend.status_code == 200, pend.text
    names = [p["tool_name"] for p in pend.json()["pending_tool_calls"]]
    assert names == ["wire_money"]

    # Approve -> the tool fires once and the workflow completes.
    appr = client.post(
        f"/v1/runs/{run_id}/approve", json={"workspace_id": "acme"}
    )
    assert appr.status_code == 200, appr.text
    done = _poll(client, run_id)
    assert done["status"] == "SUCCEEDED", done
    assert len(_per_run_tools.WIRE_CALLS) == 1
    assert (done.get("metadata") or {}).get("route") == ["step1", "step2"]


def test_workflow_run_reject_does_not_fire_the_tool(client: TestClient) -> None:
    """Rejecting a paused workflow member never runs the gated tool."""
    step1 = _store_agent(client, "step1", tools=["wire_money"])
    step2 = _store_agent(client, "step2")
    wf_id = client.post(
        "/v1/workflows",
        json={"workspace_id": "acme", "name": "pipeline", "members": [step1, step2]},
    ).json()["id"]
    run_id = client.post(
        f"/v1/workflows/{wf_id}/run",
        json={"workspace_id": "acme", "subject_id": "s1", "prompt": "go"},
    ).json()["run_id"]
    assert _poll(client, run_id)["status"] == "AWAITING_APPROVAL"

    rej = client.post(f"/v1/runs/{run_id}/reject", json={"workspace_id": "acme"})
    assert rej.status_code == 200, rej.text
    done = _poll(client, run_id)
    assert done["status"] in ("SUCCEEDED", "FAILED")
    assert not _per_run_tools.WIRE_CALLS  # the gated tool was NEVER executed
