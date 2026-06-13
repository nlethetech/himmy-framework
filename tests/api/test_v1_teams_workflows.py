"""T3b — /v1 teams + workflows: workspace-scoped CRUD + run over the shared run machinery.

These tests pin the UNIT 4 acceptance:

* ``POST /v1/teams`` stores a team whose member ``agent_id``s are validated to EXIST and be
  in the SAME workspace (an unknown / cross-workspace member is a 404 at create);
* ``POST /v1/teams/{id}/run`` executes the multi-agent orchestration on the per-run runtime
  via ``RunAppService`` and returns a canonical :class:`RunRecord` that COMPLETES (polled
  via ``GET /v1/runs/{id}``), recording the team lineage in run metadata;
* deleting a stored agent still referenced by a team / workflow is refused (409 — the
  production reference_finder the T2e delete path needs);
* a workflow run works (the durable linear pipeline) and records a graph checkpoint id;
* a cross-workspace team / workflow read is a 404 (authenticated regime); over HTTP with no
  auth (the zero-config default) scoping is inert (the standalone posture);
* the T0.4 per-workspace quota is enforced on the team/workflow run fan-out (429).

The HTTP tests run offline (stub provider, in-RAM container, no authenticator), so RBAC +
workspace scoping are inert and the operator path applies; the authenticated isolation is
asserted with a tenant-bound API key where the boundary is real.
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


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A clean project root with isolated teams/workflows + run stores."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_TEAMS_PATH", str(tmp_path / "teams.db"))
    monkeypatch.setenv("HIMMY_WORKFLOWS_PATH", str(tmp_path / "workflows.db"))
    monkeypatch.setenv(
        "HIMMY_GRAPH_CHECKPOINTS_PATH", str(tmp_path / "graph_checkpoints.db")
    )
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
    """An offline /v1 client over a fresh in-RAM container (no auth → operator).

    Entered as a context manager so the app lifespan is active: a team/workflow run
    dispatches through ``RunAppService`` on the lifespan-established loop.
    """
    with TestClient(create_app(ApiContainer.build_default())) as test_client:
        yield test_client


def _store_agent(
    client: TestClient, name: str = "helper", workspace: str = "acme"
) -> str:
    """Store a minimal /v1 agent and return its agent_id."""
    res = client.post(
        "/v1/agents",
        json={"workspace_id": workspace, "spec": {"name": name, "description": "h"}},
    )
    assert res.status_code == 201, res.text
    return res.json()["agent_id"]


def _team_body(members: list[str], **over: object) -> dict:
    base: dict = {
        "workspace_id": "acme",
        "name": "Research squad",
        "kind": "multi_agent",
        "members": members,
    }
    base.update(over)
    return base


def _workflow_body(members: list[str], **over: object) -> dict:
    base: dict = {
        "workspace_id": "acme",
        "name": "Brief pipeline",
        "members": members,
    }
    base.update(over)
    return base


def _poll_run(client: TestClient, run_id: str, workspace: str = "acme") -> dict:
    deadline = time.monotonic() + 15.0
    got: dict = {}
    while time.monotonic() < deadline:
        got = client.get(
            f"/v1/runs/{run_id}", params={"workspace_id": workspace}
        ).json()
        if got.get("status") in ("SUCCEEDED", "FAILED", "AWAITING_APPROVAL"):
            return got
        time.sleep(0.02)
    return got


# ----------------------------------------------------------------- teams CRUD


def test_create_and_read_team(client: TestClient) -> None:
    """POST /v1/teams stores a workspace-scoped team of ordered stored agents."""
    a1 = _store_agent(client, "triage")
    a2 = _store_agent(client, "writer")
    res = client.post("/v1/teams", json=_team_body([a1, a2]))
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["members"] == [a1, a2]
    assert body["workspace_id"] == "acme"
    assert body["kind"] == "multi_agent"

    tid = body["id"]
    got = client.get(f"/v1/teams/{tid}", params={"workspace_id": "acme"})
    assert got.status_code == 200
    assert got.json()["id"] == tid

    listing = client.get("/v1/teams", params={"workspace_id": "acme"}).json()
    assert [t["id"] for t in listing["items"]] == [tid]


def test_update_and_delete_team(client: TestClient) -> None:
    """PUT replaces members/kind; DELETE removes it; a re-DELETE is 404."""
    a1 = _store_agent(client, "a")
    a2 = _store_agent(client, "b")
    tid = client.post("/v1/teams", json=_team_body([a1])).json()["id"]

    updated = client.put(
        f"/v1/teams/{tid}",
        json={
            "workspace_id": "acme",
            "name": "renamed",
            "kind": "group_chat",
            "members": [a1, a2],
        },
    )
    assert updated.status_code == 200
    assert updated.json()["kind"] == "group_chat"
    assert updated.json()["members"] == [a1, a2]

    assert (
        client.delete(f"/v1/teams/{tid}", params={"workspace_id": "acme"}).status_code
        == 200
    )
    assert (
        client.get(f"/v1/teams/{tid}", params={"workspace_id": "acme"}).status_code
        == 404
    )
    assert (
        client.delete(f"/v1/teams/{tid}", params={"workspace_id": "acme"}).status_code
        == 404
    )


def test_team_create_is_idempotent(client: TestClient) -> None:
    """A re-POST with the same idempotency_key returns the prior team, no duplicate."""
    a1 = _store_agent(client)
    body = {**_team_body([a1]), "idempotency_key": "once"}
    first = client.post("/v1/teams", json=body)
    assert first.status_code == 201
    second = client.post("/v1/teams", json=body)
    assert second.status_code == 201
    assert second.json()["id"] == first.json()["id"]
    listing = client.get("/v1/teams", params={"workspace_id": "acme"}).json()
    assert listing["total"] == 1


# ------------------------------------------------- referential member checks


def test_team_unknown_member_is_404(client: TestClient) -> None:
    """A team must reference stored agents that exist in the workspace (404 otherwise)."""
    a1 = _store_agent(client)
    res = client.post("/v1/teams", json=_team_body([a1, "does-not-exist"]))
    assert res.status_code == 404


def test_team_cannot_reference_another_workspaces_agent(client: TestClient) -> None:
    """Binding a team to an agent stored in a DIFFERENT workspace is a 404."""
    a1 = _store_agent(client, workspace="acme")
    other = _store_agent(client, name="other", workspace="globex")
    res = client.post("/v1/teams", json=_team_body([a1, other], workspace_id="acme"))
    assert res.status_code == 404


def test_empty_members_is_422(client: TestClient) -> None:
    """A team requires at least one member."""
    res = client.post("/v1/teams", json=_team_body([]))
    assert res.status_code == 422


def test_duplicate_members_is_422(client: TestClient) -> None:
    """Member agent_ids must be distinct."""
    a1 = _store_agent(client)
    res = client.post("/v1/teams", json=_team_body([a1, a1]))
    assert res.status_code == 422


# --------------------------------------------------------- team run -> canonical


def test_team_run_completes_and_is_in_canonical_runs(client: TestClient) -> None:
    """POST /v1/teams/{id}/run executes the orchestration and returns a completing run."""
    a1 = _store_agent(client, "triage")
    a2 = _store_agent(client, "writer")
    tid = client.post("/v1/teams", json=_team_body([a1, a2])).json()["id"]

    res = client.post(f"/v1/teams/{tid}/run", json={"workspace_id": "acme", "prompt": "hi"})
    assert res.status_code == 200, res.text
    run_id = res.json()["run_id"]
    assert res.json()["status"] in ("QUEUED", "RUNNING")

    final = _poll_run(client, run_id)
    assert final["status"] == "SUCCEEDED", final
    meta = final.get("metadata") or {}
    assert meta.get("orchestration") == "team"
    assert meta.get("team_id") == tid
    assert meta.get("member_agent_ids") == [a1, a2]

    # The run is visible in the canonical /v1 runs list for the tenant workspace.
    runs = client.get("/v1/runs", params={"workspace_id": "acme"}).json()
    assert any(r["run_id"] == run_id for r in runs["items"]), runs


def test_team_run_group_chat_kind(client: TestClient) -> None:
    """A group_chat team also runs to completion through the run machinery."""
    a1 = _store_agent(client, "alice")
    a2 = _store_agent(client, "bob")
    tid = client.post(
        "/v1/teams", json=_team_body([a1, a2], kind="group_chat")
    ).json()["id"]
    res = client.post(
        f"/v1/teams/{tid}/run", json={"workspace_id": "acme", "prompt": "discuss"}
    )
    assert res.status_code == 200
    final = _poll_run(client, res.json()["run_id"])
    assert final["status"] == "SUCCEEDED", final


def test_run_unknown_team_is_404(client: TestClient) -> None:
    res = client.post("/v1/teams/nope/run", json={"workspace_id": "acme", "prompt": "x"})
    assert res.status_code == 404


def test_run_team_with_deleted_member_is_404(client: TestClient) -> None:
    """If a member was removed (via cascade) since create, the run is a clean 404."""
    a1 = _store_agent(client, "x")
    a2 = _store_agent(client, "y")
    tid = client.post("/v1/teams", json=_team_body([a1, a2])).json()["id"]
    # cascade-delete a member out from under the team.
    assert (
        client.delete(
            f"/v1/agents/{a2}", params={"workspace_id": "acme", "cascade": True}
        ).status_code
        == 200
    )
    res = client.post(f"/v1/teams/{tid}/run", json={"workspace_id": "acme", "prompt": "x"})
    assert res.status_code == 404


# ------------------------------------------------------- workflows CRUD + run


def test_create_and_run_workflow(client: TestClient) -> None:
    """POST /v1/workflows stores a pipeline; POST /{id}/run runs it to completion."""
    a1 = _store_agent(client, "step1")
    a2 = _store_agent(client, "step2")
    res = client.post("/v1/workflows", json=_workflow_body([a1, a2]))
    assert res.status_code == 201, res.text
    wid = res.json()["id"]
    assert res.json()["members"] == [a1, a2]

    run = client.post(
        f"/v1/workflows/{wid}/run", json={"workspace_id": "acme", "prompt": "go"}
    )
    assert run.status_code == 200, run.text
    final = _poll_run(client, run.json()["run_id"])
    assert final["status"] == "SUCCEEDED", final
    meta = final.get("metadata") or {}
    assert meta.get("orchestration") == "workflow"
    assert meta.get("workflow_id") == wid
    # A workflow always runs the durable graph pipeline → a checkpoint id is recorded.
    assert meta.get("graph_checkpoint_id")


def test_workflow_unknown_member_is_404(client: TestClient) -> None:
    a1 = _store_agent(client)
    res = client.post("/v1/workflows", json=_workflow_body([a1, "ghost"]))
    assert res.status_code == 404


def test_run_unknown_workflow_is_404(client: TestClient) -> None:
    res = client.post(
        "/v1/workflows/nope/run", json={"workspace_id": "acme", "prompt": "x"}
    )
    assert res.status_code == 404


# ------------------------------------------------ referenced-agent delete (409)


def test_delete_agent_referenced_by_team_is_409(client: TestClient) -> None:
    """Deleting a stored agent still wired to a team is refused (409, must_fix)."""
    a1 = _store_agent(client)
    client.post("/v1/teams", json=_team_body([a1]))

    res = client.delete(f"/v1/agents/{a1}", params={"workspace_id": "acme"})
    assert res.status_code == 409, res.text

    # cascade removes it anyway (the caller owns cleanup).
    res2 = client.delete(
        f"/v1/agents/{a1}", params={"workspace_id": "acme", "cascade": True}
    )
    assert res2.status_code == 200


def test_delete_agent_referenced_by_workflow_is_409(client: TestClient) -> None:
    """Deleting a stored agent still wired to a workflow is refused (409)."""
    a1 = _store_agent(client)
    client.post("/v1/workflows", json=_workflow_body([a1]))
    res = client.delete(f"/v1/agents/{a1}", params={"workspace_id": "acme"})
    assert res.status_code == 409, res.text


# ----------------------------------------- T0.4 per-workspace quota on the run


def test_team_run_enforces_workspace_quota(project: Path) -> None:
    """A burst of team runs past the per-workspace outstanding cap is rejected with 429."""
    container = ApiContainer.build_default()
    # Pin the outstanding cap to 1 so a second concurrent run is rejected.
    container.run_app._workspace_max_outstanding = 1
    with TestClient(create_app(container)) as client:
        a1 = _store_agent(client, "slow")
        tid = client.post("/v1/teams", json=_team_body([a1])).json()["id"]
        # Saturate the single slot by holding it: the first run is admitted; manually
        # reserve to make the second hit the cap deterministically.
        container.run_app._ws_outstanding["acme"] = 1
        res = client.post(
            f"/v1/teams/{tid}/run", json={"workspace_id": "acme", "prompt": "x"}
        )
        assert res.status_code == 429, res.text


# ------------------------------------------ authenticated cross-tenant isolation


def _tenant_client(project: Path, key: str = "key-acme") -> TestClient:
    """A client bound to a tenant API key (a NON-operator multi-tenant caller)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-acme": Principal.build(
                "acme-user",
                tenant_ids=["acme"],
                roles=["operator"],
                auth_method="apikey",
            ),
            "key-globex": Principal.build(
                "globex-user",
                tenant_ids=["globex"],
                roles=["operator"],
                auth_method="apikey",
            ),
        }
    )
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": key})
    return client


def test_cross_workspace_team_access_is_404_authenticated(project: Path) -> None:
    """A team is 404 to a different tenant over HTTP (the authenticated regime)."""
    acme = _tenant_client(project, key="key-acme")
    globex = _tenant_client(project, key="key-globex")

    agent_id = acme.post(
        "/v1/agents", json={"workspace_id": "acme", "spec": {"name": "secret"}}
    ).json()["agent_id"]
    tid = acme.post(
        "/v1/teams", json=_team_body([agent_id], workspace_id="acme")
    ).json()["id"]

    assert (
        acme.get(f"/v1/teams/{tid}", params={"workspace_id": "acme"}).status_code == 200
    )
    # Asking for acme's workspace as globex is a 403 (not authorized for that tenant).
    assert (
        globex.get(f"/v1/teams/{tid}", params={"workspace_id": "acme"}).status_code
        == 403
    )
    # In its own scope the team simply does not exist → 404.
    assert (
        globex.get(f"/v1/teams/{tid}", params={"workspace_id": "globex"}).status_code
        == 404
    )
    listed = globex.get("/v1/teams", params={"workspace_id": "globex"}).json()
    assert all(t["id"] != tid for t in listed["items"])


def test_team_write_requires_run_write_permission(project: Path) -> None:
    """A viewer (run:read only) cannot create or run a team (403)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "viewer-key": Principal.build(
                "v", tenant_ids=["acme"], roles=["viewer"], auth_method="apikey"
            )
        }
    )
    viewer = TestClient(app)
    viewer.headers.update({"x-himmy-internal-key": "viewer-key"})

    assert viewer.get("/v1/teams", params={"workspace_id": "acme"}).status_code == 200
    res = viewer.post("/v1/teams", json=_team_body(["any"]))
    assert res.status_code == 403
