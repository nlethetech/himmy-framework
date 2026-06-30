"""WS-bola — team/workflow run endpoints honour the subject WRITE gate (red-team r4).

A team / workflow run stamps a canonical :class:`RunRecord` with the request body's
``subject_id``. The sibling ``POST /v1/runs`` and ``POST /v1/threads/{id}/messages``
paths gate that field via :func:`enforce_subject_write`, but the two orchestration run
endpoints originally took ``subject_id`` straight from the body — letting a
``subject_scoped`` principal STAMP an orchestration run under ANOTHER data subject in the
SAME tenant (an attribution / lineage / right-to-erasure scope poisoning BOLA write).

These tests pin the fix: a ``subject_scoped`` (non-``tenant_admin``) principal launching a
team / workflow run under a foreign ``subject_id`` is **403**; under its OWN subject it is
allowed; and the offline / ``all_tenants`` path is byte-unchanged (the gate short-circuits).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api import teams_store as svc
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.studio_runs import reset_run_store


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """An authenticated app with a subject_scoped operator key in a shared tenant."""
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
    with TestClient(create_app(ApiContainer.build_default())) as test_client:
        test_client.app.state.authenticator = ApiKeyAuthenticator(  # type: ignore[attr-defined]
            key_principals={
                # A subject_scoped operator: may write ONLY under its own subject "subj-a".
                "key-a": Principal.build(
                    "subj-a",
                    tenant_ids=["acme"],
                    roles=["operator"],
                    auth_method="apikey",
                    subject_scoped=True,
                ),
            }
        )
        yield test_client.app
    svc.reset_teams_store()
    svc.reset_workflows_store()
    svc.reset_graph_checkpoint_store()


def _client(app: Any, key: str) -> TestClient:
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": key})
    return c


def _store_agent(client: TestClient) -> str:
    res = client.post(
        "/v1/agents",
        json={"workspace_id": "acme", "spec": {"name": "a", "description": "h"}},
    )
    assert res.status_code == 201, res.text
    return res.json()["agent_id"]


def _create_team(client: TestClient, member: str) -> str:
    res = client.post(
        "/v1/teams",
        json={
            "workspace_id": "acme",
            "name": "squad",
            "kind": "multi_agent",
            "members": [member],
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["id"]


def _create_workflow(client: TestClient, member: str) -> str:
    res = client.post(
        "/v1/workflows",
        json={"workspace_id": "acme", "name": "flow", "members": [member]},
    )
    assert res.status_code == 201, res.text
    return res.json()["id"]


def test_subject_scoped_cannot_run_team_under_foreign_subject(app: Any) -> None:
    """A subject_scoped operator is 403 launching a TEAM run under another subject."""
    c = _client(app, "key-a")
    team_id = _create_team(c, _store_agent(c))
    resp = c.post(
        f"/v1/teams/{team_id}/run",
        json={"workspace_id": "acme", "subject_id": "victim-bob", "prompt": "hi"},
    )
    assert resp.status_code == 403, resp.text


def test_subject_scoped_cannot_run_workflow_under_foreign_subject(app: Any) -> None:
    """A subject_scoped operator is 403 launching a WORKFLOW run under another subject."""
    c = _client(app, "key-a")
    workflow_id = _create_workflow(c, _store_agent(c))
    resp = c.post(
        f"/v1/workflows/{workflow_id}/run",
        json={"workspace_id": "acme", "subject_id": "victim-bob", "prompt": "hi"},
    )
    assert resp.status_code == 403, resp.text


def test_subject_scoped_may_run_under_own_subject(app: Any) -> None:
    """A subject_scoped operator launching under its OWN subject is NOT blocked (no 403)."""
    c = _client(app, "key-a")
    team_id = _create_team(c, _store_agent(c))
    resp = c.post(
        f"/v1/teams/{team_id}/run",
        json={"workspace_id": "acme", "subject_id": "subj-a", "prompt": "hi"},
    )
    assert resp.status_code != 403, resp.text


def test_offline_team_workflow_run_subject_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INVARIANT: offline / no-auth run under ANY subject is byte-unchanged (gate inert)."""
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
    with TestClient(create_app(ApiContainer.build_default())) as c:
        agent_id = _store_agent(c)
        team_id = _create_team(c, agent_id)
        # An anonymous (all_tenants) caller may stamp ANY subject — the gate short-circuits.
        resp = c.post(
            f"/v1/teams/{team_id}/run",
            json={"workspace_id": "acme", "subject_id": "anyone", "prompt": "hi"},
        )
        assert resp.status_code != 403, resp.text
    svc.reset_teams_store()
    svc.reset_workflows_store()
    svc.reset_graph_checkpoint_store()
