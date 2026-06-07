"""HTTP-level integration tests for the new Studio endpoints + error paths.

Drives the FastAPI app via TestClient (Host=testserver is allowed by the guard),
covering connections, memory, knowledge, evals, workflows, approvals, and lineage —
their happy paths and the 404/409/400/422 error responses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_knowledge, studio_memory
from himmy.api.app import create_app
from himmy.api.studio_approvals import reset_checkpoint_store
from himmy.api.studio_runs import reset_run_store
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_MEMORY_PATH", str(tmp_path / "memory.db"))
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    reset_run_store()
    reset_checkpoint_store()
    studio_memory.reset_memory_service()
    studio_knowledge.reset_kb_service()
    yield TestClient(create_app())
    configure_secrets(None)
    studio_memory.reset_memory_service()
    studio_knowledge.reset_kb_service()


# ---- connections --------------------------------------------------------


def test_connections_crud(client: TestClient) -> None:
    assert {c["type"] for c in client.get("/api/studio/connections").json()} >= {
        "email",
        "telegram",
        "web_search",
    }
    r = client.put(
        "/api/studio/connections/telegram",
        json={"fields": {"bot_token": "123:abc", "chat_id": "42"}},
    )
    assert r.status_code == 200 and r.json()["configured"] is True
    assert "123:abc" not in r.text  # secret never echoed
    assert (
        client.delete("/api/studio/connections/telegram").json()["configured"] is False
    )


def test_connections_unknown_type_404(client: TestClient) -> None:
    assert client.get("/api/studio/connections/slack").status_code == 404
    assert (
        client.put("/api/studio/connections/slack", json={"fields": {}}).status_code
        == 404
    )


def test_connections_read_only_backend_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    configure_secrets(EnvSecrets())  # not writable
    try:
        c = TestClient(create_app())
        r = c.put(
            "/api/studio/connections/telegram", json={"fields": {"bot_token": "x"}}
        )
        assert r.status_code == 409
    finally:
        configure_secrets(None)


# ---- memory -------------------------------------------------------------


def test_memory_endpoints(client: TestClient) -> None:
    add = client.post(
        "/api/studio/memory", json={"text": "We keep ducks", "subject_id": "farm"}
    )
    assert add.status_code == 200
    assert "farm" in client.get("/api/studio/memory/subjects").json()
    items = client.get("/api/studio/memory?subject=farm").json()
    assert items[0]["text"] == "We keep ducks"
    recall = client.post(
        "/api/studio/memory/recall", json={"query": "ducks", "subject_id": "farm"}
    )
    assert recall.status_code == 200
    assert client.delete(f"/api/studio/memory/{items[0]['memory_id']}").json()["ok"]


def test_memory_input_caps_422(client: TestClient) -> None:
    # text over the 20k cap → validation error
    assert (
        client.post("/api/studio/memory", json={"text": "x" * 20_001}).status_code
        == 422
    )
    # top_k out of range
    assert (
        client.post(
            "/api/studio/memory/recall", json={"query": "q", "top_k": 999}
        ).status_code
        == 422
    )


# ---- knowledge ----------------------------------------------------------


def test_knowledge_endpoints(client: TestClient) -> None:
    kb = client.post("/api/studio/knowledge", json={"name": "docs"}).json()
    ing = client.post(
        f"/api/studio/knowledge/{kb['kb_id']}/ingest",
        json={"text": "Ducks lay ~300 eggs a year.", "title": "ducks"},
    )
    assert ing.status_code == 200
    assert client.get("/api/studio/knowledge").json()[0]["documents"] == 1
    search = client.post(
        f"/api/studio/knowledge/{kb['kb_id']}/search", json={"query": "eggs"}
    )
    assert search.status_code == 200 and isinstance(search.json(), list)
    assert client.delete(f"/api/studio/knowledge/{kb['kb_id']}").json()["ok"]


# ---- evals & workflows (discovery) --------------------------------------


def test_eval_discovery_and_bad_path(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "demo.eval.yaml").write_text(
        "name: demo\ncases:\n  - input: { prompt: hi }\n"
    )
    assert client.get("/api/studio/evals").json()[0]["name"] == "demo"
    # running an unknown suite → 400
    bad = client.post(
        "/api/studio/evals/run",
        json={"suite_path": "nope.eval.yaml", "agent_path": "nope.yaml"},
    )
    assert bad.status_code == 400


def test_workflow_discovery(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "wf.workflow.yaml").write_text(
        "name: wf\nsteps:\n  - name: s\n    subtask: do it\n"
    )
    found = client.get("/api/studio/workflows").json()
    assert found[0]["name"] == "wf" and found[0]["steps"][0]["subtask"] == "do it"


# ---- approvals & lineage ------------------------------------------------


def test_approvals_empty_and_404(client: TestClient) -> None:
    assert client.get("/api/studio/approvals").json() == []
    assert client.get("/api/studio/approvals/nope").status_code == 404


def test_lineage_404_for_unknown_run(client: TestClient) -> None:
    assert client.get("/api/studio/runs/nope/lineage").status_code == 404
