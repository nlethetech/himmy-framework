"""Tests for Studio lineage: the provenance transform + the graph router."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_lineage as sl
from himmy.api.app import create_app
from himmy.api.studio_runs import (
    CognitionStep,
    ModelUsage,
    StudioRun,
    get_run_store,
    reset_run_store,
)
from himmy.entities.records import EntityRecord


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    yield
    reset_run_store()


def test_run_lineage_reshapes_trace(store: None) -> None:
    get_run_store().save(
        StudioRun(
            id="r1",
            created_at="t",
            agent_name="mailer",
            provider="ollama",
            prompt="email bob",
            output="Sent to bob.",
            input_tokens=100,
            output_tokens=20,
            usage_by_model=[
                ModelUsage(model="ollama:qwen", input_tokens=100, output_tokens=20)
            ],
            steps=[
                CognitionStep(
                    seq=1, kind="tool", name="send_email", read_only=False, result="ok"
                ),
                CognitionStep(
                    seq=2,
                    kind="grounding",
                    source="memory",
                    query="bob",
                    citations=[{"text": "Bob is the farm vet", "similarity": 0.8}],
                ),
                CognitionStep(seq=3, kind="delegate", worker="ops", task="schedule"),
            ],
        )
    )
    view = sl.run_lineage("r1")
    assert view is not None
    assert view.agent == "mailer"
    assert view.answer == "Sent to bob."
    assert view.models == ["ollama:qwen"]
    assert view.tokens == 120
    assert [t.name for t in view.tools] == ["send_email"]
    assert view.tools[0].read_only is False
    assert view.evidence[0].source == "memory"
    assert view.evidence[0].text == "Bob is the farm vet"
    assert view.delegates[0].worker == "ops"


def test_run_lineage_unknown(store: None) -> None:
    assert sl.run_lineage("nope") is None


# ---- /api/studio/lineage router --------------------------------------------


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    yield TestClient(create_app())
    reset_run_store()


def _registry(client: TestClient) -> Any:
    return client.app.state.container.entity_registry  # type: ignore[attr-defined]


def _seed_thread_graph(registry: Any, thread_id: str) -> dict[str, EntityRecord]:
    """A tiny realistic lineage: message → thread → persona / snapshot."""
    thread = EntityRecord.create(
        stable_id=thread_id,
        version=1,
        kind="chat_thread",
        payload={"name": "desk thread"},
    )
    persona = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="persona",
        payload={"name": "Analyst"},
    )
    message = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="message",
        payload={"content": "what moved the market today?"},
    )
    snapshot = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="context_snapshot",
        payload={"summary": "NEPSE flat, banking up"},
    )
    for rec in (thread, persona, message, snapshot):
        registry.register(rec)
    registry.link(
        from_record_id=thread.record_id,
        to_record_id=persona.record_id,
        relation="uses_persona",
    )
    registry.link(
        from_record_id=message.record_id,
        to_record_id=thread.record_id,
        relation="in_thread",
    )
    registry.link(
        from_record_id=thread.record_id,
        to_record_id=snapshot.record_id,
        relation="built_from",
    )
    return {
        "thread": thread,
        "persona": persona,
        "message": message,
        "snapshot": snapshot,
    }


def test_graph_by_entity_record_id(client: TestClient) -> None:
    recs = _seed_thread_graph(_registry(client), str(uuid.uuid4()))
    r = client.get(
        "/api/studio/lineage/graph",
        params={"entity_id": recs["thread"].record_id},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["root_id"] == recs["thread"].record_id
    assert {n["kind"] for n in body["nodes"]} == {
        "chat_thread",
        "persona",
        "message",
        "context_snapshot",
    }
    # wire shape: edges use "from"/"to", nodes carry label + created_at
    edge = body["edges"][0]
    assert set(edge) == {"from", "to", "relation"}
    assert {e["relation"] for e in body["edges"]} == {
        "uses_persona",
        "in_thread",
        "built_from",
    }
    by_id = {n["id"]: n for n in body["nodes"]}
    assert by_id[recs["persona"].record_id]["label"] == "Analyst"
    assert by_id[recs["thread"].record_id]["created_at"]
    assert body["truncated"] is False
    assert body["node_count"] == 4 and body["edge_count"] == 3


def test_graph_by_stable_id_resolves_latest(client: TestClient) -> None:
    thread_id = str(uuid.uuid4())
    recs = _seed_thread_graph(_registry(client), thread_id)
    r = client.get("/api/studio/lineage/graph", params={"entity_id": thread_id})
    assert r.status_code == 200
    assert r.json()["root_id"] == recs["thread"].record_id


def test_graph_depth_bounds_walk(client: TestClient) -> None:
    registry = _registry(client)
    # a 4-link chain: n0 -> n1 -> n2 -> n3 -> n4
    chain = [
        EntityRecord.create(
            stable_id=str(uuid.uuid4()),
            version=1,
            kind="message",
            payload={"content": f"hop {i}"},
        )
        for i in range(5)
    ]
    for rec in chain:
        registry.register(rec)
    for a, b in zip(chain, chain[1:], strict=False):
        registry.link(
            from_record_id=a.record_id,
            to_record_id=b.record_id,
            relation="derived_from",
        )
    r = client.get(
        "/api/studio/lineage/graph",
        params={"entity_id": chain[0].record_id, "depth": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["node_count"] == 3  # root + 2 hops
    assert body["truncated"] is True  # deeper records remained


def test_graph_node_cap_80(client: TestClient) -> None:
    registry = _registry(client)
    hub = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="chat_thread",
        payload={"name": "hub"},
    )
    registry.register(hub)
    for i in range(100):
        leaf = EntityRecord.create(
            stable_id=str(uuid.uuid4()),
            version=1,
            kind="message",
            payload={"content": f"m{i}"},
        )
        registry.register(leaf)
        registry.link(
            from_record_id=leaf.record_id,
            to_record_id=hub.record_id,
            relation="in_thread",
        )
    r = client.get("/api/studio/lineage/graph", params={"entity_id": hub.record_id})
    assert r.status_code == 200
    body = r.json()
    assert body["node_count"] == 80  # hard cap
    assert body["truncated"] is True
    assert body["root_id"] == hub.record_id
    assert any(n["id"] == hub.record_id for n in body["nodes"])  # root kept
    # no dangling edges to pruned nodes
    kept = {n["id"] for n in body["nodes"]}
    assert all(e["from"] in kept and e["to"] in kept for e in body["edges"])


def test_graph_param_validation(client: TestClient) -> None:
    assert client.get("/api/studio/lineage/graph").status_code == 400
    assert (
        client.get(
            "/api/studio/lineage/graph",
            params={"entity_id": "a", "run_id": "b"},
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/studio/lineage/graph", params={"entity_id": "a", "depth": 0}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/studio/lineage/graph", params={"entity_id": "a", "depth": 7}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/studio/lineage/graph", params={"entity_id": "x" * 201}
        ).status_code
        == 422
    )


def test_graph_unknown_entity_404(client: TestClient) -> None:
    r = client.get("/api/studio/lineage/graph", params={"entity_id": str(uuid.uuid4())})
    assert r.status_code == 404
    assert "no entity found" in r.json()["detail"]


def test_graph_by_studio_run_thread(client: TestClient) -> None:
    thread_id = str(uuid.uuid4())
    _seed_thread_graph(_registry(client), thread_id)
    get_run_store().save(
        StudioRun(id="run9", created_at="t", prompt="hi", thread_id=thread_id)
    )
    r = client.get("/api/studio/lineage/graph", params={"run_id": "run9"})
    assert r.status_code == 200
    assert r.json()["node_count"] == 4


def test_graph_run_not_projected_404(client: TestClient) -> None:
    get_run_store().save(
        StudioRun(id="run10", created_at="t", prompt="hi", thread_id=str(uuid.uuid4()))
    )
    r = client.get("/api/studio/lineage/graph", params={"run_id": "run10"})
    assert r.status_code == 404
    assert "not projected" in r.json()["detail"]


def test_graph_run_without_thread_404(client: TestClient) -> None:
    get_run_store().save(StudioRun(id="run11", created_at="t", prompt="hi"))
    r = client.get("/api/studio/lineage/graph", params={"run_id": "run11"})
    assert r.status_code == 404
    assert "no thread" in r.json()["detail"]


def test_graph_unknown_run_404(client: TestClient) -> None:
    r = client.get("/api/studio/lineage/graph", params={"run_id": "ghost"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_entity_detail_with_links(client: TestClient) -> None:
    recs = _seed_thread_graph(_registry(client), str(uuid.uuid4()))
    r = client.get(f"/api/studio/lineage/entity/{recs['thread'].record_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "chat_thread"
    assert body["label"] == "desk thread"
    assert body["payload"] == {"name": "desk thread"}
    out_relations = {ln["relation"] for ln in body["links_out"]}
    assert out_relations == {"uses_persona", "built_from"}
    in_links = body["links_in"]
    assert len(in_links) == 1
    assert in_links[0]["relation"] == "in_thread"
    assert in_links[0]["other_kind"] == "message"
    assert in_links[0]["other_id"] == recs["message"].record_id


def test_entity_detail_truncates_huge_payload(client: TestClient) -> None:
    registry = _registry(client)
    rec = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="context_snapshot",
        payload={"summary": "x" * 10_000, "items": list(range(500))},
    )
    registry.register(rec)
    r = client.get(f"/api/studio/lineage/entity/{rec.record_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["payload"]["summary"].endswith("… [truncated]")
    assert len(body["payload"]["summary"]) < 10_000
    assert len(body["payload"]["items"]) == 101  # 100 + truncation marker
    assert body["payload"]["items"][-1].startswith("…")


def test_entity_detail_unknown_404(client: TestClient) -> None:
    r = client.get(f"/api/studio/lineage/entity/{uuid.uuid4()}")
    assert r.status_code == 404
