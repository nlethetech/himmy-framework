"""Tests for the API kernel: the FastAPI BFF over the application services."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app


def _client() -> TestClient:
    return TestClient(create_app(ApiContainer.build_default()))


def test_health_ok() -> None:
    """The liveness probe returns ok."""
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def _await_run(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    """Poll the run endpoint until the run reaches a terminal status."""
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get(f"/v1/runs/{run_id}").json()
        if last.get("status") in ("SUCCEEDED", "FAILED"):
            return last
        time.sleep(0.02)
    return last


def test_create_and_read_run() -> None:
    """POST /v1/runs creates a background run that completes and is readable."""
    client = _client()
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "Analyst", "description": "careful"},
        "task": {"title": "brief", "prompt": "Summarize ACME"},
    }
    resp = client.post("/v1/runs", json=body)
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "QUEUED"
    run_id = run["run_id"]

    final = _await_run(client, run_id)
    assert final["status"] == "SUCCEEDED"
    assert final["output_text"]

    # List + events + thread endpoints (list returns a paged envelope, AAEO-8).
    listed = client.get("/v1/runs", params={"workspace_id": "w1"}).json()
    assert "items" in listed and "total" in listed
    assert any(r["run_id"] == run_id for r in listed["items"])
    events = client.get(f"/v1/runs/{run_id}/events").json()
    assert isinstance(events, list) and events
    thread = client.get(f"/v1/runs/{run_id}/thread").json()
    assert thread["thread_id"]


def test_run_idempotency_via_api() -> None:
    """Re-posting the same idempotency_key returns the same run id."""
    client = _client()
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "Analyst"},
        "task": {"title": "t", "prompt": "hi"},
        "idempotency_key": "abc",
    }
    first = client.post("/v1/runs", json=body).json()
    second = client.post("/v1/runs", json=body).json()
    assert first["run_id"] == second["run_id"]


def test_get_unknown_run_404() -> None:
    """An unknown run id yields a 404."""
    client = _client()
    assert client.get("/v1/runs/does-not-exist").status_code == 404


def test_run_lineage_endpoint_json_and_dot() -> None:
    """GET /v1/runs/{id}/lineage traces provenance as JSON and as Graphviz DOT."""
    client = _client()
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "Analyst", "description": "careful"},
        "task": {"title": "brief", "prompt": "Summarize ACME"},
    }
    run_id = client.post("/v1/runs", json=body).json()["run_id"]
    _await_run(client, run_id)

    graph = client.get(
        f"/v1/runs/{run_id}/lineage", params={"workspace_id": "w1"}
    ).json()
    kinds = {node["kind"] for node in graph["nodes"].values()}
    assert {"chat_thread", "persona", "prompt"} <= kinds
    assert any(edge["relation"] == "uses_persona" for edge in graph["edges"])
    assert graph["truncated"] is False

    dot = client.get(
        f"/v1/runs/{run_id}/lineage", params={"workspace_id": "w1", "format": "dot"}
    )
    assert dot.headers["content-type"].startswith("text/plain")
    assert dot.text.startswith("digraph lineage {")

    # Tenant isolation + unknown run both 404.
    assert (
        client.get(
            f"/v1/runs/{run_id}/lineage", params={"workspace_id": "other"}
        ).status_code
        == 404
    )
    assert client.get("/v1/runs/nope/lineage").status_code == 404


def test_recommendation_provenance_endpoint() -> None:
    """GET /v1/recommendations/{id}/provenance traces a rec to its run + persona."""
    from tests.conftest import run_async

    container = ApiContainer.build_default()
    client = TestClient(create_app(container))
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "Analyst", "description": "careful"},
        "task": {"title": "brief", "prompt": "Summarize ACME"},
    }
    run_id = client.post("/v1/runs", json=body).json()["run_id"]
    _await_run(client, run_id)

    async def _graph_a_recommendation() -> str:
        # The stub doesn't emit a RecommendationEnvelope; graph one from the same
        # run so its real thread_id wires derived_from to the run's hub.
        run = await container.run_app.get_run(run_id, workspace_id="w1")
        run.output_structured = {
            "recommendations": [
                {"kind": "buy", "title": "Accumulate", "confidence": 0.6}
            ]
        }
        items = await container.recommendation_app.extract_from_run(run)
        return items[0].recommendation_id

    rec_id = run_async(_graph_a_recommendation())

    graph = client.get(
        f"/v1/recommendations/{rec_id}/provenance", params={"workspace_id": "w1"}
    ).json()
    kinds = {node["kind"] for node in graph["nodes"].values()}
    assert {"recommendation", "chat_thread", "persona"} <= kinds
    assert any(edge["relation"] == "derived_from" for edge in graph["edges"])

    dot = client.get(
        f"/v1/recommendations/{rec_id}/provenance",
        params={"workspace_id": "w1", "format": "dot"},
    )
    assert dot.text.startswith("digraph lineage {")

    # Tenant isolation + unknown id both 404.
    assert (
        client.get(
            f"/v1/recommendations/{rec_id}/provenance", params={"workspace_id": "zzz"}
        ).status_code
        == 404
    )
    assert client.get("/v1/recommendations/none/provenance").status_code == 404


def test_context_field_upsert_and_list() -> None:
    """Context fields upsert and list back for a subject."""
    client = _client()
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "fields": [{"key": "company", "value": "ACME", "confidence": 0.9}],
    }
    resp = client.post("/v1/context/fields:upsert", json=body)
    assert resp.status_code == 200
    assert resp.json()["upserted"] == 1
    listed = client.get("/v1/context/fields", params={"subject_id": "s1"}).json()
    assert any(f["key"] == "company" for f in listed)


def test_build_and_get_snapshot() -> None:
    """A snapshot builds from a spec and is retrievable by id."""
    client = _client()
    body = {
        "subject_id": "s1",
        "build_spec": {"keys": [{"key": "company", "required": True}]},
    }
    resp = client.post("/v1/context/snapshots:build", json=body)
    assert resp.status_code == 200
    snap = resp.json()
    assert snap["subject_id"] == "s1"
    fetched = client.get(f"/v1/context/snapshots/{snap['snapshot_id']}")
    assert fetched.status_code == 200
    assert client.get("/v1/context/snapshots/missing").status_code == 404


def test_recommendations_list_and_patch_404() -> None:
    """Recommendations list (empty initially); patching unknown id 404s."""
    client = _client()
    listed = client.get("/v1/recommendations", params={"workspace_id": "w1"}).json()
    assert listed["items"] == [] and listed["total"] == 0
    resp = client.patch("/v1/recommendations/missing", json={"status": "ACCEPTED"})
    assert resp.status_code == 404


def test_dashboard_summary_endpoint() -> None:
    """The dashboard summary endpoint returns the aggregate shape."""
    client = _client()
    resp = client.get(
        "/v1/dashboard/summary",
        params={"subject_id": "s1", "workspace_id": "w1"},
    )
    assert resp.status_code == 200
    summary = resp.json()
    assert "context" in summary
    assert "runs" in summary
    assert "recommendations" in summary
