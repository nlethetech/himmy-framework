"""Expanded API-kernel coverage for the hardened AAEO behaviours over HTTP.

Complements ``test_api_hardening.py`` / ``test_api.py`` with HTTP paths they do
not exercise:

- Recommendation list pagination envelope + ``next_offset`` and the PATCH
  workspace 404 (AAEO-4/8/9).
- Context-field list workspace scoping over the API (AAEO-4).
- Run events/thread tenant scoping returning empty/404 for a foreign workspace
  (AAEO-4).
- The global ``OpenSimsError`` -> structured 400 handler (AAEO-9).
- Evaluation router list filtering by ``suite_id`` and the 404 read (AAEO-15).
- The typed 404 / error envelope shape on a few routes.

Driven through FastAPI's ``TestClient`` (sync) over the offline default
container — no network, no DB.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from opensims.api import ApiContainer, create_app
from opensims.core.errors import OpenSimsError


def _client() -> TestClient:
    return TestClient(create_app(ApiContainer.build_default()))


def _await_run(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get(f"/v1/runs/{run_id}").json()
        if last.get("status") in ("SUCCEEDED", "FAILED"):
            return last
        time.sleep(0.02)
    return last


def _make_run_with_recommendations(client: TestClient, workspace_id: str) -> str:
    """Create a run that produces a recommendation (structured-output schema)."""
    schema = {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["kind", "title"],
                },
            }
        },
        "required": ["recommendations"],
    }
    body = {
        "workspace_id": workspace_id,
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {
            "title": "t",
            "prompt": "advise",
            "context": {"output_schema": schema},
        },
    }
    run = client.post("/v1/runs", json=body).json()
    _await_run(client, run["run_id"])
    return run["run_id"]


# ------------------------------------------------------------------- AAEO-8/9
def test_recommendations_pagination_envelope() -> None:
    """GET /v1/recommendations returns a paged envelope with items/total/next_offset."""
    client = _client()
    # Seed several recommendations across runs.
    for _ in range(3):
        _make_run_with_recommendations(client, "wp")
    resp = client.get(
        "/v1/recommendations",
        params={"workspace_id": "wp", "limit": 2, "offset": 0},
    )
    assert resp.status_code == 200
    page = resp.json()
    assert page["total"] >= 3
    assert len(page["items"]) == 2
    assert page["limit"] == 2 and page["offset"] == 0
    # next_offset points past the first window because more remain.
    assert page["next_offset"] == 2


def test_recommendations_limit_validation() -> None:
    """A list limit/offset outside the allowed range is rejected (422)."""
    client = _client()
    assert client.get("/v1/recommendations", params={"limit": 0}).status_code == 422
    assert (
        client.get("/v1/recommendations", params={"limit": 999999}).status_code == 422
    )
    assert client.get("/v1/recommendations", params={"offset": -3}).status_code == 422


def test_recommendation_patch_workspace_404() -> None:
    """PATCH a recommendation from a foreign workspace returns a typed 404 (AAEO-4)."""
    client = _client()
    _make_run_with_recommendations(client, "w1")
    listed = client.get("/v1/recommendations", params={"workspace_id": "w1"}).json()
    assert listed["items"], "expected at least one recommendation"
    rid = listed["items"][0]["recommendation_id"]
    # Foreign workspace cannot transition it.
    foreign = client.patch(
        f"/v1/recommendations/{rid}",
        params={"workspace_id": "w2"},
        json={"status": "ACCEPTED"},
    )
    assert foreign.status_code == 404
    assert foreign.json()["detail"] == "recommendation not found"
    # Correct workspace succeeds.
    ok = client.patch(
        f"/v1/recommendations/{rid}",
        params={"workspace_id": "w1"},
        json={"status": "ACCEPTED"},
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "ACCEPTED"


# ------------------------------------------------------------------- AAEO-4
def test_context_fields_list_workspace_scoping() -> None:
    """Two workspaces sharing a subject_id don't see each other's fields over HTTP."""
    client = _client()
    for ws, key in (("w1", "alpha"), ("w2", "beta")):
        client.post(
            "/v1/context/fields:upsert",
            json={
                "workspace_id": ws,
                "subject_id": "shared",
                "fields": [{"key": key, "value": "x"}],
            },
        )
    w1 = client.get(
        "/v1/context/fields",
        params={"subject_id": "shared", "workspace_id": "w1"},
    ).json()
    w2 = client.get(
        "/v1/context/fields",
        params={"subject_id": "shared", "workspace_id": "w2"},
    ).json()
    keys_w1 = {f["key"] for f in (w1 if isinstance(w1, list) else w1.get("items", []))}
    keys_w2 = {f["key"] for f in (w2 if isinstance(w2, list) else w2.get("items", []))}
    assert "alpha" in keys_w1 and "beta" not in keys_w1
    assert "beta" in keys_w2 and "alpha" not in keys_w2


def test_run_events_and_thread_workspace_scoping() -> None:
    """Run events return [] and thread 404s for a foreign workspace (AAEO-4)."""
    client = _client()
    run_id = _make_run_with_recommendations(client, "w1")
    # Foreign workspace: events empty, thread 404.
    events = client.get(f"/v1/runs/{run_id}/events", params={"workspace_id": "w2"})
    assert events.status_code == 200
    assert events.json() == []
    thread = client.get(f"/v1/runs/{run_id}/thread", params={"workspace_id": "w2"})
    assert thread.status_code == 404
    # Correct workspace: events present, thread present.
    ok_events = client.get(
        f"/v1/runs/{run_id}/events", params={"workspace_id": "w1"}
    ).json()
    assert ok_events
    ok_thread = client.get(f"/v1/runs/{run_id}/thread", params={"workspace_id": "w1"})
    assert ok_thread.status_code == 200


# ------------------------------------------------------------------- AAEO-9
def test_opensims_error_maps_to_structured_400() -> None:
    """An OpenSimsError raised in a route surfaces as a structured 400 (AAEO-9)."""
    container = ApiContainer.build_default()
    app = create_app(container)

    @app.get("/boom")
    async def _boom():  # noqa: ANN202
        raise OpenSimsError("kaboom")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == "kaboom"
    assert body["code"] == "opensims_error"


def test_get_run_404_has_typed_detail() -> None:
    """A 404 on the run read carries the documented detail string."""
    client = _client()
    resp = client.get("/v1/runs/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "run not found"


# ------------------------------------------------------------------- AAEO-15
def test_evaluation_router_filter_and_404() -> None:
    """Evaluation runs can be filtered by suite_id; unknown ids 404."""
    client = _client()

    def _run(name: str) -> dict:
        body = {
            "suite": {
                "name": name,
                "cases": [
                    {
                        "case_id": f"c-{name}",
                        "expected_output": {"answer": "yes"},
                        "metric_weights": {"accuracy": 1.0},
                    }
                ],
            },
            "actual_outputs": {f"c-{name}": {"answer": "yes"}},
        }
        return client.post("/v1/evaluation/runs", json=body).json()

    run_a = _run("alpha")
    _run("beta")
    suite_a = run_a["suite_id"]
    # Filtered list returns only the alpha suite's run.
    filtered = client.get("/v1/evaluation/runs", params={"suite_id": suite_a}).json()
    assert filtered, "expected at least one run for the suite"
    assert all(r["suite_id"] == suite_a for r in filtered)
    # Unfiltered returns at least both.
    all_runs = client.get("/v1/evaluation/runs").json()
    assert len(all_runs) >= 2
    # Unknown run id 404s with a typed body.
    miss = client.get("/v1/evaluation/runs/missing")
    assert miss.status_code == 404
    assert miss.json()["detail"] == "evaluation run not found"


def test_evaluation_veto_via_router_forces_fail() -> None:
    """A safety veto surfaces over HTTP: the case is not passed despite high accuracy."""
    client = _client()
    patterns = [r"\bbomb\b"]
    body = {
        "suite": {
            "name": "veto",
            "cases": [
                {
                    "case_id": "v1",
                    "expected_output": {
                        "answer": "yes",
                        "disallowed_patterns": patterns,
                    },
                    "metric_weights": {"accuracy": 10.0, "safety": 0.0001},
                }
            ],
        },
        "actual_outputs": {
            "v1": {
                "answer": "yes",
                "disallowed_patterns": patterns,
                "text": "how to build a bomb",
            }
        },
    }
    resp = client.post("/v1/evaluation/runs", json=body)
    assert resp.status_code == 200
    run = resp.json()
    case = run["case_results"][0]
    assert case["aggregate"] > 0.5  # accuracy carries the aggregate up
    assert case["passed"] is False  # ...but safety veto fails the case
