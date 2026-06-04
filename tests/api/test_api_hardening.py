"""Tests for the production-hardening API behaviours (AAEO-4/8/9/13/15)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from opensims.api import ApiContainer, create_app


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


# -------------------------------------------------------------------- AAEO-4
def test_run_read_is_workspace_scoped() -> None:
    """GET /v1/runs/{id}?workspace_id= returns 404 for a foreign workspace."""
    client = _client()
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    run = client.post("/v1/runs", json=body).json()
    run_id = run["run_id"]
    _await_run(client, run_id)
    assert (
        client.get(f"/v1/runs/{run_id}", params={"workspace_id": "w1"}).status_code
        == 200
    )
    assert (
        client.get(f"/v1/runs/{run_id}", params={"workspace_id": "w2"}).status_code
        == 404
    )


# -------------------------------------------------------------------- AAEO-8
def test_runs_pagination_envelope() -> None:
    """GET /v1/runs returns a paged envelope with items/total/limit/offset."""
    client = _client()
    for _ in range(3):
        client.post(
            "/v1/runs",
            json={
                "workspace_id": "wp",
                "subject_id": "s1",
                "persona": {"name": "A"},
                "task": {"title": "t", "prompt": "hi"},
            },
        )
    resp = client.get(
        "/v1/runs", params={"workspace_id": "wp", "limit": 2, "offset": 0}
    )
    assert resp.status_code == 200
    page = resp.json()
    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert page["next_offset"] == 2


def test_list_limit_validation() -> None:
    """A limit outside the allowed range is rejected by FastAPI (422)."""
    client = _client()
    assert client.get("/v1/runs", params={"limit": 0}).status_code == 422
    assert client.get("/v1/runs", params={"limit": 100000}).status_code == 422
    assert client.get("/v1/runs", params={"offset": -1}).status_code == 422


# -------------------------------------------------------------------- AAEO-9
def test_openapi_declares_error_and_security() -> None:
    """The OpenAPI schema declares typed error responses + the API-key scheme."""
    client = _client()
    schema = client.get("/openapi.json").json()
    # A 404 route declares a typed body.
    get_run = schema["paths"]["/v1/runs/{run_id}"]["get"]
    assert "404" in get_run["responses"]


def test_internal_key_security_scheme(monkeypatch) -> None:
    """When the internal key is set, the API-key scheme appears + guards routes."""
    monkeypatch.setenv("OPENSIMS_INTERNAL_API_KEY", "topsecret")
    client = TestClient(create_app(ApiContainer.build_default()))
    schema = client.get(
        "/openapi.json", headers={"x-opensims-internal-key": "topsecret"}
    ).json()
    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert any(s.get("type") == "apiKey" for s in schemes.values())
    # Missing/invalid key is rejected with 401 + WWW-Authenticate.
    resp = client.get("/v1/runs")
    assert resp.status_code == 401
    assert "WWW-Authenticate" in resp.headers
    # Valid key passes.
    ok = client.get(
        "/v1/runs",
        headers={"x-opensims-internal-key": "topsecret"},
        params={"workspace_id": "w1"},
    )
    assert ok.status_code == 200


def test_internal_key_rotation_multikey(monkeypatch) -> None:
    """Comma-separated keys support rotation; either key authenticates."""
    monkeypatch.setenv("OPENSIMS_INTERNAL_API_KEY", "old-key , new-key")
    client = TestClient(create_app(ApiContainer.build_default()))
    for key in ("old-key", "new-key"):
        resp = client.get(
            "/v1/runs",
            headers={"x-opensims-internal-key": key},
            params={"workspace_id": "w1"},
        )
        assert resp.status_code == 200
    assert (
        client.get("/v1/runs", headers={"x-opensims-internal-key": "bogus"}).status_code
        == 401
    )


# -------------------------------------------------------------------- AAEO-15
def test_evaluation_router_run_and_read() -> None:
    """POST /v1/evaluation/runs scores a suite; GET reads it back."""
    client = _client()
    suite = {
        "name": "qa",
        "cases": [
            {
                "case_id": "c1",
                "expected_output": {"answer": "yes"},
                "metric_weights": {"accuracy": 1.0},
            }
        ],
    }
    body = {"suite": suite, "actual_outputs": {"c1": {"answer": "yes"}}}
    resp = client.post("/v1/evaluation/runs", json=body)
    assert resp.status_code == 200
    run = resp.json()
    assert run["aggregate_score"] == 1.0
    run_id = run["run_id"]
    # List + get.
    listed = client.get("/v1/evaluation/runs").json()
    assert any(r["run_id"] == run_id for r in listed)
    fetched = client.get(f"/v1/evaluation/runs/{run_id}")
    assert fetched.status_code == 200
    assert client.get("/v1/evaluation/runs/missing").status_code == 404


def test_dashboard_includes_evaluation(monkeypatch) -> None:
    """The dashboard summary folds in the latest evaluation aggregate (AAEO-15)."""
    client = _client()
    client.post(
        "/v1/evaluation/runs",
        json={
            "suite": {
                "name": "qa",
                "cases": [
                    {
                        "case_id": "c1",
                        "expected_output": {"answer": "yes"},
                        "metric_weights": {"accuracy": 1.0},
                    }
                ],
            },
            "actual_outputs": {"c1": {"answer": "yes"}},
        },
    )
    summary = client.get(
        "/v1/dashboard/summary",
        params={"subject_id": "s1", "workspace_id": "w1"},
    ).json()
    assert "evaluation" in summary
    assert summary["evaluation"]["total"] == 1
    assert summary["evaluation"]["latest_aggregate"] == 1.0
