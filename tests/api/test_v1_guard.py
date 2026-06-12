"""The rebinding/cross-site guard covers the unauth /v1/* surface, not just Studio.

``POST /v1/runs`` background-executes an agent and the rest of ``/v1/*`` reads
runs/threads/context — so the same Host + Origin/Referer loopback guard that protects
``/api/studio`` must protect ``/v1`` too, or a DNS-rebound malicious page reaches it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app())


def _run_body() -> dict:
    return {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "p", "description": "d", "instructions": ["i"]},
        "task": {"title": "t", "prompt": "do"},
    }


def test_v1_loopback_host_allowed(client: TestClient) -> None:
    r = client.get("/v1/runs", params={"workspace_id": "w1"}, headers={"host": "127.0.0.1:8765"})
    assert r.status_code == 200


def test_v1_non_loopback_host_blocked(client: TestClient) -> None:
    r = client.get("/v1/runs", params={"workspace_id": "w1"}, headers={"host": "evil.example.com"})
    assert r.status_code == 403
    assert r.json()["detail"] == "host not allowed"


def test_v1_cross_origin_blocked(client: TestClient) -> None:
    r = client.get(
        "/v1/runs",
        params={"workspace_id": "w1"},
        headers={"host": "127.0.0.1", "origin": "http://evil.example.com"},
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "cross-origin blocked"


def test_v1_runs_post_blocked_from_rebound_origin(client: TestClient) -> None:
    # Agent execution must NOT be reachable by a rebound (non-loopback Host) origin.
    r = client.post("/v1/runs", json=_run_body(), headers={"host": "attacker.test"})
    assert r.status_code == 403


def test_health_remains_open_for_probes(client: TestClient) -> None:
    # Liveness probes hit /health with the pod IP as Host — it must stay reachable.
    assert client.get("/health", headers={"host": "10.0.0.5"}).status_code == 200


def test_v1_guard_disabled_via_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HIMMY_STUDIO_GUARD", "0")
    monkeypatch.chdir(tmp_path)
    c = TestClient(create_app())
    r = c.get("/v1/runs", params={"workspace_id": "w1"}, headers={"host": "evil.example.com"})
    assert r.status_code == 200


def test_v1_allow_extra_host_via_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HIMMY_STUDIO_ALLOW_HOSTS", "proxy.internal")
    monkeypatch.chdir(tmp_path)
    c = TestClient(create_app())
    r = c.get(
        "/v1/runs", params={"workspace_id": "w1"}, headers={"host": "proxy.internal"}
    )
    assert r.status_code == 200
