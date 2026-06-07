"""The Studio API rejects non-loopback Host + cross-origin requests (anti-rebinding)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app())


def test_loopback_host_allowed(client: TestClient) -> None:
    r = client.get("/api/studio/connections", headers={"host": "127.0.0.1:8800"})
    assert r.status_code == 200


def test_non_loopback_host_blocked(client: TestClient) -> None:
    r = client.get("/api/studio/connections", headers={"host": "evil.example.com"})
    assert r.status_code == 403


def test_cross_origin_blocked(client: TestClient) -> None:
    r = client.get(
        "/api/studio/connections",
        headers={"host": "127.0.0.1", "origin": "http://evil.example.com"},
    )
    assert r.status_code == 403


def test_same_origin_allowed(client: TestClient) -> None:
    r = client.get(
        "/api/studio/connections",
        headers={"host": "127.0.0.1", "origin": "http://127.0.0.1:8800"},
    )
    assert r.status_code == 200


def test_non_studio_path_unaffected(client: TestClient) -> None:
    # the guard only applies to /api/studio
    assert (
        client.get("/health", headers={"host": "evil.example.com"}).status_code == 200
    )


def test_guard_disabled_via_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HIMMY_STUDIO_GUARD", "0")
    monkeypatch.chdir(tmp_path)
    c = TestClient(create_app())
    assert (
        c.get(
            "/api/studio/connections", headers={"host": "evil.example.com"}
        ).status_code
        == 200
    )
