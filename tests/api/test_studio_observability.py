"""Observability: health probe + request-id header + clean 500 on unhandled errors."""

from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from himmy.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app())


def test_studio_health(client: TestClient) -> None:
    r = client.get("/api/studio/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "secrets_writable" in body
    assert set(body["providers"]) == {"claude_cli", "ollama"}


def test_request_id_header_present(client: TestClient) -> None:
    r = client.get("/api/studio/health")
    assert r.headers.get("x-request-id")


def test_inbound_request_id_is_echoed(client: TestClient) -> None:
    r = client.get("/api/studio/health", headers={"x-request-id": "trace-123"})
    assert r.headers.get("x-request-id") == "trace-123"


def test_unhandled_error_becomes_clean_500(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app()
    boom = APIRouter()

    @boom.get("/api/studio/_boom")
    async def _boom() -> dict:
        raise RuntimeError("kaboom")

    app.include_router(boom)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/studio/_boom")
    assert r.status_code == 500
    assert r.json()["detail"] == "internal server error"
    assert r.json()["request_id"]  # the id is surfaced for tracing
