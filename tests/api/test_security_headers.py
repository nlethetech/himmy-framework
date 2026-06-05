"""WS3.4 — security response headers + opt-in strict CORS."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app


def test_security_headers_present_by_default() -> None:
    client = TestClient(create_app(ApiContainer.build_default()))
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert "max-age" in resp.headers["Strict-Transport-Security"]


def test_hsts_can_be_disabled(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_HSTS", "0")
    client = TestClient(create_app(ApiContainer.build_default()))
    resp = client.get("/health")
    assert "Strict-Transport-Security" not in resp.headers


def test_cors_denied_by_default() -> None:
    client = TestClient(create_app(ApiContainer.build_default()))
    resp = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


def test_cors_allows_configured_origin(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_CORS_ORIGINS", "https://app.example")
    client = TestClient(create_app(ApiContainer.build_default()))
    resp = client.get("/health", headers={"Origin": "https://app.example"})
    assert resp.headers.get("access-control-allow-origin") == "https://app.example"
