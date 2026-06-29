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


@pytest.mark.parametrize("token", ["off", "OFF", "no", "n", "false", "0"])
def test_guard_disabled_via_canonical_falsy_tokens(tmp_path, monkeypatch, token) -> None:
    """reattack-r7: HIMMY_STUDIO_GUARD routes through the canonical falsy parser.

    The prior ad-hoc tuple ``("0","false","no")`` omitted the canonical ``off``/``n``
    tokens, so ``HIMMY_STUDIO_GUARD=off`` silently kept the guard ON — a posture-vocabulary
    divergence from the sibling ``HIMMY_STUDIO_AUTH`` switch. Every canonical falsy spelling
    must now disable the guard (the non-loopback Host that is otherwise 403'd is allowed).
    """
    monkeypatch.setenv("HIMMY_STUDIO_GUARD", token)
    monkeypatch.chdir(tmp_path)
    c = TestClient(create_app())
    assert (
        c.get(
            "/api/studio/connections", headers={"host": "evil.example.com"}
        ).status_code
        == 200
    )


def test_guard_stays_on_for_unrecognised_token(tmp_path, monkeypatch) -> None:
    """An unrecognised value (typo) must keep the default-ON guard active (fail-closed)."""
    monkeypatch.setenv("HIMMY_STUDIO_GUARD", "of")  # typo for 'off'
    monkeypatch.chdir(tmp_path)
    c = TestClient(create_app())
    assert (
        c.get(
            "/api/studio/connections", headers={"host": "evil.example.com"}
        ).status_code
        == 403
    )
