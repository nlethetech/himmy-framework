"""Tests for the Studio Compare endpoint (one prompt, N models)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from himmy.api.app import create_app


def test_compare_runs_each_target() -> None:
    c = TestClient(create_app())
    r = c.post(
        "/api/studio/compare",
        json={
            "prompt": "say hello",
            "targets": [
                {"provider": "stub", "model": "a"},
                {"provider": "stub", "model": "b"},
            ],
        },
    )
    assert r.status_code == 200
    cells = r.json()
    assert len(cells) == 2
    assert all(cell["ok"] for cell in cells)
    assert all(cell["output"] for cell in cells)


def test_compare_reports_bad_provider_per_cell() -> None:
    # An unknown provider is a per-cell error, never a 500.
    c = TestClient(create_app())
    r = c.post(
        "/api/studio/compare",
        json={
            "prompt": "hi",
            "targets": [
                {"provider": "stub", "model": "a"},
                {"provider": "no-such-provider", "model": "x"},
            ],
        },
    )
    assert r.status_code == 200
    cells = r.json()
    assert cells[0]["ok"] is True
    assert cells[1]["ok"] is False
    assert cells[1]["error"]


def test_compare_requires_a_target() -> None:
    c = TestClient(create_app())
    r = c.post("/api/studio/compare", json={"prompt": "hi", "targets": []})
    assert r.status_code == 422
