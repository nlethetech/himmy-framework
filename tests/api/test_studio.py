"""Tests for the Himmy Studio API + SPA serving (Phase 0)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app, studio_is_built


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_doctor_endpoint_shape(client: TestClient) -> None:
    """GET /api/studio/doctor returns the structured diagnostics report."""
    r = client.get("/api/studio/doctor")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "python",
        "version",
        "extras",
        "providers",
        "keys",
        "guardrails",
        "has_real_model",
        "has_agent",
        "next_step",
    ):
        assert key in body
    # providers always include the two local CLIs we probe
    names = {p["name"] for p in body["providers"]}
    assert {"claude", "ollama"} <= names
    assert body["next_step"]["kind"] in {"install_model", "scaffold", "run"}


def test_doctor_matches_cli_report() -> None:
    """The API serves exactly the shared diagnostics snapshot (no drift from CLI)."""
    from himmy.runtime.diagnostics import collect_doctor_report

    client = TestClient(create_app())
    api_body = client.get("/api/studio/doctor").json()
    direct = collect_doctor_report().to_dict()
    # version/python/guardrails are stable within a process
    assert api_body["version"] == direct["version"]
    assert api_body["guardrails"] == direct["guardrails"]
    assert [e["module"] for e in api_body["extras"]] == [
        e["module"] for e in direct["extras"]
    ]


def test_unknown_api_paths_404_not_swallowed(client: TestClient) -> None:
    """The SPA fallback must never mask a real API 404."""
    assert client.get("/api/studio/does-not-exist").status_code == 404
    assert client.get("/v1/nope").status_code == 404


@pytest.mark.skipif(
    not studio_is_built(),
    reason="Studio SPA not built (run `cd studio && npm run build`)",
)
class TestSpaServing:
    """Only runs when the frontend has been built into the package."""

    def test_spa_shell_served_at_root(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert '<div id="root">' in r.text

    def test_client_route_falls_back_to_index(self, client: TestClient) -> None:
        # A client-side route the server has no handler for serves the SPA shell.
        r = client.get("/doctor")
        assert r.status_code == 200
        assert '<div id="root">' in r.text

    def test_hashed_assets_served(self, client: TestClient) -> None:
        import re

        shell = client.get("/").text
        m = re.search(r"/assets/(index-[^\"]+\.js)", shell)
        assert m, "no hashed JS asset referenced in the shell"
        r = client.get(f"/assets/{m.group(1)}")
        assert r.status_code == 200
        assert "javascript" in r.headers["content-type"]

    def test_health_not_shadowed_by_spa(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
