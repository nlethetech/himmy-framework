"""Regression: the DNS-rebinding guard fails CLOSED on a missing/empty Host header.

Friend-audit finding #2: the loopback guard was ``if host and host not in allowed`` —
an empty/absent ``Host`` parses to ``""`` (falsy) and SKIPPED the check, bypassing the
DNS-rebinding/cross-site protection. The fix rejects an empty Host too (``if not host or
host not in allowed``), mirroring the Origin/Referer fail-closed handling. These tests
FAIL against the old behavior (empty Host → allowed) and PASS with the fix.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    app = create_app()

    @app.get("/api/studio/_host_probe")
    async def _probe() -> dict[str, bool]:  # pragma: no cover - trivial
        return {"ok": True}

    return TestClient(app)


def test_empty_host_rejected(client: TestClient) -> None:
    """A guarded request with an empty Host must fail closed → 403."""
    r = client.get("/api/studio/_host_probe", headers={"host": ""})
    assert r.status_code == 403
    assert r.json()["detail"] == "host not allowed"


def test_foreign_host_rejected(client: TestClient) -> None:
    """A non-loopback Host (DNS-rebinding target) is rejected → 403 (unchanged)."""
    r = client.get("/api/studio/_host_probe", headers={"host": "evil.example.com"})
    assert r.status_code == 403


def test_valid_loopback_host_allowed(client: TestClient) -> None:
    """The allowed (TestClient default) Host still passes the guard → 200."""
    r = client.get("/api/studio/_host_probe")  # default Host: testserver (allowed)
    assert r.status_code == 200
    assert r.json() == {"ok": True}
