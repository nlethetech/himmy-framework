"""Regression: the cross-site guard fails CLOSED on opaque/absent Origin (CSRF).

Finding #3 (csrf-origin-null): the guard middleware used to skip its same-origin
check when ``Origin``/``Referer`` was the opaque value ``null``/``file://`` (which
parse to an empty host) or absent entirely on a state-changing route. A sandboxed
iframe on a malicious page sends ``Origin: null``, so this let a cross-site page
drive guarded mutating routes (e.g. approving a paused dangerous action).

These tests FAIL against the old fail-open behavior and PASS with the fix that
rejects a present-but-opaque Origin and a mutating request with no Origin/Referer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    app = create_app()

    # A guarded ( ``/api/studio`` prefix ) state-changing route with no side
    # effects, so the test isolates the middleware's CSRF decision. Routes added
    # after ``create_app`` still pass through the installed guard middleware.
    @app.post("/api/studio/_csrf_probe")
    async def _probe() -> dict[str, bool]:  # pragma: no cover - trivial
        return {"ok": True}

    # ``host`` defaults to the allowed "testserver" Host; we never override it so
    # the Host check passes and only the Origin/Referer logic is exercised.
    return TestClient(app)


def test_opaque_null_origin_rejected(client: TestClient) -> None:
    """Origin: null (sandboxed iframe) must be treated as cross-site → 403."""
    r = client.post("/api/studio/_csrf_probe", headers={"origin": "null"})
    assert r.status_code == 403


def test_file_scheme_origin_rejected(client: TestClient) -> None:
    """Origin: file:// has no host and must be rejected, not skipped → 403."""
    r = client.post("/api/studio/_csrf_probe", headers={"origin": "file://"})
    assert r.status_code == 403


def test_opaque_null_referer_rejected(client: TestClient) -> None:
    """A present-but-opaque Referer is also treated as cross-site → 403."""
    r = client.post("/api/studio/_csrf_probe", headers={"referer": "null"})
    assert r.status_code == 403


def test_mutating_request_without_origin_or_referer_rejected(
    client: TestClient,
) -> None:
    """A guarded mutating request with neither Origin nor Referer fails closed → 403."""
    r = client.post("/api/studio/_csrf_probe")
    assert r.status_code == 403


def test_same_origin_post_allowed(client: TestClient) -> None:
    """A correct same-origin Origin on the mutating route is still allowed → 200."""
    r = client.post(
        "/api/studio/_csrf_probe",
        headers={"origin": "http://testserver"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_idempotent_get_without_origin_still_allowed(client: TestClient) -> None:
    """Non-mutating (GET) guarded requests without Origin are unaffected → not 403."""
    # The probe is POST-only, so a GET 405s, but crucially NOT 403 from the guard:
    # the fail-closed-on-missing-Origin rule only applies to mutating methods.
    r = client.get("/api/studio/_csrf_probe")
    assert r.status_code != 403
