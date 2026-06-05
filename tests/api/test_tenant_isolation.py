"""WS1.0 — the tenant-isolation (IDOR) fix: workspace comes from the principal.

With a tenant-bound principal (a mapped API key), the caller can only touch its own
workspace; passing another tenant's ``workspace_id`` is denied (403), not honored.
The offline/no-auth default is unchanged (an ANONYMOUS, all-tenants principal).
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal


def _app_with_mapped_keys() -> TestClient:
    """An app whose keys bind callers to specific tenants (closes the IDOR)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-a": Principal.build(
                "user-a",
                tenant_ids=["tenant-a"],
                roles=["operator"],
                auth_method="apikey",
            ),
            "key-b": Principal.build(
                "user-b",
                tenant_ids=["tenant-b"],
                roles=["operator"],
                auth_method="apikey",
            ),
        }
    )
    return TestClient(app)


def _client_as(key: str) -> TestClient:
    c = _app_with_mapped_keys()
    c.headers.update({"x-himmy-internal-key": key})
    return c


def _create_run(client: TestClient, workspace: str) -> str:
    body = {
        "workspace_id": workspace,
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    resp = client.post("/v1/runs", json=body)
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/runs/{run_id}").json().get("status") in (
            "SUCCEEDED",
            "FAILED",
        ):
            break
        time.sleep(0.02)
    return run_id


def test_unauthenticated_request_is_rejected() -> None:
    client = _app_with_mapped_keys()  # no key header
    assert (
        client.get("/v1/runs", params={"workspace_id": "tenant-a"}).status_code == 401
    )


def test_caller_can_read_its_own_tenant() -> None:
    client = _client_as("key-a")
    run_id = _create_run(client, "tenant-a")
    # Explicit own workspace, and the implied (single-tenant) default, both work.
    assert (
        client.get(
            f"/v1/runs/{run_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 200
    )
    assert client.get(f"/v1/runs/{run_id}").status_code == 200


def test_cross_tenant_read_is_denied_not_leaked() -> None:
    """The IDOR: caller A passing tenant-b is 403 (denied), never served."""
    client_a = _client_as("key-a")
    run_id = _create_run(client_a, "tenant-a")
    # Same caller A, but asking for tenant-b → 403 (authorization), not a leak.
    assert (
        client_a.get(
            f"/v1/runs/{run_id}", params={"workspace_id": "tenant-b"}
        ).status_code
        == 403
    )
    # A genuinely different tenant's key cannot reach A's run either.
    client_b = _client_as("key-b")
    assert (
        client_b.get(
            f"/v1/runs/{run_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 403
    )
    # And B reading under its own tenant simply doesn't find A's run (404).
    assert (
        client_b.get(
            f"/v1/runs/{run_id}", params={"workspace_id": "tenant-b"}
        ).status_code
        == 404
    )


def test_cross_tenant_create_is_denied() -> None:
    """Caller A cannot create a run inside tenant-b."""
    client_a = _client_as("key-a")
    body = {
        "workspace_id": "tenant-b",
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    assert client_a.post("/v1/runs", json=body).status_code == 403


def test_list_is_pinned_to_the_callers_tenant() -> None:
    client_a = _client_as("key-a")
    _create_run(client_a, "tenant-a")
    # Omitting workspace_id resolves to the caller's single tenant (not "all").
    listed = client_a.get("/v1/runs").json()
    assert all(it["workspace_id"] == "tenant-a" for it in listed["items"])


def test_offline_default_preserves_legacy_behavior() -> None:
    """No authenticator → ANONYMOUS all-tenants → caller-supplied workspace honored."""
    client = TestClient(create_app(ApiContainer.build_default()))
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    run_id = client.post("/v1/runs", json=body).json()["run_id"]
    assert (
        client.get(f"/v1/runs/{run_id}", params={"workspace_id": "w1"}).status_code
        == 200
    )
    assert (
        client.get(f"/v1/runs/{run_id}", params={"workspace_id": "w2"}).status_code
        == 404
    )
