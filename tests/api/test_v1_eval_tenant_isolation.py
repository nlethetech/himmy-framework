"""WS1.0 — the /v1/evaluation tenant-isolation (IDOR) guarantee.

A tenant-bound principal (a mapped API key) can run + read its OWN evaluation
runs, but can never reach another tenant's run by id: passing another tenant's
``workspace_id`` is denied (403), and a different tenant's key cannot read A's
run under any workspace (403 for a workspace it isn't entitled to, 404 under its
own). The cross-tenant eval IDOR is closed at the ``resolve_workspace`` choke
point (himmy/api/routers/evaluation.py) backed by a workspace-filtered storage
lookup (himmy/services/storage/*.py:get_evaluation_run).

Mirrors tests/api/test_tenant_isolation.py (the /v1/runs IDOR test).
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal


def _mapped_app() -> Any:
    """An app whose keys bind callers to specific tenants (closes the IDOR).

    A and B authenticate against the SAME app/storage so cross-tenant isolation
    is enforced by the workspace filter, not by accidentally-separate stores.
    """
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
    return app


def _app_with_mapped_keys() -> TestClient:
    return TestClient(_mapped_app())


def _client_for(app: Any, key: str) -> TestClient:
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": key})
    return c


def _client_as(key: str) -> TestClient:
    """A standalone client (its own app/storage) authenticated as ``key``."""
    return _client_for(_mapped_app(), key)


def _suite_body(workspace: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "suite": {
            "suite_id": "suite-1",
            "name": "Smoke",
            "cases": [
                {
                    "case_id": "c1",
                    "input": {"prompt": "hi"},
                    "expected_output": {"text": "hi"},
                }
            ],
        },
        "actual_outputs": {"c1": {"text": "hi"}},
    }
    if workspace is not None:
        body["workspace_id"] = workspace
    return body


def _run_suite(client: TestClient, workspace: str) -> str:
    """Run an eval suite as the caller, returning the persisted run id."""
    resp = client.post("/v1/evaluation/runs", json=_suite_body(workspace))
    assert resp.status_code == 200, resp.text
    run_id: str = resp.json()["run_id"]
    return run_id


def test_unauthenticated_eval_request_is_rejected() -> None:
    client = _app_with_mapped_keys()  # no key header
    assert (
        client.get(
            "/v1/evaluation/runs", params={"workspace_id": "tenant-a"}
        ).status_code
        == 401
    )


def test_caller_can_run_and_read_its_own_tenant_eval() -> None:
    """No false-positive lockout: A runs + reads its own eval run."""
    client = _client_as("key-a")
    run_id = _run_suite(client, "tenant-a")
    # Explicit own workspace, and the implied (single-tenant) default, both work.
    assert (
        client.get(
            f"/v1/evaluation/runs/{run_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 200
    )
    assert client.get(f"/v1/evaluation/runs/{run_id}").status_code == 200
    # The run is listed under A's tenant.
    listed = client.get("/v1/evaluation/runs").json()
    assert any(it["run_id"] == run_id for it in listed)
    assert all(it["workspace_id"] == "tenant-a" for it in listed)


def test_cross_tenant_eval_read_is_denied_not_leaked() -> None:
    """The eval IDOR: B cannot read A's run, and A can't reach tenant-b."""
    app = _mapped_app()
    client_a = _client_for(app, "key-a")
    run_id = _run_suite(client_a, "tenant-a")

    # Same caller A, but asking for tenant-b → 403 (authorization), not a leak.
    assert (
        client_a.get(
            f"/v1/evaluation/runs/{run_id}", params={"workspace_id": "tenant-b"}
        ).status_code
        == 403
    )
    # A genuinely different tenant's key (SAME storage) cannot reach A's run.
    client_b = _client_for(app, "key-b")
    assert (
        client_b.get(
            f"/v1/evaluation/runs/{run_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 403
    )
    # And B reading under its OWN tenant simply doesn't find A's run (404).
    assert (
        client_b.get(
            f"/v1/evaluation/runs/{run_id}", params={"workspace_id": "tenant-b"}
        ).status_code
        == 404
    )
    # B's default (single-tenant) read also doesn't surface A's run.
    assert client_b.get(f"/v1/evaluation/runs/{run_id}").status_code == 404


def test_cross_tenant_eval_run_is_denied() -> None:
    """Caller A cannot run/persist an eval inside tenant-b."""
    client_a = _client_as("key-a")
    assert (
        client_a.post(
            "/v1/evaluation/runs", json=_suite_body("tenant-b")
        ).status_code
        == 403
    )


def test_eval_list_is_pinned_to_the_callers_tenant() -> None:
    """A's list never includes B's runs even when B has run its own suite."""
    app = _mapped_app()
    client_a = _client_for(app, "key-a")
    client_b = _client_for(app, "key-b")
    a_run = _run_suite(client_a, "tenant-a")
    b_run = _run_suite(client_b, "tenant-b")

    a_listed = client_a.get("/v1/evaluation/runs").json()
    a_ids = {it["run_id"] for it in a_listed}
    assert a_run in a_ids
    assert b_run not in a_ids
    assert all(it["workspace_id"] == "tenant-a" for it in a_listed)
