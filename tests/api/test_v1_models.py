"""T3d — the /v1 model catalog + compare + diagnostics surfaces.

Covers the three REST surfaces over the shared :mod:`himmy.services.inference.compare`
seam: ``GET /v1/models`` (global catalog), ``POST /v1/models/compare`` (workspace-scoped,
T0.4-quota-admitted fan-out), and ``GET /v1/diagnostics`` (global infra health). RBAC and
workspace scoping are exercised with mapped API keys; the offline default is unchanged.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal


def _offline_client() -> TestClient:
    return TestClient(create_app(ApiContainer.build_default()))


def _client_with_role(role: str, *, tenant: str = "t") -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=[tenant], roles=[role], auth_method="apikey"
            )
        }
    )
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    return client


# ----------------------------------------------------------------- GET /v1/models


def test_models_catalog_shape_offline() -> None:
    body = _offline_client().get("/v1/models").json()
    providers = {p["provider"] for p in body}
    assert {"ollama", "claude-cli"} <= providers
    for p in body:
        assert "available" in p and isinstance(p["models"], list)


def test_models_catalog_carries_no_secret_material(
    monkeypatch,
) -> None:
    # Even with a provider key set in the environment, the catalog never echoes it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    r = _offline_client().get("/v1/models")
    assert r.status_code == 200
    assert "sk-ant-supersecret" not in r.text


def test_models_catalog_read_gated() -> None:
    # viewer holds model:read; a role-less principal is denied.
    assert _client_with_role("viewer").get("/v1/models").status_code == 200
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={"k": Principal.build("u", tenant_ids=["t"], roles=[])}
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    assert c.get("/v1/models").status_code == 403


# ----------------------------------------------------------- POST /v1/models/compare


def _compare_body(workspace: str = "t") -> dict:
    return {
        "workspace_id": workspace,
        "prompt": "say hello",
        "targets": [
            {"provider": "stub", "model": "a"},
            {"provider": "stub", "model": "b"},
        ],
    }


def test_compare_offline_runs_each_target() -> None:
    r = _offline_client().post("/v1/models/compare", json=_compare_body())
    assert r.status_code == 200
    body = r.json()
    assert body["workspace_id"] == "t"
    results = body["results"]
    assert len(results) == 2
    assert all(cell["ok"] for cell in results)
    assert all(cell["output"] for cell in results)


def test_compare_reports_bad_provider_per_cell() -> None:
    body = _compare_body()
    body["targets"][1] = {"provider": "no-such-provider", "model": "x"}
    r = _offline_client().post("/v1/models/compare", json=body)
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False and results[1]["error"]


def test_compare_requires_a_target() -> None:
    body = _compare_body()
    body["targets"] = []
    assert (
        _offline_client().post("/v1/models/compare", json=body).status_code == 422
    )


def test_compare_write_gated() -> None:
    # viewer has model:read but NOT model:write → 403; operator can compare.
    assert (
        _client_with_role("viewer")
        .post("/v1/models/compare", json=_compare_body())
        .status_code
        == 403
    )
    assert (
        _client_with_role("operator")
        .post("/v1/models/compare", json=_compare_body())
        .status_code
        == 200
    )


def test_compare_workspace_scoping_is_enforced() -> None:
    # A tenant-bound operator cannot compare against a workspace it does not own.
    c = _client_with_role("operator", tenant="tenant-a")
    r = c.post("/v1/models/compare", json=_compare_body(workspace="tenant-b"))
    assert r.status_code == 403


def test_compare_over_quota_is_a_429() -> None:
    # T0.4: a workspace already at its outstanding-run cap is rejected with a 429,
    # not allowed to pin further provider fan-out.
    app = create_app(ApiContainer.build_default())
    run_app = app.state.container.run_app
    run_app._workspace_max_outstanding = 1
    # Occupy the only slot for this workspace, so the compare admission fails.
    run_app._admit_workspace_run("t")
    try:
        c = TestClient(app)
        r = c.post("/v1/models/compare", json=_compare_body())
        assert r.status_code == 429
        assert r.json()["code"] == "run_quota_exceeded"
    finally:
        run_app._release_workspace_run("t")


def test_compare_releases_its_slot_after_running() -> None:
    # A completed compare must not leak its workspace slot (so the next request is fine).
    app = create_app(ApiContainer.build_default())
    run_app = app.state.container.run_app
    run_app._workspace_max_outstanding = 1
    c = TestClient(app)
    assert c.post("/v1/models/compare", json=_compare_body()).status_code == 200
    assert run_app.workspace_outstanding("t") == 0
    # And a second one still works (the slot was released).
    assert c.post("/v1/models/compare", json=_compare_body()).status_code == 200


# ----------------------------------------------------------- GET /v1/diagnostics


def test_diagnostics_global_health_shape() -> None:
    body = _offline_client().get("/v1/diagnostics").json()
    assert "doctor" in body and "storage" in body
    assert body["storage"]["backend"] in {"sqlite", "postgresql"}
    names = {f["name"] for f in body["db_files"]}
    assert {"spine", "conversations", "v1_approvals", "run_store"} <= names
    assert isinstance(body["scheduler_active"], bool)
    # The doctor sub-report never carries key VALUES, only presence flags.
    for key in body["doctor"]["keys"]:
        assert set(key) == {"name", "present"}


def test_diagnostics_redacts_a_postgres_password(monkeypatch) -> None:
    monkeypatch.setenv(
        "HIMMY_DATABASE_URL", "postgresql://user:supersecret@db.internal:5432/himmy"
    )
    r = _offline_client().get("/v1/diagnostics")
    assert r.status_code == 200
    assert "supersecret" not in r.text
    assert r.json()["storage"]["backend"] == "postgresql"


def test_diagnostics_read_gated() -> None:
    assert _client_with_role("auditor").get("/v1/diagnostics").status_code == 200
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={"k": Principal.build("u", tenant_ids=["t"], roles=[])}
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    assert c.get("/v1/diagnostics").status_code == 403
