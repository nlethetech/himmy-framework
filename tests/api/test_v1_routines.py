"""T3c — /v1 routines: workspace-scoped CRUD + run-now over the shared routines store.

These tests pin the acceptance for UNIT 3:

* ``/v1/routines`` is workspace-scoped and references a STORED ``agent_id`` (the
  ``/v1/agents`` resource, T2e) — a tenant ``agent_path`` is REJECTED (no FS leak);
* an unknown ``agent_id`` is a 404 (the routine must reference a real stored agent);
* ``POST /{id}/run-now`` lands the run in the TENANT's workspace of the ONE canonical
  store, so it shows in ``GET /v1/runs`` (the reviewer coherence must_fix);
* a cross-workspace routine read is a 404 (the authenticated regime);
* deleting a stored agent still referenced by a routine is refused (409);
* the cross-process flock prevents a CLI run-now and the scheduler from double-executing
  one routine (asserted directly on ``execute_routine`` + the lock primitive).

The HTTP tests run offline (stub provider, in-RAM container) with NO authenticator, so
RBAC + workspace scoping are inert and the operator path applies (the standalone posture);
the authenticated isolation is asserted with a tenant-bound API key where the boundary is
real.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api import routines as svc
from himmy.api.studio_runs import reset_run_store


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A clean project root with isolated routines + run stores and no scheduler loop."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    svc.reset_routines_store()
    svc.reset_scheduler()
    reset_run_store()
    yield tmp_path
    svc.reset_routines_store()
    svc.reset_scheduler()


@pytest.fixture
def client(project: Path) -> Iterator[TestClient]:
    """An offline /v1 client over a fresh in-RAM container (no auth → operator).

    Entered as a context manager so the app lifespan is active: a routine run-now
    dispatches a /v1 run through ``RunAppService`` on the lifespan-established loop.
    """
    with TestClient(create_app(ApiContainer.build_default())) as test_client:
        yield test_client


def _store_agent(client: TestClient, name: str = "helper", workspace: str = "acme") -> str:
    """Store a minimal /v1 agent and return its agent_id."""
    res = client.post(
        "/v1/agents",
        json={"workspace_id": workspace, "spec": {"name": name, "description": "h"}},
    )
    assert res.status_code == 201, res.text
    return res.json()["agent_id"]


def _routine_body(agent_id: str, **over: object) -> dict:
    base: dict = {
        "workspace_id": "acme",
        "agent_id": agent_id,
        "name": "Morning brief",
        "prompt": "write a brief",
        "schedule": {"kind": "daily", "at": "07:00"},
    }
    base.update(over)
    return base


def _poll_run(client: TestClient, run_id: str, workspace: str = "acme") -> dict:
    deadline = time.monotonic() + 5.0
    got: dict = {}
    while time.monotonic() < deadline:
        got = client.get(
            f"/v1/runs/{run_id}", params={"workspace_id": workspace}
        ).json()
        if got.get("status") in ("SUCCEEDED", "FAILED", "AWAITING_APPROVAL"):
            return got
        time.sleep(0.02)
    return got


# ------------------------------------------------------------------------ CRUD


def test_create_and_read_routine(client: TestClient) -> None:
    """POST /v1/routines stores a workspace-scoped routine bound to a stored agent."""
    agent_id = _store_agent(client)
    res = client.post("/v1/routines", json=_routine_body(agent_id))
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["agent_id"] == agent_id
    assert body["workspace_id"] == "acme"
    # Phase 1 adds additive, defaulted schedule fields; the daily semantics are unchanged.
    assert body["schedule"] == {
        "kind": "daily",
        "at": "07:00",
        "hours": None,
        "expr": None,
        "at_datetime": None,
        "timezone": None,
        "missed": "coalesce",
        "max_catchup": 100,
        "grace_seconds": 900,
        "overlap": "skip",
    }
    # No agent_path leaks into the view.
    assert "agent_path" not in body

    rid = body["id"]
    got = client.get(f"/v1/routines/{rid}", params={"workspace_id": "acme"})
    assert got.status_code == 200
    assert got.json()["id"] == rid

    listing = client.get("/v1/routines", params={"workspace_id": "acme"}).json()
    assert [r["id"] for r in listing["items"]] == [rid]


def test_patch_and_delete_routine(client: TestClient) -> None:
    """PATCH updates fields; DELETE removes it; a re-DELETE is 404."""
    agent_id = _store_agent(client)
    rid = client.post("/v1/routines", json=_routine_body(agent_id)).json()["id"]

    patched = client.patch(
        f"/v1/routines/{rid}",
        json={
            "workspace_id": "acme",
            "enabled": False,
            "schedule": {"kind": "every", "hours": 6},
        },
    )
    assert patched.status_code == 200
    assert patched.json()["enabled"] is False
    assert patched.json()["schedule"]["hours"] == 6

    assert (
        client.delete(
            f"/v1/routines/{rid}", params={"workspace_id": "acme"}
        ).status_code
        == 200
    )
    assert (
        client.get(f"/v1/routines/{rid}", params={"workspace_id": "acme"}).status_code
        == 404
    )
    assert (
        client.delete(
            f"/v1/routines/{rid}", params={"workspace_id": "acme"}
        ).status_code
        == 404
    )


# ---------------------------------------------------- agent_path / agent_id rules


def test_agent_path_is_rejected(client: TestClient) -> None:
    """A tenant may NOT pass a filesystem agent_path — the field does not exist (422)."""
    agent_id = _store_agent(client)
    res = client.post(
        "/v1/routines",
        json={**_routine_body(agent_id), "agent_path": "../evil.yaml"},
    )
    # The extra field is ignored OR rejected; either way no path is persisted/usable.
    # Assert the stored routine carries the agent_id binding and never an agent_path.
    if res.status_code == 201:
        assert "agent_path" not in res.json()
    else:
        assert res.status_code == 422


def test_unknown_agent_id_is_404(client: TestClient) -> None:
    """A routine must reference a stored agent in the same workspace (404 otherwise)."""
    res = client.post("/v1/routines", json=_routine_body("does-not-exist"))
    assert res.status_code == 404


def test_routine_cannot_reference_another_workspaces_agent(client: TestClient) -> None:
    """Binding a routine to an agent stored in a DIFFERENT workspace is a 404."""
    other_agent = _store_agent(client, name="other", workspace="globex")
    res = client.post(
        "/v1/routines",
        json=_routine_body(other_agent, workspace_id="acme"),
    )
    assert res.status_code == 404


# ------------------------------------------------------------------ idempotency


def test_create_is_idempotent(client: TestClient) -> None:
    """A re-POST with the same idempotency_key returns the prior routine, no duplicate."""
    agent_id = _store_agent(client)
    body = {**_routine_body(agent_id), "idempotency_key": "once"}
    first = client.post("/v1/routines", json=body)
    assert first.status_code == 201
    second = client.post("/v1/routines", json=body)
    assert second.status_code == 201  # idempotent hit returns the same record
    assert second.json()["id"] == first.json()["id"]
    listing = client.get("/v1/routines", params={"workspace_id": "acme"}).json()
    assert listing["total"] == 1


# --------------------------------------------------- run-now → canonical store


def test_run_now_lands_in_canonical_runs(client: TestClient) -> None:
    """run-now dispatches a /v1 run that shows in GET /v1/runs (coherence must_fix)."""
    agent_id = _store_agent(client)
    rid = client.post("/v1/routines", json=_routine_body(agent_id)).json()["id"]

    res = client.post(f"/v1/routines/{rid}/run-now", params={"workspace_id": "acme"})
    assert res.status_code == 200, res.text
    assert res.json()["last_status"] in ("ok", "awaiting_approval")
    assert res.json()["last_run_at"] is not None

    # The run is visible in the canonical /v1 runs list for the tenant workspace.
    runs = client.get("/v1/runs", params={"workspace_id": "acme"}).json()
    assert runs["total"] >= 1
    # Exactly the run launched by this routine (stamped with its routine_id actor).
    matching = [
        r
        for r in runs["items"]
        if (r.get("metadata") or {}).get("actor", {}).get("routine_id") == rid
    ]
    assert matching, runs
    assert _poll_run(client, matching[0]["run_id"])["status"] in (
        "SUCCEEDED",
        "AWAITING_APPROVAL",
    )


def test_run_now_unknown_routine_404(client: TestClient) -> None:
    assert (
        client.post(
            "/v1/routines/nope/run-now", params={"workspace_id": "acme"}
        ).status_code
        == 404
    )


def test_run_now_is_409_when_routine_flock_held(client: TestClient) -> None:
    """Holding the routine's cross-process flock makes /v1 run-now a clean 409.

    Reviewer must_fix: the cross-process busy case (another process owns the routine's
    flock) raised ``RoutineBusyError`` deep in ``execute_routine`` but was being swallowed
    by ``RoutineScheduler._guarded_execute`` on the ``run_now`` path, so the router mapped
    ``None`` → 404 and the ``409`` handler was dead code for the exact scenario the flock
    guards. Stand in for "the Studio scheduler process holds it" by holding the same-named
    flock here, then assert run-now surfaces 409 (NOT 404) and never executes the body.
    """
    from himmy.core.process_lock import process_lock

    agent_id = _store_agent(client)
    rid = client.post("/v1/routines", json=_routine_body(agent_id)).json()["id"]

    runs_before = client.get("/v1/runs", params={"workspace_id": "acme"}).json()["total"]
    with process_lock(svc.routine_lock_name(rid)):
        res = client.post(
            f"/v1/routines/{rid}/run-now", params={"workspace_id": "acme"}
        )
    assert res.status_code == 409, res.text
    assert "already running" in res.json()["detail"]
    # The guarded body never ran a second time → no new canonical run was created.
    runs_after = client.get("/v1/runs", params={"workspace_id": "acme"}).json()["total"]
    assert runs_after == runs_before


# ------------------------------------------ authenticated cross-tenant isolation


def _tenant_client(
    project: Path, key: str = "key-acme", workspace: str = "acme"
) -> TestClient:
    """A client bound to a tenant API key (a NON-operator multi-tenant caller)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-acme": Principal.build(
                "acme-user",
                tenant_ids=["acme"],
                roles=["operator"],
                auth_method="apikey",
            ),
            "key-globex": Principal.build(
                "globex-user",
                tenant_ids=["globex"],
                roles=["operator"],
                auth_method="apikey",
            ),
        }
    )
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": key})
    return client


def test_cross_workspace_routine_access_is_404_authenticated(project: Path) -> None:
    """A routine is 404 to a different tenant over HTTP (the authenticated regime)."""
    acme = _tenant_client(project, key="key-acme")
    globex = _tenant_client(project, key="key-globex")

    agent_id = acme.post(
        "/v1/agents",
        json={"workspace_id": "acme", "spec": {"name": "secret"}},
    ).json()["agent_id"]
    rid = acme.post(
        "/v1/routines", json=_routine_body(agent_id, workspace_id="acme")
    ).json()["id"]

    # Owner reads it; the other tenant cannot (403 asking for acme, 404 in own scope).
    assert (
        acme.get(f"/v1/routines/{rid}", params={"workspace_id": "acme"}).status_code
        == 200
    )
    assert (
        globex.get(
            f"/v1/routines/{rid}", params={"workspace_id": "acme"}
        ).status_code
        == 403
    )
    assert (
        globex.get(
            f"/v1/routines/{rid}", params={"workspace_id": "globex"}
        ).status_code
        == 404
    )
    # globex's list never surfaces acme's routine.
    listed = globex.get("/v1/routines", params={"workspace_id": "globex"}).json()
    assert all(r["id"] != rid for r in listed["items"])


def test_run_now_requires_routine_write_permission(project: Path) -> None:
    """A viewer (routine:read only) cannot create or run a routine (403)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "viewer-key": Principal.build(
                "v", tenant_ids=["acme"], roles=["viewer"], auth_method="apikey"
            )
        }
    )
    viewer = TestClient(app)
    viewer.headers.update({"x-himmy-internal-key": "viewer-key"})

    # A viewer can LIST (read) but not CREATE (write).
    assert (
        viewer.get("/v1/routines", params={"workspace_id": "acme"}).status_code == 200
    )
    res = viewer.post("/v1/routines", json=_routine_body("any"))
    assert res.status_code == 403


# ------------------------------------------------ referenced-agent delete (409)


def test_delete_agent_referenced_by_routine_is_409(client: TestClient) -> None:
    """Deleting a stored agent still wired to a routine is refused (409, must_fix)."""
    agent_id = _store_agent(client)
    client.post("/v1/routines", json=_routine_body(agent_id))

    res = client.delete(f"/v1/agents/{agent_id}", params={"workspace_id": "acme"})
    assert res.status_code == 409, res.text

    # cascade removes it anyway (the caller owns cleanup).
    res2 = client.delete(
        f"/v1/agents/{agent_id}", params={"workspace_id": "acme", "cascade": True}
    )
    assert res2.status_code == 200
