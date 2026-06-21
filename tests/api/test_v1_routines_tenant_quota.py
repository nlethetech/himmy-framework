"""T3 — per-tenant routine QUOTAS + cross-tenant ISOLATION on ``/v1/routines``.

These pin the multi-tenancy acceptance for the scheduling layer:

* a tenant at its ENABLED-routine cap (``HIMMY_TENANT_MAX_ROUTINES``) is refused with a
  clear 429 AT CREATE — one tenant cannot flood the shared scheduler with routines;
* a routine asking for more catch-up backfill than the tenant ceiling
  (``HIMMY_TENANT_MAX_CATCHUP``) is clamped DOWN — one tenant's launch cannot storm the
  shared queue;
* the quota is PER-TENANT: tenant A hitting its cap does not affect tenant B;
* routine CRUD/list is workspace-scoped — tenant A cannot see/read/delete B's routine;
* with no cap configured (and the reserved ``__local__`` workspace) behaviour is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api import routines as svc
from himmy.api.studio_runs import reset_run_store


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
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
    with TestClient(create_app(ApiContainer.build_default())) as test_client:
        yield test_client


def _store_agent(client: TestClient, name: str, workspace: str) -> str:
    res = client.post(
        "/v1/agents",
        json={"workspace_id": workspace, "spec": {"name": name, "description": "h"}},
    )
    assert res.status_code == 201, res.text
    return res.json()["agent_id"]


def _routine_body(agent_id: str, workspace: str, name: str, **over: object) -> dict:
    base: dict = {
        "workspace_id": workspace,
        "agent_id": agent_id,
        "name": name,
        "prompt": "do the thing",
        "schedule": {"kind": "daily", "at": "07:00"},
    }
    base.update(over)
    return base


# --------------------------------------------------------- max-routines quota


def test_tenant_max_routines_enforced_at_create(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant at its enabled-routine cap is refused at create with a 429."""
    monkeypatch.setenv("HIMMY_TENANT_MAX_ROUTINES", "2")
    agent = _store_agent(client, "a", "acme")
    r1 = client.post("/v1/routines", json=_routine_body(agent, "acme", "one"))
    r2 = client.post("/v1/routines", json=_routine_body(agent, "acme", "two"))
    assert r1.status_code == 201 and r2.status_code == 201
    r3 = client.post("/v1/routines", json=_routine_body(agent, "acme", "three"))
    assert r3.status_code == 429, r3.text
    assert "cap is 2" in r3.json()["detail"]


def test_disabled_routines_do_not_count_toward_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap is on ENABLED routines — a disabled routine frees a slot."""
    monkeypatch.setenv("HIMMY_TENANT_MAX_ROUTINES", "1")
    agent = _store_agent(client, "a", "acme")
    r1 = client.post(
        "/v1/routines", json=_routine_body(agent, "acme", "one", enabled=False)
    )
    assert r1.status_code == 201
    # The disabled one does not count, so an enabled create still fits under cap=1.
    r2 = client.post("/v1/routines", json=_routine_body(agent, "acme", "two"))
    assert r2.status_code == 201


def test_quota_is_per_tenant(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tenant A hitting its cap does NOT block tenant B (no shared-counter leak)."""
    monkeypatch.setenv("HIMMY_TENANT_MAX_ROUTINES", "1")
    a_agent = _store_agent(client, "a", "acme")
    b_agent = _store_agent(client, "b", "globex")
    assert (
        client.post("/v1/routines", json=_routine_body(a_agent, "acme", "a1")).status_code
        == 201
    )
    # acme is now at cap; a second acme create is 429.
    assert (
        client.post("/v1/routines", json=_routine_body(a_agent, "acme", "a2")).status_code
        == 429
    )
    # globex is independent and still admits its first routine.
    assert (
        client.post(
            "/v1/routines", json=_routine_body(b_agent, "globex", "b1")
        ).status_code
        == 201
    )


def test_no_cap_by_default_does_not_clip_normal_load(client: TestClient) -> None:
    """With the generous default cap, a handful of routines all create cleanly."""
    agent = _store_agent(client, "a", "acme")
    for i in range(5):
        res = client.post("/v1/routines", json=_routine_body(agent, "acme", f"r{i}"))
        assert res.status_code == 201


# --------------------------------------------------------- max-catchup clamp


def test_tenant_catchup_ceiling_clamps_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A routine asking for more catch-up than the tenant ceiling is clamped down."""
    monkeypatch.setenv("HIMMY_TENANT_MAX_CATCHUP", "10")
    agent = _store_agent(client, "a", "acme")
    body = _routine_body(
        agent,
        "acme",
        "greedy",
        schedule={
            "kind": "every",
            "hours": 1,
            "missed": "run_each",
            "max_catchup": 5000,
        },
    )
    res = client.post("/v1/routines", json=body)
    assert res.status_code == 201, res.text
    stored = svc.get_routines_store().get(res.json()["id"], workspace_id="acme")
    assert stored is not None
    assert stored.schedule.max_catchup == 10  # clamped to the tenant ceiling


def test_catchup_within_ceiling_is_untouched(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIMMY_TENANT_MAX_CATCHUP", "100")
    agent = _store_agent(client, "a", "acme")
    body = _routine_body(
        agent,
        "acme",
        "modest",
        schedule={
            "kind": "every",
            "hours": 1,
            "missed": "run_each",
            "max_catchup": 20,
        },
    )
    res = client.post("/v1/routines", json=body)
    assert res.status_code == 201
    stored = svc.get_routines_store().get(res.json()["id"], workspace_id="acme")
    assert stored is not None and stored.schedule.max_catchup == 20


# --------------------------------------------------------- cross-tenant isolation


def test_routine_list_is_workspace_scoped(client: TestClient) -> None:
    """Tenant A's list never contains tenant B's routine."""
    a_agent = _store_agent(client, "a", "acme")
    b_agent = _store_agent(client, "b", "globex")
    client.post("/v1/routines", json=_routine_body(a_agent, "acme", "a1"))
    client.post("/v1/routines", json=_routine_body(b_agent, "globex", "b1"))
    a_list = client.get("/v1/routines", params={"workspace_id": "acme"}).json()
    b_list = client.get("/v1/routines", params={"workspace_id": "globex"}).json()
    assert {r["name"] for r in a_list["items"]} == {"a1"}
    assert {r["name"] for r in b_list["items"]} == {"b1"}


def test_routine_get_is_404_across_tenant(client: TestClient) -> None:
    """Tenant B cannot read tenant A's routine by id (404, not a leak)."""
    a_agent = _store_agent(client, "a", "acme")
    rid = client.post(
        "/v1/routines", json=_routine_body(a_agent, "acme", "secret")
    ).json()["id"]
    # Reading it as globex is a 404.
    res = client.get(f"/v1/routines/{rid}", params={"workspace_id": "globex"})
    assert res.status_code == 404
    # And acme can read it.
    assert (
        client.get(f"/v1/routines/{rid}", params={"workspace_id": "acme"}).status_code
        == 200
    )


def test_routine_delete_is_404_across_tenant(client: TestClient) -> None:
    """Tenant B cannot delete tenant A's routine (404; A's routine survives)."""
    a_agent = _store_agent(client, "a", "acme")
    rid = client.post(
        "/v1/routines", json=_routine_body(a_agent, "acme", "keep")
    ).json()["id"]
    res = client.request(
        "DELETE", f"/v1/routines/{rid}", params={"workspace_id": "globex"}
    )
    assert res.status_code == 404
    # A's routine is intact.
    assert (
        client.get(f"/v1/routines/{rid}", params={"workspace_id": "acme"}).status_code
        == 200
    )
