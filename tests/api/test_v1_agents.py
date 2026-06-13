"""T2e — /v1 stored agent definitions: CRUD + run-by-agent_id.

These tests pin the acceptance for UNIT 4:

* ``POST /v1/agents`` stores a workspace-scoped :class:`AgentSpec`; a cross-tenant GET
  is 404 in an authenticated deployment;
* ``POST /v1/runs`` with ``agent_id`` runs the stored spec WITH ITS TOOLS (proving the
  per-run tool-bearing runtime resolves a stored spec) and records an agent lineage
  edge;
* a tenant spec carrying ``tools_module`` / ``http_tools`` / ``mcp_servers`` is REJECTED
  on the write (the RCE/SSRF surface stays closed, T0.3);
* the inline-persona run path is byte-unchanged;
* create is idempotent; deleting a referenced agent is refused (409).

The HTTP tests run offline (stub provider, in-RAM store) with NO authenticator, so RBAC
+ workspace scoping are inert and the operator path applies (the standalone posture).
The authenticated-deployment isolation is asserted directly against the service +
storage (where the tenant boundary is real), per the plan's two-regimes guard.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.application.services import (
    AgentDefAppService,
    AgentDefReferencedError,
)
from himmy.config.agent_spec import AgentSpec
from himmy.config.spec_sanitizer import SpecSanitizationError
from himmy.core.events import EventType
from himmy.entities.registry import EntityRegistry
from himmy.services.storage.service import StorageService
from tests.application import _per_run_tools
from tests.conftest import run_async

_TOOLS_MODULE = "tests.application._per_run_tools"


@pytest.fixture
def client() -> Iterator[TestClient]:
    """An offline /v1 client over a fresh in-RAM container (no auth → operator).

    Entered as a context manager so the app lifespan is active: an agent-by-id run
    resolves its per-run runtime via ``asyncio.to_thread`` and must complete across
    requests on the lifespan-established shared loop.
    """
    with TestClient(create_app(ApiContainer.build_default())) as test_client:
        yield test_client


def _spec_body(name: str = "analyst", **extra: object) -> dict:
    """A minimal AgentSpec POST body (a clean tenant spec, no privileged fields)."""
    spec = {"name": name, "description": "A stored analyst.", **extra}
    return {"workspace_id": "acme", "spec": spec}


def _poll_terminal(client: TestClient, run_id: str, workspace_id: str = "acme") -> dict:
    """Poll a /v1 run to a terminal status (fire-and-forget background execution)."""
    deadline = time.monotonic() + 5.0
    got: dict = {}
    while time.monotonic() < deadline:
        got = client.get(
            f"/v1/runs/{run_id}", params={"workspace_id": workspace_id}
        ).json()
        if got["status"] in ("SUCCEEDED", "FAILED", "AWAITING_APPROVAL"):
            return got
        time.sleep(0.02)
    return got


# --------------------------------------------------------------------- CRUD


def test_post_agent_stores_a_workspace_scoped_spec(client: TestClient) -> None:
    """POST /v1/agents stores the spec and returns a durable agent_id (201)."""
    res = client.post("/v1/agents", json=_spec_body())
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["agent_id"]
    assert body["workspace_id"] == "acme"
    assert body["name"] == "analyst"
    assert body["spec"]["name"] == "analyst"

    # It is then readable + listable in the same workspace.
    agent_id = body["agent_id"]
    got = client.get(f"/v1/agents/{agent_id}", params={"workspace_id": "acme"})
    assert got.status_code == 200
    assert got.json()["agent_id"] == agent_id

    listing = client.get("/v1/agents", params={"workspace_id": "acme"})
    assert listing.status_code == 200
    ids = [a["agent_id"] for a in listing.json()["items"]]
    assert agent_id in ids


def test_put_agent_replaces_the_spec(client: TestClient) -> None:
    """PUT /v1/agents/{id} replaces the spec, preserving agent_id + created_at."""
    created = client.post("/v1/agents", json=_spec_body()).json()
    agent_id = created["agent_id"]

    updated = client.put(
        f"/v1/agents/{agent_id}",
        json={
            "workspace_id": "acme",
            "spec": {"name": "analyst-v2", "description": "Now sharper."},
        },
    )
    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["agent_id"] == agent_id
    assert body["name"] == "analyst-v2"
    assert body["created_at"] == created["created_at"]


def test_put_unknown_agent_is_404(client: TestClient) -> None:
    """PUT for an agent that does not exist is a 404."""
    res = client.put(
        "/v1/agents/does-not-exist",
        json={"workspace_id": "acme", "spec": {"name": "x"}},
    )
    assert res.status_code == 404


def test_delete_agent_removes_it(client: TestClient) -> None:
    """DELETE removes the agent; a subsequent GET is 404; a re-DELETE is 404."""
    agent_id = client.post("/v1/agents", json=_spec_body()).json()["agent_id"]
    res = client.delete(f"/v1/agents/{agent_id}", params={"workspace_id": "acme"})
    assert res.status_code == 200
    assert res.json()["status"] == "deleted"
    assert (
        client.get(f"/v1/agents/{agent_id}", params={"workspace_id": "acme"}).status_code
        == 404
    )
    assert (
        client.delete(
            f"/v1/agents/{agent_id}", params={"workspace_id": "acme"}
        ).status_code
        == 404
    )


def test_get_unknown_agent_is_404(client: TestClient) -> None:
    """GET for an unknown agent id is a 404."""
    res = client.get("/v1/agents/nope", params={"workspace_id": "acme"})
    assert res.status_code == 404


# ------------------------------------------------------------- idempotency


def test_create_is_idempotent(client: TestClient) -> None:
    """A re-POST with the same idempotency_key returns the prior record (201→200)."""
    body = {**_spec_body(), "idempotency_key": "once"}
    first = client.post("/v1/agents", json=body)
    assert first.status_code == 201
    second = client.post("/v1/agents", json=body)
    assert second.status_code == 200  # existing, not re-created
    assert second.json()["agent_id"] == first.json()["agent_id"]

    # Exactly one stored agent for the workspace.
    listing = client.get("/v1/agents", params={"workspace_id": "acme"}).json()
    assert len([a for a in listing["items"] if a["idempotency_key"] == "once"]) == 1


# --------------------------------------------------------- run by agent_id


def test_run_by_agent_id_executes_the_stored_agent_with_tools(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /v1/runs with agent_id runs the stored spec WITH ITS TOOLS (T2e headline).

    The agent is OPERATOR-provisioned with ``tools_module`` (offline → the caller is the
    operator and the opt-in flag is set), so the only way its tool can fire is the
    per-run tool-bearing runtime resolved from the STORED spec. The recording tool's
    ``CALLS`` + a TOOL_CALLED event prove the stored agent's tools executed.
    """
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _per_run_tools.CALLS.clear()

    created = client.post(
        "/v1/agents",
        json={
            "workspace_id": "acme",
            "spec": {
                "name": "probe",
                "tools_module": _TOOLS_MODULE,
                "tools": ["ping"],
            },
        },
    )
    assert created.status_code == 201, created.text
    agent_id = created.json()["agent_id"]
    # The operator-provisioned privileged field survived the write sanitizer.
    assert created.json()["spec"]["tools_module"] == _TOOLS_MODULE

    run = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "agent_id": agent_id,
            "task": {"title": "probe", "prompt": "please ping"},
        },
    )
    assert run.status_code == 200, run.text
    run_id = run.json()["run_id"]
    terminal = _poll_terminal(client, run_id)
    assert terminal["status"] == "SUCCEEDED", terminal
    assert terminal["metadata"].get("agent_id") == agent_id
    assert terminal["metadata"].get("agent_name") == "probe"

    # The stored agent's tool ACTUALLY fired on the per-run runtime.
    assert _per_run_tools.CALLS, "the stored agent's tool never fired"

    # And a TOOL_CALLED event landed in the canonical store the BFF reads.
    events = client.get(
        f"/v1/runs/{run_id}/events", params={"workspace_id": "acme"}
    ).json()
    kinds = {e.get("event_type") for e in events}
    assert EventType.TOOL_CALLED.value in {str(k) for k in kinds}


def test_run_by_agent_id_records_agent_lineage_edge(client: TestClient) -> None:
    """A run launched by agent_id links its thread -> the agent node in the spine (T2e)."""
    agent_id = client.post(
        "/v1/agents",
        json={"workspace_id": "acme", "spec": {"name": "calc", "tool_packs": ["utils"]}},
    ).json()["agent_id"]

    run = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "agent_id": agent_id,
            "task": {"title": "calc", "prompt": "what is 2+2"},
        },
    )
    run_id = run.json()["run_id"]
    terminal = _poll_terminal(client, run_id)
    assert terminal["status"] == "SUCCEEDED", terminal

    # The shared spine carries a run_of_agent edge to the registered agent node.
    registry = client.app.state.container.entity_registry
    from himmy.entities.records import stable_id_for

    agent_sid = stable_id_for(f"acme:{agent_id}", namespace="agent")

    async def _has_edge() -> bool:
        from himmy.application.services import _maybe_await

        agent_node = await _maybe_await(registry.get_latest(agent_sid))
        if agent_node is None:
            return False
        incoming = await _maybe_await(registry.links_to(agent_node.record_id))
        return any(link.relation == "run_of_agent" for link in incoming)

    assert run_async(_has_edge()), "missing run_of_agent lineage edge"


def test_run_with_unknown_agent_id_is_404(client: TestClient) -> None:
    """POST /v1/runs with an agent_id that does not exist is a 404."""
    res = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "agent_id": "nope",
            "task": {"title": "t", "prompt": "hi"},
        },
    )
    assert res.status_code == 404


def test_run_requires_exactly_one_of_persona_or_agent_id(client: TestClient) -> None:
    """Supplying both (or neither) persona and agent_id is a 422 (mutual exclusion)."""
    both = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "persona": {"name": "p"},
            "agent_id": "x",
            "task": {"title": "t", "prompt": "hi"},
        },
    )
    assert both.status_code == 422
    neither = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "task": {"title": "t", "prompt": "hi"},
        },
    )
    assert neither.status_code == 422


# ------------------------------------------------------------- back-compat


def test_inline_persona_run_path_byte_unchanged(client: TestClient) -> None:
    """The legacy inline-persona POST /v1/runs path still works unchanged (back-compat)."""
    res = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "persona": {"name": "analyst", "instructions": ["Be brief."]},
            "task": {"title": "greet", "prompt": "say hi"},
        },
    )
    assert res.status_code == 200, res.text
    run_id = res.json()["run_id"]
    terminal = _poll_terminal(client, run_id)
    assert terminal["status"] == "SUCCEEDED", terminal
    # No stored-agent metadata on an inline-persona run.
    assert "agent_id" not in terminal["metadata"]


# ----------------------------------------------------------------- security


def _tenant_client(key: str = "key-acme", workspace: str = "acme") -> TestClient:
    """A client bound to a real tenant API key (a NON-operator multi-tenant caller).

    With an authenticator configured the principal is tenant-bound (``all_tenants`` is
    False), so the spec sanitizer's operator path does NOT apply and a privileged-field
    spec is rejected — exactly the multi-tenant posture the RCE/SSRF guard defends.
    """
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            key: Principal.build(
                "tenant-user",
                tenant_ids=[workspace],
                roles=["operator"],
                auth_method="apikey",
            )
        }
    )
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": key})
    return client


@pytest.mark.parametrize(
    ("field", "spec_extra"),
    [
        ("tools_module", {"tools_module": _TOOLS_MODULE}),
        (
            "http_tools",
            {
                "http_tools": [
                    {"name": "x", "base_url": "http://10.0.0.1", "path": "/"}
                ]
            },
        ),
        ("mcp_servers", {"mcp_servers": [{"command": "echo"}]}),
    ],
)
def test_tenant_spec_with_privileged_field_is_rejected_on_write(
    field: str, spec_extra: dict
) -> None:
    """A TENANT POST /v1/agents carrying a privileged field is rejected 422 (T0.3).

    Done over an AUTHENTICATED deployment (a tenant-bound, non-operator API key), so the
    sanitizer actually bites — the RCE/SSRF surface must stay closed for a real
    multi-tenant caller. The recording-tool module is never imported.
    """
    client = _tenant_client()
    res = client.post(
        "/v1/agents",
        json={"workspace_id": "acme", "spec": {"name": "evil", **spec_extra}},
    )
    assert res.status_code == 422, res.text
    assert field in res.json()["detail"]
    # Nothing was stored for the rejected write.
    listing = client.get("/v1/agents", params={"workspace_id": "acme"}).json()
    assert listing["items"] == []


def test_cross_tenant_agent_get_is_404_authenticated() -> None:
    """A stored agent is 404 to a different tenant over HTTP (the authenticated regime)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-a": Principal.build(
                "user-a", tenant_ids=["tenant-a"], roles=["admin"], auth_method="apikey"
            ),
            "key-b": Principal.build(
                "user-b", tenant_ids=["tenant-b"], roles=["admin"], auth_method="apikey"
            ),
        }
    )
    client_a = TestClient(app)
    client_a.headers.update({"x-himmy-internal-key": "key-a"})
    client_b = TestClient(app)
    client_b.headers.update({"x-himmy-internal-key": "key-b"})

    agent_id = client_a.post(
        "/v1/agents",
        json={"workspace_id": "tenant-a", "spec": {"name": "secret"}},
    ).json()["agent_id"]

    # Owner reads it; the other tenant gets 403 (asking for tenant-a) / 404 (own tenant).
    assert (
        client_a.get(
            f"/v1/agents/{agent_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 200
    )
    assert (
        client_b.get(
            f"/v1/agents/{agent_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 403
    )
    assert (
        client_b.get(
            f"/v1/agents/{agent_id}", params={"workspace_id": "tenant-b"}
        ).status_code
        == 404
    )
    # B's list never surfaces A's agent.
    listed_b = client_b.get("/v1/agents", params={"workspace_id": "tenant-b"}).json()
    assert all(a["agent_id"] != agent_id for a in listed_b["items"])


# ------------------------------------------ authenticated-deployment isolation


@pytest.mark.asyncio
async def test_cross_workspace_get_is_none_authenticated() -> None:
    """A stored agent is invisible to another workspace (the 404 path, authenticated).

    Asserted at the service+storage layer where the tenant boundary is REAL (the HTTP
    layer's scoping is inert offline). ``acme``'s agent must read as None for ``globex``.
    """
    storage = StorageService()
    svc = AgentDefAppService(storage=storage, entity_registry=EntityRegistry())

    stored, created = await svc.save_agent_def(
        AgentSpec(name="secret"), workspace_id="acme"
    )
    assert created

    # Same workspace: found. Other workspace: None (→ the router returns 404).
    assert await svc.get_agent_def(stored.agent_id, workspace_id="acme") is not None
    assert await svc.get_agent_def(stored.agent_id, workspace_id="globex") is None
    # And it is not in globex's list.
    assert await svc.list_agent_defs(workspace_id="globex") == []


@pytest.mark.asyncio
async def test_cross_workspace_delete_is_noop() -> None:
    """A delete from the wrong workspace removes nothing (tenant-scoped)."""
    storage = StorageService()
    svc = AgentDefAppService(storage=storage)
    stored, _ = await svc.save_agent_def(AgentSpec(name="a"), workspace_id="acme")

    assert await svc.delete_agent_def(stored.agent_id, workspace_id="globex") is False
    # Still present for the owning workspace.
    assert await svc.get_agent_def(stored.agent_id, workspace_id="acme") is not None


# ------------------------------------------------------ referential delete


@pytest.mark.asyncio
async def test_delete_referenced_agent_is_refused() -> None:
    """Deleting an agent referenced by a team/routine is refused (409, must_fix).

    A reference_finder reporting a referencing team makes delete raise
    :class:`AgentDefReferencedError`; ``cascade=True`` overrides it.
    """
    storage = StorageService()

    async def _finder(agent_id: str, _ws: str) -> list[str]:
        return [f"team:weekly-digest references {agent_id}"]

    svc = AgentDefAppService(storage=storage, reference_finder=_finder)
    stored, _ = await svc.save_agent_def(AgentSpec(name="a"), workspace_id="acme")

    with pytest.raises(AgentDefReferencedError):
        await svc.delete_agent_def(stored.agent_id, workspace_id="acme")
    # Still there.
    assert await svc.get_agent_def(stored.agent_id, workspace_id="acme") is not None

    # Cascade override removes it.
    assert (
        await svc.delete_agent_def(
            stored.agent_id, workspace_id="acme", cascade=True
        )
        is True
    )


@pytest.mark.asyncio
async def test_service_save_rejects_tenant_privileged_spec() -> None:
    """The service write choke point rejects a tenant spec with tools_module (T0.3)."""
    storage = StorageService()
    svc = AgentDefAppService(storage=storage)
    with pytest.raises(SpecSanitizationError):
        await svc.save_agent_def(
            AgentSpec(name="evil", tools_module=_TOOLS_MODULE), workspace_id="acme"
        )
    # Nothing persisted.
    assert await svc.list_agent_defs(workspace_id="acme") == []
