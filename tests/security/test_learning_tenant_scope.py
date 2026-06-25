"""Security regression: the Studio Learning panel is tenant-scoped (Finding #15).

Before the fix the Learning router hardcoded ``_PANEL_WORKSPACE=None``, so on a
multi-tenant server every ``studio:use`` principal mined tool reputation across the
ENTIRE event store and could pin/mute/reset another tenant's tools. The fix resolves
the effective ``workspace_id`` from the verified principal
(:func:`himmy.api.auth.context.resolve_workspace`) and threads it into both the GET
read path (``build_learning_report`` + override ``load``) and the POST write paths
(``set_override`` / ``set_trust_feedback``).

These tests stand up two tenant-bound principals over one shared in-memory event store
+ spine and assert that tenant A cannot READ tenant B's learning data, and a tenant-A
WRITE never lands on tenant B's overrides. They FAIL against the old (hardcoded-None)
behaviour and PASS with the per-principal scoping.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.principal import Principal
from himmy.config.project import HIMMY_SPINE_PATH_ENV
from himmy.core.events import EventType, RunEvent
from tests.conftest import run_async

_TENANT_HEADER = "x-test-tenant"


class _TenantAuthenticator:
    """A tenant-binding authenticator that reads the tenant from a test header.

    Mirrors a real mapped-key / OIDC authenticator: each request resolves to a
    principal bound to exactly ONE tenant (``all_tenants=False``), which is what
    drives :func:`resolve_workspace` to pin the request to that tenant's workspace.
    Holds the ``admin`` role purely so the ``studio:use`` guard passes — the scoping
    under test is tenant isolation, not RBAC.
    """

    #: Marks this authenticator as tenant-binding for the multi-tenant posture check.
    binds_tenants = True

    async def authenticate(self, request: object) -> Principal:
        tenant = request.headers.get(_TENANT_HEADER, "tenant-a")  # type: ignore[attr-defined]
        return Principal.build(
            subject=f"user-{tenant}",
            tenant_ids=[tenant],
            roles=["admin"],
            all_tenants=False,
            auth_method="test",
        )


def _seed(store: object, tool: str, *, ok: bool, n: int, workspace_id: str) -> None:
    """Append n TOOL_COMPLETED/TOOL_FAILED events for one tool in one tenant's workspace."""
    for _ in range(n):
        run_async(
            store.append_event(  # type: ignore[attr-defined]
                RunEvent(
                    event_type=(
                        EventType.TOOL_COMPLETED if ok else EventType.TOOL_FAILED
                    ),
                    payload={"tool_name": tool},
                    workspace_id=workspace_id,
                )
            )
        )


@pytest.fixture()
def isolated_spine(tmp_path, monkeypatch):
    """Point the override store's durable spine at a per-test temp ``spine.db``."""
    monkeypatch.setenv(HIMMY_SPINE_PATH_ENV, str(tmp_path / "spine.db"))
    yield


@pytest.fixture()
def tenant_client(isolated_spine):
    """A TestClient whose app authenticates each request to a per-header tenant."""
    container = ApiContainer.build_default()
    app = create_app(container)
    # Swap in the tenant-binding authenticator so RBAC + tenant scoping are live
    # (create_app builds with no auth on the zero-config test path).
    app.state.authenticator = _TenantAuthenticator()
    client = TestClient(app)
    return container, client


def _get(client: TestClient, tenant: str):
    return client.get("/api/studio/learning", headers={_TENANT_HEADER: tenant}).json()


def _tool_names(body: dict) -> set[str]:
    return {t["tool_name"] for t in body["tools"]}


def test_read_is_scoped_to_the_calling_tenant(tenant_client) -> None:
    """Tenant A's learning report must EXCLUDE tenant B's tools (no cross-tenant read)."""
    container, client = tenant_client
    store = container.storage
    _seed(store, "alpha_tool", ok=True, n=4, workspace_id="tenant-a")
    _seed(store, "beta_tool", ok=True, n=4, workspace_id="tenant-b")

    a_body = _get(client, "tenant-a")
    b_body = _get(client, "tenant-b")

    # Each tenant sees only its own tool — the bug let A see beta_tool too.
    assert _tool_names(a_body) == {"alpha_tool"}
    assert _tool_names(b_body) == {"beta_tool"}
    assert "beta_tool" not in _tool_names(a_body)
    assert "alpha_tool" not in _tool_names(b_body)


def test_write_does_not_leak_across_tenants(tenant_client) -> None:
    """A tenant-A override must NOT appear in tenant B's report (no cross-tenant write)."""
    container, client = tenant_client
    store = container.storage
    # Same tool name observed (flaky) in BOTH tenants so the override would be visible
    # in either report if it landed on the wrong subject.
    _seed(store, "shared_tool", ok=False, n=9, workspace_id="tenant-a")
    _seed(store, "shared_tool", ok=True, n=1, workspace_id="tenant-a")
    _seed(store, "shared_tool", ok=False, n=9, workspace_id="tenant-b")
    _seed(store, "shared_tool", ok=True, n=1, workspace_id="tenant-b")

    # Tenant A pins shared_tool -> not flaky, labelled "pinned" for A only.
    resp = client.post(
        "/api/studio/learning/overrides",
        headers={_TENANT_HEADER: "tenant-a", "origin": "http://testserver"},
        json={"tool_name": "shared_tool", "action": "pin"},
    )
    assert resp.status_code == 200
    a_row = next(t for t in resp.json()["tools"] if t["tool_name"] == "shared_tool")
    assert a_row["override"] == "pinned"
    assert a_row["flaky"] is False

    # Tenant B must be unaffected: no override, still flaky.
    b_body = _get(client, "tenant-b")
    b_row = next(t for t in b_body["tools"] if t["tool_name"] == "shared_tool")
    assert b_row["override"] is None, "tenant-A override leaked into tenant-B"
    assert b_row["flaky"] is True
    assert b_body["flaky_count"] == 1


def test_trust_feedback_toggle_is_tenant_scoped(tenant_client) -> None:
    """Flipping the trust gate for tenant A must not flip it for tenant B."""
    _container, client = tenant_client

    on = client.post(
        "/api/studio/learning/trust-feedback",
        headers={_TENANT_HEADER: "tenant-a", "origin": "http://testserver"},
        json={"enabled": True},
    )
    assert on.status_code == 200
    assert on.json()["trust_feedback"] is True

    # Tenant B's gate is untouched.
    assert _get(client, "tenant-b")["trust_feedback"] is False
    # Tenant A's gate persisted (shared spine, scoped subject).
    assert _get(client, "tenant-a")["trust_feedback"] is True
