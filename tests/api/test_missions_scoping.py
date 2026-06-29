"""Tenant/subject (IDOR/BOLA) scoping for the Studio Missions surface.

The process-local mission registry historically carried NO tenant/subject stamp, so once an
authenticator binds a caller to a tenant, that caller could list / read / steer / interrupt
ANOTHER tenant's background mission by id — the exact recurrence class the data-scoping drain
closes. These tests pin the by-construction property: a mission is stamped at start time with
the OWNING tenant + subject derived from the verified principal, and a cross-tenant /
cross-subject read is a uniform 404 (existence never leaked) while the list omits it.

The offline / ``all_tenants`` path is asserted byte-unchanged: with no authenticator the
mission is unowned (``workspace_id``/``subject_id`` are ``None``) and every reader sees it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer
from himmy.api.app import create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy
from himmy.api.missions import Mission, MissionRegistry


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _tenant_app(
    *,
    tenant_ids: list[str],
    roles: list[str],
    grants: dict[str, list[str]],
    subject: str = "u",
    subject_scoped: bool = False,
) -> tuple[TestClient, MissionRegistry]:
    """A built app authenticated as one tenant/subject-bound principal, plus its registry."""
    container = ApiContainer.build_default()
    app = create_app(container)
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                subject,
                tenant_ids=tenant_ids,
                roles=roles,
                auth_method="apikey",
                subject_scoped=subject_scoped,
            )
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping(grants)
    registry = MissionRegistry()
    app.state.mission_registry = registry
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    return client, registry


def _seed(
    registry: MissionRegistry,
    *,
    mid: str,
    workspace_id: str | None,
    subject_id: str | None = None,
    status: str = "done",
) -> None:
    """Inject a finished mission stamped to a given owner (no model run needed)."""
    registry._missions[mid] = Mission(
        id=mid,
        agent="helper",
        agent_path="agent.yaml",
        prompt="hi",
        provider="stub",
        model=None,
        plan_mode=False,
        created_at="2026-06-29T00:00:00+00:00",
        workspace_id=workspace_id,
        subject_id=subject_id,
        status=status,
    )


# --------------------------------------------------------------- tenant axis (IDOR)


def test_cross_tenant_mission_is_invisible(project: Path) -> None:
    """A tenant-bound caller cannot list/read/steer/interrupt another tenant's mission."""
    # The caller holds BOTH read and write on missions, so a cross-tenant refusal is the
    # SCOPE fold (404), not merely the write-permission guard (403) tripping first.
    client, registry = _tenant_app(
        tenant_ids=["t"],
        roles=["operator"],
        grants={
            "operator": [
                "studio.console:read",
                "studio.missions:read",
                "studio.missions:write",
            ]
        },
    )
    _seed(registry, mid="mine", workspace_id="t", status="running")
    _seed(registry, mid="theirs", workspace_id="other", status="running")

    # List: only the same-tenant mission shows.
    items = client.get("/api/studio/missions").json()["items"]
    ids = {m["id"] for m in items}
    assert ids == {"mine"}

    # By-id: own is 200, foreign is a uniform 404 (existence never leaked).
    assert client.get("/api/studio/missions/mine").status_code == 200
    assert client.get("/api/studio/missions/theirs").status_code == 404
    # Stream of a foreign mission is refused before the SSE starts.
    assert client.get("/api/studio/missions/theirs/stream").status_code == 404
    # Command paths fold the same way (a uniform 404 — NOT the 409 a real running mission
    # in scope would give, which would confirm existence; NOT a 403 since we hold write).
    assert (
        client.post(
            "/api/studio/missions/theirs/steer", json={"text": "hi"}
        ).status_code
        == 404
    )
    assert client.post("/api/studio/missions/theirs/interrupt").status_code == 404


def test_tenant_admin_sees_only_own_tenants(project: Path) -> None:
    """A principal bound to two tenants sees both, but never a third tenant's mission."""
    client, registry = _tenant_app(
        tenant_ids=["a", "b"],
        roles=["reader"],
        grants={"reader": ["studio.console:read", "studio.missions:read"]},
    )
    _seed(registry, mid="ma", workspace_id="a")
    _seed(registry, mid="mb", workspace_id="b")
    _seed(registry, mid="mc", workspace_id="c")

    ids = {m["id"] for m in client.get("/api/studio/missions").json()["items"]}
    assert ids == {"ma", "mb"}
    assert client.get("/api/studio/missions/mc").status_code == 404


# -------------------------------------------------------------- subject axis (BOLA)


def test_subject_scoped_principal_cannot_read_other_subject(project: Path) -> None:
    """A ``subject_scoped`` caller (no tenant_admin) sees only its OWN subject's mission."""
    client, registry = _tenant_app(
        tenant_ids=["t"],
        roles=["reader"],
        grants={"reader": ["studio.console:read", "studio.missions:read"]},
        subject="alice",
        subject_scoped=True,
    )
    _seed(registry, mid="mine", workspace_id="t", subject_id="alice")
    _seed(registry, mid="bob", workspace_id="t", subject_id="bob")

    ids = {m["id"] for m in client.get("/api/studio/missions").json()["items"]}
    assert ids == {"mine"}
    assert client.get("/api/studio/missions/mine").status_code == 200
    assert client.get("/api/studio/missions/bob").status_code == 404


def test_tenant_admin_crosses_subjects_within_tenant(project: Path) -> None:
    """A ``tenant_admin`` (subject_scoped) reads every subject in its own tenant."""
    client, registry = _tenant_app(
        tenant_ids=["t"],
        roles=["reader", "tenant_admin"],
        grants={"reader": ["studio.console:read", "studio.missions:read"]},
        subject="admin",
        subject_scoped=True,
    )
    _seed(registry, mid="a1", workspace_id="t", subject_id="alice")
    _seed(registry, mid="b1", workspace_id="t", subject_id="bob")
    # ...but still NOT another tenant's mission.
    _seed(registry, mid="x1", workspace_id="other", subject_id="alice")

    ids = {m["id"] for m in client.get("/api/studio/missions").json()["items"]}
    assert ids == {"a1", "b1"}
    assert client.get("/api/studio/missions/x1").status_code == 404


# ------------------------------------------------------------------ start stamping


def test_started_mission_is_stamped_with_caller_scope(project: Path) -> None:
    """POST stamps the mission with the principal's tenant + subject (not client input)."""
    client, registry = _tenant_app(
        tenant_ids=["t"],
        roles=["admin"],
        grants={"admin": ["*:*"]},
        subject="alice",
        subject_scoped=True,
    )
    r = client.post(
        "/api/studio/missions",
        json={"agent_path": "agent.yaml", "prompt": "2+2?", "provider": "stub"},
    )
    assert r.status_code == 200, r.text
    mission = registry.get(r.json()["mission_id"])
    assert mission.workspace_id == "t"
    assert mission.subject_id == "alice"


# --------------------------------------------------------------------- offline no-op


def test_offline_missions_are_unowned_and_visible(project: Path) -> None:
    """With no authenticator a mission is unowned and every reader sees it (byte-unchanged)."""
    app = create_app(ApiContainer.build_default())
    registry = MissionRegistry()
    app.state.mission_registry = registry
    # A mission seeded with a workspace stamp is STILL visible offline: the ANONYMOUS /
    # all_tenants principal's scoping is a strict no-op (authorize_studio_object/object
    # both return True), so the zero-config console shows everything exactly as before.
    _seed(registry, mid="m1", workspace_id="anything", subject_id="anyone")
    with TestClient(app) as client:
        items = client.get("/api/studio/missions").json()["items"]
        assert {m["id"] for m in items} == {"m1"}
        assert client.get("/api/studio/missions/m1").status_code == 200

    # And a freshly started mission is unowned (None/None) on the offline path.
    offline_app = create_app()
    with TestClient(offline_app) as client:
        r = client.post(
            "/api/studio/missions",
            json={"agent_path": "agent.yaml", "prompt": "hi", "provider": "stub"},
        )
        assert r.status_code == 200, r.text
        m = offline_app.state.mission_registry.get(r.json()["mission_id"])
        assert m.workspace_id is None and m.subject_id is None
