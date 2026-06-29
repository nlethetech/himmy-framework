"""Studio hardening: RBAC gating of /api/studio + symlink-safe spec resolution.

Covers the two halves of the Studio trust boundary once the app leaves loopback:

* With an authenticator configured, every Studio route requires its per-surface
  permission (``studio.console:read``, ``studio.connections:write``, …; admin's
  ``*:*`` qualifies for all); without one, the zero-config behavior is unchanged.
* :func:`resolve_spec_path` rejects any symlinked component below the project
  root, in addition to the existing ``..``/containment check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy
from himmy.api.studio_service import resolve_spec_path


# ------------------------------------------------------ studio RBAC gating
def _app_with_key(*roles: str) -> FastAPI:
    """An app whose mapped API key ``"k"`` is bound to ``roles``."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=list(roles), auth_method="apikey"
            )
        }
    )
    return app


def _client_with_key(app: FastAPI) -> TestClient:
    """A client for ``app`` with the mapped key pre-attached."""
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


def test_no_auth_studio_unchanged() -> None:
    """Zero-config (no authenticator): Studio answers without any credentials."""
    c = TestClient(create_app(ApiContainer.build_default()))
    assert c.get("/api/studio/health").status_code == 200


def test_auth_configured_unauthenticated_is_401() -> None:
    """With auth on, a request without a key never reaches Studio."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(shared_keys={"s"})
    assert TestClient(app).get("/api/studio/health").status_code == 401


def test_auth_configured_role_without_studio_grant_is_403() -> None:
    """An authenticated principal with NO Studio grant is denied everywhere.

    Studio is now gated per-surface (``studio.console:read``, ``studio.runs:read``,
    …) rather than by one coarse ``studio:use``; a custom role holding only a ``/v1``
    grant carries none of those, so every Studio route 403s.
    """
    app = _app_with_key("v1_only")
    app.state.access_policy = AccessPolicy.from_mapping({"v1_only": ["run:read"]})
    c = _client_with_key(app)
    assert c.get("/api/studio/health").status_code == 403
    assert c.get("/api/studio/connections").status_code == 403
    assert c.get("/api/studio/agents").status_code == 403


def test_auth_configured_admin_passes() -> None:
    """``admin`` (``*:*``) covers ``studio:use``."""
    c = _client_with_key(_app_with_key("admin"))
    assert c.get("/api/studio/health").status_code == 200


def test_shared_internal_key_passes() -> None:
    """The shared trusted-boundary key maps to admin, so Studio keeps working."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(shared_keys={"s"})
    c = TestClient(app)
    r = c.get("/api/studio/health", headers={"x-himmy-internal-key": "s"})
    assert r.status_code == 200


def test_custom_policy_can_grant_studio_read() -> None:
    """An operator-defined role granting ``studio.console:read`` unlocks the console.

    The granular replacement for the old coarse ``studio:use``: the read baseline the
    main router carries is ``studio.console:read``, so a role holding it can browse
    Studio's read surfaces (but not the ``/v1`` resource, nor any mutating route).
    """
    app = _app_with_key("studio_user")
    app.state.access_policy = AccessPolicy.from_mapping(
        {"studio_user": ["studio.console:read", "studio.connections:read"]}
    )
    c = _client_with_key(app)
    assert c.get("/api/studio/health").status_code == 200
    assert c.get("/api/studio/connections").status_code == 200
    # …but not the /v1 surface (no other grants).
    assert c.get("/v1/runs", params={"workspace_id": "t"}).status_code == 403
    # …and not a mutating Studio route (no studio.connections:write).
    assert (
        c.put("/api/studio/connections/email", json={"fields": {}}).status_code == 403
    )


def _app_with_policy(role: str, perms: list[str]) -> FastAPI:
    """An app whose mapped key is bound to ``role`` granting exactly ``perms``."""
    app = _app_with_key(role)
    app.state.access_policy = AccessPolicy.from_mapping({role: perms})
    return app


def test_per_surface_read_isolates_models_from_gmail() -> None:
    """A role granted ONE surface's read cannot read a DIFFERENT surface (r6 split).

    The Studio main-router GET routes no longer collapse to one coarse
    ``studio.console:read``: each carries its OWN ``studio.<surface>:read``. A role
    holding the console baseline + ``studio.models:read`` (but NOT the Google grant the
    Gmail/Calendar reads require) is 200 on the model catalog yet 403 on the Gmail and
    Calendar reads — proving the surfaces are independently gated, not a single bucket.
    """
    app = _app_with_policy(
        "models_only", ["studio.console:read", "studio.models:read"]
    )
    c = _client_with_key(app)
    # Granted surface → 200.
    assert c.get("/api/studio/models").status_code == 200
    # Un-granted sensitive surfaces (folded under studio.google) → 403.
    assert c.get("/api/studio/google/gmail").status_code == 403
    assert c.get("/api/studio/google/calendar").status_code == 403
    assert c.get("/api/studio/google").status_code == 403
    # …and the inverse: a connections-read role cannot read models.
    app2 = _app_with_policy(
        "conn_only", ["studio.console:read", "studio.connections:read"]
    )
    c2 = _client_with_key(app2)
    assert c2.get("/api/studio/connections").status_code == 200
    assert c2.get("/api/studio/models").status_code == 403


def test_per_surface_runs_read_split() -> None:
    """The run readers require ``studio.runs:read``, not just the console baseline.

    Before r6 the ``/runs`` GET family inherited only ``studio.console:read``; now each
    declares ``studio.runs:read``. A role with the console baseline but no runs grant is
    403 on run history/analytics/detail, while one granted the runs read is 200.
    """
    app = _app_with_policy("console_only", ["studio.console:read"])
    c = _client_with_key(app)
    assert c.get("/api/studio/runs").status_code == 403
    assert c.get("/api/studio/runs/analytics").status_code == 403
    assert c.get("/api/studio/runs/does-not-exist").status_code == 403

    app2 = _app_with_policy(
        "runs_reader", ["studio.console:read", "studio.runs:read"]
    )
    c2 = _client_with_key(app2)
    assert c2.get("/api/studio/runs").status_code == 200
    assert c2.get("/api/studio/runs/analytics").status_code == 200
    # A missing run is 404 (past the 403 guard), never 403, for a granted reader.
    assert c2.get("/api/studio/runs/does-not-exist").status_code == 404


def test_admin_reads_every_studio_surface() -> None:
    """``admin`` (``*:*``) satisfies every per-surface read guard."""
    c = _client_with_key(_app_with_key("admin"))
    for path in (
        "/api/studio/models",
        "/api/studio/runs",
        "/api/studio/runs/analytics",
        "/api/studio/connections",
        "/api/studio/google",
        "/api/studio/approvals",
    ):
        assert c.get(path).status_code == 200, path


def test_default_browse_roles_grant_benign_reads_not_sensitive() -> None:
    """viewer/operator/auditor get the benign surface reads; google/connections stay out.

    The DEFAULT_RBAC browse roles are spliced with ``studio.<surface>:read`` for every
    surface EXCEPT the un-partitionable operator/global-store surfaces
    (``_STUDIO_GLOBAL_STORE_RESOURCES``: connections, google, memory, …). So a default
    ``viewer`` reads models/runs/approvals but is 403 on the operator's connection and
    Google surfaces — least-privilege, no per-deployment policy file needed.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        # Benign tenant-facing reads granted by default.
        assert c.get("/api/studio/models").status_code == 200, role
        assert c.get("/api/studio/runs").status_code == 200, role
        assert c.get("/api/studio/approvals").status_code == 200, role
        # Sensitive operator surfaces withheld by default (admin-only).
        assert c.get("/api/studio/connections").status_code == 403, role
        assert c.get("/api/studio/google").status_code == 403, role
        assert c.get("/api/studio/google/gmail").status_code == 403, role


def test_tenant_bound_runs_reader_cannot_cross_tenants() -> None:
    """A tenant-bound principal with ``studio.runs:read`` only sees its own workspace.

    The per-surface read guard is the surface gate, NOT the tenant boundary: the run
    readers stay tenant-filtered (``studio_tenant_filter``), so a principal bound to
    tenant ``t`` cannot fetch a run stamped to another workspace by id — it 404s rather
    than leaking existence, even though the role DOES hold ``studio.runs:read``.
    """
    from himmy.api import ApiContainer, create_app
    from himmy.api.auth.apikey import ApiKeyAuthenticator

    container = ApiContainer.build_default()
    app = create_app(container)
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=["runs_reader"], auth_method="apikey"
            )
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping(
        {"runs_reader": ["studio.console:read", "studio.runs:read"]}
    )
    c = _client_with_key(app)
    # Seed a run owned by a DIFFERENT workspace into the canonical store.
    import anyio

    from himmy.services.storage.models import RunRecord, RunStatus

    other = RunRecord(
        run_id="other-tenant-run",
        workspace_id="other",
        subject_id="someone-else",
        status=RunStatus.SUCCEEDED,
    )
    anyio.run(container.storage.save_run, other)
    # The granted reader is 404 (not 200, not 403) on the cross-tenant run by id.
    assert c.get("/api/studio/runs/other-tenant-run").status_code == 404
    # And the list does not include the other tenant's run.
    body = c.get("/api/studio/runs").json()
    assert all(item["run_id"] != "other-tenant-run" for item in body["items"])


def test_escape_hatch_disables_studio_rbac(monkeypatch: pytest.MonkeyPatch) -> None:
    """HIMMY_STUDIO_AUTH=off (dangerous) skips the permission check, not authn."""
    monkeypatch.setenv("HIMMY_STUDIO_AUTH", "off")
    app = _app_with_key("viewer")
    assert _client_with_key(app).get("/api/studio/health").status_code == 200
    # Authentication itself still applies: no key is still a 401.
    assert TestClient(app).get("/api/studio/health").status_code == 401


# ------------------------------------------------- symlink-safe spec paths
def test_resolve_spec_path_plain_file(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text("name: a\n")
    assert resolve_spec_path("agent.yaml", tmp_path).name == "agent.yaml"


def test_resolve_spec_path_traversal_rejected(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (tmp_path / "outside.yaml").write_text("name: evil\n")
    with pytest.raises(ValueError, match="escapes project root"):
        resolve_spec_path("../outside.yaml", root)


def test_resolve_spec_path_symlink_to_outside_rejected(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    secret = tmp_path / "secret.yaml"
    secret.write_text("name: evil\n")
    (root / "link.yaml").symlink_to(secret)
    with pytest.raises(ValueError):
        resolve_spec_path("link.yaml", root)


def test_resolve_spec_path_symlink_inside_root_rejected(tmp_path: Path) -> None:
    """Even a link whose destination stays under the root is refused."""
    real = tmp_path / "real.yaml"
    real.write_text("name: ok\n")
    (tmp_path / "alias.yaml").symlink_to(real)
    with pytest.raises(ValueError, match="symlinked path rejected"):
        resolve_spec_path("alias.yaml", tmp_path)


def test_resolve_spec_path_symlinked_dir_component_rejected(tmp_path: Path) -> None:
    """A symlinked directory below the root is refused, even pointing inside."""
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "a.yaml").write_text("name: a\n")
    (tmp_path / "shortcut").symlink_to(agents)
    with pytest.raises(ValueError, match="symlinked path rejected"):
        resolve_spec_path("shortcut/a.yaml", tmp_path)
    # The real path is still fine.
    assert resolve_spec_path("agents/a.yaml", tmp_path).name == "a.yaml"
