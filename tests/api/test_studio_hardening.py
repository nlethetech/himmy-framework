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
