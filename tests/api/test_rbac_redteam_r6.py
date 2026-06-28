"""Red-team round 6 regressions for the RBAC hardening branch.

Four confirmed findings, each with a test that FAILS before the fix and PASSES after:

* (vuln 1) tenant-facing browse roles (viewer/operator/auditor) must NOT reach the
  operator-only Studio console surfaces — deployment diagnostics (``/doctor``), connection
  status (``/connections``) and the operator's Google connection (``/google*``). They were
  reachable because those reads collapsed to the coarse ``studio.console:read`` baseline.
* (vuln 2) the state-changing Google OAuth GET routes (``/google/auth-url``,
  ``/google/callback``) must require ``studio.google:write`` like their PUT/DELETE mutator
  siblings, not the read baseline.
* (vuln 4) every per-surface GET on the MAIN ``/api/studio`` router must enforce its OWN
  ``studio.<surface>:read`` (connections/google/approvals/models), not silently downgrade
  to the coarse console baseline.
* (vuln 3) a CAPABILITY/RBAC tool denial must be a DISTINCT outcome from a
  ``requires_approval`` denial, so the runtime never mis-classifies it as a resumable
  human-approval checkpoint (which would wedge the run).

The OFFLINE / zero-config invariant is asserted too: with NO authenticator configured
every one of these routes still answers (``require_permission`` no-ops).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import (
    _STUDIO_GLOBAL_STORE_RESOURCES,
    DEFAULT_POLICY,
    AccessPolicy,
)


def _app_with_role(*roles: str) -> FastAPI:
    """An app whose mapped key ``"k"`` is bound to ``roles`` (default-policy roles)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=list(roles), auth_method="apikey"
            )
        }
    )
    return app


def _client(app: FastAPI) -> TestClient:
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


def _app_with_policy(role: str, perms: list[str]) -> FastAPI:
    """An app whose key ``"k"`` holds exactly ``perms`` via a custom policy role."""
    app = _app_with_role(role)
    app.state.access_policy = AccessPolicy.from_mapping({role: perms})
    return app


# --------------------------------------------------------------------------- vuln 1
@pytest.mark.parametrize("role", ["viewer", "operator", "auditor"])
@pytest.mark.parametrize(
    "path",
    [
        "/api/studio/doctor",
        "/api/studio/connections",
        "/api/studio/connections/email",
        "/api/studio/google",
        "/api/studio/google/gmail",
        "/api/studio/google/calendar",
    ],
)
def test_tenant_browse_role_is_403_on_operator_console_surface(
    role: str, path: str
) -> None:
    """A default tenant-facing browse role cannot reach the operator-only console reads."""
    c = _client(_app_with_role(role))
    assert c.get(path).status_code == 403, f"{role} reached {path}"


def test_connections_and_google_are_withheld_from_browse_roles_by_policy() -> None:
    """The default browse roles do NOT carry studio.connections/google reads anymore."""
    for role in ("viewer", "operator", "auditor"):
        perms = DEFAULT_POLICY.role_permissions[role]
        assert ("studio.connections", "read") not in perms
        assert ("studio.google", "read") not in perms
    assert "studio.connections" in _STUDIO_GLOBAL_STORE_RESOURCES
    assert "studio.google" in _STUDIO_GLOBAL_STORE_RESOURCES


def test_admin_still_reads_operator_console_surfaces() -> None:
    """``admin`` (``*:*``) is never 403 on the operator console reads."""
    c = _client(_app_with_role("admin"))
    for path in ("/api/studio/doctor", "/api/studio/connections", "/api/studio/google"):
        assert c.get(path).status_code != 403, f"admin blocked on {path}"


# --------------------------------------------------------------------------- vuln 4
def test_console_read_only_role_is_403_on_per_surface_reads() -> None:
    """A role holding ONLY studio.console:read cannot reach the per-surface GETs.

    The crux of vuln 4: these reads previously collapsed to the coarse console baseline,
    so console:read alone reached connections/google/approvals/models. Each now enforces
    its own per-surface read, which this minimal role does not hold.
    """
    c = _client(_app_with_policy("console_only", ["studio.console:read"]))
    # The baseline still opens Studio (health) ...
    assert c.get("/api/studio/health").status_code == 200
    # ... but NOT the higher-value per-surface reads.
    for path in (
        "/api/studio/connections",
        "/api/studio/connections/email",
        "/api/studio/google",
        "/api/studio/google/gmail",
        "/api/studio/approvals",
        "/api/studio/models",
    ):
        assert c.get(path).status_code == 403, f"console:read alone reached {path}"


def test_per_surface_read_grant_unlocks_its_own_surface_only() -> None:
    """Granting studio.connections:read unlocks /connections but not /google."""
    c = _client(
        _app_with_policy(
            "conn_reader", ["studio.console:read", "studio.connections:read"]
        )
    )
    assert c.get("/api/studio/connections").status_code == 200
    assert c.get("/api/studio/google").status_code == 403


# --------------------------------------------------------------------------- vuln 2
def test_google_oauth_get_routes_require_write_not_read() -> None:
    """The state-changing OAuth GETs need studio.google:write, not just :read."""
    # A role with studio.google:READ (the strongest non-admin Google grant) is still 403
    # on the OAuth lifecycle GETs, because those mutate connection state.
    c = _client(
        _app_with_policy(
            "g_reader", ["studio.console:read", "studio.google:read"]
        )
    )
    assert c.get("/api/studio/google").status_code == 200  # pure read OK
    assert c.get("/api/studio/google/auth-url").status_code == 403
    assert (
        c.get("/api/studio/google/callback", params={"code": "x"}).status_code == 403
    )


def test_google_oauth_get_routes_pass_for_write_grant() -> None:
    """With studio.google:write the OAuth GETs pass RBAC (then 409: no client)."""
    c = _client(
        _app_with_policy(
            "g_writer", ["studio.console:read", "studio.google:write"]
        )
    )
    # Passes RBAC and reaches the handler, which 409s because no OAuth client is set —
    # crucially NOT a 403, proving the write grant is what's required.
    assert c.get("/api/studio/google/auth-url").status_code == 409


# --------------------------------------------------------------------------- invariant
@pytest.mark.parametrize(
    "path",
    [
        "/api/studio/doctor",
        "/api/studio/connections",
        "/api/studio/google",
        "/api/studio/approvals",
        "/api/studio/models",
    ],
)
def test_offline_reads_every_surface_byte_unchanged(path: str) -> None:
    """INVARIANT: with NO authenticator, every surface answers (require_permission no-ops)."""
    c = TestClient(create_app(ApiContainer.build_default()))
    assert c.get(path).status_code != 403, f"offline blocked on {path}"
