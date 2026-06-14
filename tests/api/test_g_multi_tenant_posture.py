"""G1 + G2: tenant-binding authenticator capability + multi-tenant fail-closed posture.

G1 adds ``binds_tenants`` to the authenticator capability (OIDC → True; API keys →
True only when tenant-mapped) plus a demotable shared key. G2 wires a fail-closed
multi-tenant posture into ``_enforce_auth_posture`` BEFORE the off-loopback early
return, so a shared-key-only deploy (which would otherwise take the early return and
mint every caller an all-tenants admin) is refused.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.auth.apikey import (
    DEMOTED_SHARED_KEY_ROLES,
    ApiKeyAuthenticator,
)
from himmy.api.auth.context import build_authenticator, is_multi_tenant
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import DEFAULT_POLICY
from himmy.core.errors import HimmyError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every posture-relevant env var so each test starts from the offline default."""
    for var in (
        "HIMMY_MULTI_TENANT",
        "HIMMY_AUTH_MODE",
        "HIMMY_INTERNAL_API_KEY",
        "HIMMY_API_KEYS_FILE",
        "HIMMY_ALLOW_UNAUTHENTICATED",
        "HIMMY_ALLOW_OPERATOR_SPEC_TOOLS",
        "HIMMY_DATABASE_URL",
        "HIMMY_DURABLE_STORAGE",
        "HIMMY_BIND_HOST",
    ):
        monkeypatch.delenv(var, raising=False)


# ----------------------------------------------------------------- G1: binds_tenants


def test_shared_only_apikey_binds_nobody() -> None:
    auth = ApiKeyAuthenticator(shared_keys={"shared"})
    assert auth.binds_tenants is False


def test_mapped_apikey_binds_tenants() -> None:
    principal = Principal.build("u1", tenant_ids=["t1"], roles=["operator"])
    auth = ApiKeyAuthenticator(key_principals={"mapped": principal})
    assert auth.binds_tenants is True


def test_mixed_apikey_binds_tenants() -> None:
    # A mapped key makes the authenticator tenant-binding even alongside a shared key.
    principal = Principal.build("u1", tenant_ids=["t1"])
    auth = ApiKeyAuthenticator(
        shared_keys={"shared"}, key_principals={"mapped": principal}
    )
    assert auth.binds_tenants is True


def test_oidc_binds_tenants_capability() -> None:
    from himmy.api.auth.oidc import OidcAuthenticator

    # Class-level capability, readable without constructing (no IdP needed).
    assert getattr(OidcAuthenticator, "binds_tenants", False) is True


def test_custom_authenticator_without_member_reads_false() -> None:
    # G1 must_fix: a legacy authenticator lacking the member must read False, not raise.
    class _Legacy:
        async def authenticate(self, request: Any) -> Any:  # pragma: no cover
            ...

    assert getattr(_Legacy(), "binds_tenants", False) is False


# ------------------------------------------------------------ G1: demotable shared key


@pytest.mark.asyncio
async def test_shared_key_default_is_all_tenants_admin() -> None:
    auth = ApiKeyAuthenticator(shared_keys={"shared"})
    principal = await auth.authenticate(_Req("shared"))
    assert principal.all_tenants is True
    assert principal.roles == frozenset({"admin"})


@pytest.mark.asyncio
async def test_shared_key_demoted_is_operator_only_no_tenants() -> None:
    auth = ApiKeyAuthenticator(
        shared_keys={"shared"}, shared_key_roles=DEMOTED_SHARED_KEY_ROLES
    )
    principal = await auth.authenticate(_Req("shared"))
    assert principal.all_tenants is False
    assert principal.tenant_ids == frozenset()
    assert principal.roles == frozenset(DEMOTED_SHARED_KEY_ROLES)


def test_demoted_role_authorizes_ops_routes_not_403_everywhere() -> None:
    # G1 must_fix: the demoted role must actually grant SOMETHING under DEFAULT_POLICY,
    # else the key is useless. Operator covers the run/agent/diagnostics ops surface.
    principal = Principal.build("k", roles=list(DEMOTED_SHARED_KEY_ROLES))
    assert DEFAULT_POLICY.authorize(principal, "run", "read")
    assert DEFAULT_POLICY.authorize(principal, "agent", "write")
    assert DEFAULT_POLICY.authorize(principal, "diagnostics", "read")
    # ... but it is NOT an unrestricted admin.
    assert not DEFAULT_POLICY.authorize(principal, "audit", "read")


# ----------------------------------------------------- G2: is_multi_tenant signal


def test_is_multi_tenant_default_false() -> None:
    assert is_multi_tenant() is False


def test_is_multi_tenant_via_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    assert is_multi_tenant() is True


def test_is_multi_tenant_via_any_auth_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # ANY non-empty mode engages strictness (incl. the values.yaml 'apikey' example).
    monkeypatch.setenv("HIMMY_AUTH_MODE", "apikey")
    assert is_multi_tenant() is True


def test_is_multi_tenant_none_mode_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_AUTH_MODE", "none")
    assert is_multi_tenant() is False


def test_build_authenticator_demotes_shared_key_under_multi_tenant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # A mapped key keeps binds_tenants True (so the deploy is allowed) but the shared key
    # is still demoted to operator-only under the multi-tenant posture.
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "shared-secret")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    auth = build_authenticator()
    assert isinstance(auth, ApiKeyAuthenticator)
    assert auth.binds_tenants is True
    assert auth._shared_key_roles == DEMOTED_SHARED_KEY_ROLES


# ------------------------------------------------- G2: fail-closed posture refusals


def test_multi_tenant_shared_key_only_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The CORE must_fix: the multi-tenant block runs BEFORE the line-130 early return,
    # so a shared-key-only deploy (authenticator != None) is still refused.
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "shared-secret")
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "bind callers to tenants" in str(exc.value)


def test_multi_tenant_anonymous_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Multi-tenant declared but NO authenticator at all → also refused.
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "bind callers to tenants" in str(exc.value)


def test_multi_tenant_with_mapped_keys_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    app = create_app()
    assert app.state.authenticator is not None
    assert app.state.authenticator.binds_tenants is True


def test_multi_tenant_rejects_allow_unauthenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("HIMMY_ALLOW_UNAUTHENTICATED", "1")
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "HIMMY_ALLOW_UNAUTHENTICATED" in str(exc.value)


def test_multi_tenant_rejects_operator_spec_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "true")
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "HIMMY_ALLOW_OPERATOR_SPEC_TOOLS" in str(exc.value)


def test_single_box_default_starts_byte_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No posture env at all → the offline single-box default must start unchanged.
    app = create_app()
    assert app.state.authenticator is None
    assert TestClient(app).get("/health").status_code == 200


def test_auth_mode_apikey_without_tenant_keys_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Documented behavior: any non-empty HIMMY_AUTH_MODE engages strictness, so a
    # previously-working shared-key-only deploy that also sets a mode is now refused.
    monkeypatch.setenv("HIMMY_AUTH_MODE", "apikey")
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "shared-secret")
    with pytest.raises(HimmyError):
        create_app()


class _Req:
    """A minimal stand-in for a Starlette Request carrying just the key header."""

    def __init__(self, key: str) -> None:
        self.headers = {"x-himmy-internal-key": key}
        self.client = None
