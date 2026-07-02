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
        "HIMMY_STUDIO_AUTH",
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


def test_is_multi_tenant_via_flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Red-team r3: HIMMY_MULTI_TENANT=on MUST engage the posture.

    'on' is the natural on/off convention and is already accepted by the consuming
    sanitizers (apikey._env_truthy / spec_sanitizer._truthy). A divergent detector that
    missed 'on' silently left a shared key a cross-tenant super-admin and skipped the
    startup refusal.
    """
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "on")
    assert is_multi_tenant() is True


def test_multi_tenant_on_shared_key_only_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HIMMY_MULTI_TENANT=on + shared-key-only is refused (posture engages for 'on')."""
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "on")
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "shared-secret")
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "bind callers to tenants" in str(exc.value)


def test_multi_tenant_on_demotes_shared_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """HIMMY_MULTI_TENANT=on demotes a co-configured shared key (G1) just like '1'."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "on")
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "shared-secret")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    auth = build_authenticator()
    assert isinstance(auth, ApiKeyAuthenticator)
    assert auth._shared_key_roles == DEMOTED_SHARED_KEY_ROLES


def test_multi_tenant_rejects_operator_spec_tools_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Red-team r3: HIMMY_ALLOW_OPERATOR_SPEC_TOOLS=on must ALSO trip the startup refusal.

    The consuming sanitizer (spec_sanitizer._truthy) treats 'on' as enabled, so the
    posture check must use the SAME truthy vocabulary or the RCE/SSRF opt-in slips past
    the fail-closed guard.
    """
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "on")
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "HIMMY_ALLOW_OPERATOR_SPEC_TOOLS" in str(exc.value)


def test_multi_tenant_rejects_allow_unauthenticated_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """HIMMY_ALLOW_UNAUTHENTICATED=on trips the startup refusal (truthy-parity)."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("HIMMY_ALLOW_UNAUTHENTICATED", "on")
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "HIMMY_ALLOW_UNAUTHENTICATED" in str(exc.value)


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


# ------------------ sec-r1: binds_concrete_tenants closes the all-tenants-only-file gap


def test_all_tenants_only_file_does_not_bind_concrete_tenants() -> None:
    """A keys file of ONLY all-tenants records satisfies binds_tenants but NOT concrete.

    Regression: the multi-tenant guard was satisfied by a single all-tenants-admin key,
    giving false assurance of isolation. The stricter ``binds_concrete_tenants`` sees through
    it — an all-tenants-only file binds no concrete tenant.
    """
    principal = Principal.build("admin", all_tenants=True)
    auth = ApiKeyAuthenticator(key_principals={"k": principal})
    assert auth.binds_tenants is True  # historical: any mapped key counts
    assert auth.binds_concrete_tenants is False  # but it binds no concrete tenant


def test_concrete_tenant_key_binds_concrete_tenants() -> None:
    """A key scoped to a real tenant binds concrete tenants (the honest multi-tenant posture)."""
    principal = Principal.build("u1", tenant_ids=["t1"])
    auth = ApiKeyAuthenticator(key_principals={"k": principal})
    assert auth.binds_concrete_tenants is True


def test_multi_tenant_all_tenants_only_file_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """HIMMY_MULTI_TENANT + an all-tenants-only keys file is refused (no false assurance).

    Regression for the sec-r1 finding: under an EXPLICIT multi-tenant posture a keys file whose
    every record is all-tenants would run every caller as an all-tenants admin — exactly the
    posture the guard promises to refuse. It must now fail closed.
    """
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"admin": {"subject": "admin", "all_tenants": True}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "concrete tenant" in str(exc.value)


def test_single_agent_apikey_deploy_still_boots_on_all_tenants_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The documented one-secret single-agent deploy (auth mode, NO HIMMY_MULTI_TENANT) boots.

    The stricter concrete-tenant check applies ONLY when HIMMY_MULTI_TENANT is explicitly
    declared. The intentional one-key deploy (apikey mode alone) is unchanged — it still boots
    on a single all-tenants key, so the offline/one-agent contract is preserved.
    """
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"admin": {"subject": "admin", "all_tenants": True}})
    )
    monkeypatch.setenv("HIMMY_AUTH_MODE", "apikey")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    # No HIMMY_MULTI_TENANT → the stricter concrete-tenant check does not engage.
    app = create_app()  # boots, no refusal
    assert app is not None


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


# --------------------------------------- P0 #4: Studio auth kill-switch lockdown


def test_multi_tenant_rejects_studio_auth_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # Even with tenant-binding auth, HIMMY_STUDIO_AUTH=off under a multi-tenant posture
    # re-opens the operator console to any authenticated principal → refused at startup.
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("HIMMY_STUDIO_AUTH", "off")
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "HIMMY_STUDIO_AUTH" in str(exc.value)


def test_single_box_studio_auth_off_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No multi-tenant posture → HIMMY_STUDIO_AUTH=off is the documented single-user
    # escape hatch and MUST NOT block startup.
    monkeypatch.setenv("HIMMY_STUDIO_AUTH", "off")
    app = create_app()
    assert app.state.authenticator is None
    assert TestClient(app).get("/health").status_code == 200


# ------------------------------------------- P0 #4: OpenAPI auto-docs lockdown gate


def test_docs_open_on_offline_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Zero-config offline default: the interactive docs + schema stay ENABLED.
    app = create_app()
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_docs_locked_when_authenticator_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # With auth configured, /openapi.json + /docs are suppressed (404) but /health stays open.
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    app = create_app()
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
    # /health is still a registered route (the docs gate only drops the schema/docs
    # routes); it answers 200 to an authenticated caller rather than 404.
    assert client.get("/health", headers={"x-himmy-internal-key": "mapped"}).status_code == 200


def test_auth_mode_apikey_without_tenant_keys_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Documented behavior: any non-empty HIMMY_AUTH_MODE engages strictness, so a
    # previously-working shared-key-only deploy that also sets a mode is now refused.
    monkeypatch.setenv("HIMMY_AUTH_MODE", "apikey")
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "shared-secret")
    with pytest.raises(HimmyError):
        create_app()


# ----------------------- red-team r2: keys-file-only (no env flag) engages the posture
# A per-tenant HIMMY_API_KEYS_FILE is multi-tenant IN FACT even with NO HIMMY_MULTI_TENANT
# / HIMMY_AUTH_MODE set. The env-flag-only detector silently skipped the whole posture for
# such a deploy, so (a) a co-configured shared key stayed an all-tenants admin (the G1 hole
# re-opened) and (b) HIMMY_STUDIO_AUTH=off was NOT refused. Both must now engage off the
# authenticator's binds_tenants capability.


def test_keys_file_only_demotes_shared_key_without_env_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A mapped keys file alone (no HIMMY_MULTI_TENANT) still DEMOTES a shared key (G1)."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    # Deliberately NO HIMMY_MULTI_TENANT / HIMMY_AUTH_MODE — only a tenant-mapped keys file.
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "shared-secret")
    assert is_multi_tenant() is False  # the env flag is genuinely absent
    auth = build_authenticator()
    assert isinstance(auth, ApiKeyAuthenticator)
    assert auth.binds_tenants is True
    # The shared key must be demoted (NOT an all-tenants admin) purely off binds_tenants.
    assert auth._shared_key_roles == DEMOTED_SHARED_KEY_ROLES


def test_keys_file_only_refuses_studio_auth_off_without_env_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A mapped keys file alone makes HIMMY_STUDIO_AUTH=off a startup refusal (BOLA fix)."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("HIMMY_STUDIO_AUTH", "off")
    assert is_multi_tenant() is False
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "HIMMY_STUDIO_AUTH" in str(exc.value)


def test_keys_file_only_refuses_allow_unauthenticated_without_env_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A mapped keys file alone makes HIMMY_ALLOW_UNAUTHENTICATED a startup refusal."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("HIMMY_ALLOW_UNAUTHENTICATED", "1")
    assert is_multi_tenant() is False
    with pytest.raises(HimmyError) as exc:
        create_app()
    assert "HIMMY_ALLOW_UNAUTHENTICATED" in str(exc.value)


def test_keys_file_only_clean_deploy_still_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A mapped keys file with no kill-switches starts cleanly (the posture is not over-broad)."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(
        json.dumps({"mapped": {"subject": "u1", "tenant_ids": ["t1"]}})
    )
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys_file))
    app = create_app()
    assert app.state.authenticator is not None
    assert app.state.authenticator.binds_tenants is True


class _Req:
    """A minimal stand-in for a Starlette Request carrying just the key header."""

    def __init__(self, key: str) -> None:
        self.headers = {"x-himmy-internal-key": key}
        self.client = None
