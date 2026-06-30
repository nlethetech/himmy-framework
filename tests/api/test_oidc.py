"""WS1.1 — OIDC/JWT auth, verified offline with a self-signed key + fixture JWKS."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

jwt = pytest.importorskip("jwt")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from himmy.api import ApiContainer, create_app  # noqa: E402
from himmy.api.auth.base import AuthError  # noqa: E402
from himmy.api.auth.context import build_authenticator  # noqa: E402
from himmy.api.auth.oidc import OidcAuthenticator  # noqa: E402
from tests.conftest import run_async  # noqa: E402

_ISS = "https://issuer.example"
_AUD = "himmy-api"
_KID = "test-key-1"

# One keypair for the whole module (RSA keygen is slow).
_PRIV = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks(kid: str = _KID) -> dict[str, Any]:
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(_PRIV.public_key()))
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": [jwk]}


def _token(
    *,
    kid: str = _KID,
    iss: str = _ISS,
    aud: str = _AUD,
    sub: str = "alice",
    exp_delta: int = 3600,
    include_exp: bool = True,
    key: Any = None,
    extra: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {"iss": iss, "aud": aud, "sub": sub, "iat": now}
    if include_exp:
        payload["exp"] = now + exp_delta
    payload.update(extra or {})
    return jwt.encode(payload, key or _PRIV, algorithm="RS256", headers={"kid": kid})


def _auth(**over: Any) -> OidcAuthenticator:
    opts: dict[str, Any] = {"tenant_claim": "tenant"}
    opts.update(over)
    return OidcAuthenticator(issuer=_ISS, audience=_AUD, jwks=_jwks(), **opts)


class _Req:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.client = type("C", (), {"host": "9.9.9.9"})()


def _authn(auth: OidcAuthenticator, token: str) -> Any:
    return run_async(auth.authenticate(_Req({"authorization": f"Bearer {token}"})))


# ----------------------------------------------------------------- happy path
def test_valid_token_maps_claims_to_principal() -> None:
    p = _authn(
        _auth(),
        _token(extra={"tenant": "t1", "roles": ["operator"], "scope": "read write"}),
    )
    assert p.subject == "alice"
    assert p.tenant_ids == frozenset({"t1"})
    assert "operator" in p.roles
    assert {"read", "write"} <= p.scopes
    assert p.auth_method == "oidc"
    assert p.source_ip == "9.9.9.9"


def test_tenant_claim_list_yields_multiple_tenants() -> None:
    p = _authn(_auth(), _token(extra={"tenant": ["a", "b"], "roles": ["viewer"]}))
    assert p.tenant_ids == frozenset({"a", "b"})


def test_admin_role_grants_all_tenants() -> None:
    auth = _auth(all_tenants_roles=["platform-admin"])
    p = _authn(auth, _token(extra={"roles": ["platform-admin"]}))
    assert p.all_tenants is True


def test_nested_roles_claim_path() -> None:
    auth = _auth(roles_claim="realm_access.roles")
    p = _authn(auth, _token(extra={"realm_access": {"roles": ["operator"]}}))
    assert "operator" in p.roles


# -------------------------------------------------------------- rejection path
def test_expired_token_is_rejected() -> None:
    with pytest.raises(AuthError):
        _authn(_auth(), _token(exp_delta=-10))


def test_missing_exp_is_rejected() -> None:
    with pytest.raises(AuthError):
        _authn(_auth(), _token(include_exp=False))


def test_wrong_audience_is_rejected() -> None:
    with pytest.raises(AuthError):
        _authn(_auth(), _token(aud="some-other-api"))


def test_wrong_issuer_is_rejected() -> None:
    with pytest.raises(AuthError):
        _authn(_auth(), _token(iss="https://evil.example"))


def test_bad_signature_is_rejected() -> None:
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthError):
        _authn(_auth(), _token(key=other))


def test_missing_bearer_is_rejected() -> None:
    with pytest.raises(AuthError):
        run_async(_auth().authenticate(_Req({})))


def test_unknown_kid_triggers_jwks_refresh() -> None:
    """An unknown kid forces a JWKS refresh (key rotation) before giving up."""

    class _Refreshing:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        async def get(self, *, force: bool = False) -> dict[str, Any]:
            self.calls.append(force)
            return _jwks() if force else {"keys": []}

    provider = _Refreshing()
    auth = OidcAuthenticator(
        issuer=_ISS, audience=_AUD, jwks=provider, tenant_claim="tenant"
    )
    p = _authn(auth, _token(extra={"tenant": "t"}))
    assert p.subject == "alice"
    assert provider.calls == [False, True]  # tried cache, then forced a refresh


# -------------------------------------------------------------- BFF integration
def _oidc_app() -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = _auth()
    return TestClient(app)


def test_valid_token_reaches_a_route() -> None:
    client = _oidc_app()
    token = _token(extra={"tenant": "t", "roles": ["operator"]})
    resp = client.get(
        "/v1/runs",
        headers={"Authorization": f"Bearer {token}"},
        params={"workspace_id": "t"},
    )
    assert resp.status_code == 200


def test_oidc_roles_drive_rbac() -> None:
    client = _oidc_app()
    body = {
        "workspace_id": "t",
        "subject_id": "s",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    viewer = _token(sub="v", extra={"tenant": "t", "roles": ["viewer"]})
    assert (
        client.post(
            "/v1/runs", headers={"Authorization": f"Bearer {viewer}"}, json=body
        ).status_code
        == 403
    )
    operator = _token(sub="o", extra={"tenant": "t", "roles": ["operator"]})
    assert (
        client.post(
            "/v1/runs", headers={"Authorization": f"Bearer {operator}"}, json=body
        ).status_code
        == 200
    )


def test_cross_tenant_denied_under_oidc() -> None:
    client = _oidc_app()
    token = _token(extra={"tenant": "t", "roles": ["operator"]})
    # Token bound to tenant t cannot read tenant other → 403.
    assert (
        client.get(
            "/v1/runs",
            headers={"Authorization": f"Bearer {token}"},
            params={"workspace_id": "other"},
        ).status_code
        == 403
    )


def test_openapi_advertises_bearer_scheme() -> None:
    schemes = _auth().openapi_security_scheme()
    assert schemes["himmyOidc"]["scheme"] == "bearer"
    assert schemes["himmyOidc"]["type"] == "http"


# ------------------------------------------------------------------- from_env
def test_build_authenticator_selects_oidc(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_AUTH_MODE", "oidc")
    monkeypatch.setenv("HIMMY_OIDC_ISSUER", _ISS)
    monkeypatch.setenv("HIMMY_OIDC_AUDIENCE", _AUD)
    monkeypatch.setenv("HIMMY_OIDC_JWKS_URL", "https://issuer.example/jwks.json")
    auth = build_authenticator()
    assert isinstance(auth, OidcAuthenticator)


def test_oidc_from_env_requires_issuer_and_audience(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_AUTH_MODE", "oidc")
    monkeypatch.delenv("HIMMY_OIDC_ISSUER", raising=False)
    monkeypatch.delenv("HIMMY_OIDC_AUDIENCE", raising=False)
    with pytest.raises(AuthError):
        build_authenticator()


# ----------------------------------------------- red-team r2: OIDC subject-scoping opt-in
def test_oidc_principal_is_not_subject_scoped_by_default() -> None:
    """INVARIANT: a default OIDC principal is NOT subject_scoped (byte-unchanged model).

    Without the opt-in knob a tenant-bound OIDC caller stays workspace-scoped (the
    documented model), so ``may_access_subject`` is True for any subject (no narrowing).
    """
    p = _authn(_auth(), _token(extra={"tenant": "t1", "roles": ["operator"]}))
    assert p.subject_scoped is False
    # No BOLA narrowing → may read another subject's object within the tenant.
    assert p.may_access_subject("someone-else") is True


def test_oidc_subject_scoped_opt_in_narrows_to_own_subject() -> None:
    """With ``subject_scoped=True`` a tenant-bound OIDC caller is narrowed to its own sub."""
    auth = _auth(subject_scoped=True)
    p = _authn(auth, _token(sub="alice", extra={"tenant": "t1", "roles": ["operator"]}))
    assert p.subject_scoped is True
    # BOLA gate now engages: only its OWN subject is readable.
    assert p.may_access_subject("alice") is True
    assert p.may_access_subject("bob") is False
    # A subject-less legacy object is still allowed.
    assert p.may_access_subject(None) is True


def test_oidc_subject_scoped_admin_role_crosses_subjects() -> None:
    """An ``all_tenants`` (admin-role) OIDC principal is never narrowed even when opted in."""
    auth = _auth(subject_scoped=True, all_tenants_roles=["platform-admin"])
    p = _authn(auth, _token(sub="alice", extra={"roles": ["platform-admin"]}))
    assert p.all_tenants is True
    # all_tenants short-circuits the BOLA gate → reads any subject.
    assert p.may_access_subject("bob") is True


def test_oidc_from_env_reads_subject_scoped_flag(monkeypatch: Any) -> None:
    """``HIMMY_OIDC_SUBJECT_SCOPED`` opts the env-built OIDC authenticator into BOLA."""
    monkeypatch.setenv("HIMMY_AUTH_MODE", "oidc")
    monkeypatch.setenv("HIMMY_OIDC_ISSUER", _ISS)
    monkeypatch.setenv("HIMMY_OIDC_AUDIENCE", _AUD)
    monkeypatch.setenv("HIMMY_OIDC_JWKS_URL", "https://issuer.example/jwks.json")
    monkeypatch.setenv("HIMMY_OIDC_SUBJECT_SCOPED", "1")
    auth = build_authenticator()
    assert isinstance(auth, OidcAuthenticator)
    assert auth._subject_scoped is True
    # Default (flag absent) stays off.
    monkeypatch.delenv("HIMMY_OIDC_SUBJECT_SCOPED", raising=False)
    auth2 = build_authenticator()
    assert isinstance(auth2, OidcAuthenticator)
    assert auth2._subject_scoped is False
