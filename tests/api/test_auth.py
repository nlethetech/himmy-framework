"""Unit tests for the auth seam: Principal, ApiKeyAuthenticator, key mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from himmy.api.auth.apikey import (
    ApiKeyAuthenticator,
    load_key_principals,
)
from himmy.api.auth.base import AuthError
from himmy.api.auth.principal import ANONYMOUS, Principal
from tests.conftest import run_async


class _Req:
    """A minimal stand-in for a Starlette Request (headers + client)."""

    def __init__(self, headers: dict[str, str], host: str = "1.2.3.4") -> None:
        self.headers = headers
        self.client = type("C", (), {"host": host})()


# ----------------------------------------------------------------- Principal
def test_principal_tenant_entitlements() -> None:
    p = Principal.build("u", tenant_ids=["a", "b"], roles=["operator"])
    assert p.may_access("a") and p.may_access("b")
    assert not p.may_access("c")
    assert p.default_tenant() is None  # two tenants → no implied default
    assert Principal.build("u", tenant_ids=["only"]).default_tenant() == "only"
    assert p.has_role("operator") and not p.has_role("admin")


def test_anonymous_is_all_tenants() -> None:
    assert ANONYMOUS.all_tenants is True
    assert ANONYMOUS.may_access("anything") is True


# -------------------------------------------------- may_access_subject (BOLA)
def test_may_access_subject_noop_for_default_and_anonymous() -> None:
    """The INVARIANT: no subject narrowing for offline / non-subject-scoped principals."""
    # ANONYMOUS / all_tenants → always True (offline byte-unchanged).
    assert ANONYMOUS.may_access_subject("anyone") is True
    assert ANONYMOUS.may_access_subject(None) is True
    # A tenant-bound but NOT subject-scoped operator (the default multi-user workspace)
    # reads every subject — subject narrowing is opt-in.
    op = Principal.build("u", tenant_ids=["t"], roles=["operator"])
    assert op.may_access_subject("someone-else") is True


def test_may_access_subject_narrows_only_when_subject_scoped() -> None:
    p = Principal.build(
        "subj-a", tenant_ids=["t"], roles=["operator"], subject_scoped=True
    )
    assert p.may_access_subject("subj-a") is True  # its own
    assert p.may_access_subject("subj-b") is False  # another subject → denied
    assert p.may_access_subject(None) is True  # legacy subject-less resource


def test_may_access_subject_tenant_admin_crosses_subjects() -> None:
    admin = Principal.build(
        "admin",
        tenant_ids=["t"],
        roles=["operator", "tenant_admin"],
        subject_scoped=True,
    )
    assert admin.may_access_subject("subj-a") is True
    assert admin.may_access_subject("subj-b") is True


# -------------------------------------------------------- ApiKeyAuthenticator
def test_shared_key_yields_all_tenants_admin() -> None:
    auth = ApiKeyAuthenticator(shared_keys={"sek"})
    p = run_async(auth.authenticate(_Req({"x-himmy-internal-key": "sek"})))
    assert p.all_tenants is True
    assert "admin" in p.roles
    assert p.source_ip == "1.2.3.4"


def test_mapped_key_yields_bound_principal() -> None:
    bound = Principal.build("svc-a", tenant_ids=["a"], roles=["operator"])
    auth = ApiKeyAuthenticator(key_principals={"ka": bound})
    p = run_async(auth.authenticate(_Req({"x-himmy-internal-key": "ka"})))
    assert p.tenant_ids == frozenset({"a"})
    assert p.all_tenants is False
    assert p.source_ip == "1.2.3.4"  # stamped from the request


def test_missing_key_raises() -> None:
    auth = ApiKeyAuthenticator(shared_keys={"sek"})
    with pytest.raises(AuthError):
        run_async(auth.authenticate(_Req({})))


def test_invalid_key_raises() -> None:
    auth = ApiKeyAuthenticator(shared_keys={"sek"})
    with pytest.raises(AuthError):
        run_async(auth.authenticate(_Req({"x-himmy-internal-key": "nope"})))


def test_no_keys_configured_is_an_error() -> None:
    with pytest.raises(AuthError):
        ApiKeyAuthenticator()


def test_load_key_principals_from_file(tmp_path: Path) -> None:
    spec: dict[str, Any] = {
        "k1": {"subject": "svc1", "tenant_ids": ["t1"], "roles": ["operator"]},
        "k2": {"tenant_ids": ["t2", "t3"], "roles": ["viewer"]},
    }
    f = tmp_path / "keys.json"
    f.write_text(json.dumps(spec))
    mapped = load_key_principals(f)
    assert mapped["k1"].subject == "svc1"
    assert mapped["k1"].tenant_ids == frozenset({"t1"})
    assert mapped["k2"].tenant_ids == frozenset({"t2", "t3"})
    assert "viewer" in mapped["k2"].roles
