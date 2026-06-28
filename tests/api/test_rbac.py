"""WS1.2 — RBAC: roles → permissions, enforced per route (deny-by-default)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import (
    DEFAULT_POLICY,
    AccessPolicy,
    RbacPolicyError,
    _parse_perm,
    lint_policy,
    load_policy,
)


# ------------------------------------------------------------ policy unit tests
def _p(*roles: str) -> Principal:
    return Principal.build("u", tenant_ids=["t"], roles=list(roles))


def test_default_role_matrix() -> None:
    pol = DEFAULT_POLICY
    # viewer: reads, no writes, no audit.
    assert pol.authorize(_p("viewer"), "run", "read")
    assert not pol.authorize(_p("viewer"), "run", "write")
    assert not pol.authorize(_p("viewer"), "audit", "read")
    # operator: reads + writes operational, still no audit.
    assert pol.authorize(_p("operator"), "run", "write")
    assert pol.authorize(_p("operator"), "context", "write")
    assert not pol.authorize(_p("operator"), "audit", "read")
    # auditor: reads incl. audit, no writes.
    assert pol.authorize(_p("auditor"), "audit", "read")
    assert not pol.authorize(_p("auditor"), "run", "write")
    # admin: everything via wildcard.
    assert pol.authorize(_p("admin"), "anything", "destroy")


def test_no_role_is_denied_by_default() -> None:
    assert not DEFAULT_POLICY.authorize(_p(), "run", "read")


def test_multiple_roles_union() -> None:
    assert DEFAULT_POLICY.authorize(_p("viewer", "operator"), "run", "write")


def test_load_policy_from_file(tmp_path: Path) -> None:
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({"custom": ["run:read", "widget:*"]}))
    pol = load_policy(f)
    assert pol.authorize(_p("custom"), "widget", "anything")
    assert not pol.authorize(_p("custom"), "run", "write")


def test_wildcard_action_and_resource() -> None:
    pol = AccessPolicy.from_mapping({"r": ["run:*"], "a": ["*:read"]})
    assert pol.authorize(_p("r"), "run", "write")
    assert not pol.authorize(_p("r"), "context", "read")
    assert pol.authorize(_p("a"), "context", "read")
    assert not pol.authorize(_p("a"), "context", "write")


# --------------------------------------------- fail-closed permission parsing (P0)
def test_trailing_colon_no_longer_widens() -> None:
    """A typo 'run:' must FAIL CLOSED, not silently widen to ('run', '*')."""
    with pytest.raises(RbacPolicyError):
        _parse_perm("run:")


def test_empty_half_rejected() -> None:
    """Both ':read' and an empty spec fail closed (no emptiness-as-wildcard)."""
    with pytest.raises(RbacPolicyError):
        _parse_perm(":read")
    with pytest.raises(RbacPolicyError):
        _parse_perm("")
    with pytest.raises(RbacPolicyError):
        _parse_perm("noseparator")


def test_literal_wildcard_still_parses() -> None:
    """A wildcard written as the literal '*' is fine; emptiness is not."""
    assert _parse_perm("run:*") == ("run", "*")
    assert _parse_perm("*:read") == ("*", "read")
    assert _parse_perm("*:*") == ("*", "*")


def test_from_mapping_names_offending_role_on_bad_perm() -> None:
    """A malformed perm names the role that carries it."""
    with pytest.raises(RbacPolicyError, match="broken"):
        AccessPolicy.from_mapping({"broken": ["run:"]})


def test_from_mapping_rejects_non_list_perms() -> None:
    with pytest.raises(RbacPolicyError, match="badrole"):
        AccessPolicy.from_mapping({"badrole": "run:read"})  # type: ignore[dict-item]


def test_from_mapping_rejects_non_string_perm() -> None:
    with pytest.raises(RbacPolicyError, match="r1"):
        AccessPolicy.from_mapping({"r1": [123]})  # type: ignore[list-item]


def test_from_mapping_rejects_non_dict_top_level() -> None:
    with pytest.raises(RbacPolicyError):
        AccessPolicy.from_mapping(["run:read"])  # type: ignore[arg-type]


def test_empty_policy_warns_lockout() -> None:
    """An empty {} policy parses but WARNS loudly (it locks everyone out)."""
    _policy, errors, warnings = lint_policy({})
    assert errors == []
    assert any("EMPTY" in w for w in warnings)


def test_nonadmin_wildcard_warns() -> None:
    """A non-admin role granted '*:*' / '*:<action>' warns (likely over-grant)."""
    _policy, errors, warnings = lint_policy({"weak": ["*:read"]})
    assert errors == []
    assert any("wildcard" in w and "weak" in w for w in warnings)
    # admin's wildcard is expected — no wildcard warning for it.
    _p2, _e2, w2 = lint_policy({"admin": ["*:*"]})
    assert not any("wildcard" in w and "admin" in w for w in w2)


def test_lint_unknown_token_warns_typo() -> None:
    """An unknown resource/action token is flagged as a likely typo (warning only)."""
    _policy, errors, warnings = lint_policy({"role": ["recommendaton:read"]})
    assert errors == []
    assert any("typo" in w for w in warnings)


def test_lint_collects_errors_without_raising() -> None:
    """lint_policy reports all errors at once and never raises."""
    policy, errors, _warnings = lint_policy({"r": ["run:", ":read"]})
    assert policy is None
    assert len(errors) == 2
    assert all("r" in e for e in errors)


def test_load_policy_malformed_json_raises_clean_error(tmp_path: Path) -> None:
    """A broken file raises a clean RbacPolicyError, never a raw JSONDecodeError 500."""
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")
    with pytest.raises(RbacPolicyError, match="invalid JSON"):
        load_policy(f)


def test_load_policy_non_object_raises_clean_error(tmp_path: Path) -> None:
    f = tmp_path / "list.json"
    f.write_text("[1, 2, 3]")
    with pytest.raises(RbacPolicyError, match="must be a JSON object"):
        load_policy(f)


def test_malformed_rbac_file_fails_startup_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken HIMMY_RBAC_FILE raises a clean HimmyError at create_app, not a 500."""
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({"role": ["run:"]}))  # trailing-colon widen attempt
    monkeypatch.setenv("HIMMY_RBAC_FILE", str(f))
    with pytest.raises(RbacPolicyError):
        create_app(ApiContainer.build_default())


# ------------------------------------------------------------- route enforcement
def _client(role: str) -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=[role], auth_method="apikey"
            )
        }
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


def _create_body() -> dict:
    return {
        "workspace_id": "t",
        "subject_id": "s",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }


def test_viewer_cannot_create_a_run() -> None:
    assert _client("viewer").post("/v1/runs", json=_create_body()).status_code == 403


def test_viewer_can_list_runs() -> None:
    assert (
        _client("viewer").get("/v1/runs", params={"workspace_id": "t"}).status_code
        == 200
    )


def test_operator_can_create_a_run() -> None:
    assert _client("operator").post("/v1/runs", json=_create_body()).status_code == 200


def test_roleless_principal_is_forbidden_everywhere() -> None:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={"k": Principal.build("u", tenant_ids=["t"], roles=[])}
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    assert c.get("/v1/runs", params={"workspace_id": "t"}).status_code == 403


def test_offline_default_bypasses_rbac() -> None:
    """No authenticator configured → RBAC off → zero-config behavior unchanged."""
    c = TestClient(create_app(ApiContainer.build_default()))
    assert c.post("/v1/runs", json=_create_body()).status_code == 200
