"""Tests for the CLI RBAC layer (himmy/cli/rbac_cmd.py).

These exercise the REAL framework primitives: an :class:`AccessPolicy` built from
the CLI catalogue, the wildcard-aware ``_covers`` / ``authorize`` check, and the
``gate_tool_for_role`` enforcement hook. Roles only restrict; admin is the
zero-config default so existing single-user CLIs keep full access.
"""

from __future__ import annotations

import argparse
import json

import pytest

from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy, _covers
from himmy.cli import rbac_cmd


def _ns(**kw):
    return argparse.Namespace(**kw)


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    """Run every test from a clean tmp cwd so a stray himmy.toml [rbac]/[permissions]
    in the runner's working directory can't leak into role/policy resolution."""
    monkeypatch.chdir(tmp_path)


def _clean_env(monkeypatch):
    monkeypatch.delenv("HIMMY_ROLE", raising=False)
    monkeypatch.delenv("HIMMY_RBAC_FILE", raising=False)


# --- the real policy primitives -------------------------------------------------


def test_policy_built_from_real_access_policy():
    policy = rbac_cmd.cli_access_policy()
    assert isinstance(policy, AccessPolicy)
    # admin's wildcard is the real frozenset of (resource, action) tuples.
    assert _covers(policy.role_permissions["admin"], "tool_gated", "execute")
    assert _covers(policy.role_permissions["operator"], "tool_gated", "execute")
    assert not _covers(policy.role_permissions["viewer"], "tool_gated", "execute")


def test_gate_admin_allows_gated_tool(monkeypatch):
    _clean_env(monkeypatch)
    policy = rbac_cmd.cli_access_policy()
    admin = Principal.build(subject="a", roles=["admin"], all_tenants=True)
    assert rbac_cmd.gate_tool_for_role(admin, policy, "shell", True) == "allow"


def test_gate_operator_prompts_gated_tool(monkeypatch):
    _clean_env(monkeypatch)
    policy = rbac_cmd.cli_access_policy()
    op = Principal.build(subject="o", roles=["operator"])
    # gated → prompt (may run, but needs approval)
    assert rbac_cmd.gate_tool_for_role(op, policy, "shell", True) == "prompt"
    # ungated → allow outright
    assert rbac_cmd.gate_tool_for_role(op, policy, "search", False) == "allow"


def test_gate_viewer_denies_gated_tool(monkeypatch):
    _clean_env(monkeypatch)
    policy = rbac_cmd.cli_access_policy()
    viewer = Principal.build(subject="v", roles=["viewer"])
    # An approval-gated tool fails closed for a viewer.
    assert rbac_cmd.gate_tool_for_role(viewer, policy, "shell", True) == "deny"
    # But a plain (ungated) read-style tool is allowed.
    assert rbac_cmd.gate_tool_for_role(viewer, policy, "search", False) == "allow"


def test_cli_permits_delegates_to_authorize(monkeypatch):
    _clean_env(monkeypatch)
    policy = rbac_cmd.cli_access_policy()
    viewer = Principal.build(subject="v", roles=["viewer"])
    assert rbac_cmd.cli_permits(viewer, policy, "run", "read") is True
    assert rbac_cmd.cli_permits(viewer, policy, "code", "execute") is False


# --- principal precedence -------------------------------------------------------


def test_principal_default_is_admin(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = rbac_cmd.cli_principal(_ns(), start=tmp_path)
    assert p.roles == frozenset({"admin"})
    assert p.all_tenants is True


def test_principal_toml_role(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    (tmp_path / "himmy.toml").write_text('[rbac]\nrole = "operator"\n')
    p = rbac_cmd.cli_principal(_ns(), start=tmp_path)
    assert p.roles == frozenset({"operator"})
    assert p.all_tenants is False


def test_principal_env_beats_toml(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    (tmp_path / "himmy.toml").write_text('[rbac]\nrole = "operator"\n')
    monkeypatch.setenv("HIMMY_ROLE", "viewer")
    p = rbac_cmd.cli_principal(_ns(), start=tmp_path)
    assert p.roles == frozenset({"viewer"})


def test_principal_flag_beats_env_and_toml(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    (tmp_path / "himmy.toml").write_text('[rbac]\nrole = "operator"\n')
    monkeypatch.setenv("HIMMY_ROLE", "viewer")
    p = rbac_cmd.cli_principal(_ns(role="admin"), start=tmp_path)
    assert p.roles == frozenset({"admin"})


def test_principal_malformed_toml_falls_back(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    (tmp_path / "himmy.toml").write_text("this is = = not valid toml [[[")
    p = rbac_cmd.cli_principal(_ns(), start=tmp_path)
    assert p.roles == frozenset({"admin"})


# --- custom HIMMY_RBAC_FILE -----------------------------------------------------


def test_custom_rbac_file_is_loaded(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    rbac_file = tmp_path / "rbac.json"
    rbac_file.write_text(json.dumps({"custom": ["tool_gated:execute"]}))
    monkeypatch.setenv("HIMMY_RBAC_FILE", str(rbac_file))
    policy = rbac_cmd.cli_access_policy()
    p = Principal.build(subject="c", roles=["custom"])
    # Honours the operator's custom file via the real build_access_policy().
    assert rbac_cmd.gate_tool_for_role(p, policy, "shell", True) == "prompt"
    # A role absent from the custom file is denied (deny-by-default).
    other = Principal.build(subject="x", roles=["nobody"])
    assert rbac_cmd.gate_tool_for_role(other, policy, "shell", True) == "deny"


# --- whoami output --------------------------------------------------------------


def test_whoami_admin(monkeypatch, capsys):
    _clean_env(monkeypatch)
    rc = rbac_cmd.cmd_whoami(_ns(role="admin"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "role:" in out
    assert "admin" in out
    assert "unrestricted" in out
    assert "run" in out  # gated-tools line


def test_whoami_viewer_shows_denied(monkeypatch, capsys):
    _clean_env(monkeypatch)
    rc = rbac_cmd.cmd_whoami(_ns(role="viewer"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "viewer" in out
    assert "run:read" in out
    assert "NOT run" in out  # viewer cannot run gated tools


def test_whoami_unknown_role_no_grants(monkeypatch, capsys):
    _clean_env(monkeypatch)
    rc = rbac_cmd.cmd_whoami(_ns(role="ghost"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "ghost" in out
    assert "none" in out


# --- enforcement: registry re-gating + prompt_approval fail-closed -------------


def _gated_registry():
    from himmy.services.tools.registry import ToolRegistry, register_local_tool

    reg = ToolRegistry()
    register_local_tool(
        reg,
        name="wire_money",
        handler=lambda args: {"sent": True},
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=True,
    )
    return reg


def test_enforce_role_admin_is_noop(monkeypatch):
    """The default admin re-gates nothing — existing sessions are unaffected."""
    _clean_env(monkeypatch)
    reg = _gated_registry()
    # Even after an allowlist ungated it, admin leaves it as-is.
    from himmy.cli import permissions

    permissions.apply_allowlist(reg, ["wire_money"])
    assert reg.get("wire_money").requires_approval is False
    regated = rbac_cmd.enforce_role_on_registry(reg, role=None)
    assert regated == []
    assert reg.get("wire_money").requires_approval is False


def test_enforce_role_viewer_regates_allowlisted_gated_tool(monkeypatch):
    """A viewer can't slip a gated tool past the allowlist: it is re-gated (fail closed)."""
    _clean_env(monkeypatch)
    reg = _gated_registry()
    from himmy.cli import permissions

    # The real callers snapshot born-gated status BEFORE the allowlist ungates any.
    born = rbac_cmd.born_gated_names(reg)
    assert born == {"wire_money"}
    permissions.apply_allowlist(reg, ["wire_money"])  # would ungated it
    assert reg.get("wire_money").requires_approval is False
    regated = rbac_cmd.enforce_role_on_registry(reg, role="viewer", born_gated=born)
    assert "wire_money" in regated
    assert reg.get("wire_money").requires_approval is True  # forced back closed


def _mixed_registry():
    """A registry with one ungated plain tool + one born-gated tool."""
    from himmy.services.tools.registry import ToolRegistry, register_local_tool

    reg = ToolRegistry()
    register_local_tool(
        reg,
        name="calculator",
        handler=lambda args: {"r": 4},
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=False,
    )
    register_local_tool(
        reg,
        name="wire_money",
        handler=lambda args: {"sent": True},
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=True,
    )
    return reg


def test_born_gated_names_snapshots_gated_only():
    """born_gated_names returns exactly the tools approval-gated as built."""
    reg = _mixed_registry()
    assert rbac_cmd.born_gated_names(reg) == {"wire_money"}
    assert rbac_cmd.born_gated_names(None) == set()


def test_enforce_role_viewer_keeps_ungated_tools(monkeypatch):
    """Finding 1: a viewer KEEPS plain ungated tools (it holds tool:execute) and
    only loses approval-gated ones — the ungated tool is never re-gated/denied."""
    _clean_env(monkeypatch)
    reg = _mixed_registry()
    born = rbac_cmd.born_gated_names(reg)
    lines: list[str] = []
    regated = rbac_cmd.enforce_role_on_registry(
        reg, role="viewer", born_gated=born, on_line=lines.append
    )
    # Only the gated tool is re-gated; the plain calculator is untouched.
    assert regated == ["wire_money"]
    assert reg.get("calculator").requires_approval is False
    assert reg.get("wire_money").requires_approval is True
    # The deny line names only the gated tool, and never calls calculator gated.
    assert any("wire_money" in line for line in lines)
    assert not any("calculator" in line for line in lines)


def test_enforce_role_viewer_denies_plain_tool_without_tool_execute(
    monkeypatch, tmp_path
):
    """A role lacking tool:execute IS denied even a plain tool — with a message that
    says 'lacks tool:execute', not 'approval-gated'."""
    _clean_env(monkeypatch)
    rbac_file = tmp_path / "rbac.json"
    # 'locked' grants nothing for tools at all.
    rbac_file.write_text(json.dumps({"locked": ["run:read"]}))
    monkeypatch.setenv("HIMMY_RBAC_FILE", str(rbac_file))
    reg = _mixed_registry()
    born = rbac_cmd.born_gated_names(reg)
    lines: list[str] = []
    regated = rbac_cmd.enforce_role_on_registry(
        reg, role="locked", born_gated=born, on_line=lines.append
    )
    assert "calculator" in regated
    assert reg.get("calculator").requires_approval is True  # re-gated closed
    plain_line = next(line for line in lines if "calculator" in line)
    assert "lacks tool:execute" in plain_line
    assert "approval-gated" not in plain_line


def test_deny_message_gated_vs_plain():
    """deny_message distinguishes gated (approval-gated) from plain (lacks tool:execute)."""
    viewer = Principal.build(subject="v", roles=["viewer"])
    gated = rbac_cmd.deny_message(viewer, "wire_money", gated=True)
    plain = rbac_cmd.deny_message(viewer, "calculator", gated=False)
    assert "approval-gated" in gated
    assert "lacks tool:execute" in plain
    assert "approval-gated" not in plain


def test_enforce_role_fallback_uses_current_gating(monkeypatch):
    """With no born_gated supplied, enforcement falls back to the registry's current
    requires_approval (correct when nothing has ungated a born-gated tool)."""
    _clean_env(monkeypatch)
    reg = _mixed_registry()
    regated = rbac_cmd.enforce_role_on_registry(reg, role="viewer")
    # calculator (ungated) kept; wire_money (gated) re-gated.
    assert regated == ["wire_money"]
    assert reg.get("calculator").requires_approval is False


def _paused_checkpoint():
    """A real paused agent loop at a gated tool; returns (cp_store, checkpoint_id)."""
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona
    from tests.conftest import run_async
    from tests.runtime.test_hitl import _hitl_runtime

    rt_obj, cp_store = _hitl_runtime()
    paused = run_async(
        rt_obj.run_agent_loop(
            Persona(name="t"),
            Task(
                title="t",
                prompt="wire",
                context={
                    "tool_names": ["wire_money"],
                    "response_format": "AUTO_TOOLS",
                },
            ),
            hitl=True,
        )
    )
    return cp_store, paused.checkpoint_id


def test_prompt_approval_viewer_denied_before_prompt(monkeypatch):
    """A viewer is DENIED a gated tool before any approve? prompt (fail closed)."""
    _clean_env(monkeypatch)
    from himmy.cli.repl import prompt_approval

    cp_store, cid = _paused_checkpoint()
    lines: list[str] = []
    prompted = {"asked": False}

    def _never(_p):
        prompted["asked"] = True
        return "y"

    decision = prompt_approval(
        cp_store, cid, input_fn=_never, on_line=lines.append, role="viewer"
    )
    assert decision is False  # fail closed
    assert prompted["asked"] is False  # never reached the prompt
    assert any("may not run wire_money" in line for line in lines)


def test_prompt_approval_operator_still_prompts(monkeypatch):
    """An operator may run the gated tool but still sees the normal approve? prompt."""
    _clean_env(monkeypatch)
    from himmy.cli.repl import prompt_approval

    cp_store, cid = _paused_checkpoint()
    lines: list[str] = []
    decision = prompt_approval(
        cp_store, cid, input_fn=lambda _p: "y", on_line=lines.append, role="operator"
    )
    assert decision is True
    assert any("approval required" in line and "wire_money" in line for line in lines)


def test_prompt_approval_admin_default_prompts(monkeypatch):
    """The default admin role keeps the normal HITL prompt (RBAC never auto-approves)."""
    _clean_env(monkeypatch)
    from himmy.cli.repl import prompt_approval

    cp_store, cid = _paused_checkpoint()
    lines: list[str] = []
    decision = prompt_approval(
        cp_store, cid, input_fn=lambda _p: "y", on_line=lines.append, role=None
    )
    assert decision is True
    assert any("approval required" in line for line in lines)


# --- `himmy rbac validate` CLI -------------------------------------------------


def _validate(file_path) -> int:
    return rbac_cmd.cmd_rbac(_ns(action="validate", file=str(file_path)))


def test_rbac_validate_happy_path(monkeypatch, tmp_path, capsys):
    _clean_env(monkeypatch)
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({"viewer": ["run:read"], "admin": ["*:*"]}))
    rc = _validate(f)
    out = capsys.readouterr().out
    assert rc == 0
    assert "policy OK" in out


def test_rbac_validate_trailing_colon_errors(monkeypatch, tmp_path, capsys):
    """A hand-edited 'run:' is rejected (exit non-zero), naming the role."""
    _clean_env(monkeypatch)
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({"oops": ["run:"]}))
    rc = _validate(f)
    out = capsys.readouterr().out
    assert rc == 1
    assert "oops" in out
    assert "1 error" in out


def test_rbac_validate_empty_policy_warns(monkeypatch, tmp_path, capsys):
    """An empty {} validates (no error) but warns about the lock-out."""
    _clean_env(monkeypatch)
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({}))
    rc = _validate(f)
    out = capsys.readouterr().out
    assert rc == 0
    assert "EMPTY" in out


def test_rbac_validate_nonadmin_wildcard_warns(monkeypatch, tmp_path, capsys):
    _clean_env(monkeypatch)
    f = tmp_path / "rbac.json"
    f.write_text(json.dumps({"weak": ["*:read"]}))
    rc = _validate(f)
    out = capsys.readouterr().out
    assert rc == 0  # warning, not an error
    assert "wildcard" in out
    assert "weak" in out


def test_rbac_validate_missing_file(monkeypatch, tmp_path, capsys):
    _clean_env(monkeypatch)
    rc = _validate(tmp_path / "nope.json")
    out = capsys.readouterr().out
    assert rc == 1
    assert "no such file" in out


def test_rbac_validate_bad_json(monkeypatch, tmp_path, capsys):
    _clean_env(monkeypatch)
    f = tmp_path / "rbac.json"
    f.write_text("{not json")
    rc = _validate(f)
    out = capsys.readouterr().out
    assert rc == 1
    assert "invalid JSON" in out


def test_rbac_validate_unknown_action(monkeypatch, tmp_path, capsys):
    rc = rbac_cmd.cmd_rbac(_ns(action="bogus"))
    out = capsys.readouterr().out
    assert rc == 2
    assert "unknown rbac action" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
