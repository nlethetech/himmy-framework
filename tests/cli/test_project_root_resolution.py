"""The CLI resolves RBAC role / budget cap / tool allowlist from the PROJECT ROOT.

A run launched from a subdirectory (``cd subdir && himmy run``) must still pick up
the project's ``himmy.toml`` — the role, the session budget cap, and the
``[permissions] auto_approve`` allowlist — instead of silently dropping them
(falling back to admin / no-budget / empty allowlist) while still writing the
protected spine. These readers resolve the file via
:func:`himmy.config.project.find_project_root`, the same mechanism ``spine.db`` and
``agent.yaml`` use; ``himmy.toml`` is itself a project-root marker.
"""

from __future__ import annotations

import argparse

from himmy.cli import permissions, rbac_cmd


def _project_with_toml(tmp_path):
    """A project root carrying a himmy.toml with role + budget + allowlist, and a
    nested subdirectory to launch from. Returns (root, subdir)."""
    (tmp_path / "himmy.toml").write_text(
        "[rbac]\n"
        'role = "viewer"\n'
        "\n"
        "[limits]\n"
        "session_budget = 2.5\n"
        "\n"
        "[permissions]\n"
        'auto_approve = ["wire_money"]\n'
    )
    subdir = tmp_path / "pkg" / "deep"
    subdir.mkdir(parents=True)
    return tmp_path, subdir


def test_budget_cap_enforced_from_subdirectory(tmp_path, monkeypatch):
    """A ``[limits] session_budget`` set at the project root is still read when the
    CLI is invoked from a nested subdirectory (would be ``None`` resolving from cwd)."""
    root, subdir = _project_with_toml(tmp_path)
    monkeypatch.chdir(subdir)
    assert permissions.load_session_budget() == 2.5


def test_allowlist_enforced_from_subdirectory(tmp_path, monkeypatch):
    """A ``[permissions] auto_approve`` allowlist set at the project root is still
    read from a nested subdirectory (would be ``[]`` resolving from cwd)."""
    root, subdir = _project_with_toml(tmp_path)
    monkeypatch.chdir(subdir)
    assert permissions.load_auto_approve() == ["wire_money"]


def test_rbac_role_enforced_from_subdirectory(tmp_path, monkeypatch):
    """The ``[rbac] role`` set at the project root is still resolved from a nested
    subdirectory — the CLI must NOT fall back to admin (which would drop enforcement
    while still writing the protected spine)."""
    monkeypatch.delenv("HIMMY_ROLE", raising=False)
    monkeypatch.delenv("HIMMY_RBAC_FILE", raising=False)
    root, subdir = _project_with_toml(tmp_path)
    monkeypatch.chdir(subdir)
    p = rbac_cmd.cli_principal(argparse.Namespace())
    assert p.roles == frozenset({"viewer"})
    assert p.all_tenants is False


def test_persist_auto_approve_writes_project_root_toml(tmp_path, monkeypatch):
    """An ``always`` answer given from a subdirectory persists into the SAME
    project-root ``himmy.toml`` the reader later consults — not a stray cwd file."""
    root, subdir = _project_with_toml(tmp_path)
    monkeypatch.chdir(subdir)
    changed = permissions.persist_auto_approve("transfer_funds")
    assert changed is True
    # No stray himmy.toml was created in the subdirectory.
    assert not (subdir / "himmy.toml").is_file()
    # The new tool landed in the project-root file and is now readable from the subdir.
    assert "transfer_funds" in permissions.load_auto_approve()
