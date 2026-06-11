"""Tests for ``himmy mcp`` (himmy/cli/mcp_cmd.py).

list/add/remove round-trip a temp ``agent.yaml`` (asserting the YAML survives), and
``test`` is exercised against the in-repo mock stdio MCP server (a real connect →
tools/list → disconnect) plus the clean failure path for a bogus command.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest
import yaml

from himmy.cli import mcp_cmd

_MOCK_SERVER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "mcp", "mock_mcp_server.py"
)


def _ns(**kw: object) -> argparse.Namespace:
    """An argparse.Namespace with the attrs the handlers read (defaults filled)."""
    base: dict[str, object] = {
        "action": None,
        "name": None,
        "file": None,
        "command": None,
        "arg": None,
        "env": None,
        "cwd": None,
        "prefix": None,
        "requires_approval": False,
        "tool": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def agent_yaml(tmp_path: Path) -> Path:
    """A minimal valid agent.yaml in tmp_path."""
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {"name": "demo", "description": "test agent", "instructions": ["hi"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- add


def test_add_appends_and_round_trips(
    agent_yaml: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = mcp_cmd.cmd_mcp(
        _ns(
            action="add",
            name="fs",
            file=str(agent_yaml),
            command=sys.executable,
            arg=["-c", "print('hi')"],
        )
    )
    assert rc == 0
    raw = yaml.safe_load(agent_yaml.read_text())
    # The other keys survive untouched.
    assert raw["name"] == "demo"
    assert raw["description"] == "test agent"
    assert raw["instructions"] == ["hi"]
    # The server landed under its friendly name (prefix) + args preserved.
    servers = raw["mcp_servers"]
    assert len(servers) == 1
    assert servers[0]["command"] == sys.executable
    assert servers[0]["args"] == ["-c", "print('hi')"]
    assert servers[0]["prefix"] == "fs_"
    # And the whole file still loads through the real spec loader.
    from himmy.config.agent_spec import load_agent_spec

    spec = load_agent_spec(str(agent_yaml))
    assert len(spec.mcp_servers) == 1


def test_documented_arg_dash_form_parses(agent_yaml: Path) -> None:
    """Finding 4: the documented `--arg=-y` form parses (a dash-leading value needs
    the `=` so argparse keeps it as the arg, not an unknown option flag)."""
    from himmy.cli.__main__ import build_parser

    parser = build_parser()
    # The form the docstring now shows works end-to-end.
    args = parser.parse_args(
        ["mcp", "add", "gh", "--command", "npx", "--arg=-y", "--arg", "@scope/server"]
    )
    assert args.arg == ["-y", "@scope/server"]
    # The OLD broken form (`--arg -y`) is a parse error (proves the doc fix matters).
    with pytest.raises(SystemExit):
        parser.parse_args(["mcp", "add", "gh", "--command", "npx", "--arg", "-y"])


def test_docstring_documents_arg_equals_form() -> None:
    """The module docstring documents the runnable `--arg=-y` form, not `--arg -y`."""
    assert "--arg=-y" in (mcp_cmd.__doc__ or "")
    assert "--arg -y " not in (mcp_cmd.__doc__ or "")


def test_add_requires_command(agent_yaml: Path) -> None:
    rc = mcp_cmd.cmd_mcp(_ns(action="add", name="x", file=str(agent_yaml)))
    assert rc == 2
    # Nothing was written.
    raw = yaml.safe_load(agent_yaml.read_text())
    assert "mcp_servers" not in raw


def test_add_rejects_duplicate(agent_yaml: Path) -> None:
    args = _ns(action="add", name="fs", file=str(agent_yaml), command="echo")
    assert mcp_cmd.cmd_mcp(args) == 0
    # Second add with the same name fails and leaves a single entry.
    assert mcp_cmd.cmd_mcp(args) == 1
    raw = yaml.safe_load(agent_yaml.read_text())
    assert len(raw["mcp_servers"]) == 1


# -------------------------------------------------------------------------- list


def test_list_empty(agent_yaml: Path, capsys: pytest.CaptureFixture) -> None:
    rc = mcp_cmd.cmd_mcp(_ns(action="list", file=str(agent_yaml)))
    assert rc == 0
    # Plain stdout carries no rows when there are no servers.
    assert capsys.readouterr().out.strip() == ""


def test_list_shows_servers(agent_yaml: Path, capsys: pytest.CaptureFixture) -> None:
    mcp_cmd.cmd_mcp(_ns(action="add", name="fs", file=str(agent_yaml), command="echo"))
    capsys.readouterr()  # drop the add output
    rc = mcp_cmd.cmd_mcp(_ns(action="list", file=str(agent_yaml)))
    assert rc == 0
    out = capsys.readouterr().out
    # The data line (stdout, tab-separated) names the server.
    assert "fs\t" in out
    assert "echo" in out


def test_list_no_agent_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # cwd with no agent.yaml anywhere upward is handled cleanly, not a crash.
    monkeypatch.chdir(tmp_path)
    rc = mcp_cmd.cmd_mcp(_ns(action="list"))
    assert rc == 1


# ------------------------------------------------------------------------ remove


def test_remove_round_trips(agent_yaml: Path) -> None:
    mcp_cmd.cmd_mcp(_ns(action="add", name="fs", file=str(agent_yaml), command="echo"))
    rc = mcp_cmd.cmd_mcp(_ns(action="remove", name="fs", file=str(agent_yaml)))
    assert rc == 0
    raw = yaml.safe_load(agent_yaml.read_text())
    # Empty mcp_servers is dropped entirely; the rest of the file survives.
    assert "mcp_servers" not in raw
    assert raw["name"] == "demo"


def test_remove_unknown(agent_yaml: Path) -> None:
    rc = mcp_cmd.cmd_mcp(_ns(action="remove", name="nope", file=str(agent_yaml)))
    assert rc == 1


# -------------------------------------------------------------------------- test


def test_test_connects_to_mock_server(
    agent_yaml: Path, capsys: pytest.CaptureFixture
) -> None:
    # Add the in-repo mock stdio MCP server, then really connect to it.
    add = mcp_cmd.cmd_mcp(
        _ns(
            action="add",
            name="mock",
            file=str(agent_yaml),
            command=sys.executable,
            arg=[_MOCK_SERVER],
        )
    )
    assert add == 0
    capsys.readouterr()
    rc = mcp_cmd.cmd_mcp(_ns(action="test", name="mock", file=str(agent_yaml)))
    assert rc == 0
    out = capsys.readouterr().out
    # The mock advertises echo + add → 2 tools, prefixed mock_.
    assert "mock\t2" in out


def test_test_bogus_command_fails_cleanly(
    agent_yaml: Path, capsys: pytest.CaptureFixture
) -> None:
    mcp_cmd.cmd_mcp(
        _ns(
            action="add",
            name="bad",
            file=str(agent_yaml),
            command="himmy-no-such-binary-xyz",
        )
    )
    capsys.readouterr()
    rc = mcp_cmd.cmd_mcp(_ns(action="test", name="bad", file=str(agent_yaml)))
    assert rc == 1
    err = capsys.readouterr().err
    assert "failed to connect" in err


def test_test_unknown_server(agent_yaml: Path) -> None:
    rc = mcp_cmd.cmd_mcp(_ns(action="test", name="ghost", file=str(agent_yaml)))
    assert rc == 1


# ------------------------------------------------------------------------ dispatch


def test_unknown_action(agent_yaml: Path) -> None:
    rc = mcp_cmd.cmd_mcp(_ns(action="frobnicate", file=str(agent_yaml)))
    assert rc == 2


def test_default_action_is_list(agent_yaml: Path) -> None:
    # action=None defaults to list.
    rc = mcp_cmd.cmd_mcp(_ns(action=None, file=str(agent_yaml)))
    assert rc == 0
