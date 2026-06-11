"""Tests for `himmy agents` (spec listing) and `himmy validate` (spec linting)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from himmy.cli.__main__ import main

GOOD = """\
name: scout
description: Watches things.
provider: stub
tool_packs: [web, utils]
instructions:
  - Be brief.
"""

TEAM = """\
entry: boss
members:
  - name: boss
    role: orchestrator
  - name: worker
    role: specialist
"""


def test_agents_lists_agent_team_and_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """All three kinds show up: agent with caps, team with members, broken with reason."""
    (tmp_path / "agent.yaml").write_text(GOOD)
    (tmp_path / "crew.team.yaml").write_text(TEAM)
    (tmp_path / "bad.agent.yaml").write_text("just a string")
    # a loose team file (Studio-style, content-classified) and unrelated yaml noise
    (tmp_path / "default-team.yaml").write_text(TEAM)
    (tmp_path / "docker-compose.yaml").write_text("services:\n  db:\n    image: pg\n")
    monkeypatch.chdir(tmp_path)

    assert main(["agents"]) == 0
    out = capsys.readouterr().out
    assert "scout" in out and "web, utils" in out
    assert "2 member(s)" in out
    assert "BROKEN" in out and "bad.agent.yaml" in out
    assert "default-team.yaml" in out
    assert "docker-compose" not in out


def test_agents_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--json emits a machine-readable listing."""
    (tmp_path / "agent.yaml").write_text(GOOD)
    assert main(["agents", str(tmp_path), "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert entries[0]["kind"] == "agent"
    assert entries[0]["name"] == "scout"


def test_agents_empty_dir_points_at_creation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty directory suggests init/new instead of printing nothing."""
    assert main(["agents", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "no agent specs" in captured.out
    assert "himmy init" in captured.err


def test_validate_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A clean spec validates with exit 0."""
    spec = tmp_path / "agent.yaml"
    spec.write_text(GOOD)
    assert main(["validate", str(spec)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_did_you_mean_key_and_pack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Typo'd top-level keys and tool packs get did-you-mean hints, exit 1."""
    spec = tmp_path / "agent.yaml"
    spec.write_text("name: x\ndescription: y\ntool_pack: [web]\ntool_packs: [webz]\n")
    assert main(["validate", str(spec)]) == 1
    out = capsys.readouterr().out
    assert "unknown key 'tool_pack'" in out
    assert "did you mean 'tool_packs'" in out
    assert "unknown tool pack 'webz'" in out
    assert "did you mean 'web'" in out


def test_validate_unknown_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = tmp_path / "agent.yaml"
    spec.write_text("name: x\nprovider: claude\n")
    assert main(["validate", str(spec)]) == 1
    out = capsys.readouterr().out
    assert "unknown provider 'claude'" in out
    assert "did you mean 'claude-cli'" in out


def test_validate_discovers_nearest_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bare `himmy validate` walks upward like `himmy run` does."""
    (tmp_path / "agent.yaml").write_text(GOOD)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert main(["validate"]) == 0
    assert str(tmp_path / "agent.yaml") in capsys.readouterr().out


def test_validate_no_spec_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["validate"]) == 2
    assert "no agent.yaml" in capsys.readouterr().err


def test_validate_bad_yaml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "agent.yaml"
    spec.write_text("name: [unclosed\n")
    assert main(["validate", str(spec)]) == 1
    assert "not valid YAML" in capsys.readouterr().out
