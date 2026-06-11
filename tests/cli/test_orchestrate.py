"""Tests for ``himmy orchestrate`` — multi-agent modes over the REAL orchestrators.

All offline (the zero-config stub): every mode routes a 2-member team.yaml through the
corresponding framework orchestrator and we assert both members actually participated,
that the synthesized result + per-agent contribution summary print, and that the input
validation (missing prompt/file, bad mode) and the ``--json`` shape behave.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from himmy.cli import orchestrate

_TEAM_YAML = """\
entry: planner
members:
  - name: planner
    description: Draft an initial answer to the question.
    handoffs: [writer]
  - name: writer
    description: Improve and finalize the answer.
"""


def _write_team(tmp_path: Path, body: str = _TEAM_YAML) -> str:
    path = tmp_path / "team.yaml"
    path.write_text(body)
    return str(path)


def _ns(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "mode": "hierarchical",
        "prompt": "What is 2+2?",
        "file": None,
        "provider": None,
        "model": None,
        "json": False,
        "max_rounds": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------- happy paths


def test_hierarchical_runs_real_orchestrator_both_members(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--mode hierarchical` routes the team through MultiAgentOrchestrator; the
    entry member runs and hands off to the writer, so BOTH participate."""
    args = _ns(file=_write_team(tmp_path), mode="hierarchical")
    rc = orchestrate.cmd_orchestrate(args)
    out = capsys.readouterr()
    assert rc == 0
    # The handoff route (stderr) names both members.
    assert "planner" in out.err
    assert "writer" in out.err
    # A synthesized answer prints on stdout.
    assert out.out.strip() != ""


def test_group_chat_runs_real_orchestrator_both_members(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--mode group_chat` runs GroupChatOrchestrator; round-robin gives every member
    a turn within max_rounds, so both speak."""
    args = _ns(file=_write_team(tmp_path), mode="group_chat", max_rounds=4)
    rc = orchestrate.cmd_orchestrate(args)
    out = capsys.readouterr()
    assert rc == 0
    assert "planner" in out.err and "writer" in out.err


def test_graph_mode_runs_stategraph_both_nodes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--mode graph` compiles a real StateGraph (each member a node) and runs both
    nodes in sequence to END."""
    args = _ns(file=_write_team(tmp_path), mode="graph")
    rc = orchestrate.cmd_orchestrate(args)
    out = capsys.readouterr()
    assert rc == 0
    # Both nodes appear in the route + contributions.
    assert "planner" in out.err and "writer" in out.err


def test_reflection_mode_drafts_then_reviews(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--mode reflection` drafts with the entry member and revises with the reviewer
    member via the real reflect() pass; both contribute."""
    args = _ns(file=_write_team(tmp_path), mode="reflection")
    rc = orchestrate.cmd_orchestrate(args)
    out = capsys.readouterr()
    assert rc == 0
    assert "planner" in out.err and "writer" in out.err


def test_json_output_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--json` emits a machine-readable record with mode/route/contributions."""
    args = _ns(file=_write_team(tmp_path), mode="graph", json=True)
    rc = orchestrate.cmd_orchestrate(args)
    out = capsys.readouterr()
    assert rc == 0
    payload = json.loads(out.out)
    assert payload["mode"] == "graph"
    assert payload["route"]  # non-empty
    agents = {c["agent"] for c in payload["contributions"]}
    assert {"planner", "writer"} <= agents
    # Transcript carries both members' turns.
    transcript_agents = {t["agent"] for t in payload["transcript"]}
    assert {"planner", "writer"} <= transcript_agents


# ---------------------------------------------------------------- edge cases


def test_missing_prompt_errors(tmp_path: Path) -> None:
    args = _ns(file=_write_team(tmp_path), prompt=None)
    assert orchestrate.cmd_orchestrate(args) == 2


def test_missing_file_errors() -> None:
    args = _ns(file=None)
    assert orchestrate.cmd_orchestrate(args) == 2


def test_unknown_mode_errors(tmp_path: Path) -> None:
    args = _ns(file=_write_team(tmp_path), mode="banana")
    assert orchestrate.cmd_orchestrate(args) == 2


def test_nonexistent_team_file_errors_cleanly(tmp_path: Path) -> None:
    """A missing team file fails with a clean exit code, not a traceback."""
    args = _ns(file=str(tmp_path / "nope.yaml"))
    rc = orchestrate.cmd_orchestrate(args)
    assert rc == 2


def test_single_member_reflection_uses_default_reviewer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A one-member team in reflection mode falls back to a default reviewer persona
    rather than crashing."""
    solo = "entry: solo\nmembers:\n  - name: solo\n    description: Answer it.\n"
    args = _ns(file=_write_team(tmp_path, solo), mode="reflection")
    rc = orchestrate.cmd_orchestrate(args)
    out = capsys.readouterr()
    assert rc == 0
    assert "solo" in out.err
    assert "reviewer" in out.err


# ----------------------------------------------- Finding 2: orchestrate enforces RBAC


def test_orchestrate_enforces_rbac_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`himmy orchestrate --role viewer` enforces RBAC on the team registry, so a
    viewer can't slip a gated tool by switching entry point to orchestrate."""
    seen: dict[str, Any] = {}
    import himmy.cli.rbac_cmd as rbac_cmd

    real = rbac_cmd.enforce_role_on_registry

    def _spy(registry: Any, *, role: Any = None, **kw: Any) -> Any:
        seen["role"] = role
        seen["called"] = True
        return real(registry, role=role, **kw)

    monkeypatch.setattr(rbac_cmd, "enforce_role_on_registry", _spy)
    args = _ns(file=_write_team(tmp_path), mode="graph", role="viewer")
    rc = orchestrate.cmd_orchestrate(args)
    assert rc == 0
    assert seen.get("called") is True
    assert seen["role"] == "viewer"


# ----------------------------------------------------------------- REPL hook


class _FakeRepl:
    def __init__(self, file: str | None) -> None:
        self._c = orchestrate.styles(None)  # plain styles (non-TTY)
        self.args = argparse.Namespace(file=file, provider=None, model=None)
        self.messages: list[str] = []

    def _eprint(self, msg: str) -> None:
        self.messages.append(msg)


def test_slash_orchestrate_usage_without_args(tmp_path: Path) -> None:
    repl = _FakeRepl(file=_write_team(tmp_path))
    orchestrate.slash_orchestrate(repl, [])
    assert any("usage" in m for m in repl.messages)


def test_slash_orchestrate_unknown_mode(tmp_path: Path) -> None:
    repl = _FakeRepl(file=_write_team(tmp_path))
    orchestrate.slash_orchestrate(repl, ["banana", "hello"])
    assert any("unknown mode" in m for m in repl.messages)


def test_slash_orchestrate_no_team_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no team.yaml here
    repl = _FakeRepl(file=None)
    orchestrate.slash_orchestrate(repl, ["hierarchical", "do it"])
    assert any("no team.yaml" in m for m in repl.messages)


def test_slash_orchestrate_runs_a_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repl = _FakeRepl(file=_write_team(tmp_path))
    orchestrate.slash_orchestrate(repl, ["graph", "answer", "this"])
    out = capsys.readouterr()
    # The run printed a route line + answer.
    assert "planner" in out.err and "writer" in out.err


# ------------------------------------------------- Finding 5: team-spec fallback

_AGENT_YAML = """\
name: helper
description: A single agent, NOT a team (no members/entry).
tool_packs: [utils]
"""


def test_is_team_spec_distinguishes_team_from_agent(tmp_path: Path) -> None:
    team = _write_team(tmp_path)
    agent = tmp_path / "agent.yaml"
    agent.write_text(_AGENT_YAML)
    assert orchestrate._is_team_spec(team) is True
    assert orchestrate._is_team_spec(str(agent)) is False
    assert orchestrate._is_team_spec(str(tmp_path / "nope.yaml")) is False


def test_slash_orchestrate_falls_back_to_team_yaml_when_file_is_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A chat started with -f agent.yaml must NOT mis-load the agent as a team —
    /orchestrate falls back to ./team.yaml when repl.args.file isn't a team spec."""
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent.yaml"
    agent.write_text(_AGENT_YAML)
    _write_team(tmp_path)  # a real team.yaml in cwd
    repl = _FakeRepl(file=str(agent))  # the chat's --file is the AGENT spec
    orchestrate.slash_orchestrate(repl, ["graph", "answer", "this"])
    out = capsys.readouterr()
    # The team.yaml members ran (proof we fell back), not a crash on the agent spec.
    assert "planner" in out.err and "writer" in out.err


def test_slash_orchestrate_agent_file_and_no_team_yaml_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-f agent.yaml and NO ./team.yaml → the no-team hint, not a mis-load."""
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent.yaml"
    agent.write_text(_AGENT_YAML)
    repl = _FakeRepl(file=str(agent))
    orchestrate.slash_orchestrate(repl, ["hierarchical", "do it"])
    assert any("no team.yaml" in m for m in repl.messages)
