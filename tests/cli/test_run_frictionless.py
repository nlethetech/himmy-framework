"""Tests for the composability layer of `himmy run`: positional prompt, piped
stdin, upward agent.yaml discovery, and errors landing on stderr."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from himmy.cli.__main__ import main

STUB_SPEC = """\
name: nearby
description: Discovered by walking up from cwd.
provider: stub
instructions:
  - Be brief.
"""


def test_run_positional_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    """`himmy run hello there` works without -p."""
    code = main(["run", "--provider", "stub", "--name", "t", "hello", "there"])
    assert code == 0
    assert "hello there" in capsys.readouterr().out


def test_run_piped_stdin_is_the_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no prompt args, piped stdin becomes the prompt."""
    monkeypatch.setattr("sys.stdin", io.StringIO("piped question"))
    code = main(["run", "--provider", "stub", "--name", "t"])
    assert code == 0
    assert "piped question" in capsys.readouterr().out


def test_run_piped_stdin_appends_to_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cat diff | himmy run "review"` → prompt + the piped content."""
    monkeypatch.setattr("sys.stdin", io.StringIO("the diff body"))
    code = main(["run", "--provider", "stub", "--name", "t", "review", "this"])
    assert code == 0
    out = capsys.readouterr().out
    assert "review this" in out
    assert "the diff body" in out


def test_run_no_prompt_anywhere_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    """No positional, no -p, no piped stdin → usage error on stderr."""
    code = main(["run", "--provider", "stub", "--name", "t"])
    assert code == 2
    assert "prompt" in capsys.readouterr().err


def test_run_discovers_agent_yaml_upward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without -f, the nearest agent.yaml above cwd is found and announced."""
    (tmp_path / "agent.yaml").write_text(STUB_SPEC)
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    code = main(["run", "hello"])
    assert code == 0
    captured = capsys.readouterr()
    assert f"using {tmp_path / 'agent.yaml'}" in captured.err


def test_run_explicit_file_beats_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """-f always wins over a discovered agent.yaml."""
    (tmp_path / "agent.yaml").write_text(STUB_SPEC)
    other = tmp_path / "other.yaml"
    other.write_text(STUB_SPEC.replace("nearby", "explicit"))
    monkeypatch.chdir(tmp_path)

    code = main(["run", "-f", str(other), "hello"])
    assert code == 0
    assert "using " not in capsys.readouterr().err


def test_run_named_agent_skips_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--name keeps the ad-hoc agent even when an agent.yaml is nearby."""
    (tmp_path / "agent.yaml").write_text(STUB_SPEC)
    monkeypatch.chdir(tmp_path)

    code = main(["run", "--provider", "stub", "--name", "adhoc", "hello"])
    assert code == 0
    assert "using " not in capsys.readouterr().err


def test_errors_go_to_stderr_not_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A HimmyError (e.g. missing spec file) must never land in piped stdout."""
    code = main(["run", "-f", str(tmp_path / "missing.yaml"), "-p", "hi"])
    assert code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "error:" not in captured.out
