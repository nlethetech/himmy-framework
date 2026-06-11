"""Tests for the frontier-CLI entry points: `himmy "question"` and the house agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.cli import commands
from himmy.cli.__main__ import build_parser, main
from himmy.cli.wizard import ProviderChoice


def test_command_names_introspection() -> None:
    """The dispatcher sees every registered subcommand (can't drift)."""
    from himmy.cli.__main__ import _command_names

    names = _command_names(build_parser())
    assert {"run", "chat", "new", "agents", "validate", "doctor"} <= names


def test_bare_words_become_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`himmy yo whats up` (no subcommand) answers via the run path."""
    monkeypatch.chdir(tmp_path)  # no agent.yaml anywhere above tmp
    monkeypatch.setattr(
        commands,
        "_house_spec",
        lambda: commands.AgentSpec(name="house", instructions=["echo"]),
    )
    code = main(["yo", "whats", "up"])
    assert code == 0
    assert "yo whats up" in capsys.readouterr().out


def test_single_token_typo_gets_did_you_mean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A near-miss of a real command is flagged, not silently asked to the model."""
    code = main(["validte"])
    assert code == 2
    err = capsys.readouterr().err
    assert "did you mean 'validate'" in err


def test_known_commands_still_dispatch(capsys: pytest.CaptureFixture[str]) -> None:
    """`himmy doctor` is never mistaken for a question."""
    assert main(["doctor"]) == 0
    assert "himmy doctor" in capsys.readouterr().out


def test_house_spec_uses_detected_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a real provider detected, the house agent gets the core keyless packs."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        lambda: [ProviderChoice(key="claude-cli", label="x", model="sonnet")],
    )
    spec = commands._house_spec()
    assert spec.provider == "claude-cli"
    assert spec.model == "sonnet"
    assert "web" in spec.tool_packs and "memory" in spec.tool_packs
    assert spec.memory is True


def test_house_spec_stub_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub-only machine: house agent stays minimal but functional."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        lambda: [ProviderChoice(key="stub", label="stub")],
    )
    spec = commands._house_spec()
    assert spec.provider is None  # auto → offline stub
    assert spec.tool_packs == ["utils"]
    assert spec.memory is False


def test_explicit_provider_keeps_adhoc_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`himmy run --provider stub` without a name never builds the house agent."""
    monkeypatch.chdir(tmp_path)
    called: list[bool] = []
    real = commands._house_spec
    monkeypatch.setattr(commands, "_house_spec", lambda: called.append(True) or real())
    assert main(["run", "--provider", "stub", "hello"]) == 0
    assert not called
