"""Tests for the interactive `himmy init` wizard (himmy.cli.wizard)."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.cli import wizard
from himmy.cli.__main__ import main
from himmy.cli.wizard import ProviderChoice, run_wizard
from himmy.runtime import from_spec


def _scripted(answers: list[str]):
    """An input_fn that replays ``answers`` in order (fails loudly if exhausted)."""
    queue = list(answers)

    def _input(prompt: str) -> str:
        assert queue, f"wizard asked an unexpected extra question: {prompt!r}"
        return queue.pop(0)

    return _input


@pytest.fixture
def stub_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make provider detection deterministic: only the stub is 'installed'."""
    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        lambda: [ProviderChoice(key="stub", label="stub — offline")],
    )


def test_wizard_defaults_write_loadable_spec(
    tmp_path: Path, stub_only: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """All-default answers produce ONE agent.yaml that load_spec_file accepts."""
    answers = _scripted(
        [
            "",  # name → directory name
            "",  # description → default
            "",  # provider choice → default (stub)
            "",  # tool packs → web, utils
            "",  # memory → no
        ]
    )
    assert run_wizard(tmp_path, input_fn=answers) == 0

    dest = tmp_path / "agent.yaml"
    assert dest.exists()
    spec = from_spec.load_spec_file(str(dest))
    assert spec.name == tmp_path.resolve().name
    assert spec.provider == "stub"
    assert spec.tool_packs == ["web", "utils"]
    assert spec.memory is False
    # only the spec file is written — the wizard's output is minimal by design
    assert sorted(p.name for p in tmp_path.iterdir()) == ["agent.yaml"]
    assert f"wrote {dest}" in capsys.readouterr().out


def test_wizard_custom_answers(tmp_path: Path, stub_only: None) -> None:
    """Named agent, packs by number+name mix, memory on."""
    answers = _scripted(
        [
            "scout",  # name
            "Watches the news.",  # description
            "1",  # provider choice
            "news, 1",  # packs: name + number (1 == web)
            "y",  # memory
        ]
    )
    assert run_wizard(tmp_path, input_fn=answers) == 0
    spec = from_spec.load_spec_file(str(tmp_path / "agent.yaml"))
    assert spec.name == "scout"
    assert spec.description == "Watches the news."
    assert spec.tool_packs == ["news", "web"]
    assert spec.memory is True


def test_wizard_reasks_on_unknown_pack(tmp_path: Path, stub_only: None) -> None:
    """A typo'd pack name re-asks instead of writing a broken spec."""
    answers = _scripted(
        [
            "",  # name
            "",  # description
            "",  # provider
            "webz",  # packs: unknown → re-ask
            "none",  # packs: explicitly none
            "",  # memory
        ]
    )
    assert run_wizard(tmp_path, input_fn=answers) == 0
    spec = from_spec.load_spec_file(str(tmp_path / "agent.yaml"))
    assert spec.tool_packs == []


def test_wizard_abort_writes_nothing(tmp_path: Path, stub_only: None) -> None:
    """Ctrl-C anywhere aborts with 130 and leaves the directory untouched."""

    def _interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    assert run_wizard(tmp_path, input_fn=_interrupt) == 130
    assert not (tmp_path / "agent.yaml").exists()


def test_wizard_refuses_overwrite_unless_confirmed(
    tmp_path: Path, stub_only: None
) -> None:
    """An existing agent.yaml is only replaced after an explicit yes."""
    dest = tmp_path / "agent.yaml"
    dest.write_text("name: keep-me\n")

    answers = _scripted(["n"])  # overwrite? → no
    assert run_wizard(tmp_path, input_fn=answers) == 1
    assert dest.read_text() == "name: keep-me\n"

    answers = _scripted(["y", "", "", "", "", ""])  # overwrite? → yes, then defaults
    assert run_wizard(tmp_path, input_fn=answers) == 0
    assert "keep-me" not in dest.read_text()


def test_detect_provider_choices_stub_always_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing installed and no keys, the stub is the single (last) option."""
    monkeypatch.setattr(wizard.shutil, "which", lambda _name: None)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    choices = wizard.detect_provider_choices()
    assert [c.key for c in choices] == ["stub"]


def test_detect_provider_choices_prefers_claude_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `claude` binary on PATH is offered first, ahead of keys and the stub."""
    monkeypatch.setattr(
        wizard.shutil,
        "which",
        lambda name: "/usr/local/bin/claude" if name == "claude" else None,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    keys = [c.key for c in wizard.detect_provider_choices()]
    assert keys == ["claude-cli", "openai", "stub"]


def test_init_non_tty_keeps_classic_scaffold(tmp_path: Path) -> None:
    """Without a TTY (pytest), `himmy init` writes the classic example files."""
    target = tmp_path / "proj"
    assert main(["init", str(target)]) == 0
    assert (target / "agent.yaml").exists()
    assert (target / "tools.py").exists()


def test_init_routes_to_wizard_when_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a TTY is available, the default `himmy init` runs the wizard."""
    called: dict[str, object] = {}

    def _fake_wizard(target: Path, *, force: bool = False, input_fn=input) -> int:
        called["target"] = target
        called["force"] = force
        return 0

    monkeypatch.setattr(wizard, "wizard_available", lambda: True)
    monkeypatch.setattr(wizard, "run_wizard", _fake_wizard)
    assert main(["init", str(tmp_path / "proj")]) == 0
    assert called["target"] == tmp_path / "proj"

    # --classic must bypass the wizard even on a TTY
    called.clear()
    assert main(["init", "--classic", str(tmp_path / "classic")]) == 0
    assert not called
    assert (tmp_path / "classic" / "agent.yaml").exists()
