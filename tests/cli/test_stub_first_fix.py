"""`model: default` resolves against the machine so a real backend beats the stub headline.

On a box where `himmy doctor` reports claude-cli/ollama `[ok]`, a spec that named no
provider and left `model: default` used to fall through to the offline stub — the misleading
`[stub:…]` headline. These tests pin the run-time resolution (:func:`_resolve_default_provider`)
and the classic/non-TTY scaffold stamp (:func:`_stamp_scaffold_provider`): a detected backend is
wired in, explicit config is never touched, and a stub-only machine keeps the honest fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from himmy.cli import commands
from himmy.cli.wizard import ProviderChoice


def _stub_detect(*choices: ProviderChoice):
    """A `detect_provider_choices` stand-in returning ``choices`` (stub appended if absent)."""
    result = list(choices)
    if not result or result[-1].key != "stub":
        result.append(ProviderChoice(key="stub", label="stub"))
    return lambda: result


def _run_args(spec_file: Path) -> argparse.Namespace:
    """Minimal `himmy run -f` namespace with no CLI provider/model override."""
    return argparse.Namespace(
        file=str(spec_file), name=None, instruction=None, provider=None, model=None
    )


# ------------------------------------------------------------ run-time resolution


def test_model_default_resolves_to_detected_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider-less `model: default` spec picks up the best detected backend at run time."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        _stub_detect(ProviderChoice(key="claude-cli", label="x", model="sonnet")),
    )
    spec_file = tmp_path / "agent.yaml"
    spec_file.write_text("name: a\nmodel: default\n")
    spec = commands._spec_from_args(_run_args(spec_file))
    assert spec.provider == "claude-cli"
    assert spec.model == "sonnet"


def test_detected_provider_without_model_keeps_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detected choice carrying no model leaves `model: default` (provider's own default)."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        _stub_detect(ProviderChoice(key="ollama", label="x", model=None)),
    )
    spec_file = tmp_path / "agent.yaml"
    spec_file.write_text("name: a\nmodel: default\n")
    spec = commands._spec_from_args(_run_args(spec_file))
    assert spec.provider == "ollama"
    assert spec.model == "default"


def test_nothing_detected_stays_on_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub-only machine: the spec is left untouched on the honest offline stub."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(wizard, "detect_provider_choices", _stub_detect())
    spec_file = tmp_path / "agent.yaml"
    spec_file.write_text("name: a\nmodel: default\n")
    spec = commands._spec_from_args(_run_args(spec_file))
    assert spec.provider is None
    assert spec.model == "default"


def test_explicit_spec_provider_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `provider:` in the YAML is never clobbered by the machine probe."""
    import himmy.cli.wizard as wizard

    # If the probe ran it would return claude-cli; the explicit stub must still win.
    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        _stub_detect(ProviderChoice(key="claude-cli", label="x", model="sonnet")),
    )
    spec_file = tmp_path / "agent.yaml"
    spec_file.write_text("name: a\nprovider: stub\nmodel: default\n")
    spec = commands._spec_from_args(_run_args(spec_file))
    assert spec.provider == "stub"


def test_explicit_spec_model_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned `model:` (not ``default``) short-circuits the probe entirely."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        _stub_detect(ProviderChoice(key="claude-cli", label="x", model="sonnet")),
    )
    spec_file = tmp_path / "agent.yaml"
    spec_file.write_text("name: a\nmodel: qwen2.5:7b-instruct\n")
    spec = commands._spec_from_args(_run_args(spec_file))
    assert spec.provider is None  # provider unset, but pinned model blocks resolution
    assert spec.model == "qwen2.5:7b-instruct"


def test_explicit_cli_override_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--provider` flag wins over the probe (and over any spec resolution)."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        _stub_detect(ProviderChoice(key="claude-cli", label="x", model="sonnet")),
    )
    spec_file = tmp_path / "agent.yaml"
    spec_file.write_text("name: a\nmodel: default\n")
    args = argparse.Namespace(
        file=str(spec_file),
        name=None,
        instruction=None,
        provider="ollama",
        model=None,
    )
    spec = commands._spec_from_args(args)
    assert spec.provider == "ollama"


# --------------------------------------------------- classic / non-TTY scaffold


def test_classic_scaffold_stamps_detected_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`himmy init --classic` stamps the detected backend into the written agent.yaml."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        _stub_detect(ProviderChoice(key="claude-cli", label="x", model="sonnet")),
    )
    args = argparse.Namespace(
        directory=str(tmp_path), template=None, team=None, classic=True, force=False
    )
    assert commands.cmd_init(args) == 0
    raw = yaml.safe_load((tmp_path / "agent.yaml").read_text())
    assert raw["provider"] == "claude-cli"
    assert raw["model"] == "sonnet"
    assert "wired claude-cli" in capsys.readouterr().err


def test_classic_scaffold_stub_leaves_default_with_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing detected: the scaffold keeps `model: default` + prints a clean note."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(wizard, "detect_provider_choices", _stub_detect())
    args = argparse.Namespace(
        directory=str(tmp_path), template=None, team=None, classic=True, force=False
    )
    assert commands.cmd_init(args) == 0
    raw = yaml.safe_load((tmp_path / "agent.yaml").read_text())
    assert "provider" not in raw  # commented line stays commented, never a fake provider
    assert raw["model"] == "default"
    assert "no real backend detected" in capsys.readouterr().err


def test_template_scaffold_is_not_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--template` starter already pins its own provider/model — never re-stamped."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        _stub_detect(ProviderChoice(key="claude-cli", label="x", model="sonnet")),
    )
    args = argparse.Namespace(
        directory=str(tmp_path),
        template="researcher",
        team=None,
        classic=False,
        force=False,
    )
    assert commands.cmd_init(args) == 0
    raw = yaml.safe_load((tmp_path / "agent.yaml").read_text())
    # The researcher template ships its own explicit ollama backend, untouched.
    assert raw["provider"] == "ollama"
