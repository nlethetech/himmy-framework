"""Tests for the himmy CLI (himmy.cli). Everything runs offline against the stub."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.cli.__main__ import main
from himmy.cli.provider import ProviderError, build_inference_for
from himmy.services.inference.service import InferenceService


def test_provider_factory_stub() -> None:
    """The explicit stub provider builds an InferenceService offline."""
    assert isinstance(build_inference_for("stub"), InferenceService)


def test_provider_factory_default_is_offline() -> None:
    """provider=None delegates to build_inference (stub when no keys/extra)."""
    assert isinstance(build_inference_for(None), InferenceService)


def test_provider_factory_unknown_raises() -> None:
    """An unknown provider name is a clear ProviderError."""
    with pytest.raises(ProviderError):
        build_inference_for("nope")


def test_run_prints_output(capsys: pytest.CaptureFixture[str]) -> None:
    """`himmy run --provider stub` prints the deterministic stub reply."""
    code = main(["run", "--provider", "stub", "--name", "t", "-p", "hello there"])
    out = capsys.readouterr().out
    assert code == 0
    assert "hello there" in out  # stub echoes the last user message


def test_run_requires_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    """`himmy run` without a prompt exits non-zero with a message."""
    code = main(["run", "--provider", "stub", "--name", "t"])
    assert code == 2
    assert "prompt" in capsys.readouterr().err


def test_chat_single_message(capsys: pytest.CaptureFixture[str]) -> None:
    """`himmy chat --message` runs one turn and prints a reply."""
    code = main(["chat", "--provider", "stub", "--name", "t", "--message", "ping"])
    assert code == 0
    assert "ping" in capsys.readouterr().out


def test_doctor_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """`himmy doctor` reports a status table and exits 0."""
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert code == 0
    assert "himmy doctor" in out
    assert "optional extras" in out


def test_init_scaffolds_and_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`himmy init` writes the spec + tools, and the spec then runs end-to-end."""
    target = tmp_path / "demo"
    assert main(["init", str(target)]) == 0
    assert (target / "agent.yaml").exists()
    assert (target / "tools.py").exists()
    capsys.readouterr()  # drain

    code = main(
        ["run", "-f", str(target / "agent.yaml"), "--provider", "stub", "-p", "hi"]
    )
    assert code == 0


def test_init_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second `himmy init` without --force refuses to clobber existing files."""
    target = tmp_path / "demo"
    assert main(["init", str(target)]) == 0
    capsys.readouterr()
    code = main(["init", str(target)])
    assert code == 1
    assert "already exist" in capsys.readouterr().err
    # --force overwrites cleanly.
    assert main(["init", str(target), "--force"]) == 0


def test_tools_command_lists_packs(capsys: pytest.CaptureFixture[str]) -> None:
    """`himmy tools` lists the built-in packs and their tools."""
    code = main(["tools"])
    out = capsys.readouterr().out
    assert code == 0
    assert "web_search" in out
    assert "calculator" in out


def test_run_with_tool_packs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A spec declaring tool_packs registers them and runs end-to-end (stub)."""
    (tmp_path / "agent.yaml").write_text(
        "name: calc\ntool_packs: [utils]\ntools: [calculator]\n"
    )
    code = main(
        ["run", "-f", str(tmp_path / "agent.yaml"), "--provider", "stub", "-p", "2+2"]
    )
    assert code == 0


def test_run_structured_output_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A spec with an output_schema prints structured JSON from the stub."""
    (tmp_path / "agent.yaml").write_text(
        "name: extractor\n"
        "output_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    sentiment: {type: string}\n"
        "  required: [sentiment]\n"
    )
    code = main(
        ["run", "-f", str(tmp_path / "agent.yaml"), "--provider", "stub", "-p", "x"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "sentiment" in out
