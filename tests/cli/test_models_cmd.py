"""T3d — the ``himmy models`` + ``himmy compare`` CLI over the shared compare seam.

Both commands drive the SAME :mod:`himmy.services.inference.compare` helpers Studio and
``/v1`` use, so this asserts the CLI surface end-to-end (catalog rendering, prov:model
lexing, the side-by-side compare, and the JSON modes). The deterministic ``stub`` provider
keeps it offline.
"""

from __future__ import annotations

import json

import pytest

from himmy.cli.__main__ import build_parser, main
from himmy.cli.models_cmd import _parse_target

# --------------------------------------------------------------- registration


def test_models_and_compare_are_registered_subcommands() -> None:
    from himmy.cli.__main__ import _command_names

    names = _command_names(build_parser())
    assert {"models", "compare"} <= names


# ------------------------------------------------------------ prov:model lexing


def test_parse_target_handles_a_colon_in_the_model() -> None:
    t = _parse_target("ollama:qwen2.5:3b-instruct")
    assert t is not None
    assert t.provider == "ollama"
    assert t.model == "qwen2.5:3b-instruct"  # only the FIRST colon splits


@pytest.mark.parametrize("bad", ["", "noseparator", ":model", "provider:", "  "])
def test_parse_target_rejects_malformed(bad: str) -> None:
    assert _parse_target(bad) is None


# ------------------------------------------------------------------ himmy models


def test_models_lists_providers(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "claude-cli" in out
    assert "ollama" in out


def test_models_json_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models", "--json"]) == 0
    out = capsys.readouterr().out
    catalog = json.loads(out)
    providers = {entry["provider"] for entry in catalog}
    assert {"ollama", "claude-cli"} <= providers


# ----------------------------------------------------------------- himmy compare


def test_compare_runs_each_target(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["compare", "-m", "stub:a", "-m", "stub:b", "say hello"])
    assert code == 0
    out = capsys.readouterr().out
    assert "stub:a" in out and "stub:b" in out
    # The stub echoes the prompt, so both cells render an output line.
    assert out.count("Acknowledged") == 2


def test_compare_accepts_a_comma_list(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["compare", "--models", "stub:a,stub:b", "hi"])
    assert code == 0
    out = capsys.readouterr().out
    assert "stub:a" in out and "stub:b" in out


def test_compare_json_includes_usage(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["compare", "-m", "stub:a", "say hi", "--json"])
    assert code == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "stub" and row["model"] == "a"
    assert row["ok"] is True
    assert isinstance(row["input_tokens"], int)


def test_compare_bad_target_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["compare", "-m", "noseparator", "hi"])
    assert code == 2
    assert "must be provider:model" in capsys.readouterr().err


def test_compare_without_a_target_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["compare", "hi"])
    assert code == 2
    assert "at least one model is required" in capsys.readouterr().err


def test_compare_reports_a_bad_provider_per_cell(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["compare", "-m", "stub:a", "-m", "no-such-provider:x", "hi"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ERROR" in out  # the bad cell renders an error, the good one an output
    assert "Acknowledged" in out
