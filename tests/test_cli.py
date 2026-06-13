"""Tests for the himmy CLI (himmy.cli). Everything runs offline against the stub."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.cli.__main__ import main
from himmy.cli.provider import (
    PROVIDERS,
    ProviderError,
    build_inference_for,
    build_manager_for,
)
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


def test_openrouter_in_provider_choices() -> None:
    """`openrouter` is a first-class --provider/agent.yaml choice."""
    assert "openrouter" in PROVIDERS


def test_provider_factory_openrouter_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without OPENROUTER_API_KEY, openrouter raises a clear, named ProviderError."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError) as exc:
        build_inference_for("openrouter")
    assert "OPENROUTER_API_KEY" in str(exc.value)


def test_provider_factory_openrouter_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a key set, openrouter builds an InferenceService wired for OpenRouter.

    No network: we only assert the manager is configured with the OpenRouter
    base_url, the provided key, the default model, and the `openrouter` name.
    OpenRouter now runs through the direct OpenAI-compatible manager (no pydantic-ai),
    so the model id passes through verbatim with no ``openai:`` prefix.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    # Must NOT require OPENAI_API_KEY to be set as well.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert isinstance(build_inference_for("openrouter"), InferenceService)

    manager = build_manager_for("openrouter")
    assert manager.provider_name == "openrouter"
    assert manager._base_url == "https://openrouter.ai/api/v1"
    assert manager._api_key == "sk-or-test"
    # The model id passes through verbatim (no provider-prefix rewriting).
    assert manager.resolve("default") == "mistralai/mistral-small-3.2-24b-instruct"


def test_provider_factory_openrouter_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit --model overrides the OpenRouter default, passed through verbatim."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    manager = build_manager_for("openrouter", "anthropic/claude-3.5-sonnet")
    assert manager.resolve("default") == "anthropic/claude-3.5-sonnet"
    assert manager._api_key == "sk-or-test"


def test_openai_compatible_in_provider_choices() -> None:
    """`openai-compatible` is a first-class --provider/agent.yaml choice."""
    assert "openai-compatible" in PROVIDERS


def test_provider_factory_openai_compatible_missing_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a base_url env, openai-compatible raises a clear ProviderError."""
    monkeypatch.delenv("HIMMY_OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    with pytest.raises(ProviderError) as exc:
        build_manager_for("openai-compatible", "some/model")
    assert "base_url" in str(exc.value)


def test_provider_factory_openai_compatible_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai-compatible requires an explicit model (no silent default)."""
    monkeypatch.setenv("HIMMY_OPENAI_COMPAT_BASE_URL", "https://api.groq.com/openai/v1")
    with pytest.raises(ProviderError) as exc:
        build_manager_for("openai-compatible")
    assert "--model" in str(exc.value)


def test_provider_factory_openai_compatible_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai-compatible raises a clear, key-redacting error when no key is set."""
    monkeypatch.setenv(
        "HIMMY_OPENAI_COMPAT_BASE_URL", "https://api.together.xyz/v1"
    )
    for name in (
        "HIMMY_OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ProviderError) as exc:
        build_manager_for("openai-compatible", "meta-llama/Llama-3-70b")
    # The error names the key env var, never echoes a (missing) secret value.
    assert "HIMMY_OPENAI_COMPAT_API_KEY" in str(exc.value)


def test_provider_factory_openai_compatible_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With base_url + key + model, openai-compatible builds a verbatim-passthrough mgr.

    Covers the Sarvam/Groq/Together shape: one provider, custom endpoint, model id
    passed through unchanged, key sourced from the secrets layer.
    """
    monkeypatch.setenv("HIMMY_OPENAI_COMPAT_BASE_URL", "https://api.sarvam.ai/v1")
    monkeypatch.setenv("HIMMY_OPENAI_COMPAT_API_KEY", "sk-compat-test")
    manager = build_manager_for("openai-compatible", "sarvam-m")

    assert manager.provider_name == "openai-compatible"
    assert manager._base_url == "https://api.sarvam.ai/v1"
    assert manager._api_key == "sk-compat-test"
    assert manager.resolve("default") == "sarvam-m"


def test_provider_factory_openrouter_attribution_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter's optional HTTP-Referer / X-Title are forwarded when set, not required."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://himmy.example")
    monkeypatch.setenv("OPENROUTER_X_TITLE", "Himmy Agent")
    manager = build_manager_for("openrouter")
    assert manager._default_headers == {
        "HTTP-Referer": "https://himmy.example",
        "X-Title": "Himmy Agent",
    }

    # Unset → no headers (attribution is optional, never required).
    monkeypatch.delenv("OPENROUTER_HTTP_REFERER", raising=False)
    monkeypatch.delenv("OPENROUTER_X_TITLE", raising=False)
    assert build_manager_for("openrouter")._default_headers == {}


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


def test_team_command_routes_and_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`himmy team` runs a team.yaml, routing a prompt through a handoff."""
    (tmp_path / "team.yaml").write_text(
        "entry: triage\n"
        "members:\n"
        "  - name: triage\n"
        "    description: route\n"
        "    handoffs: [writer]\n"
        "  - name: writer\n"
        "    description: write\n"
    )
    code = main(
        ["team", "-f", str(tmp_path / "team.yaml"), "--provider", "stub", "-p", "hi"]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "triage → writer" in err  # the routing trail


def test_init_team_scaffold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`himmy init --team` writes a team.yaml."""
    target = tmp_path / "t"
    assert main(["init", "--team", str(target)]) == 0
    assert (target / "team.yaml").exists()
    assert not (target / "agent.yaml").exists()


def test_run_trace_prints_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`himmy run --trace` prints a timeline and saves it to .himmy/trace.db."""
    monkeypatch.chdir(tmp_path)
    code = main(["run", "--provider", "stub", "--name", "t", "-p", "hi", "--trace"])
    err = capsys.readouterr().err
    assert code == 0
    assert "--- trace ---" in err
    assert "run started" in err
    assert (tmp_path / ".himmy" / "trace.db").exists()
    # `himmy trace` then lists the saved run.
    assert main(["trace"]) == 0
    assert "recent runs" in capsys.readouterr().out


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
