"""Tests for `himmy new` (himmy.cli.new) — AI-drafted agent specs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from himmy.cli import new as new_mod
from himmy.cli.__main__ import main
from himmy.cli.wizard import ProviderChoice
from himmy.core import HimmyError
from himmy.runtime import from_spec

GOOD_DRAFT = {
    "name": "feed-watcher",
    "description": "Watches RSS feeds and summarizes new stories.",
    "role": "News Assistant",
    "instructions": ["Check the feeds.", "Summarize concisely."],
    "tool_packs": ["news", "web"],
    "memory": True,
}


@pytest.fixture
def claude_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend claude-cli is the best detected backend."""
    monkeypatch.setattr(
        new_mod,
        "detect_provider_choices",
        lambda: [ProviderChoice(key="claude-cli", label="claude", model="sonnet")],
        raising=False,
    )
    # cmd_new imports it from wizard at call time — patch the source module too.
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        lambda: [ProviderChoice(key="claude-cli", label="claude", model="sonnet")],
    )


def _patch_reply(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> list[dict]:
    """Stub the model call; returns the list of recorded (provider, model) calls."""
    calls: list[dict] = []
    queue = list(replies)

    def _fake_complete(provider: str, model: str | None, system: str, user: str) -> str:
        calls.append({"provider": provider, "model": model, "user": user})
        assert queue, "unexpected extra model call"
        return queue.pop(0)

    monkeypatch.setattr(new_mod, "_complete", _fake_complete)
    return calls


def test_new_writes_validated_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claude_detected: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clean draft is sanitized, validated, and written where -o points."""
    calls = _patch_reply(monkeypatch, [json.dumps(GOOD_DRAFT)])
    dest = tmp_path / "agent.yaml"

    assert main(["new", "watch", "my", "feeds", "-o", str(dest)]) == 0
    assert calls[0]["provider"] == "claude-cli"
    assert calls[0]["model"] == "sonnet"

    spec = from_spec.load_spec_file(str(dest))
    assert spec.name == "feed-watcher"
    assert spec.tool_packs == ["news", "web"]
    assert spec.memory is True
    # provider is pinned to the detected backend, never model-chosen
    assert spec.provider == "claude-cli"
    assert f"wrote {dest}" in capsys.readouterr().out


def test_new_dry_run_prints_yaml_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claude_detected: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run prints YAML to stdout and writes nothing."""
    _patch_reply(monkeypatch, [json.dumps(GOOD_DRAFT)])
    monkeypatch.chdir(tmp_path)

    assert main(["new", "watch feeds", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "name: feed-watcher" in out
    assert not (tmp_path / "agent.yaml").exists()


def test_new_strips_hallucinated_keys_and_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claude_detected: None
) -> None:
    """Unknown keys, unknown packs, and a model-chosen provider are dropped."""
    draft = dict(GOOD_DRAFT)
    draft["provider"] = "openai"  # model tries to pick — must be overridden
    draft["tool_packs"] = ["news", "hackernews", "web"]  # hackernews isn't real
    draft["dangerous_setting"] = True  # not a spec key
    _patch_reply(monkeypatch, [json.dumps(draft)])
    dest = tmp_path / "agent.yaml"

    assert main(["new", "watch feeds", "-o", str(dest), "--yes"]) == 0
    spec = from_spec.load_spec_file(str(dest))
    assert spec.provider == "claude-cli"
    assert spec.tool_packs == ["news", "web"]
    assert "dangerous_setting" not in dest.read_text()


def test_new_retries_once_on_bad_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claude_detected: None
) -> None:
    """A non-JSON first reply triggers exactly one corrective retry."""
    calls = _patch_reply(
        monkeypatch, ["Sure! Here is your agent:", json.dumps(GOOD_DRAFT)]
    )
    dest = tmp_path / "agent.yaml"
    assert main(["new", "watch feeds", "-o", str(dest), "--yes"]) == 0
    assert len(calls) == 2
    assert "previous reply was invalid" in calls[1]["user"]
    assert dest.exists()


def test_new_fails_cleanly_when_drafts_stay_bad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claude_detected: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two bad replies → exit 1 with a clear error, nothing written."""
    _patch_reply(monkeypatch, ["nope", "still nope"])
    dest = tmp_path / "agent.yaml"
    assert main(["new", "watch feeds", "-o", str(dest)]) == 1
    assert "could not draft a valid spec" in capsys.readouterr().err
    assert not dest.exists()


def test_new_refuses_stub_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no real model on the machine, `new` points at doctor instead."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        lambda: [ProviderChoice(key="stub", label="stub")],
    )
    assert main(["new", "anything"]) == 1
    assert "himmy doctor" in capsys.readouterr().err


def test_new_refuses_silent_overwrite_non_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claude_detected: None
) -> None:
    """Existing output + no TTY + no --yes → refuse, keep the file."""
    _patch_reply(monkeypatch, [json.dumps(GOOD_DRAFT)])
    dest = tmp_path / "agent.yaml"
    dest.write_text("name: keep-me\n")
    assert main(["new", "watch feeds", "-o", str(dest)]) == 1
    assert dest.read_text() == "name: keep-me\n"


def test_extract_json_tolerates_fences() -> None:
    """```json fenced replies still parse."""
    fenced = '```json\n{"name": "x"}\n```'
    assert new_mod._extract_json(fenced) == {"name": "x"}
    with pytest.raises(HimmyError):
        new_mod._extract_json("no json here")
