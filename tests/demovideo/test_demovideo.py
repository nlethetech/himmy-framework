"""demo-video: script model, scaffold, recorder plumbing, CLI, and the skill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from himmy.cli.__main__ import main
from himmy.core.errors import HimmyError
from himmy.demovideo import DemoScript, scaffold
from himmy.demovideo.recorder import (
    build_stitch_command,
    load_script,
    write_script_js,
)
from himmy.demovideo.scaffold import _STARTER_SCRIPT
from himmy.skills.builtin import BUILTIN_SKILLS

# --- the script model -------------------------------------------------------------


def test_starter_script_validates() -> None:
    script = DemoScript.from_dict(_STARTER_SCRIPT)
    assert script.chapter_ids() == ["boot", "entry01", "outro"]
    assert script.brand.prompt == "▲"


def test_unknown_step_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DemoScript.from_dict({"chapters": [{"id": "x", "steps": [{"do": "explode"}]}]})


def test_extra_fields_are_rejected() -> None:
    # Typos in script.json must fail loudly, not record a broken film.
    with pytest.raises(ValidationError):
        DemoScript.from_dict({"chapters": [{"id": "x", "stepz": []}]})


# --- scaffold ---------------------------------------------------------------------


def test_scaffold_writes_workspace(tmp_path: Path) -> None:
    written = scaffold(tmp_path / "demo")
    names = {p.name for p in written}
    assert {"player.html", "script.json", "README.md"} <= names
    # The scaffolded script must itself be valid.
    script = load_script(tmp_path / "demo")
    assert script.chapters


def test_scaffold_is_idempotent_and_never_clobbers(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    scaffold(target)
    edited = (target / "script.json").read_text() + "\n"
    (target / "script.json").write_text(edited)
    again = scaffold(target)
    assert (target / "script.json").read_text() == edited
    assert not any(p.name == "script.json" for p in again)


def test_player_ships_as_package_data() -> None:
    import himmy.demovideo as dv

    player = Path(dv.__file__).parent / "player.html"
    text = player.read_text(encoding="utf-8")
    assert "window.DEMO_SCRIPT" in text
    assert "__sceneDone" in text  # the recorder's completion signal


# --- recorder plumbing (no playwright/ffmpeg needed) --------------------------------


def test_write_script_js_serializes_script(tmp_path: Path) -> None:
    scaffold(tmp_path)
    script = load_script(tmp_path)
    js = write_script_js(tmp_path, script).read_text(encoding="utf-8")
    assert js.startswith("window.DEMO_SCRIPT = ")
    payload = json.loads(js.removeprefix("window.DEMO_SCRIPT = ").removesuffix(";\n"))
    assert [c["id"] for c in payload["chapters"]] == script.chapter_ids()


def test_build_stitch_command_orders_clips(tmp_path: Path) -> None:
    clips = [tmp_path / "01_boot.webm", tmp_path / "02_entry01.webm"]
    cmd = build_stitch_command(clips, tmp_path / "demo.mp4")
    assert cmd[0] == "ffmpeg"
    assert cmd.index(str(clips[0])) < cmd.index(str(clips[1]))
    assert "concat=n=2:v=1:a=0" in " ".join(cmd)
    assert str(tmp_path / "demo.mp4") == cmd[-1]


def test_load_script_missing_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(HimmyError, match="script.json"):
        load_script(tmp_path)


# --- CLI --------------------------------------------------------------------------


def test_cli_scaffolds_workspace(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["demo-video", str(tmp_path / "demo")]) == 0
    out = capsys.readouterr().out
    assert "script.json" in out
    assert "--render" in out  # tells the user the next step
    assert (tmp_path / "demo" / "player.html").exists()


# --- the skill --------------------------------------------------------------------


def test_cli_video_skill_is_registered_and_honest() -> None:
    skill = BUILTIN_SKILLS["cli_video"]
    assert "files" in skill.tool_packs
    joined = " ".join(skill.instructions)
    assert "NEVER fabricate" in joined  # the doctrine survives edits
    assert "himmy demo-video" in joined  # points at the real workflow
    assert "live side effects" in joined  # footage is read-only/paper mode only
    # `guardrails` is the NAMED registry (pii, injection, …) — free text breaks
    # resolve; behavioural rules belong in instructions.
    assert skill.guardrails == []


def test_cli_video_skill_resolves_into_an_agent() -> None:
    # The regression that found the guardrails bug: resolving must not raise.
    from himmy.config.agent_spec import AgentSpec

    spec = AgentSpec.model_validate({"name": "video-maker", "skills": ["cli_video"]})
    assert spec.skills == ["cli_video"]
