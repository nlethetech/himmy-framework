"""Tier 2: YAML skill authoring, discovery, and project-overlays-builtin precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.skills import (
    BUILTIN_SKILLS,
    SkillFileError,
    build_skill_registry,
    load_skill_dir,
    load_skill_file,
)


def test_load_skill_file_defaults_name_to_stem(tmp_path: Path) -> None:
    f = tmp_path / "web_research.yaml"
    f.write_text("description: from yaml\ntool_packs: [web]\n")
    skill = load_skill_file(f)
    assert skill.name == "web_research"  # defaulted from the file stem
    assert skill.tool_packs == ["web"]


def test_load_skill_file_rejects_bad_field(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("name: bad\nnonsense: 1\n")  # extra="forbid" on the model
    with pytest.raises(SkillFileError):
        load_skill_file(f)


def test_load_skill_dir_loads_all_yaml(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: a\n")
    (tmp_path / "b.yml").write_text("name: b\n")
    (tmp_path / "ignore.txt").write_text("not a skill")
    names = sorted(s.name for s in load_skill_dir(tmp_path))
    assert names == ["a", "b"]


def test_build_registry_overlays_project_over_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    # Shadow a built-in by name, plus add a brand-new project skill.
    (skills_dir / "web_research.yaml").write_text(
        "name: web_research\ndescription: PROJECT OVERRIDE\ntool_packs: [web]\n"
    )
    (skills_dir / "my_custom.yaml").write_text("name: my_custom\ntool_packs: [utils]\n")
    monkeypatch.chdir(tmp_path)
    registry = build_skill_registry()
    assert registry.get("web_research").description == "PROJECT OVERRIDE"  # shadowed
    assert registry.get("my_custom") is not None  # discovered
    assert "summarize" in registry  # built-ins still present


def test_build_registry_without_project_dir_is_just_builtins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no ./skills here
    registry = build_skill_registry()
    assert set(registry.names()) == set(BUILTIN_SKILLS)


def test_env_path_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ext = tmp_path / "shared_skills"
    ext.mkdir()
    (ext / "shared.yaml").write_text("name: shared\ntool_packs: [utils]\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_SKILLS_PATH", str(ext))
    registry = build_skill_registry()
    assert registry.get("shared") is not None
