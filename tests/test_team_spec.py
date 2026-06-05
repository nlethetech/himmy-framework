"""Tests for the declarative team spec (himmy.config.team_spec)."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.config import TeamSpec, build_team, load_team_spec


def _write(tmp_path: Path) -> Path:
    (tmp_path / "team.yaml").write_text(
        "entry: triage\n"
        "members:\n"
        "  - name: triage\n"
        "    description: route it\n"
        "    handoffs: [writer]\n"
        "  - name: writer\n"
        "    description: write it\n"
        "    role: Writer\n"
    )
    return tmp_path / "team.yaml"


def test_load_team_spec(tmp_path: Path) -> None:
    spec = load_team_spec(_write(tmp_path))
    assert spec.entry == "triage"
    assert [m.name for m in spec.members] == ["triage", "writer"]
    assert spec.members[0].handoffs == ["writer"]


def test_build_team_wires_personas_and_edges(tmp_path: Path) -> None:
    team, registry = build_team(load_team_spec(_write(tmp_path)))
    assert team.entry == "triage"
    triage = team.require("triage")
    assert triage.handoffs == ["writer"]
    assert team.require("writer").persona.role == "Writer"


def test_build_team_registers_tool_packs(tmp_path: Path) -> None:
    (tmp_path / "team.yaml").write_text(
        "entry: r\n"
        "members:\n"
        "  - name: r\n"
        "    description: research\n"
        "    tool_packs: [utils]\n"
        "    tools: [calculator]\n"
    )
    _team, registry = build_team(load_team_spec(tmp_path / "team.yaml"))
    assert registry.get("calculator") is not None


def test_non_mapping_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_team_spec(tmp_path / "bad.yaml")


def test_team_spec_model_roundtrip() -> None:
    spec = TeamSpec.model_validate(
        {"entry": "a", "members": [{"name": "a", "delegates": ["b"]}, {"name": "b"}]}
    )
    assert spec.members[0].delegates == ["b"]
