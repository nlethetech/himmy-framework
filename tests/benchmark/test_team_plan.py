"""Unit tests for the benchmark team-spec parser (himmy.benchmark.team)."""

from __future__ import annotations

import pytest

from himmy.benchmark.team import (
    GROUP_CHAT_SELECTORS,
    TEAM_MODES,
    build_team_plan,
)


def _handoff_spec() -> dict:
    return {
        "mode": "handoff",
        "entry": "router",
        "members": [
            {"name": "router", "handoffs": ["math"], "persona": {"description": "r"}},
            {"name": "math", "tools": ["calculator"], "persona": {"description": "m"}},
        ],
    }


def test_builds_handoff_team() -> None:
    plan = build_team_plan(_handoff_spec())
    assert plan.mode == "handoff"
    assert plan.team.entry == "router"
    assert [m.name for m in plan.team.members] == ["router", "math"]
    assert plan.team.require("router").handoffs == ["math"]
    assert plan.tool_names == ["calculator"]


def test_defaults_entry_to_first_member() -> None:
    spec = {"members": [{"name": "solo", "persona": {}}]}
    plan = build_team_plan(spec)
    assert plan.mode == "handoff"  # default mode
    assert plan.team.entry == "solo"


def test_builds_group_chat_with_selector_and_speakers() -> None:
    spec = {
        "mode": "group_chat",
        "selector": "round_robin",
        "max_rounds": 3,
        "speakers": ["a", "b"],
        "members": [
            {"name": "a", "persona": {}},
            {"name": "b", "persona": {}},
        ],
    }
    plan = build_team_plan(spec)
    assert plan.mode == "group_chat"
    assert plan.selector == "round_robin"
    assert plan.max_rounds == 3
    assert plan.speakers == ["a", "b"]


def test_unknown_mode_raises() -> None:
    spec = _handoff_spec()
    spec["mode"] = "telepathy"
    with pytest.raises(ValueError, match="unknown team mode"):
        build_team_plan(spec)


def test_empty_members_raises() -> None:
    with pytest.raises(ValueError, match="non-empty 'members'"):
        build_team_plan({"members": []})


def test_unknown_entry_raises() -> None:
    spec = _handoff_spec()
    spec["entry"] = "ghost"
    with pytest.raises(ValueError, match="entry .* is not a declared member"):
        build_team_plan(spec)


def test_edge_to_unknown_member_raises() -> None:
    spec = _handoff_spec()
    spec["members"][0]["handoffs"] = ["ghost"]
    with pytest.raises(ValueError, match="unknown peer 'ghost'"):
        build_team_plan(spec)


def test_delegate_to_unknown_member_raises() -> None:
    spec = _handoff_spec()
    spec["members"][0]["delegates"] = ["ghost"]
    with pytest.raises(ValueError, match="unknown peer 'ghost'"):
        build_team_plan(spec)


def test_unknown_group_chat_selector_raises() -> None:
    spec = {
        "mode": "group_chat",
        "selector": "telepathy",
        "members": [{"name": "a", "persona": {}}],
    }
    with pytest.raises(ValueError, match="unknown group_chat selector"):
        build_team_plan(spec)


def test_unknown_speaker_raises() -> None:
    spec = {
        "mode": "group_chat",
        "speakers": ["ghost"],
        "members": [{"name": "a", "persona": {}}],
    }
    with pytest.raises(ValueError, match="speaker 'ghost' is not a declared member"):
        build_team_plan(spec)


def test_constants_exposed() -> None:
    assert "handoff" in TEAM_MODES
    assert "group_chat" in TEAM_MODES
    assert "round_robin" in GROUP_CHAT_SELECTORS
    assert "llm" in GROUP_CHAT_SELECTORS
