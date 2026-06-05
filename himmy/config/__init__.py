"""Declarative configuration: load agents from files instead of hand-wiring code."""

from __future__ import annotations

from himmy.config.agent_spec import AgentSpec, load_agent_spec
from himmy.config.team_spec import (
    TeamMemberSpec,
    TeamSpec,
    build_team,
    load_team_spec,
)

__all__ = [
    "AgentSpec",
    "load_agent_spec",
    "TeamSpec",
    "TeamMemberSpec",
    "load_team_spec",
    "build_team",
]
