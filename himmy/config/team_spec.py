"""Declarative agent teams: define a multi-agent team in YAML, run it without code.

A ``team.yaml`` describes members (each a persona with its own tools, model, and
handoff/delegate edges) and the entry member. :func:`load_team_spec` reads it and
:func:`build_team` turns it into an :class:`~himmy.orchestrators.AgentTeam` plus a
:class:`~himmy.services.tools.registry.ToolRegistry` pre-loaded with every member's
toolkit packs / custom tools — ready to hand to a ``MultiAgentOrchestrator``::

    entry: triage
    members:
      - name: triage
        description: Route the request to the right specialist.
        handoffs: [researcher, writer]
      - name: researcher
        description: Gather facts with web search.
        tool_packs: [web]
        tools: [web_search, web_fetch]
        handoffs: [writer]
      - name: writer
        description: Write the final answer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel

from himmy.agents.personas.persona import Persona

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.orchestrators import AgentTeam
    from himmy.services.tools.registry import ToolRegistry


class TeamMemberSpec(BaseModel):
    """Declarative description of one team member."""

    name: str
    description: str = ""
    instructions: list[str] = []
    role: str | None = None
    model: str = "default"
    tools: list[str] = []
    tool_packs: list[str] = []
    tools_module: str | None = None
    handoffs: list[str] = []
    delegates: list[str] = []


class TeamSpec(BaseModel):
    """Declarative description of a team: members + the entry member."""

    members: list[TeamMemberSpec]
    entry: str


def load_team_spec(path: str | Path) -> TeamSpec:
    """Load a :class:`TeamSpec` from a YAML file."""
    raw = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"team spec {path} must be a YAML mapping")
    return TeamSpec.model_validate(raw)


def _persona_for(member: TeamMemberSpec) -> Persona:
    """Build a Persona for a member (role folded into metadata)."""
    metadata: dict[str, Any] = {}
    if member.role:
        metadata["role"] = member.role
    return Persona(
        name=member.name,
        description=member.description,
        instructions=list(member.instructions),
        metadata=metadata,
    )


def build_team(
    spec: TeamSpec,
    *,
    toolkit_config: Any = None,
    resolve_tools_module: Callable[[str], Callable[[Any], Any]] | None = None,
) -> tuple[AgentTeam, ToolRegistry]:
    """Build an :class:`AgentTeam` + a :class:`ToolRegistry` loaded with member tools.

    Each member's ``tool_packs`` are registered onto the shared registry (so its tools
    resolve at run time), and an optional ``tools_module`` is wired via
    ``resolve_tools_module`` (the CLI passes its dotted-path resolver). The returned
    registry is also where the orchestrator registers the synthetic handoff/delegate
    tools.
    """
    from himmy.orchestrators import AgentTeam, TeamMember
    from himmy.services.tools.registry import ToolRegistry
    from himmy.toolkit import ToolkitConfig, register_packs

    cfg = toolkit_config or ToolkitConfig.from_env()
    registry = ToolRegistry()
    seen_packs: set[str] = set()
    members: list[TeamMember] = []

    for member in spec.members:
        fresh_packs = [p for p in member.tool_packs if p not in seen_packs]
        if fresh_packs:
            register_packs(registry, fresh_packs, cfg)
            seen_packs.update(fresh_packs)
        if member.tools_module and resolve_tools_module is not None:
            resolve_tools_module(member.tools_module)(registry)
        members.append(
            TeamMember(
                name=member.name,
                persona=_persona_for(member),
                tools=list(member.tools),
                model_key=member.model,
                handoffs=list(member.handoffs),
                delegates=list(member.delegates),
            )
        )

    return AgentTeam(members=members, entry=spec.entry), registry


__all__ = ["TeamSpec", "TeamMemberSpec", "load_team_spec", "build_team"]
