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

from pydantic import BaseModel, ConfigDict

from himmy.agents.personas.persona import Persona
from himmy.config.agent_spec import parse_spec_yaml, validate_spec
from himmy.config.mcp_spec import MCPServerConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.orchestrators import AgentTeam
    from himmy.services.tools.registry import ToolRegistry


class TeamMemberSpec(BaseModel):
    """Declarative description of one team member."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    instructions: list[str] = []
    role: str | None = None
    # per member: claude-cli | ollama | pydantic-ai | openrouter | stub
    provider: str | None = None
    model: str = "default"
    tools: list[str] = []
    tool_packs: list[str] = []
    tools_module: str | None = None
    handoffs: list[str] = []
    delegates: list[str] = []


def dispatch_key_for(provider: str | None, model: str) -> str:
    """The model_key a member/graph-node routes on: ``<provider>:<model>`` or plain model.

    This is the key :func:`build_team_inference` registers each backend under, so every
    execution path (multi-agent, group-chat, AND the linear graph runner) must build its
    per-step ``LLMConfig.model_key`` from here — otherwise a plain model name misses the
    ``<provider>:<model>`` key and silently falls through to the stub default.
    """
    return f"{provider}:{model}" if provider else model


def _dispatch_key(member: TeamMemberSpec) -> str:
    """The model_key a member routes on (see :func:`dispatch_key_for`)."""
    return dispatch_key_for(member.provider, member.model)


class TeamSpec(BaseModel):
    """Declarative description of a team: members + the entry member.

    ``extra="forbid"`` so a typo'd top-level field fails loudly rather than being
    silently dropped (see :class:`AgentSpec`).
    """

    model_config = ConfigDict(extra="forbid")

    members: list[TeamMemberSpec]
    entry: str
    mcp_servers: list[MCPServerConfig] = []  # stdio MCP servers shared by the team


def load_team_spec(path: str | Path) -> TeamSpec:
    """Load a :class:`TeamSpec` from a YAML file.

    Malformed YAML or a schema error is re-raised as a :class:`HimmyError` naming
    the file, not a raw parser/pydantic traceback.
    """
    spec_path = Path(path).expanduser()
    raw = parse_spec_yaml(spec_path, kind="team spec")
    return validate_spec(TeamSpec, raw, spec_path, kind="team spec")


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
                model_key=_dispatch_key(member),
                handoffs=list(member.handoffs),
                delegates=list(member.delegates),
            )
        )

    return AgentTeam(members=members, entry=spec.entry), registry


def build_team_inference(
    spec: TeamSpec,
    *,
    default_provider: str | None = None,
    default_model: str | None = None,
) -> Any:
    """Build the team's :class:`InferenceService`, mixing providers across members.

    When a member declares its own ``provider``, the team runs on a
    :class:`~himmy.services.inference.multi_provider.MultiProviderClientManager` that
    dispatches each member's ``model_key`` to its own backend (e.g. a Claude-CLI brain +
    local Ollama workers). When no member sets a provider, a single backend is used (the
    CLI ``--provider``/``--model``, or the framework default).
    """
    from himmy.cli.provider import build_inference_for, build_manager_for
    from himmy.services.inference.multi_provider import MultiProviderClientManager
    from himmy.services.inference.service import InferenceService

    if not any(m.provider for m in spec.members):
        return build_inference_for(default_provider, default_model)

    managers: dict[str, Any] = {}
    for member in spec.members:
        if member.provider:
            key = _dispatch_key(member)
            if key not in managers:
                model = None if member.model == "default" else member.model
                managers[key] = build_manager_for(member.provider, model)
    managers["default"] = build_manager_for(default_provider, default_model)
    return InferenceService(MultiProviderClientManager(managers, default_key="default"))


__all__ = [
    "TeamSpec",
    "TeamMemberSpec",
    "load_team_spec",
    "build_team",
    "build_team_inference",
]
