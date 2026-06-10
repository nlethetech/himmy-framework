"""Build a multi-agent team (and pick its orchestrator) from a benchmark task spec.

A benchmark task may declare a small ``team`` instead of running as a single agent.
The declarative shape mirrors :class:`~himmy.orchestrators.AgentTeam` /
:class:`~himmy.orchestrators.TeamMember` but stays plain YAML — so a suite author writes
personas, tool sets, and the collaboration topology (handoff / delegation / group chat)
without importing any runtime types.

This module is the parser + orchestrator selector; the runner owns the registry/runtime
wiring and the run itself. Heavy imports (orchestrators, personas) are deferred into the
function bodies to match the rest of :mod:`himmy.benchmark` (no import cycles).

YAML ``team`` block::

    team:
      mode: handoff            # handoff (default) | group_chat
      entry: router            # required for handoff; the first speaker for group_chat
      members:
        - name: router
          persona: { description: "...", instructions: ["..."] }
          tools: [calculator]          # tool names (from the task's packs)
          handoffs: [math, weather]    # binds transfer_to_math / transfer_to_weather
          delegates: [worker]          # binds ask_worker
          model_key: default
      # group_chat only:
      selector: round_robin    # round_robin (default) | llm
      max_rounds: 4
      speakers: [a, b]         # optional speaking roster (default: all members)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.orchestrators import AgentTeam

#: Recognized collaboration modes a task's ``team.mode`` may name.
TEAM_MODES = frozenset({"handoff", "group_chat"})
#: Recognized group-chat selectors.
GROUP_CHAT_SELECTORS = frozenset({"round_robin", "llm"})


@dataclass(frozen=True)
class TeamPlan:
    """A parsed team task: the built team plus how to orchestrate it.

    ``mode`` is ``"handoff"`` (a :class:`MultiAgentOrchestrator`, covering both peer
    handoff and supervisor delegation) or ``"group_chat"`` (a
    :class:`GroupChatOrchestrator`). The remaining fields configure the chosen
    orchestrator; ``tool_names`` is every tool any member references (so the runner
    knows which packs' tools must be present before wiring).
    """

    team: AgentTeam
    mode: str
    tool_names: list[str] = field(default_factory=list)
    selector: str = "round_robin"
    max_rounds: int = 8
    speakers: list[str] | None = None


def build_team_plan(spec: dict[str, Any]) -> TeamPlan:
    """Parse a task's ``team`` block into a :class:`TeamPlan` (fail-loud on a bad spec).

    Builds the :class:`~himmy.orchestrators.AgentTeam` and validates the topology:
    a mode in :data:`TEAM_MODES`, a non-empty member list, a known ``entry`` member, and
    handoff/delegate edges that point at real members. A malformed team raises
    ``ValueError`` here (the runner records it as a trial error, not a crash).
    """
    from himmy.agents.personas.persona import Persona
    from himmy.orchestrators import AgentTeam, TeamMember

    mode = str(spec.get("mode", "handoff"))
    if mode not in TEAM_MODES:
        raise ValueError(
            f"unknown team mode {mode!r}; expected one of {sorted(TEAM_MODES)}"
        )
    raw_members = spec.get("members") or []
    if not raw_members:
        raise ValueError("team requires a non-empty 'members' list")

    members: list[TeamMember] = []
    tool_names: list[str] = []
    for raw in raw_members:
        name = str(raw["name"])
        persona_spec = dict(raw.get("persona") or {})
        persona = Persona(
            name=name,
            description=str(persona_spec.get("description", "")),
            instructions=[str(i) for i in persona_spec.get("instructions", [])],
        )
        tools = [str(t) for t in raw.get("tools", [])]
        for tool in tools:
            if tool not in tool_names:
                tool_names.append(tool)
        members.append(
            TeamMember(
                name=name,
                persona=persona,
                tools=tools,
                model_key=str(raw.get("model_key", "default")),
                handoffs=[str(h) for h in raw.get("handoffs", [])],
                delegates=[str(d) for d in raw.get("delegates", [])],
            )
        )

    member_names = {m.name for m in members}
    entry = str(spec.get("entry", members[0].name))
    if entry not in member_names:
        raise ValueError(f"team entry {entry!r} is not a declared member")
    for member in members:
        for edge in (*member.handoffs, *member.delegates):
            if edge not in member_names:
                raise ValueError(
                    f"member {member.name!r} references unknown peer {edge!r}"
                )

    team = AgentTeam(members=members, entry=entry)

    selector = str(spec.get("selector", "round_robin"))
    speakers_raw = spec.get("speakers")
    speakers = [str(s) for s in speakers_raw] if speakers_raw is not None else None
    if mode == "group_chat":
        if selector not in GROUP_CHAT_SELECTORS:
            raise ValueError(
                f"unknown group_chat selector {selector!r}; "
                f"expected one of {sorted(GROUP_CHAT_SELECTORS)}"
            )
        if speakers is not None:
            for name in speakers:
                if name not in member_names:
                    raise ValueError(f"speaker {name!r} is not a declared member")

    return TeamPlan(
        team=team,
        mode=mode,
        tool_names=tool_names,
        selector=selector,
        max_rounds=int(spec.get("max_rounds", 8)),
        speakers=speakers,
    )


__all__ = ["TeamPlan", "build_team_plan", "TEAM_MODES", "GROUP_CHAT_SELECTORS"]
