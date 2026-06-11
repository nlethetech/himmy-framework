"""``himmy orchestrate`` — run a team through a real multi-agent orchestrator.

Where ``himmy team`` (see :func:`himmy.cli.commands.cmd_team`) routes a ``team.yaml``
through ONE shape (handoff/delegation), this surfaces the framework's other real
multi-agent control modes behind a single ``--mode`` selector, each routing the SAME
declarative team through the corresponding orchestrator:

* ``hierarchical`` — :class:`~himmy.orchestrators.MultiAgentOrchestrator`: a manager
  hands off / delegates over one shared audited thread (swarm + supervisor edges).
* ``group_chat`` — :class:`~himmy.orchestrators.GroupChatOrchestrator`: every member
  shares one thread and a selector (round-robin by default, or an LLM "manager") picks
  the next speaker each round.
* ``reflection`` — the entry member drafts an answer, then a reviewer member critiques
  and revises it via :func:`himmy.orchestrators.reflect` (the cheapest quality lift).
* ``graph`` — a real :class:`~himmy.orchestrators.StateGraph`: each member becomes a
  node run in sequence over a typed shared state (draft → revise → … → END), so the
  graph kernel drives genuine agent turns.

The multi-agent activity renders live through the SAME :class:`~himmy.cli.ui.LiveRunUI`
the rest of the CLI uses — ``AGENT_HANDOFF`` / ``AGENT_DELEGATED`` /
``GROUP_SPEAKER_*`` / ``GRAPH_*`` events land as they happen — and the synthesized
result plus a per-agent contribution summary print at the end. Everything defaults to
the offline stub, so this runs with no keys and no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from himmy.cli.commands import _eprint, _exec_with_mcp, _resolve_register
from himmy.cli.ui import LiveRunUI, render_markdown_lite, styles

#: The selectable orchestration modes (the ``--mode`` choices).
MODES = ("hierarchical", "group_chat", "reflection", "graph")


def _build_team_and_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    """Build ``(spec, team, registry, runtime, live)`` from a team.yaml + CLI flags.

    Mirrors :func:`himmy.cli.commands.cmd_team`'s wiring so a member may declare its
    own provider (a Claude-CLI brain over local Ollama workers), then attaches a
    :class:`~himmy.cli.ui.LiveRunUI` as the runtime's ``on_event`` so inference /
    tool / multi-agent events render live on a TTY (silent off-TTY).
    """
    from himmy import build_runtime
    from himmy.cli.rbac_cmd import enforce_role_on_registry
    from himmy.config.team_spec import build_team, build_team_inference, load_team_spec

    spec = load_team_spec(args.file)
    team, registry = build_team(spec, resolve_tools_module=_resolve_register)
    # RBAC: a restricted --role re-gates any gated tool it may not run (fail closed),
    # consistent with run/chat — so a viewer can't slip a gated tool via orchestrate.
    enforce_role_on_registry(
        registry, role=getattr(args, "role", None), on_line=_eprint
    )
    inference = build_team_inference(
        spec,
        default_provider=getattr(args, "provider", None),
        default_model=getattr(args, "model", None),
    )
    live = LiveRunUI(model_label=_model_label_for(args))
    runtime, _inference, _tools = build_runtime(
        inference=inference,
        tool_registry=registry,
        on_event=live.handle if live.enabled else None,
    )
    return spec, team, registry, runtime, live


def _model_label_for(args: argparse.Namespace) -> str:
    """A ``model · provider`` tag for the live UI from the CLI flags (team-level).

    Members may each carry their own provider/model, so there is no single spec to
    label; this reflects only the team-level ``--provider``/``--model`` overrides.
    """
    model = getattr(args, "model", None)
    provider = getattr(args, "provider", None)
    parts = [p for p in (model, provider) if p and p != "default"]
    return " · ".join(parts)


def _live_handler(live: Any) -> Any:
    """The orchestrator-facing ``on_event`` callback (live multi-agent lines)."""
    return live.handle if live.enabled else None


# ----------------------------------------------------------------- contributions


def _contributions(turns: list[tuple[str, Any]]) -> list[tuple[str, int, str]]:
    """Per-agent ``(name, turn_count, last_snippet)`` summary, in first-seen order."""
    order: list[str] = []
    counts: dict[str, int] = {}
    last: dict[str, str] = {}
    for name, turn in turns:
        if name not in counts:
            order.append(name)
            counts[name] = 0
        counts[name] += 1
        text = (getattr(turn, "output_text", None) or "").strip().replace("\n", " ")
        if text:
            last[name] = text[:80] + ("…" if len(text) > 80 else "")
    return [(name, counts[name], last.get(name, "")) for name in order]


def _print_contributions(
    contribs: list[tuple[str, int, str]], stream: Any = None
) -> None:
    """Render the per-agent contribution summary (faint, to stderr)."""
    c = styles(stream or sys.stderr)
    if not contribs:
        return
    _eprint(f"{c['dim']}contributions:{c['reset']}")
    for name, count, snippet in contribs:
        turns = "turn" if count == 1 else "turns"
        tail = f"  {c['faint']}{snippet}{c['reset']}" if snippet else ""
        _eprint(
            f"  {c['gold']}{name}{c['reset']} {c['faint']}({count} {turns}){c['reset']}{tail}"
        )


# --------------------------------------------------------------------- the modes


async def _run_hierarchical(
    team: Any, registry: Any, runtime: Any, live: Any, prompt: str
) -> dict[str, Any]:
    """Route ``prompt`` through the real handoff/delegation orchestrator."""
    from himmy.orchestrators import MultiAgentOrchestrator

    orch = MultiAgentOrchestrator(runtime, team, registry, on_event=_live_handler(live))
    result = await orch.run(prompt)
    return {
        "route": list(result.handoff_chain),
        "final_agent": result.final_agent,
        "stopped_reason": result.stopped_reason,
        "output_text": result.output_text,
        "total_cost": result.total_cost,
        "turns": list(result.turns),
        "contributions": _contributions(result.turns),
    }


async def _run_group_chat(
    team: Any,
    registry: Any,
    runtime: Any,
    live: Any,
    prompt: str,
    *,
    max_rounds: int | None,
) -> dict[str, Any]:
    """Run the team as a round-robin panel over one shared thread.

    ``allow_final_answer`` is OFF so the chat is a fixed-length panel that runs every
    member at least once (to ``max_rounds``) rather than stopping the moment the first
    speaker emits a final answer — the point of a group chat is that the whole team
    contributes. ``max_rounds`` defaults to one round per member (so each speaks once).
    """
    from himmy.orchestrators import GroupChatOrchestrator

    rounds = max_rounds if max_rounds is not None else len(team.members)
    rounds = max(1, rounds)
    orch = GroupChatOrchestrator(
        runtime,
        team,
        registry,
        on_event=_live_handler(live),
        max_rounds=rounds,
        allow_final_answer=False,
    )
    result = await orch.run(prompt)
    return {
        "route": list(result.speaker_order),
        "final_agent": result.final_speaker,
        "stopped_reason": result.stopped_reason,
        "output_text": result.output_text,
        "total_cost": result.total_cost,
        "turns": list(result.turns),
        "contributions": _contributions(result.turns),
    }


async def _run_reflection(
    team: Any, runtime: Any, live: Any, prompt: str
) -> dict[str, Any]:
    """Entry member drafts; a reviewer member critiques + revises via ``reflect``.

    The draft comes from the team's entry member (a genuine runtime turn); the
    reviewer is the next distinct member (or a default reviewer persona when the
    team has only one member). Both turns are surfaced as contributions.
    """
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.orchestrators import reflect
    from himmy.services.inference.models import LLMConfig

    drafter = team.require(team.entry)
    reviewer = next((m for m in team.members if m.name != drafter.name), None)

    draft_run = await runtime.run_task_detailed(
        drafter.persona,
        Task(
            title=f"{drafter.name}-draft",
            prompt=prompt,
            context={"model_key": drafter.model_key, "tool_names": []},
        ),
        thread=ChatThread(agent_id=drafter.persona.agent_id),
        llm_config=LLMConfig(model_key=drafter.model_key, temperature=0.1),
    )
    draft = draft_run.output_text or ""
    turns: list[tuple[str, Any]] = [(drafter.name, draft_run)]

    reviewer_persona = reviewer.persona if reviewer is not None else None
    reviewer_key = reviewer.model_key if reviewer is not None else "default"
    reviewer_name = reviewer.name if reviewer is not None else "reviewer"
    improved = await reflect(
        runtime,
        draft,
        persona=reviewer_persona,
        model_key=reviewer_key,
    )
    # reflect() runs an isolated turn; record the reviewer's contribution by wrapping
    # the revised text in a lightweight turn-shaped object for the summary.
    turns.append((reviewer_name, _TextTurn(improved)))
    return {
        "route": [drafter.name, reviewer_name],
        "final_agent": reviewer_name,
        "stopped_reason": "reflected",
        "output_text": improved,
        "total_cost": draft_run.cost,
        "turns": turns,
        "contributions": _contributions(turns),
    }


class _TextTurn:
    """A minimal turn-shaped holder (``.output_text``/``.cost``) for the summary."""

    def __init__(self, text: str) -> None:
        self.output_text = text
        self.cost = 0.0


async def _run_graph(team: Any, runtime: Any, live: Any, prompt: str) -> dict[str, Any]:
    """Run the team as a real StateGraph: each member a node, in sequence to END.

    Each member becomes a node that takes one runtime turn over the shared graph
    state, appending its answer; nodes are chained entry → … → END so the graph
    kernel drives the genuine agent turns and emits ``GRAPH_*`` audited events.
    """
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.orchestrators import END, StateGraph
    from himmy.services.inference.models import LLMConfig

    members = list(team.members)
    if not members:
        return {
            "route": [],
            "final_agent": "",
            "stopped_reason": "empty",
            "output_text": None,
            "total_cost": 0.0,
            "turns": [],
            "contributions": [],
        }

    turns: list[tuple[str, Any]] = []

    def _make_node(member: Any) -> Any:
        async def _node(state: dict[str, Any]) -> dict[str, Any]:
            prior = state.get("answers", [])
            context_note = ""
            if prior:
                context_note = "\n\nPrior answers:\n" + "\n".join(
                    f"- {a}" for a in prior
                )
            run = await runtime.run_task_detailed(
                member.persona,
                Task(
                    title=f"{member.name}-node",
                    prompt=f"{state.get('prompt', '')}{context_note}",
                    context={"model_key": member.model_key, "tool_names": []},
                ),
                thread=ChatThread(agent_id=member.persona.agent_id),
                llm_config=LLMConfig(model_key=member.model_key, temperature=0.1),
            )
            turns.append((member.name, run))
            text = run.output_text or ""
            return {"answers": [text], "last": text, "cost": run.cost}

        return _node

    graph = StateGraph(name="orchestrate")
    for member in members:
        graph.add_node(member.name, _make_node(member))
    # Chain the members in declaration order; the last routes to END.
    for src, dst in zip(members, members[1:], strict=False):
        graph.add_edge(src.name, dst.name)
    graph.add_edge(members[-1].name, END)
    graph.set_entry_point(team.entry)

    compiled = graph.compile(on_event=_live_handler(live))
    result = await compiled.invoke({"prompt": prompt, "answers": []})
    total = float(result.final_state.get("cost", 0.0))
    return {
        "route": list(result.node_sequence),
        "final_agent": result.node_sequence[-1] if result.node_sequence else "",
        "stopped_reason": result.status,
        "output_text": result.final_state.get("last"),
        "total_cost": total,
        "turns": turns,
        "contributions": _contributions(turns),
    }


# ------------------------------------------------------------------- entrypoint


def cmd_orchestrate(args: argparse.Namespace) -> int:
    """Run a ``team.yaml`` through the orchestrator selected by ``--mode``.

    Prints the synthesized result on stdout (markdown-lite on a TTY) and a faint
    per-agent route + contribution summary on stderr; ``--json`` emits the full
    machine-readable record on stdout instead. Returns 0 on success.
    """
    mode = getattr(args, "mode", None) or "hierarchical"
    if mode not in MODES:
        _eprint(f"error: unknown --mode {mode!r} (choose from {', '.join(MODES)})")
        return 2
    if not getattr(args, "prompt", None):
        _eprint("error: --prompt/-p is required for `himmy orchestrate`")
        return 2
    if not getattr(args, "file", None):
        _eprint("error: --file/-f <team.yaml> is required for `himmy orchestrate`")
        return 2

    try:
        spec, team, registry, runtime, live = _build_team_and_runtime(args)
    except FileNotFoundError:
        _eprint(f"error: team file not found: {args.file}")
        return 2
    except Exception as exc:  # noqa: BLE001 - surface a clean error, never a traceback
        _eprint(f"error: could not load team {args.file!r}: {exc}")
        return 1

    mcp_servers = getattr(spec, "mcp_servers", None) or []

    if mode == "hierarchical":
        coro_factory = lambda: _run_hierarchical(  # noqa: E731
            team, registry, runtime, live, args.prompt
        )
    elif mode == "group_chat":
        raw_rounds = getattr(args, "max_rounds", None)
        max_rounds = int(raw_rounds) if raw_rounds is not None else None
        coro_factory = lambda: _run_group_chat(  # noqa: E731
            team, registry, runtime, live, args.prompt, max_rounds=max_rounds
        )
    elif mode == "reflection":
        coro_factory = lambda: _run_reflection(team, runtime, live, args.prompt)  # noqa: E731
    else:  # graph
        coro_factory = lambda: _run_graph(team, runtime, live, args.prompt)  # noqa: E731

    out = _exec_with_mcp(coro_factory, registry, mcp_servers)
    live.finish()

    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "mode": mode,
                    "route": out["route"],
                    "final_agent": out["final_agent"],
                    "stopped_reason": out["stopped_reason"],
                    "total_cost": out["total_cost"],
                    "output_text": out["output_text"],
                    "contributions": [
                        {"agent": name, "turns": count, "last": snippet}
                        for name, count, snippet in out["contributions"]
                    ],
                    "transcript": [
                        {"agent": name, "output": getattr(turn, "output_text", None)}
                        for name, turn in out["turns"]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        live.footer()
        return 0

    c = styles(sys.stderr)
    route = " → ".join(out["route"]) or "(no agents ran)"
    _eprint(f"{c['dim']}[{mode}] route: {route}  ({out['stopped_reason']}){c['reset']}")
    _print_contributions(out["contributions"])
    print(render_markdown_lite(out["output_text"] or "", stream=sys.stdout))
    live.footer()
    return 0


# --------------------------------------------------------------------- REPL hook


def _is_team_spec(path: str) -> bool:
    """Whether ``path`` is a TEAM spec (has ``members``/``entry``), not an agent spec.

    A team.yaml declares ``members`` (and usually ``entry``); an agent.yaml does not.
    Read defensively — a missing/odd file is simply "not a team" (returns ``False``),
    never an exception, so ``/orchestrate`` falls back to ``./team.yaml`` cleanly.
    """
    try:
        import yaml

        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    return "members" in data or "entry" in data


def slash_orchestrate(repl: Any, rest: list[str]) -> Any:
    """``/orchestrate <mode> <prompt>`` — run the REPL's spec-adjacent team file.

    Looks for a ``team.yaml`` next to the REPL's working directory (or via
    ``--file`` on the REPL's args) and runs it through the chosen mode, rendering
    the route + result inline. With no team file it prints a short hint. ``rest`` is
    the already-split tail of the slash line (``["group_chat", "compare", "X"]``).
    """
    from pathlib import Path

    c = repl._c
    if not rest:
        repl._eprint(
            f"{c['dim']}usage: /orchestrate <{'|'.join(MODES)}> <prompt>{c['reset']}"
        )
        return
    mode = rest[0]
    if mode not in MODES:
        repl._eprint(
            f"{c['dim']}unknown mode {mode!r} — choose from "
            f"{', '.join(MODES)}{c['reset']}"
        )
        return
    prompt = " ".join(rest[1:]).strip()
    if not prompt:
        repl._eprint(
            f"{c['dim']}give a prompt: /orchestrate {mode} <prompt>{c['reset']}"
        )
        return
    # The chat's --file is an AGENT spec, not a team; only use it when it actually IS
    # a team spec (has 'members'/'entry'), else fall back to ./team.yaml. Otherwise
    # /orchestrate in a `himmy chat -f agent.yaml` session mis-loads the agent as a team.
    team_file = getattr(repl.args, "file", None)
    if team_file and not _is_team_spec(team_file):
        team_file = None
    if not team_file:
        candidate = Path("team.yaml")
        if candidate.is_file():
            team_file = str(candidate)
    if not team_file:
        repl._eprint(
            f"{c['dim']}no team.yaml found — scaffold one with "
            f"`himmy init --team .`{c['reset']}"
        )
        return
    ns = argparse.Namespace(
        mode=mode,
        prompt=prompt,
        file=team_file,
        provider=getattr(repl.args, "provider", None),
        model=getattr(repl.args, "model", None),
        role=getattr(repl.args, "role", None),
        json=False,
        max_rounds=None,
    )
    cmd_orchestrate(ns)


__all__ = [
    "MODES",
    "cmd_orchestrate",
    "slash_orchestrate",
]
