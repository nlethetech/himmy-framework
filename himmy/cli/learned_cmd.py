"""``himmy learned`` — show what self-learning has concluded about the agent's tools.

Self-learning silently demotes flaky tools and injects reliability hints, but until now an
operator had no way to SEE it. This command reads the durable CLI store
(``.himmy/storage.db`` — the same place a ``self_learning`` agent records its tool events)
and prints a worst-first table of per-tool reputation plus the recent reorders the loop
applied. Pure read: it changes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import TYPE_CHECKING, Any, cast

from himmy.cli.ui import styles
from himmy.services.learning.overrides import (
    OverrideKind,
    OverrideSet,
    OverrideStore,
)
from himmy.services.learning.report import (
    LearningReport,
    LearningToolRow,
    build_learning_report,
)
from himmy.services.storage.factory import StoreFactory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.storage.protocols import EventLog


def _load_override_set(workspace: str | None) -> OverrideSet:
    """Load the operator overrides for a workspace from the CLI spine (best-effort).

    Any spine/registry error degrades to an empty :class:`OverrideSet` (trust off) so the
    command falls back to pure Sprint-1 auto-learning rather than failing.
    """
    try:
        return OverrideStore.for_cli().load(workspace)
    except Exception:  # pragma: no cover - defensive: overrides never break the CLI
        return OverrideSet()


def _load_report(workspace: str | None, outcome_weight: float) -> LearningReport:
    """Open the durable CLI store and build the override-aware learning report.

    Threads the operator :class:`OverrideSet` (same ``.himmy/spine.db`` the runtime reads)
    into :func:`build_learning_report`, so the table reflects pins/mutes/resets/the
    trust-feedback gate exactly as the runtime acts on them. Best-effort throughout.
    """
    store = cast("EventLog", StoreFactory.for_cli_durable())
    override_set = _load_override_set(workspace)
    return asyncio.run(
        build_learning_report(
            store,
            workspace_id=workspace,
            outcome_weight=outcome_weight,
            override_set=override_set,
        )
    )


def _pct(score: float) -> str:
    """Format a [0,1] score as a whole-percent reliability string."""
    return f"{round(score * 100):d}%"


def _report_to_dict(report: LearningReport) -> dict[str, Any]:
    """Machine-readable view of the report (``--json``)."""
    return {
        "workspace_id": report.workspace_id,
        "floor": report.floor,
        "outcome_weight": report.outcome_weight,
        "trust_feedback": report.trust_feedback,
        "flaky_count": report.flaky_count,
        "tools": [
            {
                "tool_name": t.tool_name,
                "score": t.score,
                "operational_score": t.operational_score,
                "completed": t.completed,
                "failed": t.failed,
                "has_min_samples": t.has_min_samples,
                "flaky": t.flaky,
                "outcome_score": t.outcome_score,
                "outcome_samples": t.outcome_samples,
                "override": t.override,
            }
            for t in report.tools
        ],
        "recent_reorders": [
            {
                "when": r.when,
                "tools_reordered": r.tools_reordered,
                "deprioritised_tools": r.deprioritised_tools,
            }
            for r in report.recent_reorders
        ],
    }


def render_learning_report(
    report: LearningReport, *, show_all: bool, stream: Any = None
) -> str:
    """Render the report as a styled, worst-first table (TTY) or plain text (pipe)."""
    out = stream or sys.stdout
    c = styles(out)
    lines: list[str] = []
    lines.append(
        f"{c['bold']}{c['snow']}Himmy — what it's learned about its tools{c['reset']}"
    )

    if not report.has_data:
        lines.append("")
        lines.append(f"{c['dim']}No tool reputation yet.{c['reset']}")
        lines.append(
            f"{c['dim']}Learning accumulates as you run an agent with "
            f"{c['reset']}{c['gold']}self_learning: true{c['reset']}"
            f"{c['dim']} and it calls tools. Come back after a few runs.{c['reset']}"
        )
        return "\n".join(lines)

    floor_pct = _pct(report.floor)
    flaky = report.flaky_count
    flaky_txt = (
        f"{c['crimson']}{flaky} flaky{c['reset']}"
        if flaky
        else f"{c['green']}0 flaky{c['reset']}"
    )
    lines.append(
        f"{c['dim']}{len(report.tools)} tool(s) tracked · {flaky_txt}{c['dim']} "
        f"(below the {floor_pct} caution floor){c['reset']}"
    )
    if report.outcome_weight > 0:
        lines.append(
            f"{c['dim']}outcome quality blended at weight "
            f"{report.outcome_weight:.2f}{c['reset']}"
        )
    # The per-workspace 👍/👎 trust gate — show ON/off so the operator can see whether the
    # recorded feedback is actually being blended into reputation.
    if report.trust_feedback:
        lines.append(
            f"{c['dim']}feedback gate {c['reset']}{c['gold']}trust_feedback: on{c['reset']}"
            f"{c['dim']} — 👍/👎 blended into reputation{c['reset']}"
        )
    else:
        lines.append(
            f"{c['dim']}feedback gate trust_feedback: off — 👍/👎 recorded but inert "
            f"(turn on with {c['reset']}{c['gold']}himmy learned trust-feedback on{c['reset']}"
            f"{c['dim']}){c['reset']}"
        )
    lines.append("")

    # Tools that need an operator's eye: demoted (flaky), with any recorded failures, or
    # carrying an active operator override (so a pin/mute/reset is visibly confirmed). The
    # default view hides spotless tools; ``--all`` shows everything.
    needs_attention = [
        t
        for t in report.tools
        if t.flaky or (t.has_min_samples and t.failed > 0) or t.override is not None
    ]
    if not show_all and not needs_attention:
        lines.append("")
        lines.append(
            f"{c['green']}All {len(report.tools)} tracked tool(s) are reliable.{c['reset']}"
            f"{c['dim']} Pass {c['reset']}{c['gold']}--all{c['reset']}"
            f"{c['dim']} to see every tool's score.{c['reset']}"
        )
        rows: list[LearningToolRow] = []
    else:
        rows = report.tools if show_all else needs_attention

    if rows:
        name_w = max(max((len(t.tool_name) for t in rows), default=4), 4)
        header = f"  {'TOOL'.ljust(name_w)}   SCORE   OK   FAIL  STATUS"
        lines.append(f"{c['dim']}{header}{c['reset']}")
        for t in rows:
            # An active operator override wins the STATUS column — it reflects what the
            # runtime actually does (pinned/muted tools are never demoted; reset rebuilds).
            if t.override == "pinned":
                col, status = c["gold"], "pinned (trusted)"
            elif t.override == "ignored":
                col, status = c["dim"], "ignored"
            elif t.override == "reset":
                col, status = c["gold"], "reset — rebuilding"
            elif not t.has_min_samples:
                col, status = c["dim"], f"learning ({t.total}/3 calls)"
            elif t.flaky:
                col, status = c["crimson"], "flaky — demoted"
            elif t.failed > 0:
                col, status = c["gold"], f"watch ({t.failed} failed)"
            else:
                col, status = c["green"], "reliable"
            score_s = _pct(t.score).rjust(5)
            line = (
                f"  {col}{t.tool_name.ljust(name_w)}{c['reset']}   "
                f"{col}{score_s}{c['reset']}  "
                f"{str(t.completed).rjust(3)}  {str(t.failed).rjust(4)}  "
                f"{col}{status}{c['reset']}"
            )
            lines.append(line)
            if report.outcome_weight > 0 and t.outcome_score is not None:
                lines.append(
                    f"  {c['dim']}{' ' * name_w}   answer-quality {_pct(t.outcome_score)} "
                    f"over {t.outcome_samples} graded{c['reset']}"
                )

        hidden = len(report.tools) - len(rows)
        if not show_all and hidden > 0:
            lines.append("")
            lines.append(
                f"{c['dim']}… {hidden} reliable tool(s) hidden — pass {c['reset']}"
                f"{c['gold']}--all{c['reset']}{c['dim']} to show them.{c['reset']}"
            )

    if report.recent_reorders:
        lines.append("")
        lines.append(f"{c['bold']}{c['snow']}Recent reorders{c['reset']}")
        for r in report.recent_reorders[:5]:
            when = (r.when or "")[:19].replace("T", " ")
            demoted = ", ".join(r.deprioritised_tools) or "(none)"
            lines.append(
                f"  {c['dim']}{when}{c['reset']}  demoted: {c['crimson']}{demoted}{c['reset']}"
            )

    return "\n".join(lines)


def _confirm(message: str) -> None:
    """Print a styled one-line confirmation for an override mutation."""
    c = styles(sys.stdout)
    print(f"{c['green']}✓{c['reset']} {message}")


def _render_after_mutation(workspace: str | None) -> None:
    """Re-render the override-aware report after a mutation (best-effort)."""
    try:
        report = _load_report(workspace, 0.0)
        print()
        print(render_learning_report(report, show_all=False))
    except (
        Exception
    ):  # pragma: no cover - defensive: never let a re-render fail the verb
        pass


def _set_override_verb(
    args: argparse.Namespace, kind: OverrideKind, *, confirm: str
) -> int:
    """Shared handler for the pin / mute / reset / clear verbs (best-effort)."""
    workspace = getattr(args, "workspace", None)
    tool = args.tool_name
    try:
        if kind is OverrideKind.CLEAR:
            OverrideStore.for_cli().clear_override(
                tool, workspace_id=workspace, actor="cli", source="cli"
            )
        else:
            OverrideStore.for_cli().set_override(
                tool, kind, workspace_id=workspace, actor="cli", source="cli"
            )
    except Exception:  # pragma: no cover - defensive: overrides never break the CLI
        c = styles(sys.stdout)
        print(f"{c['crimson']}✗{c['reset']} could not update override for {tool}")
        return 1
    _confirm(confirm.format(tool=tool))
    _render_after_mutation(workspace)
    return 0


def cmd_learned_pin(args: argparse.Namespace) -> int:
    """``himmy learned pin <tool>`` — force-trust a tool (never demoted/flagged)."""
    return _set_override_verb(
        args,
        OverrideKind.PIN,
        confirm="pinned {tool} — force-trusted (never demoted, never flagged flaky)",
    )


def cmd_learned_mute(args: argparse.Namespace) -> int:
    """``himmy learned mute|ignore <tool>`` — exclude a tool from auto-learning."""
    return _set_override_verb(
        args,
        OverrideKind.MUTE,
        confirm="ignored {tool} — excluded from auto-learning (never demoted/hinted)",
    )


def cmd_learned_reset(args: argparse.Namespace) -> int:
    """``himmy learned reset <tool>`` — forget accumulated reputation; rebuild from now."""
    return _set_override_verb(
        args,
        OverrideKind.RESET,
        confirm="reset {tool} — reputation forgotten; rebuilding from now",
    )


def cmd_learned_clear(args: argparse.Namespace) -> int:
    """``himmy learned clear <tool>`` — remove any override (back to auto-learning)."""
    return _set_override_verb(
        args,
        OverrideKind.CLEAR,
        confirm="cleared override for {tool} — back to pure auto-learning",
    )


def cmd_learned_trust_feedback(args: argparse.Namespace) -> int:
    """``himmy learned trust-feedback {on|off}`` — gate the 👍/👎 outcome blend."""
    workspace = getattr(args, "workspace", None)
    enabled = args.state == "on"
    try:
        OverrideStore.for_cli().set_trust_feedback(
            enabled, workspace_id=workspace, actor="cli", source="cli"
        )
    except Exception:  # pragma: no cover - defensive
        c = styles(sys.stdout)
        print(f"{c['crimson']}✗{c['reset']} could not update trust_feedback")
        return 1
    if enabled:
        _confirm(
            "trust_feedback on — 👍/👎 now blended into reputation "
            "(at weight 0.30 unless you set outcome_weight explicitly)"
        )
    else:
        _confirm("trust_feedback off — 👍/👎 recorded but inert (Sprint-1 behaviour)")
    _render_after_mutation(workspace)
    return 0


def cmd_learned(args: argparse.Namespace) -> int:
    """``himmy learned`` — print the learning report (table or ``--json``)."""
    report = _load_report(
        getattr(args, "workspace", None), getattr(args, "outcome_weight", 0.0)
    )
    if getattr(args, "json", False):
        print(json.dumps(_report_to_dict(report), indent=2))
        return 0
    print(render_learning_report(report, show_all=getattr(args, "all", False)))
    return 0


def add_learned_parser(sub: Any) -> None:
    """Register the ``learned`` subcommand on the CLI's subparsers."""
    p = sub.add_parser(
        "learned",
        help="show what self-learning concluded about your tools (reputation + reorders)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="show healthy tools too (default: flaky tools + summary)",
    )
    p.add_argument(
        "--workspace",
        dest="workspace",
        default=None,
        help="scope the report to one workspace_id (default: all)",
    )
    p.add_argument(
        "--outcome-weight",
        dest="outcome_weight",
        type=float,
        default=0.0,
        help="blend answer-quality (OUTCOME_SCORED) into the score at this weight [0..1]",
    )
    p.add_argument("--json", action="store_true", help="machine-readable JSON report")
    # The bare ``himmy learned`` (no verb) still renders the report — verbs are optional.
    p.set_defaults(func=cmd_learned)

    verbs = p.add_subparsers(
        dest="learned_verb", metavar="{pin,mute,reset,clear,trust-feedback}"
    )

    def _add_tool_verb(
        name: str, handler: Any, help_text: str, *, aliases: list[str] | None = None
    ) -> None:
        vp = verbs.add_parser(name, help=help_text, aliases=aliases or [])
        vp.add_argument("tool_name", help="the tool to act on")
        vp.add_argument(
            "--workspace",
            dest="workspace",
            default=None,
            help="scope the override to one workspace_id (default: local)",
        )
        vp.set_defaults(func=handler)

    _add_tool_verb(
        "pin",
        cmd_learned_pin,
        "force-trust a tool — never demoted, never flagged flaky",
    )
    _add_tool_verb(
        "mute",
        cmd_learned_mute,
        "exclude a tool from auto-learning (shown as ignored)",
        aliases=["ignore"],
    )
    _add_tool_verb(
        "reset",
        cmd_learned_reset,
        "forget a tool's accumulated reputation; rebuild from now",
    )
    _add_tool_verb(
        "clear",
        cmd_learned_clear,
        "remove any override on a tool (back to auto-learning)",
    )

    tf = verbs.add_parser(
        "trust-feedback",
        help="gate the 👍/👎 outcome blend for this workspace {on|off}",
    )
    tf.add_argument(
        "state", choices=["on", "off"], help="turn the feedback blend on or off"
    )
    tf.add_argument(
        "--workspace",
        dest="workspace",
        default=None,
        help="scope the gate to one workspace_id (default: local)",
    )
    tf.set_defaults(func=cmd_learned_trust_feedback)


__all__ = [
    "add_learned_parser",
    "cmd_learned",
    "cmd_learned_clear",
    "cmd_learned_mute",
    "cmd_learned_pin",
    "cmd_learned_reset",
    "cmd_learned_trust_feedback",
    "render_learning_report",
]
