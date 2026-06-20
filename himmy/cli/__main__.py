"""The ``himmy`` command-line entry point.

Exposed as a console script (``himmy``) and as ``python -m himmy``. Subcommands wire
the offline-first runtime so the common case needs no keys::

    himmy init my-agent
    himmy run -f my-agent/agent.yaml -p "Summarize NRB's latest forex move."
    himmy chat -f my-agent/agent.yaml
    himmy serve
    himmy doctor
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from himmy import __version__
from himmy.cli import commands
from himmy.cli.audit import add_audit_parser
from himmy.cli.consent import add_consent_parser
from himmy.cli.context_cmd import add_context_parser
from himmy.cli.dashboard_cmd import add_dashboard_parser
from himmy.cli.models_cmd import cmd_compare, cmd_models
from himmy.cli.provider import PROVIDERS
from himmy.cli.recommendations_cmd import add_recommendations_parser
from himmy.cli.routines_cmd import add_routines_parser
from himmy.cli.runs_cmd import add_runs_parser
from himmy.cli.security_audit_cmd import add_seclog_parser
from himmy.core import HimmyError


def _cmd_new(args: argparse.Namespace) -> int:
    """Lazy dispatcher: `himmy new` pulls in inference machinery only when used."""
    from himmy.cli.new import cmd_new

    return cmd_new(args)


def _cmd_agents(args: argparse.Namespace) -> int:
    from himmy.cli.agents import cmd_agents

    return cmd_agents(args)


def _cmd_validate(args: argparse.Namespace) -> int:
    from himmy.cli.agents import cmd_validate

    return cmd_validate(args)


def _cmd_workflow(args: argparse.Namespace) -> int:
    from himmy.cli.workflow_cmd import cmd_workflow

    return cmd_workflow(args)


def _cmd_orchestrate(args: argparse.Namespace) -> int:
    from himmy.cli.orchestrate import cmd_orchestrate

    return cmd_orchestrate(args)


def _cmd_guardrails(args: argparse.Namespace) -> int:
    from himmy.cli.guardrails_view import cmd_guardrails

    return cmd_guardrails(args)


def _cmd_lineage(args: argparse.Namespace) -> int:
    from himmy.cli.lineage_cmd import cmd_lineage

    return cmd_lineage(args)


def _cmd_whoami(args: argparse.Namespace) -> int:
    from himmy.cli.rbac_cmd import cmd_whoami

    return cmd_whoami(args)


def _cmd_mcp(args: argparse.Namespace) -> int:
    from himmy.cli.mcp_cmd import cmd_mcp

    return cmd_mcp(args)


def _cmd_skill(args: argparse.Namespace) -> int:
    from himmy.cli.skill_dispatch import cmd_skill

    return cmd_skill(args)


def _cmd_connectors(args: argparse.Namespace) -> int:
    from himmy.cli.connectors_cmd import cmd_connectors

    return cmd_connectors(args)


def _add_agent_flags(parser: argparse.ArgumentParser) -> None:
    """Shared flags for commands that build/run an agent (run, chat)."""
    parser.add_argument("-f", "--file", help="path to an agent.yaml spec")
    parser.add_argument("--name", help="agent name when no spec file is given")
    parser.add_argument(
        "--instruction",
        action="append",
        help="instruction line (repeatable) when no spec file is given",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="inference provider (default: auto pydantic-ai→stub)",
    )
    parser.add_argument("--model", help="model key/name for the provider")
    parser.add_argument(
        "--role",
        help="RBAC role for this session; restricts gated-tool execution "
        "(else HIMMY_ROLE env / himmy.toml [rbac] role / admin)",
    )
    # Permission profile (mutually exclusive): --yolo auto-approves every gated
    # tool; --safe ignores the himmy.toml allowlist so everything gated prompts.
    perms = parser.add_mutually_exclusive_group()
    perms.add_argument(
        "--yolo",
        action="store_true",
        help="auto-approve every approval-gated tool this run (no prompts)",
    )
    perms.add_argument(
        "--safe",
        action="store_true",
        help="ignore [permissions] auto_approve — prompt for every gated tool",
    )
    parser.add_argument(
        "--budget",
        type=float,
        metavar="USD",
        help=(
            "session/run cost cap in dollars (overrides himmy.toml [limits] "
            "session_budget). run: stops the agent loop once spend reaches the cap; "
            "chat: warns at 80%% and prompts before a turn once the cap is spent"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="himmy", description="Himmy agent CLI.")
    parser.add_argument(
        "-v", "--version", action="version", version=f"himmy {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a single prompt and print the answer")
    _add_agent_flags(p_run)
    p_run.add_argument(
        "words",
        nargs="*",
        metavar="prompt",
        help="the prompt, as plain words (alternative to -p; stdin can be piped too)",
    )
    p_run.add_argument("-p", "--prompt", help="the prompt to run")
    p_run.add_argument("--json", action="store_true", help="print full result as JSON")
    p_run.add_argument(
        "--stream", action="store_true", help="stream the reply token-by-token"
    )
    p_run.add_argument(
        "--trace", action="store_true", help="print + save the run's event timeline"
    )
    p_run.add_argument(
        "--plan", action="store_true", help="plan-and-execute (decompose into steps)"
    )
    p_run.add_argument(
        "--record",
        metavar="FILE",
        help="record every model response to a replayable cassette (JSON)",
    )
    p_run.add_argument(
        "--replay",
        metavar="FILE",
        help="re-run deterministically from a recorded cassette (no provider/network)",
    )
    p_run.add_argument(
        "--persist",
        action="store_true",
        help="record this run in the canonical store (visible in `himmy runs` + /v1 + "
        "Studio); default off keeps the run in-RAM",
    )
    p_run.set_defaults(func=commands.cmd_run)

    p_chat = sub.add_parser("chat", help="interactive chat keeping one thread")
    _add_agent_flags(p_chat)
    p_chat.add_argument("--message", help="run a single turn non-interactively")
    p_chat.add_argument(
        "--session",
        help="persist/resume this conversation by id (.himmy/conversations.db)",
    )
    p_chat.add_argument(
        "-c",
        "--continue",
        dest="continue_last",
        action="store_true",
        help="continue the most recent session (the auto-saved 'last' thread)",
    )
    p_chat.add_argument(
        "--persist",
        action="store_true",
        help="record a --message turn in the canonical store (visible in `himmy runs` + "
        "/v1 + Studio); default off keeps the turn in-RAM",
    )
    p_chat.set_defaults(func=commands.cmd_chat)

    p_tg = sub.add_parser("telegram", help="run an agent as a live Telegram bot")
    _add_agent_flags(p_tg)
    p_tg.add_argument("--token", help="bot token (default: HIMMY_TELEGRAM_BOT_TOKEN)")
    p_tg.add_argument(
        "--allow-chat",
        action="append",
        metavar="CHAT_ID",
        help="restrict the bot to these chat ids (repeatable; default: "
        "HIMMY_TELEGRAM_ALLOWED_CHATS). Without an allowlist the bot answers everyone.",
    )
    p_tg.set_defaults(func=commands.cmd_telegram)

    p_team = sub.add_parser("team", help="run a multi-agent team from a team.yaml")
    p_team.add_argument("-f", "--file", required=True, help="path to a team.yaml")
    p_team.add_argument("-p", "--prompt", help="the prompt to route through the team")
    p_team.add_argument(
        "--provider", choices=PROVIDERS, help="inference provider (default: auto)"
    )
    p_team.add_argument("--model", help="model key/name for the provider")
    p_team.add_argument("--json", action="store_true", help="print the full transcript")
    p_team.set_defaults(func=commands.cmd_team)

    p_eval = sub.add_parser("eval", help="evaluate an agent/team against a suite.yaml")
    p_eval.add_argument("-f", "--file", required=True, help="path to a suite.yaml")
    p_eval.add_argument("--agent", help="path to an agent.yaml to evaluate")
    p_eval.add_argument("--team", help="path to a team.yaml to evaluate instead")
    p_eval.add_argument(
        "--provider", choices=PROVIDERS, help="inference provider (default: auto)"
    )
    p_eval.add_argument("--model", help="model key/name for the provider")
    p_eval.add_argument(
        "--json", action="store_true", help="print the full run as JSON"
    )
    p_eval.set_defaults(func=commands.cmd_eval)

    p_bench = sub.add_parser(
        "bench", help="benchmark models on a task suite (accuracy/tool-call/latency)"
    )
    p_bench.add_argument(
        "--models",
        required=True,
        help="comma list of provider:model (e.g. ollama:qwen2.5:3b-instruct,claude-cli:haiku)",
    )
    p_bench.add_argument(
        "--suite",
        help="path to a suite.yaml, or a built-in suite name "
        "(core/nepali/irrelevance/multiagent/bfcl); default: built-in core",
    )
    p_bench.add_argument(
        "--leaderboard",
        action="store_true",
        help="render a BFCL-style ranked leaderboard (Accuracy + CI, tool-call, "
        "arg-accuracy, multi-turn, abstain, McNemar) instead of the default scorecard",
    )
    p_bench.add_argument(
        "--trials", type=int, default=3, help="runs per task (more → tighter CIs)"
    )
    p_bench.add_argument(
        "--router", action="store_true", help="enable the tool router for all models"
    )
    p_bench.add_argument(
        "--extra-packs", help="comma list of distractor tool packs (to test routing)"
    )
    p_bench.add_argument("--temperature", type=float, default=0.0)
    p_bench.add_argument(
        "--judge-model",
        help="judge model for LLM-judge-tier tasks (must differ from each candidate); "
        "without it, judge-tier trials are recorded ungraded",
    )
    p_bench.add_argument(
        "--judge-provider",
        help="provider for --judge-model (default: each candidate's own provider)",
    )
    p_bench.add_argument("--json", help="also write the full results as JSON")
    p_bench.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero if any model's accuracy is below this floor (0-1), for CI",
    )
    p_bench.set_defaults(func=commands.cmd_bench)

    # models — list the local providers + models himmy can run on (shared catalog seam).
    p_models = sub.add_parser(
        "models",
        help="list local providers + models (with cached bench reliability)",
    )
    p_models.add_argument(
        "--prices",
        action="store_true",
        help="append each model's input/output $/1M from the price table",
    )
    p_models.add_argument(
        "--json", action="store_true", help="print the catalog as JSON"
    )
    p_models.set_defaults(func=cmd_models)

    # compare — one prompt across N models side-by-side (the quick path; bench is rigorous).
    p_compare = sub.add_parser(
        "compare",
        help="run one prompt across several models side-by-side (quick; bench is rigorous)",
    )
    p_compare.add_argument(
        "prompt", nargs="+", help="the prompt to send to every model"
    )
    p_compare.add_argument(
        "-m",
        "--models",
        action="append",
        metavar="PROVIDER:MODEL",
        help="a provider:model target (repeatable, or a comma list), "
        "e.g. -m ollama:llama3.2 -m claude-cli:haiku",
    )
    p_compare.add_argument(
        "--system", help="an optional system prompt sent to every model"
    )
    p_compare.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-model timeout in seconds (default: 120)",
    )
    p_compare.add_argument(
        "--json", action="store_true", help="print the results as JSON"
    )
    p_compare.set_defaults(func=cmd_compare)

    p_new = sub.add_parser(
        "new",
        help='draft an agent.yaml from a description, e.g. himmy new "tracks my rss feeds"',
    )
    p_new.add_argument(
        "description", nargs="+", help="what the agent should do, in plain English"
    )
    p_new.add_argument(
        "-o", "--output", default="agent.yaml", help="where to write the spec"
    )
    p_new.add_argument(
        "--dry-run",
        action="store_true",
        help="print the drafted YAML to stdout without writing",
    )
    p_new.add_argument(
        "--yes", action="store_true", help="write without asking (and overwrite)"
    )
    p_new.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="backend to draft with (default: best one detected on this machine)",
    )
    p_new.add_argument("--model", help="model to draft with")
    p_new.set_defaults(func=_cmd_new)

    p_agents = sub.add_parser(
        "agents", help="list the agent/team specs in a directory (default: here)"
    )
    p_agents.add_argument("directory", nargs="?", default=".", help="directory to scan")
    p_agents.add_argument(
        "--json", action="store_true", help="print the listing as JSON"
    )
    p_agents.set_defaults(func=_cmd_agents)

    p_validate = sub.add_parser(
        "validate", help="lint an agent.yaml before running it (did-you-mean hints)"
    )
    p_validate.add_argument(
        "file", nargs="?", help="spec to check (default: the nearest agent.yaml)"
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_init = sub.add_parser("init", help="scaffold an agent.yaml + tools.py")
    p_init.add_argument("directory", nargs="?", default=".", help="target directory")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.add_argument(
        "--classic",
        action="store_true",
        help="skip the interactive wizard; write the full example scaffold",
    )
    p_init.add_argument(
        "--template",
        choices=["helpdesk", "analyst", "researcher"],
        help="start from a working specialised agent (docs / API / web research)",
    )
    p_init.add_argument(
        "--team", action="store_true", help="scaffold a team.yaml instead"
    )
    p_init.set_defaults(func=commands.cmd_init)

    p_demo = sub.add_parser(
        "demo-video",
        help="scaffold or render a cinematic all-terminal product demo (MP4)",
    )
    p_demo.add_argument("directory", nargs="?", default=".", help="workspace directory")
    p_demo.add_argument(
        "--render",
        action="store_true",
        help="record the chapters in script.json and stitch demo.mp4 "
        "(needs playwright + chromium + ffmpeg)",
    )
    p_demo.add_argument("--only", help="with --render: re-record just this chapter id")
    p_demo.add_argument(
        "--output", default="demo.mp4", help="output filename (default: demo.mp4)"
    )
    p_demo.set_defaults(func=commands.cmd_demo_video)

    p_serve = sub.add_parser("serve", help="serve the FastAPI BFF (needs api extra)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=commands.cmd_serve)

    p_studio = sub.add_parser(
        "studio", help="serve Himmy Studio, the local web GUI (needs studio extra)"
    )
    p_studio.add_argument("--host", default="127.0.0.1")
    p_studio.add_argument("--port", type=int, default=8765)
    p_studio.add_argument(
        "--no-browser", action="store_true", help="don't open a browser window"
    )
    p_studio.set_defaults(func=commands.cmd_studio)

    p_doctor = sub.add_parser("doctor", help="report extras, providers, and keys")
    p_doctor.add_argument(
        "--storage",
        action="store_true",
        help="also report the storage backend, migrations, and SQLite stores",
    )
    p_doctor.set_defaults(func=commands.cmd_doctor)

    p_tools = sub.add_parser("tools", help="list built-in tool packs and their tools")
    p_tools.set_defaults(func=commands.cmd_tools)

    p_skills = sub.add_parser(
        "skills", help="list available skills (built-in + project-local)"
    )
    p_skills.add_argument(
        "name", nargs="?", help="show full detail for one skill instead of listing"
    )
    p_skills.set_defaults(func=commands.cmd_skills)

    p_prices = sub.add_parser(
        "prices", help="model price table: sync (refresh), show <model>, or list"
    )
    p_prices.add_argument(
        "action", nargs="?", choices=["sync", "show", "list"], help="default: list"
    )
    p_prices.add_argument("model", nargs="?", help="model name for `show`")
    p_prices.add_argument("--url", help="override the price source URL for `sync`")
    p_prices.set_defaults(func=commands.cmd_prices)

    p_trace = sub.add_parser("trace", help="inspect saved run traces")
    p_trace.add_argument("thread", nargs="?", help="thread id to show the timeline for")
    p_trace.add_argument("--limit", type=int, default=10, help="recent runs to list")
    p_trace.set_defaults(func=commands.cmd_trace)

    # workflow — run a declarative workflow.yaml through the orchestrator.
    p_workflow = sub.add_parser(
        "workflow", help="run a declarative workflow.yaml through the orchestrator"
    )
    wf_sub = p_workflow.add_subparsers(dest="action")
    p_wf_run = wf_sub.add_parser("run", help="run a workflow.yaml")
    p_wf_run.add_argument("-f", "--file", help="path to a workflow.yaml")
    p_wf_run.add_argument(
        "--state",
        action="append",
        metavar="KEY=VALUE",
        help="seed initial state (repeatable; value JSON-decoded if possible)",
    )
    p_wf_run.add_argument("--timeout", type=float, help="overall run timeout (seconds)")
    p_wf_run.add_argument(
        "--provider", choices=PROVIDERS, help="inference provider (default: auto)"
    )
    p_wf_run.add_argument("--model", help="model key/name for the provider")
    p_wf_run.add_argument(
        "--role",
        help="RBAC role for this run; restricts gated-tool execution "
        "(else HIMMY_ROLE env / himmy.toml [rbac] role / admin)",
    )
    p_wf_run.add_argument(
        "--json", action="store_true", help="print the full result as JSON"
    )
    p_wf_run.set_defaults(func=_cmd_workflow)
    p_wf_init = wf_sub.add_parser("init", help="scaffold a minimal workflow.yaml")
    p_wf_init.add_argument(
        "directory", nargs="?", default=".", help="where to write workflow.yaml"
    )
    p_wf_init.add_argument(
        "--force", action="store_true", help="overwrite an existing file"
    )
    p_wf_init.set_defaults(func=_cmd_workflow)
    # also allow bare `himmy workflow -f ...` (no subcommand) to run:
    p_workflow.add_argument("-f", "--file", dest="file", help=argparse.SUPPRESS)
    p_workflow.add_argument("--scaffold", action="store_true", help=argparse.SUPPRESS)
    p_workflow.add_argument("--role", dest="role", help=argparse.SUPPRESS)
    p_workflow.set_defaults(func=_cmd_workflow, action=None)

    # orchestrate — run a team.yaml through a selectable real orchestrator.
    p_orch = sub.add_parser(
        "orchestrate",
        help="run a team.yaml through a multi-agent mode "
        "(hierarchical/group_chat/reflection/graph)",
    )
    p_orch.add_argument(
        "--mode",
        choices=("hierarchical", "group_chat", "reflection", "graph"),
        default="hierarchical",
        help="orchestration mode (default: hierarchical)",
    )
    p_orch.add_argument("-f", "--file", required=True, help="path to a team.yaml")
    p_orch.add_argument("-p", "--prompt", help="the prompt to route through the team")
    p_orch.add_argument(
        "--provider", choices=PROVIDERS, help="inference provider (default: auto)"
    )
    p_orch.add_argument("--model", help="model key/name for the provider")
    p_orch.add_argument(
        "--role",
        help="RBAC role for this run; restricts gated-tool execution "
        "(else HIMMY_ROLE env / himmy.toml [rbac] role / admin)",
    )
    p_orch.add_argument(
        "--max-rounds",
        dest="max_rounds",
        type=int,
        default=None,
        help="group_chat: rounds to run (default: one per member)",
    )
    p_orch.add_argument("--json", action="store_true", help="print the full record")
    p_orch.set_defaults(func=_cmd_orchestrate)

    # guardrails — list the built-in guardrails.
    p_guardrails = sub.add_parser(
        "guardrails",
        help="list the built-in guardrails (pii, injection, nepal_pii, grounding, dlp)",
    )
    p_guardrails.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    p_guardrails.set_defaults(func=_cmd_guardrails)

    # lineage — trace a run back to its inferences/tool calls/results.
    p_lineage = sub.add_parser(
        "lineage",
        help="show a run's provenance tree (run -> inferences -> tools -> results)",
    )
    p_lineage.add_argument(
        "run_id",
        nargs="?",
        help="run id (thread id) to trace; defaults to the most recent traced run",
    )
    p_lineage.add_argument(
        "--dot",
        action="store_true",
        help="emit Graphviz DOT instead of the indented tree",
    )
    p_lineage.set_defaults(func=_cmd_lineage)

    # whoami — show the active CLI role and the permissions it grants.
    p_whoami = sub.add_parser(
        "whoami", help="show the active CLI role and the permissions it grants"
    )
    p_whoami.add_argument(
        "--role",
        help="override the active role (else HIMMY_ROLE env / himmy.toml [rbac] role "
        "/ admin)",
    )
    p_whoami.set_defaults(func=_cmd_whoami)

    # mcp — manage the stdio MCP servers wired into an agent.yaml.
    p_mcp = sub.add_parser(
        "mcp", help="manage the stdio MCP servers wired into an agent.yaml"
    )
    p_mcp.add_argument(
        "action",
        nargs="?",
        choices=["list", "add", "test", "remove"],
        help="default: list",
    )
    p_mcp.add_argument("name", nargs="?", help="friendly server name (add/test/remove)")
    p_mcp.add_argument("-f", "--file", help="path to an agent.yaml (default: nearest)")
    p_mcp.add_argument("--command", help="executable to launch (add)")
    p_mcp.add_argument(
        "--arg", action="append", help="one command arg; repeat for several (add)"
    )
    p_mcp.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="env var for the server (add)",
    )
    p_mcp.add_argument("--cwd", help="working directory for the server (add)")
    p_mcp.add_argument("--prefix", help="namespace tool names, e.g. github_ (add)")
    p_mcp.add_argument(
        "--tool", action="append", help="bind only this tool; repeat for several (add)"
    )
    p_mcp.add_argument(
        "--requires-approval",
        action="store_true",
        help="gate this server's tools behind approval (add)",
    )
    p_mcp.set_defaults(func=_cmd_mcp)

    # connectors — manage the connectors that bridge agents to Slack/Discord/etc.
    p_conn = sub.add_parser(
        "connectors",
        help="list/configure/enable/test the connectors that bridge agents to "
        "external systems",
    )
    p_conn.add_argument(
        "action",
        nargs="?",
        choices=["list", "set", "enable", "disable", "test"],
        help="default: list",
    )
    p_conn.add_argument(
        "name", nargs="?", help="connector name (set/enable/disable/test)"
    )
    p_conn.add_argument(
        "--secret",
        action="append",
        metavar="NAME=VALUE",
        help="a credential to write, by secret NAME (set; repeat for several)",
    )
    p_conn.add_argument(
        "--allowlist",
        help="comma-separated allow-list (channels/repos/senders) for this connector (set)",
    )
    p_conn.add_argument(
        "--surface",
        choices=["outbound", "inbound"],
        help="which surface to enable/disable (default: outbound)",
    )
    p_conn.set_defaults(func=_cmd_connectors)

    # skill — run a single named skill as a focused sub-agent.
    p_skill = sub.add_parser(
        "skill", help="run a named skill as a focused sub-agent on a prompt"
    )
    p_skill.add_argument(
        "name", help="the skill (capability) to run; see `himmy skills`"
    )
    p_skill.add_argument(
        "words",
        nargs="*",
        metavar="prompt",
        help="the prompt for the skill, as plain words (alternative to -p)",
    )
    p_skill.add_argument("-p", "--prompt", help="the prompt to run the skill on")
    p_skill.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="inference provider (default: auto pydantic-ai→stub)",
    )
    p_skill.add_argument("--model", help="model key/name for the provider")
    p_skill.add_argument(
        "--role",
        help="RBAC role for this run; restricts gated-tool execution "
        "(else HIMMY_ROLE env / himmy.toml [rbac] role / admin)",
    )
    p_skill.set_defaults(func=_cmd_skill)

    add_runs_parser(sub)
    add_routines_parser(sub)
    add_recommendations_parser(sub)
    add_context_parser(sub)
    add_dashboard_parser(sub)
    add_consent_parser(sub)
    add_audit_parser(sub)
    add_seclog_parser(sub)

    return parser


def _command_names(parser: argparse.ArgumentParser) -> set[str]:
    """All registered subcommand names (introspected, so it can't drift)."""
    for action in parser._actions:  # noqa: SLF001 - argparse keeps these stable
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return set(action.choices)
    return set()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command handler.

    Two frontier-CLI affordances on top of plain argparse: bare ``himmy`` on a
    real terminal opens the splash and drops into the chat REPL, and anything
    that isn't a known subcommand is treated as a question for the agent —
    ``himmy "what's the weather in Kathmandu?"`` just answers.
    """
    args_list = list(argv) if argv is not None else sys.argv[1:]
    # Top-level `himmy -c` / `himmy --continue`: open the REPL continuing the
    # most recent ('last') session, exactly like `himmy chat -c`. Handled before
    # subcommand parsing so the bare flag works (and -c can't collide with a
    # subcommand's own -c, since a subcommand name would precede it).
    if args_list and args_list[0] in ("-c", "--continue"):
        args_list = ["chat", *args_list]
    if not args_list:
        # Bare `himmy` is a first contact, not a mistake: splash, then converse.
        from himmy.cli.banner import print_banner

        code = print_banner()
        if sys.stdin.isatty() and sys.stdout.isatty():
            repl_args = argparse.Namespace(
                file=None,
                name=None,
                instruction=None,
                provider=None,
                model=None,
                role=None,
                message=None,
                session=None,
                continue_last=False,
                yolo=False,
                safe=False,
            )
            return commands.cmd_chat(repl_args)
        return code
    parser = build_parser()
    known = _command_names(parser)
    first = args_list[0]
    if not first.startswith("-") and first not in known:
        import difflib

        close = difflib.get_close_matches(first, known, n=1, cutoff=0.8)
        if close and len(args_list) <= 2:
            print(
                f"error: unknown command {first!r} — did you mean {close[0]!r}? "
                f'(quote your text to ask the agent: himmy "{first} …")',
                file=sys.stderr,
            )
            return 2
        # Not a command — it's a question. `himmy yo what's the weather` answers.
        args_list = ["run", *args_list]
    args = parser.parse_args(args_list)
    try:
        return int(args.func(args))
    except HimmyError as exc:
        # stderr, so a piped stdout never swallows the failure
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
