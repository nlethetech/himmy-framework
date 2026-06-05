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
from collections.abc import Sequence

from himmy import __version__
from himmy.cli import commands
from himmy.cli.provider import PROVIDERS
from himmy.core import HimmyError


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


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="himmy", description="Himmy agent CLI.")
    parser.add_argument(
        "-v", "--version", action="version", version=f"himmy {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a single prompt and print the answer")
    _add_agent_flags(p_run)
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
    p_run.set_defaults(func=commands.cmd_run)

    p_chat = sub.add_parser("chat", help="interactive chat keeping one thread")
    _add_agent_flags(p_chat)
    p_chat.add_argument("--message", help="run a single turn non-interactively")
    p_chat.add_argument(
        "--session", help="persist/resume this conversation by id (.himmy/sessions.db)"
    )
    p_chat.set_defaults(func=commands.cmd_chat)

    p_tg = sub.add_parser("telegram", help="run an agent as a live Telegram bot")
    _add_agent_flags(p_tg)
    p_tg.add_argument("--token", help="bot token (default: HIMMY_TELEGRAM_BOT_TOKEN)")
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
        "--suite", help="path to a suite.yaml (default: built-in core)"
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
    p_bench.add_argument("--json", help="also write the full results as JSON")
    p_bench.set_defaults(func=commands.cmd_bench)

    p_init = sub.add_parser("init", help="scaffold an agent.yaml + tools.py")
    p_init.add_argument("directory", nargs="?", default=".", help="target directory")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.add_argument(
        "--team", action="store_true", help="scaffold a team.yaml instead"
    )
    p_init.set_defaults(func=commands.cmd_init)

    p_serve = sub.add_parser("serve", help="serve the FastAPI BFF (needs api extra)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=commands.cmd_serve)

    p_doctor = sub.add_parser("doctor", help="report extras, providers, and keys")
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

    p_trace = sub.add_parser("trace", help="inspect saved run traces")
    p_trace.add_argument("thread", nargs="?", help="thread id to show the timeline for")
    p_trace.add_argument("--limit", type=int, default=10, help="recent runs to list")
    p_trace.set_defaults(func=commands.cmd_trace)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HimmyError as exc:
        print(f"error: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
