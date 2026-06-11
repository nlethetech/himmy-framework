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
from himmy.cli.provider import PROVIDERS
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

    add_consent_parser(sub)
    add_audit_parser(sub)

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
                message=None,
                session=None,
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
