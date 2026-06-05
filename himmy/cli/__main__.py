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
    p_run.set_defaults(func=commands.cmd_run)

    p_chat = sub.add_parser("chat", help="interactive chat keeping one thread")
    _add_agent_flags(p_chat)
    p_chat.add_argument("--message", help="run a single turn non-interactively")
    p_chat.set_defaults(func=commands.cmd_chat)

    p_init = sub.add_parser("init", help="scaffold an agent.yaml + tools.py")
    p_init.add_argument("directory", nargs="?", default=".", help="target directory")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.set_defaults(func=commands.cmd_init)

    p_serve = sub.add_parser("serve", help="serve the FastAPI BFF (needs api extra)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=commands.cmd_serve)

    p_doctor = sub.add_parser("doctor", help="report extras, providers, and keys")
    p_doctor.set_defaults(func=commands.cmd_doctor)

    p_tools = sub.add_parser("tools", help="list built-in tool packs and their tools")
    p_tools.set_defaults(func=commands.cmd_tools)

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
