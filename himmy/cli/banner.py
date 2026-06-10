"""The ``himmy`` no-args splash: a snow-capped mountain and the fastest next steps.

Running ``himmy`` with no arguments used to be an argparse error. Instead it now
prints the mark — a block mountain with a crimson summit (the same ▲ used across
the project's branding) — the one-line pitch, and the three commands a new user
actually needs. Color is ANSI truecolor, applied only when stdout is a TTY and
``NO_COLOR`` is unset (https://no-color.org/), so piped output stays clean.
"""

from __future__ import annotations

import os
import sys

#: (art line, role) — roles map to colors when color is enabled.
_MOUNTAIN: list[tuple[str, str]] = [
    ("        ▲", "summit"),
    ("       ◢█◣", "snow"),
    ("      ◢███◣", "snow"),
    ("     ◢█████◣", "stone"),
    ("    ◢███████◣", "stone"),
    ("   ◢█████████◣", "stone"),
]

_COLORS = {
    "summit": "\x1b[38;2;200;16;46m",  # crimson
    "snow": "\x1b[97m",  # bright white
    "stone": "\x1b[38;5;245m",  # grey
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "reset": "\x1b[0m",
}


def supports_color(stream: object = None) -> bool:
    """True when ANSI color should be used: a TTY and ``NO_COLOR`` unset."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    stream = stream if stream is not None else sys.stdout
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def render_banner(*, color: bool) -> str:
    """The splash text, with or without ANSI color codes."""
    from himmy import __version__

    c = _COLORS if color else dict.fromkeys(_COLORS, "")
    lines = [""]
    lines += [f"{c[role]}{art}{c['reset']}" for art, role in _MOUNTAIN]
    lines += [
        "",
        f"  {c['bold']}himmy{c['reset']} v{__version__} — the local-first agent framework",
        f"  {c['dim']}offline by default · zero API keys · every action audited{c['reset']}",
        "",
        "  get started:",
        f"    himmy init my-agent              {c['dim']}scaffold an agent{c['reset']}",
        f'    himmy run -f agent.yaml -p "…"   {c["dim"]}one-shot run{c["reset"]}',
        f"    himmy chat -f agent.yaml         {c['dim']}interactive thread{c['reset']}",
        f"    himmy --help                     {c['dim']}all commands{c['reset']}",
        "",
    ]
    return "\n".join(lines)


def print_banner() -> int:
    """Print the splash to stdout; the ``himmy`` no-args exit code (0)."""
    print(render_banner(color=supports_color()))
    return 0
