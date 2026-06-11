"""The ``himmy`` no-args splash: the HIMMY / AGENTS lockup and the fastest next steps.

Running ``himmy`` with no arguments used to be an argparse error. Instead it now
prints the mark — HIMMY in ANSI Shadow block letters with a crimson fill over
AGENTS in stone grey (the same lockup used in the project's demo films). On an
interactive TTY the lockup *decodes in*: every row starts as scrambled block
glyphs and resolves top-to-bottom in a ~0.7s cascade, exactly like the demo.
Piped output (and ``HIMMY_NO_ANIM=1``) gets the plain static splash.

Color: truecolor only when the terminal advertises it (``COLORTERM`` is
``truecolor``/``24bit`` — iTerm2, kitty, etc.); otherwise 256-color codes so
macOS Terminal.app renders correctly instead of smearing 24-bit sequences into
background blocks. Applied only when stdout is a TTY and ``NO_COLOR`` is unset
(https://no-color.org/), so piped output stays clean.
"""

from __future__ import annotations

import os
import random
import sys
import time

#: (art line, role) — roles map to colors when color is enabled.
_LOCKUP: list[tuple[str, str]] = [
    ("██╗  ██╗██╗███╗   ███╗███╗   ███╗██╗   ██╗", "crimson"),
    ("██║  ██║██║████╗ ████║████╗ ████║╚██╗ ██╔╝", "crimson"),
    ("███████║██║██╔████╔██║██╔████╔██║ ╚████╔╝ ", "crimson"),
    ("██╔══██║██║██║╚██╔╝██║██║╚██╔╝██║  ╚██╔╝  ", "crimson"),
    ("██║  ██║██║██║ ╚═╝ ██║██║ ╚═╝ ██║   ██║   ", "crimson"),
    ("╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚═╝     ╚═╝   ╚═╝   ", "crimson"),
    ("", "stone"),
    (" █████╗  ██████╗ ███████╗███╗   ██╗████████╗███████╗", "stone"),
    ("██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔════╝", "stone"),
    ("███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ███████╗", "stone"),
    ("██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║", "stone"),
    ("██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████║", "stone"),
    ("╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝", "stone"),
]

#: 24-bit palette — terminals that advertise truecolor via COLORTERM.
_COLORS_TRUE = {
    "crimson": "\x1b[38;2;232;41;74m",  # the demo-film crimson (#E8294A)
    "summit": "\x1b[38;2;200;16;46m",  # deep crimson (#C8102E), kept for ▲ accents
    "stone": "\x1b[38;2;151;145;127m",  # warm stone grey (#97917F)
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "reset": "\x1b[0m",
}

#: 256-color palette — everything else (macOS Terminal.app has no truecolor).
_COLORS_256 = {
    "crimson": "\x1b[38;5;197m",
    "summit": "\x1b[38;5;160m",
    "stone": "\x1b[38;5;246m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "reset": "\x1b[0m",
}

# Only glyphs the ANSI Shadow font itself uses: a mid-decode pause then looks
# like shifting letterforms, never like corruption (▓▒░ read as a broken screen).
_SCRAMBLE = "█║═╔╗╚╝"


def supports_color(stream: object = None) -> bool:
    """True when ANSI color should be used: a TTY and ``NO_COLOR`` unset."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    stream = stream if stream is not None else sys.stdout
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def palette(*, color: bool) -> dict[str, str]:
    """The escape-code palette: truecolor, 256-color, or empty strings."""
    if not color:
        return dict.fromkeys(_COLORS_TRUE, "")
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return _COLORS_TRUE
    return _COLORS_256


def _tail_lines(c: dict[str, str]) -> list[str]:
    """Everything below the lockup: pitch line and the get-started block."""
    from himmy import __version__

    return [
        "",
        f"  {c['summit']}▲{c['reset']} {c['bold']}himmy{c['reset']} v{__version__} — the local-first agent framework",
        f"  {c['dim']}offline by default · zero API keys · every action audited{c['reset']}",
        "",
        "  get started:",
        f"    himmy init my-agent              {c['dim']}scaffold an agent{c['reset']}",
        f'    himmy run -f agent.yaml -p "…"   {c["dim"]}one-shot run{c["reset"]}',
        f"    himmy chat -f agent.yaml         {c['dim']}interactive thread{c['reset']}",
        f"    himmy --help                     {c['dim']}all commands{c['reset']}",
        "",
    ]


def render_banner(*, color: bool) -> str:
    """The full splash as static text, with or without ANSI color codes."""
    c = palette(color=color)
    lines = [""]
    lines += [f"{c[role]}{art}{c['reset']}" if art else "" for art, role in _LOCKUP]
    lines += _tail_lines(c)
    return "\n".join(lines)


def _scramble_line(art: str, rng: random.Random) -> str:
    return "".join(ch if ch == " " else rng.choice(_SCRAMBLE) for ch in art)


def _animate_lockup(c: dict[str, str], *, step_s: float = 0.03, lead: int = 3) -> None:
    """Decode the lockup in place: scrambled rows resolve top-to-bottom.

    Ends with one unconditional clean repaint of every row — if a renderer
    drops or batches intermediate frames (xterm.js under a recorder, a slow
    ssh hop), the settled state is still guaranteed to be the real art.
    """
    rng = random.Random()
    out = sys.stdout
    n = len(_LOCKUP)
    out.write("\n")
    for art, _role in _LOCKUP:
        out.write(
            f"{c['dim']}{_scramble_line(art, rng)}{c['reset']}\n" if art else "\n"
        )
    out.flush()
    try:
        for step in range(n + lead):
            out.write(f"\x1b[{n}A")
            for i, (art, role) in enumerate(_LOCKUP):
                if not art:
                    out.write("\x1b[2K\n")
                elif i < step - lead + 1:
                    out.write(f"\x1b[2K{c[role]}{art}{c['reset']}\n")
                else:
                    out.write(
                        f"\x1b[2K{c['dim']}{_scramble_line(art, rng)}{c['reset']}\n"
                    )
            out.flush()
            time.sleep(step_s)
    finally:
        # Settle pass: repaint everything resolved no matter what came before —
        # a Ctrl-C mid-cascade or a renderer that dropped frames still ends clean.
        out.write(f"\x1b[{n}A")
        for art, role in _LOCKUP:
            out.write(f"\x1b[2K{c[role]}{art}{c['reset']}\n" if art else "\x1b[2K\n")
        out.flush()


def print_banner() -> int:
    """Print the splash to stdout; the ``himmy`` no-args exit code (0).

    Animated decode on an interactive TTY; static text when piped, when
    ``NO_COLOR`` is set, or when ``HIMMY_NO_ANIM`` asks for stillness.
    """
    import shutil

    color = supports_color()
    # The in-place decode rewrites rows with cursor-up; a terminal narrower than
    # the art wraps lines and breaks that math into garbage. Static splash then.
    wide_enough = shutil.get_terminal_size().columns >= max(
        len(art) for art, _ in _LOCKUP
    )
    animate = color and wide_enough and os.environ.get("HIMMY_NO_ANIM") is None
    if animate:
        c = palette(color=True)
        _animate_lockup(c)
        print("\n".join(_tail_lines(c)))
    else:
        print(render_banner(color=color))
    return 0
