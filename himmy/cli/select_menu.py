"""An interactive, arrow-key model selector for the in-chat ``/model`` command.

``/model`` with no args becomes a LIVE menu: the user scrolls the model list with
the arrow keys (or ``j``/``k``) and presses Enter to switch — no typing the full
``/model <name> <provider>``. This module is the selector; the REPL builds the
entry list (via :func:`himmy.cli.model_picker.build_picker_entries`, the same data
the static printer uses) and calls :func:`select_model`.

Design constraints (deliberate):

* **Stdlib only.** Raw key reads use ``termios`` + ``tty`` on POSIX. No curses, no
  third-party TUI deps. The REPL elsewhere uses plain ``input()`` + ``readline``.
* **Never wedge the terminal.** Raw/cbreak mode is entered under a ``try``/``finally``
  that ALWAYS restores the original ``termios`` settings via ``termios.tcsetattr`` —
  on a clean Enter, on Esc, on Ctrl-C, and on any exception. A stuck terminal is the
  worst possible failure for an interactive menu, so restoration is unconditional.
* **Testable with no real terminal.** ``read_key`` is injectable: a zero-arg callable
  returning the next key token. Tests drive navigation deterministically without a TTY.
  When ``read_key`` is ``None`` the selector reads from ``sys.stdin`` in raw mode.
* **Graceful fallback.** If raw mode is unavailable (no ``termios``, not a real TTY,
  ``isatty`` False) :func:`select_model` returns ``None`` immediately so the caller
  prints the existing static list instead.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any

# ----------------------------------------------------------------- key tokens

#: The decoded key tokens the selector understands. Raw-mode reads normalize the
#: terminal's byte sequences to these; an injected ``read_key`` returns them directly.
KEY_UP = "up"
KEY_DOWN = "down"
KEY_ENTER = "enter"
KEY_CANCEL = "cancel"


def _decode_key(token: str) -> str:
    """Map a raw key token (escape sequence / char) to a logical key, or ``""``.

    Accepts both the raw bytes a terminal emits (``"\\x1b[A"`` for Up, ``"\\r"`` for
    Enter, ``"\\x03"`` for Ctrl-C …) AND the logical tokens (``"up"``/``"down"``/
    ``"enter"``/``"cancel"``) so an injected ``read_key`` can speak either dialect.
    Unrecognized tokens return ``""`` (ignored by the loop).
    """
    if token in (KEY_UP, KEY_DOWN, KEY_ENTER, KEY_CANCEL):
        return token
    # Arrows: full ANSI escape sequences, plus vi-style j/k.
    if token in ("\x1b[A", "\x1bOA", "k"):
        return KEY_UP
    if token in ("\x1b[B", "\x1bOB", "j"):
        return KEY_DOWN
    # Enter (CR or LF).
    if token in ("\r", "\n"):
        return KEY_ENTER
    # Cancel: bare Esc, 'q', or Ctrl-C.
    if token in ("\x1b", "q", "Q", "\x03"):
        return KEY_CANCEL
    return ""


def _read_key_raw() -> str:
    """Read ONE logical key from ``sys.stdin`` already in raw/cbreak mode.

    Reads a single byte; if it is ESC, peeks (non-blocking) for a ``[``/``O`` +
    final byte to assemble an arrow sequence, otherwise treats the lone ESC as
    cancel. Returns a raw token that :func:`_decode_key` understands. Assumes the
    caller has put the fd into raw mode (and will restore it) — see ``_RawMode``.
    """
    import select

    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    # Possible escape sequence. Peek without blocking: a lone ESC (no follow-on
    # bytes within a tiny window) is a cancel; a CSI/SS3 sequence has more bytes.
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    if not rlist:
        return "\x1b"
    second = sys.stdin.read(1)
    if second not in ("[", "O"):
        return "\x1b"
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    if not rlist:
        return "\x1b"
    third = sys.stdin.read(1)
    return "\x1b" + second + third


class _RawMode:
    """Context manager that puts ``fd`` into cbreak mode and ALWAYS restores it.

    ``__enter__`` saves the current ``termios`` attributes and switches to cbreak
    (raw enough for char-at-a-time reads while keeping signal handling sane).
    ``__exit__`` restores the saved attributes via ``tcsetattr`` UNCONDITIONALLY —
    so an exception, Esc, or Ctrl-C inside the menu can never leave the terminal
    wedged. Raises ``OSError``/``ImportError`` from ``__enter__`` if the fd is not
    a real terminal; the caller catches that and falls back.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._saved: Any = None

    def __enter__(self) -> _RawMode:
        import termios
        import tty

        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._saved is None:
            return
        import termios

        # Restore on EVERY exit path. TCSADRAIN waits for pending output so the
        # final redraw lands before the cooked terminal returns.
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


def _selectable_indices(entries: Sequence[Any]) -> list[int]:
    """Indices of selectable (``kind == "model"``) rows, in display order."""
    return [i for i, e in enumerate(entries) if e.get("kind") == "model"]


def _render(
    entries: Sequence[Any],
    *,
    cursor: int,
    palette: dict[str, str],
    stream: Any,
) -> int:
    """Draw the menu to ``stream`` and return the line count (for in-place redraw).

    ``cursor`` is an index INTO ``entries`` (a model row). The cursor row gets a
    bright ``›`` marker (reverse video where styling is on); headers are dimmed
    provider names; model rows show ``<model> <provider>  <label>`` with an
    ``(active)`` tag preserved from the entry. Returns how many lines were written
    so the next redraw can move the cursor up exactly that many and overwrite.
    """
    c = palette
    reverse = c.get("reverse", "")
    lines: list[str] = []
    lines.append(f"{c.get('snow', '')}select a model "
                 f"{c.get('faint', '')}(↑/↓ move · enter switch · esc cancel)"
                 f"{c.get('reset', '')}")
    for i, e in enumerate(entries):
        if e.get("kind") == "header":
            lines.append(f"  {c.get('snow', '')}{e.get('label', '')}{c.get('reset', '')}")
            continue
        selected = i == cursor
        marker = (
            f"{c.get('crimson', '')}›{c.get('reset', '')} " if selected else "  "
        )
        label = e.get("label", "")
        active = (
            f"{c.get('green', '')} (active){c.get('reset', '')}"
            if e.get("is_active")
            else ""
        )
        body = (
            f"{c.get('gold', '')}{e.get('model', '')}{c.get('reset', '')} "
            f"{c.get('faint', '')}{e.get('provider', '')}{c.get('reset', '')}  "
            f"{c.get('faint', '')}{label}{c.get('reset', '')}{active}"
        )
        if selected and reverse:
            body = f"{reverse}{body}{c.get('reset', '')}"
        lines.append(f"  {marker}{body}")
    text = "\n".join(lines)
    stream.write(text + "\n")
    stream.flush()
    return len(lines)


def _clear(stream: Any, line_count: int) -> None:
    """Move the cursor up ``line_count`` lines and clear to end of screen.

    Lets the next :func:`_render` overwrite the previous frame in place, so the
    menu feels live rather than reprinting a growing transcript. No-op for zero
    lines. Uses bare ANSI (``\\x1b[<n>A`` up, ``\\x1b[J`` clear-to-end).
    """
    if line_count <= 0:
        return
    stream.write(f"\x1b[{line_count}A\x1b[J")
    stream.flush()


def select_model(
    entries: Sequence[Any],
    *,
    active_index: int | None = None,
    palette: dict[str, str] | None = None,
    stream: Any = None,
    read_key: Callable[[], str] | None = None,
    isatty: bool = True,
) -> tuple[str, str] | None:
    """Interactively pick a model row from ``entries``; return ``(provider, model)``.

    ``entries`` is the shared picker list (provider HEADERS + selectable MODEL rows)
    from :func:`himmy.cli.model_picker.build_picker_entries`. Arrow movement skips
    headers; Enter selects the highlighted model; Esc/``q``/Ctrl-C cancel. The
    ACTIVE model (``active_index``, an index into ``entries`` or ``None``) is
    pre-selected; otherwise the first selectable row is.

    ``read_key`` is an injectable zero-arg callable returning the next key token
    (raw bytes OR a logical token — see :func:`_decode_key`); when ``None`` the
    selector reads ``sys.stdin`` in raw mode. ``isatty`` gates raw mode: when it is
    False (or ``termios`` is unavailable, or there is no real fd) the function
    returns ``None`` immediately so the caller can fall back to the static list.

    Returns the chosen ``(provider, model)`` tuple, or ``None`` on cancel /
    non-tty / unavailable terminal. Guarantees the terminal is restored on every
    path when it drove a real raw-mode session.
    """
    stream = stream if stream is not None else sys.stderr
    palette = palette if palette is not None else {}

    selectable = _selectable_indices(entries)
    if not selectable:
        return None

    # Start on the active row when given + selectable, else the first model row.
    cursor = selectable[0]
    if active_index is not None and active_index in selectable:
        cursor = active_index

    # ----- choose the key source. An injected read_key needs no terminal at all.
    if read_key is not None:
        return _run_loop(
            entries, selectable, cursor=cursor, palette=palette,
            stream=stream, read_key=read_key,
        )

    # Real raw-mode path: refuse unless we have a true TTY + a usable fd + termios.
    if not isatty:
        return None
    try:
        import termios  # noqa: F401  (probe availability)
        import tty  # noqa: F401
    except ImportError:
        return None
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):
        return None
    if not getattr(sys.stdin, "isatty", lambda: False)():
        return None

    try:
        with _RawMode(fd):
            return _run_loop(
                entries, selectable, cursor=cursor, palette=palette,
                stream=stream, read_key=_read_key_raw,
            )
    except OSError:
        # tcgetattr/setcbreak failed → not a controllable terminal. Fall back.
        return None


def _run_loop(
    entries: Sequence[Any],
    selectable: list[int],
    *,
    cursor: int,
    palette: dict[str, str],
    stream: Any,
    read_key: Callable[[], str],
) -> tuple[str, str] | None:
    """The draw → read-key → move → redraw loop (shared by raw + injected paths).

    Separated from :func:`select_model` so the terminal-restore wrapper there stays
    a thin try/finally and this stays pure key→state logic. Draws once, then on each
    keypress moves the cursor (clamped, headers skipped) and redraws IN PLACE.
    Returns the chosen ``(provider, model)`` on Enter or ``None`` on cancel.
    """
    pos = selectable.index(cursor) if cursor in selectable else 0
    line_count = _render(
        entries, cursor=selectable[pos], palette=palette, stream=stream
    )
    while True:
        token = read_key()
        key = _decode_key(token)
        if key == KEY_CANCEL:
            return None
        if key == KEY_ENTER:
            chosen = entries[selectable[pos]]
            return (str(chosen.get("provider", "")), str(chosen.get("model", "")))
        if key == KEY_UP:
            pos = max(0, pos - 1)
        elif key == KEY_DOWN:
            pos = min(len(selectable) - 1, pos + 1)
        else:
            continue  # unknown key → no redraw, keep waiting
        _clear(stream, line_count)
        line_count = _render(
            entries, cursor=selectable[pos], palette=palette, stream=stream
        )


__all__ = ["select_model"]
