"""Desk-style live rendering for the himmy CLI: palette, spinner, run events.

The look is the one proven in the desk films (HIMMY/AGENTS lockup, crimson
prompt, gold tools, green checks on warm stone): while an agent runs, the CLI
shows what is actually happening — a braille spinner for each inference with
the model it is waiting on, then ``✓ inference ok (920ms · sonnet · claude-cli)``,
gold ``⚙ tool call`` lines as tools fire, green ``✓ tool done`` as they land —
instead of dead silence followed by a wall of text.

Everything here is decoration, never data: all live lines go to **stderr** and
only when it is a real TTY (``NO_COLOR``/pipes get nothing), so stdout stays
byte-identical for scripts. The palette mirrors the film scenes (truecolor)
with a 256-color fallback for terminals that don't advertise ``COLORTERM``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from typing import Any

#: Truecolor palette — the exact values from the demo-film scenes.
_STYLES_TRUE = {
    "crimson": "\x1b[38;2;232;41;74m",
    "gold": "\x1b[38;2;201;162;39m",
    "green": "\x1b[38;2;127;165;106m",
    "snow": "\x1b[38;2;245;242;233m",
    "ink": "\x1b[38;2;233;230;220m",
    "dim": "\x1b[38;2;151;145;127m",
    "faint": "\x1b[38;2;92;87;73m",
    "bold": "\x1b[1m",
    "reset": "\x1b[0m",
}

#: 256-color fallback (macOS Terminal.app has no truecolor).
_STYLES_256 = {
    "crimson": "\x1b[38;5;197m",
    "gold": "\x1b[38;5;178m",
    "green": "\x1b[38;5;108m",
    "snow": "\x1b[38;5;255m",
    "ink": "\x1b[38;5;254m",
    "dim": "\x1b[38;5;246m",
    "faint": "\x1b[38;5;240m",
    "bold": "\x1b[1m",
    "reset": "\x1b[0m",
}

_PLAIN = {k: "" for k in _STYLES_TRUE}

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧")

#: ASCII spinner frames for font-limited renderers (the braille glyphs tofu-box in
#: agg/GIF and other minimal fonts). Used when HIMMY_ASCII_SPINNER=1 or NO_UNICODE.
SPINNER_FRAMES_ASCII = ("|", "/", "-", "\\")


def spinner_frames() -> tuple[str, ...]:
    """Pick the spinner frame set: ASCII fallback when the env opts out of unicode.

    Set ``HIMMY_ASCII_SPINNER=1`` (or ``NO_UNICODE``) to render a simple ``|/-\\``
    spinner instead of the braille one — for terminals / video renderers whose font
    can't draw the braille glyphs (they show as tofu boxes). Default is unchanged.
    """
    if os.environ.get("HIMMY_ASCII_SPINNER") or os.environ.get("NO_UNICODE"):
        return SPINNER_FRAMES_ASCII
    return SPINNER_FRAMES


def wants_live_ui(stream: Any = None) -> bool:
    """True when the stream is a TTY and color/animation is not opted out."""
    if os.environ.get("NO_COLOR") is not None or os.environ.get("HIMMY_NO_ANIM"):
        return False
    stream = stream if stream is not None else sys.stderr
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def styles(stream: Any = None) -> dict[str, str]:
    """The ANSI style map for ``stream`` (empty strings when styling is off)."""
    if not wants_live_ui(stream):
        return dict(_PLAIN)
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return dict(_STYLES_TRUE)
    return dict(_STYLES_256)


def render_markdown_lite(text: str, *, stream: Any = None) -> str:
    """``**bold**`` → bold snow, `` `code` `` → gold, on a styled TTY only.

    Models answer in light markdown; raw ``**`` asterisks on a terminal read as
    noise. Plain streams get the text untouched (scripts see what the model said).
    """
    c = styles(stream)
    if not c["reset"]:
        return text
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: f"{c['bold']}{c['snow']}{m.group(1)}{c['reset']}",
        text,
    )
    return re.sub(
        r"`([^`\n]+)`", lambda m: f"{c['gold']}{m.group(1)}{c['reset']}", text
    )


class Spinner:
    """A braille spinner on one stderr line: ``⠋ <label>`` → ``✓ <done>``.

    Runs on a daemon thread (80ms frames, like the films) so it animates while
    the event loop waits on the model. ``stop`` clears the line and optionally
    prints the final form. No-ops entirely on non-TTY streams.
    """

    def __init__(self, label: str, *, indent: str = "    ", stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._label = label
        self._indent = indent
        self._enabled = wants_live_ui(self._stream)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = time.monotonic()

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        self.started_at = time.monotonic()

        def _spin() -> None:
            c = styles(self._stream)
            frames = spinner_frames()
            i = 0
            while not self._stop.wait(0.08):
                frame = frames[i % len(frames)]
                self._stream.write(
                    f"\r{self._indent}{c['gold']}{frame}{c['reset']}"
                    f"{c['faint']} {self._label}{c['reset']}\x1b[K"
                )
                self._stream.flush()
                i += 1

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

    def stop(self, final: str | None = None) -> None:
        """Stop animating; replace the line with ``final`` (or just clear it)."""
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        self._stream.write("\r\x1b[K")
        if final is not None:
            self._stream.write(final + "\n")
        self._stream.flush()


class LiveRunUI:
    """Render run events as they happen — the film look, live.

    An ``on_event`` async callable for the runtime: inference shows a spinner
    that resolves into the timed ✓ line, tool calls land in gold the moment
    they fire. Decoration only: it never raises into the run, counts events
    for the closing footer, and goes fully silent off-TTY.
    """

    def __init__(self, *, model_label: str = "", stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._model_label = model_label
        self.enabled = wants_live_ui(self._stream)
        self._c = styles(self._stream)
        self._spinner: Spinner | None = None
        self.events = 0
        self.tool_calls = 0
        self.cost = 0.0

    # -- helpers -----------------------------------------------------------

    def _line(self, text: str) -> None:
        self._stream.write(text + "\n")
        self._stream.flush()

    def _tool_args(self, payload: dict[str, Any]) -> str:
        args = json.dumps(payload.get("tool_args", {}), default=str)
        return args if len(args) <= 60 else args[:57] + "…"

    # -- the event hook ----------------------------------------------------

    async def handle(self, event: Any) -> None:
        if not self.enabled:
            return
        try:
            self._render(event)
        except Exception:  # noqa: BLE001 - never let decoration break a run
            pass

    def _render(self, event: Any) -> None:
        c = self._c
        kind = getattr(event.event_type, "value", str(event.event_type))
        p = event.payload or {}
        self.events += 1
        self.cost += event.cost or 0.0

        if kind == "INFERENCE_REQUESTED":
            label = "inference"
            model = p.get("model_key") or ""
            tag = self._model_label or (model if model != "default" else "")
            if tag:
                label = f"inference · {tag}"
            self._spinner = Spinner(label, stream=self._stream)
            self._spinner.start()
        elif kind in ("INFERENCE_SUCCEEDED", "INFERENCE_FAILED"):
            ms = f"{event.latency_ms:.0f}ms" if event.latency_ms else "?ms"
            tag = f" · {self._model_label}" if self._model_label else ""
            if kind == "INFERENCE_SUCCEEDED":
                final = (
                    f"    {c['green']}✓{c['reset']}{c['faint']} inference ok"
                    f"  ({ms}{tag}){c['reset']}"
                )
            else:
                err = event.error or "error"
                final = (
                    f"    {c['crimson']}✗ inference FAILED{c['reset']}"
                    f"{c['faint']}  ({ms} · {err}){c['reset']}"
                )
            if self._spinner is not None:
                self._spinner.stop(final)
                self._spinner = None
            else:
                self._line(final)
        elif kind == "TOOL_CALLED":
            self.tool_calls += 1
            self._line(
                f"    {c['dim']}⚙ tool call {c['reset']}"
                f"{c['gold']}{p.get('tool_name', '')}{c['reset']}"
                f"{c['faint']} {self._tool_args(p)}{c['reset']}"
            )
        elif kind == "TOOL_COMPLETED":
            self._line(
                f"    {c['green']}✓{c['reset']}{c['dim']} tool done {c['reset']}"
                f"{c['gold']}{p.get('tool_name', '')}{c['reset']}"
            )
        elif kind == "TOOL_FAILED":
            self._line(
                f"    {c['crimson']}✗ tool FAILED {c['reset']}"
                f"{c['gold']}{p.get('tool_name', '')}{c['reset']}"
                f"{c['faint']} {event.error or ''}{c['reset']}"
            )
        elif kind == "AGENT_TURN_STARTED":
            turn = p.get("turn")
            if turn and turn != 1:
                self._line(f"  {c['faint']}○ turn #{turn}{c['reset']}")
        elif kind == "APPROVAL_REQUIRED":
            self._line(f"  {c['gold']}⏸ approval required{c['reset']}")
        elif kind == "AGENT_HANDOFF":
            self._line(
                f"  {c['dim']}⇄ handoff {p.get('from')} → {p.get('to')}{c['reset']}"
            )
        elif kind == "AGENT_DELEGATED":
            self._line(f"    {c['dim']}⇲ delegated {p.get('worker', '')}{c['reset']}")
        elif kind == "GUARDRAIL_APPLIED":
            from himmy.cli.guardrails_view import render_guardrail_event

            self._line(render_guardrail_event(event, c))

    # -- lifecycle ---------------------------------------------------------

    def finish(self) -> None:
        """Stop any straggling spinner; add a blank spacing line before the answer."""
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None
        if self.enabled and self.events:
            self._stream.write("\n")
            self._stream.flush()

    def footer(self) -> None:
        """The faint closing line: ``— 14 events · 2 tool call(s) · $0.2520``."""
        if not self.enabled or not self.events:
            return
        c = self._c
        cost = f" · ${self.cost:.4f}" if self.cost else ""
        self._line(
            f"\n  {c['faint']}— {self.events} events · {self.tool_calls} "
            f"tool call(s){cost}{c['reset']}"
        )


class TurnRecorder:
    """An in-memory ``on_event`` sink that remembers the LAST turn's events.

    Composed alongside :class:`LiveRunUI` (which only counts events on a TTY), this
    captures the full :class:`~himmy.core.events.RunEvent` list for ``/why`` AND a
    running cumulative cost so budget accounting works even when the live UI is
    disabled (pipes/CI). A new turn (``start_turn``) clears the per-turn buffer but
    keeps the cumulative cost; events that arrive after ``start_turn`` accrue into
    both ``events`` (this turn) and ``total_cost`` (the session).
    """

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.total_cost: float = 0.0
        self.total_events: int = 0
        self.total_tool_calls: int = 0

    def start_turn(self) -> None:
        """Begin recording a new turn (clears the per-turn event buffer)."""
        self.events = []

    async def handle(self, event: Any) -> None:
        self.events.append(event)
        self.total_events += 1
        try:
            if getattr(event.event_type, "value", "") == "TOOL_CALLED":
                self.total_tool_calls += 1
            self.total_cost += event.cost or 0.0
        except Exception:  # noqa: BLE001 - accounting must never break a run
            pass


def compose_event_handlers(*handlers: Any) -> Any:
    """Fan one runtime ``on_event`` out to several handlers (Nones skipped)."""
    active = [h for h in handlers if h is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    async def _fan(event: Any) -> None:
        for h in active:
            await h(event)

    return _fan
