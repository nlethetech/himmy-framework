"""Trajectory-aware hints (P2): advise from the run's tool-call SEQUENCE, not just counts.

The P1 learned-hints adapter says "tool X failed N of its last M calls". That is a
per-tool aggregate — it is blind to ORDER and REPETITION, the two failure modes a small
model most often falls into: calling the same tool over and over (a loop) and reaching for
a known-flaky tool *first*. This advisor mines the recent ``TOOL_CALLED`` sequence (scoped
to the run's workspace, the same scope the reputation miner uses) into two structural,
privacy-safe hints:

* **looping** — one tool dominates the recent window with many back-to-back repeats, the
  classic "the model is stuck calling X again and again" pattern. The hint nudges it to
  stop and answer / try a different tool.
* **don't-call-first** — a tool that recently tends to be the FIRST call AND is below the
  reliability floor, so the model is told not to open with it.

Both are derived from tool NAMES + counts only (never args / results / prompts), mirroring
the privacy contract of the P1 hint. The advisor is best-effort and fail-open: any storage
error yields no hint, exactly like the rest of the learning kernel.

This is deliberately a separate, opt-in lever (``trajectory_hints``) layered on top of the
P1 hint: with it off the learned-hints adapter renders precisely the P1 string, byte-identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from himmy.core.events import EventType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.storage.protocols import EventLog

logger = logging.getLogger("himmy.services.learning")

#: How many of the most-recent TOOL_CALLED events the advisor inspects — the run's recent
#: call sequence. Bounded so the read is a small index seek, like the reputation window.
DEFAULT_SEQUENCE_WINDOW = 30

#: Minimum back-to-back repeats of one tool in the window before the looping hint fires.
#: 3 consecutive identical calls is the smallest run that is unambiguously a loop, not a
#: legitimate retry-then-succeed.
DEFAULT_LOOP_THRESHOLD = 3


@dataclass(frozen=True)
class TrajectorySignal:
    """The structural observations mined from the recent call sequence (all best-effort).

    ``looping_tool`` is the tool stuck in a back-to-back repeat run (or ``None``);
    ``max_repeat`` is the longest such run. ``first_call_tools`` is the set of tools that
    opened the recent window — candidates for a don't-call-first hint when also unreliable.
    """

    looping_tool: str | None
    max_repeat: int
    first_call_tools: tuple[str, ...]


class TrajectoryAdvisor:
    """Mine the recent ``TOOL_CALLED`` sequence into structural, ordering-aware hints.

    Reads the workspace-scoped recent call window from the SAME event log the reputation
    miner uses (no second backend). Every read is bounded and every error is swallowed to
    "no signal", so the advisor never breaks or slows a run.
    """

    def __init__(
        self,
        event_log: EventLog,
        *,
        window: int = DEFAULT_SEQUENCE_WINDOW,
        loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
        workspace_id: str | None = None,
    ) -> None:
        self._events = event_log
        self._window = max(2, int(window))
        self._loop_threshold = max(2, int(loop_threshold))
        self._workspace_id = workspace_id

    async def analyze(self) -> TrajectorySignal:
        """Read the recent call window and return its :class:`TrajectorySignal` (best-effort)."""
        try:
            names = await self._recent_call_names()
        except Exception:  # pragma: no cover - defensive: learning never breaks a run
            logger.debug("trajectory read failed; no signal", exc_info=True)
            return TrajectorySignal(looping_tool=None, max_repeat=0, first_call_tools=())
        return self._signal_from(names)

    async def _recent_call_names(self) -> list[str]:
        """Return the recent tool-call NAMES, oldest-first (chronological order)."""
        events = await self._events.list_events(
            event_type=EventType.TOOL_CALLED,
            workspace_id=self._workspace_id,
            limit=self._window,
            newest_first=True,
        )
        # ``newest_first`` gives most-recent-first; reverse to chronological so the
        # repeat-run + first-call analysis reads in the order the model actually called.
        names = [
            str((e.payload or {}).get("tool_name") or "")
            for e in reversed(events)
        ]
        return [n for n in names if n]

    def _signal_from(self, names: list[str]) -> TrajectorySignal:
        """Compute the looping + first-call structure from an ordered name sequence."""
        looping_tool: str | None = None
        max_repeat = 0
        if names:
            # Longest run of back-to-back identical calls.
            run_tool = names[0]
            run_len = 1
            best_tool, best_len = names[0], 1
            for name in names[1:]:
                if name == run_tool:
                    run_len += 1
                else:
                    run_tool, run_len = name, 1
                if run_len > best_len:
                    best_tool, best_len = run_tool, run_len
            if best_len >= self._loop_threshold:
                looping_tool, max_repeat = best_tool, best_len
        # The tool that opened the window is the don't-call-first candidate. We surface the
        # single most-recent opener (head of the chronological sequence); the adapter cross-
        # checks it against the reliability floor before ever emitting a hint about it.
        first_call_tools = (names[0],) if names else ()
        return TrajectorySignal(
            looping_tool=looping_tool,
            max_repeat=max_repeat,
            first_call_tools=first_call_tools,
        )


def render_trajectory_hints(
    signal: TrajectorySignal,
    *,
    unreliable_first_tools: tuple[str, ...] = (),
) -> list[str]:
    """Render a :class:`TrajectorySignal` into short, privacy-safe hint lines.

    ``unreliable_first_tools`` is the subset of ``signal.first_call_tools`` the caller has
    confirmed are below the reliability floor (the adapter passes its own reputation read),
    so a don't-call-first hint only fires for a tool that is BOTH an opener AND flaky — a
    healthy opener is left alone. Returns ``[]`` when there is nothing structural to say.
    """
    lines: list[str] = []
    if signal.looping_tool and signal.max_repeat >= 2:
        lines.append(
            f'- you appear to be looping on tool "{signal.looping_tool}" '
            f"({signal.max_repeat} calls in a row) — stop and answer with what you have, "
            "or try a different tool."
        )
    for tool in unreliable_first_tools:
        if tool in signal.first_call_tools:
            lines.append(
                f'- avoid calling "{tool}" first — it has been unreliable lately; '
                "gather context with a more reliable tool before reaching for it."
            )
    return lines


__all__ = [
    "DEFAULT_LOOP_THRESHOLD",
    "DEFAULT_SEQUENCE_WINDOW",
    "TrajectoryAdvisor",
    "TrajectorySignal",
    "render_trajectory_hints",
]
