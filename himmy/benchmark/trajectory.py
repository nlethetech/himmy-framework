"""Structured tool-call trajectory captured per benchmark trial.

The runner records the agent's tool calls as *turns of calls* — a
``list[list[RecordedCall]]`` — preserving both the argument payloads (the dominant
small-model failure is "right tool, WRONG args") and within-turn parallelism (a model
that emits two calls in one assistant turn vs two sequential turns). The flat
name-only ``list[str]`` view that the legacy trajectory graders (``first_tool`` /
``tool_sequence`` / ``max_tool_calls`` / …) run against is derived from this, so those
graders keep passing unchanged.

A :class:`Trajectory` is the canonical richer object the new argument-level and
parallelism graders (``tool_called_with`` / ``arg_equals`` / ``arg_matches`` /
``result_contains`` / ``parallel_in_turn`` / ``max_parallel``) run against. It is built
from the typed :class:`~himmy.services.inference.models.ToolCallRecord` /
:class:`~himmy.services.inference.models.ToolReturnRecord` exchanges the agent loop (and
the orchestrators) already produce, so no information is fabricated — args/results that
were never captured are simply absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from himmy.services.inference.models import ToolCallRecord, ToolReturnRecord


@dataclass(frozen=True)
class RecordedCall:
    """One tool call the agent made, with its arguments and (when known) its result.

    ``args`` is the exact argument mapping the model produced. ``result`` is the tool's
    returned content (``None`` when no return was paired by ``tool_call_id`` — e.g. the
    provider reported a call but no result was captured). Both are read-only: graders
    inspect them, never mutate them.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    has_result: bool = False


@dataclass(frozen=True)
class Trajectory:
    """The agent's tool calls grouped into the turns they were emitted in.

    ``turns`` is ordered (turn order, then within-turn call order, with repeats). The
    flat name-only sequence (:attr:`names`) and the flat call sequence (:attr:`calls`)
    are derived views the graders use; nothing is recomputed lazily beyond a flatten.
    """

    turns: list[list[RecordedCall]] = field(default_factory=list)

    @property
    def calls(self) -> list[RecordedCall]:
        """Every call across all turns, in order, with repeats."""
        return [c for turn in self.turns for c in turn]

    @property
    def names(self) -> list[str]:
        """The flat name-only sequence the legacy trajectory graders run against."""
        return [c.name for c in self.calls]

    @property
    def max_parallel(self) -> int:
        """The largest number of calls the agent emitted in a single turn."""
        return max((len(turn) for turn in self.turns), default=0)

    @classmethod
    def from_turns(
        cls,
        turns: Iterable[tuple[Iterable[ToolCallRecord], Iterable[ToolReturnRecord]]],
    ) -> Trajectory:
        """Build a trajectory from ``(tool_calls, tool_returns)`` pairs per turn.

        Results are paired to calls by ``tool_call_id`` within the same turn (the agent
        loop / orchestrators populate both lists together). A call with no matching
        return is recorded with ``has_result=False`` rather than a fabricated result.
        """
        recorded_turns: list[list[RecordedCall]] = []
        for calls, returns in turns:
            by_id: dict[str, ToolReturnRecord] = {}
            for r in returns:
                by_id.setdefault(r.tool_call_id, r)
            turn_calls: list[RecordedCall] = []
            for call in calls:
                ret = by_id.get(call.tool_call_id)
                turn_calls.append(
                    RecordedCall(
                        name=call.tool_name,
                        args=dict(call.args or {}),
                        result=(ret.content if ret is not None else None),
                        has_result=ret is not None,
                    )
                )
            recorded_turns.append(turn_calls)
        return cls(turns=recorded_turns)


__all__ = ["RecordedCall", "Trajectory"]
