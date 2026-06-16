"""Learning kernel: compute per-tool reputation from recorded run-events.

The :class:`LearningService` mines the ``TOOL_FAILED`` / ``TOOL_COMPLETED`` audit
stream (denormalised onto the P0-B indexed ``event_type`` / ``tool_name`` columns) into
a :class:`ToolReputation` per tool: a reliability score in ``[0, 1]`` plus the raw
recent counts used to phrase a learned hint. The math is deterministic given the same
events and every read is bounded (``window``) so it never scans unbounded history.

Two consumers sit on top of the service:

* the :class:`~himmy.services.learning.adapter.LearnedHintsContextAdapter` (async) reads
  reputation directly when building a context snapshot;
* the sync :class:`ToolReputationProvider` holds a *snapshot* of reputation that the sync
  :meth:`~himmy.services.tools.service.ToolService.bound_tools` reorder reads without an
  ``await`` — the snapshot is refreshed out-of-band (once when the runtime is built),
  so the per-turn binding pays no storage cost.

Every public method is 100% best-effort: any exception (store unavailable, malformed
rows) is swallowed and returns the neutral / empty result, logged at ``debug`` — learning
must never break or noticeably slow a run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from himmy.core.events import EventType, RunEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.core.events import EventSink
    from himmy.services.storage.protocols import EventLog

logger = logging.getLogger("himmy.services.learning")

#: How many of a tool's most-recent tool events the reputation read inspects. Bounds the
#: storage read so the miner never scans the whole (growing) audit stream.
DEFAULT_WINDOW = 200

#: Minimum number of completed+failed calls before a tool's score is allowed to drop
#: below the neutral prior. Below this a brand-new (or barely-used) tool stays neutral so
#: a single early failure can never bury an otherwise-fine tool.
DEFAULT_MIN_SAMPLES = 3

#: The cold-start / insufficient-sample reputation: a perfectly neutral 1.0 so an unseen
#: or barely-seen tool is NEVER deprioritised (a stable sort on all-1.0 is a no-op).
NEUTRAL_SCORE = 1.0


@dataclass(frozen=True)
class ToolReputation:
    """A tool's recent reliability, derived from its TOOL_FAILED/TOOL_COMPLETED events."""

    tool_name: str
    completed: int
    failed: int
    score: float
    #: Whether enough samples exist for ``score`` to be trusted below neutral. When
    #: False the score is the neutral prior regardless of the (sub-threshold) counts.
    has_min_samples: bool

    @property
    def total(self) -> int:
        """Total scored calls in the window (completed + failed)."""
        return self.completed + self.failed


class LearningService:
    """Mines per-tool reputation from the recorded run-event audit stream.

    Reads are bounded by ``window`` and scoped by the indexed ``event_type`` /
    ``tool_name`` columns, so a reputation read is a small index seek even on a long
    audit stream. Construction takes the SAME :class:`EventLog` handle the runtime
    already uses, so no second storage backend is opened.
    """

    def __init__(
        self,
        event_log: EventLog,
        *,
        window: int = DEFAULT_WINDOW,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._events = event_log
        self._window = max(1, int(window))
        self._min_samples = max(1, int(min_samples))

    async def get_tool_reputation(
        self, tool_names: list[str]
    ) -> dict[str, ToolReputation]:
        """Return a :class:`ToolReputation` for each named tool (best-effort).

        For every tool we read up to ``window`` of its most-recent ``TOOL_COMPLETED`` and
        ``TOOL_FAILED`` events and compute ``completed / (completed + failed)``. An unseen
        tool — or one with fewer than ``min_samples`` scored calls — is reported at the
        neutral prior (``1.0``) so brand-new tools are never punished. Any error returns
        the neutral result for the affected tool and is logged at ``debug``; the call
        never raises.
        """
        result: dict[str, ToolReputation] = {}
        for name in tool_names:
            result[name] = await self._reputation_for_tool(name)
        return result

    async def _reputation_for_tool(self, tool_name: str) -> ToolReputation:
        """Compute one tool's reputation, swallowing any storage/data error to neutral."""
        try:
            completed, failed = await self._recent_counts(tool_name)
        except Exception:  # pragma: no cover - defensive: learning never breaks a run
            logger.debug(
                "reputation read failed for tool %r; using neutral", tool_name,
                exc_info=True,
            )
            return self._neutral(tool_name)
        total = completed + failed
        if total < self._min_samples:
            # Cold-start / sub-threshold: keep the neutral prior but carry the real
            # counts so a hint can still mention them if a consumer chooses to.
            return ToolReputation(
                tool_name=tool_name,
                completed=completed,
                failed=failed,
                score=NEUTRAL_SCORE,
                has_min_samples=False,
            )
        score = completed / total if total else NEUTRAL_SCORE
        return ToolReputation(
            tool_name=tool_name,
            completed=completed,
            failed=failed,
            score=score,
            has_min_samples=True,
        )

    async def _recent_counts(self, tool_name: str) -> tuple[int, int]:
        """Return ``(completed, failed)`` over a SINGLE combined most-recent-N window.

        Reads up to ``window`` of each outcome type (a bounded index seek per type),
        merges them into one stream ordered newest-first, and keeps only the most-recent
        ``window`` across BOTH types before counting. Windowing the combined stream (not
        each type independently) is what makes the score reflect *recent* behaviour: a
        long history of completions can no longer dilute a recent burst of failures, since
        only the freshest ``window`` events of either kind survive the cut.
        """
        completed = await self._events.list_events(
            event_type=EventType.TOOL_COMPLETED,
            tool_name=tool_name,
            limit=self._window,
            newest_first=True,
        )
        failed = await self._events.list_events(
            event_type=EventType.TOOL_FAILED,
            tool_name=tool_name,
            limit=self._window,
            newest_first=True,
        )
        # Merge both newest-first reads and keep only the most-recent ``window`` events
        # across the two types (ordered by recorded timestamp, which is monotonic with
        # insertion order). Counting within that combined slice is the recency cut.
        merged = sorted(
            [(e.timestamp, EventType.TOOL_COMPLETED) for e in completed]
            + [(e.timestamp, EventType.TOOL_FAILED) for e in failed],
            key=lambda pair: pair[0],
            reverse=True,
        )[: self._window]
        n_completed = sum(1 for _, et in merged if et is EventType.TOOL_COMPLETED)
        return n_completed, len(merged) - n_completed

    @staticmethod
    def _neutral(tool_name: str) -> ToolReputation:
        """The neutral / empty reputation for a tool (no samples, perfect prior)."""
        return ToolReputation(
            tool_name=tool_name,
            completed=0,
            failed=0,
            score=NEUTRAL_SCORE,
            has_min_samples=False,
        )


class ToolReputationProvider:
    """A sync snapshot of tool reputation for the ``bound_tools`` reorder hook.

    ``ToolService.bound_tools`` is synchronous and on the per-turn inference hot path, so
    it cannot ``await`` the async :class:`LearningService`. This provider bridges the gap:
    :meth:`refresh` (async, driven out-of-band when the runtime is built) populates an
    in-memory snapshot, and :meth:`score_for` / :meth:`is_unreliable` (sync) read it with
    no I/O. An empty / un-refreshed snapshot reports every tool as neutral, so the reorder
    is a no-op until real history exists — exactly the zero-behaviour-change default.

    ``floor`` is the score below which a sufficiently-sampled tool is considered
    *unreliable* (eligible for an annotated caution).
    """

    def __init__(
        self,
        learning: LearningService,
        *,
        floor: float = 0.2,
        event_sink: EventSink | None = None,
    ) -> None:
        self._learning = learning
        self._floor = floor
        self._event_sink = event_sink
        self._snapshot: dict[str, ToolReputation] = {}

    async def refresh(self, tool_names: list[str]) -> dict[str, ToolReputation]:
        """Recompute and cache the reputation snapshot for ``tool_names`` (best-effort).

        When the fresh snapshot would actually MOVE a tool in the bound-tool order (the
        same stable score-sort :meth:`~himmy.services.tools.service.ToolService.bound_tools`
        applies produces a different sequence), a best-effort ``LEARNING_APPLIED`` event is
        emitted so the reorder is auditable. The event is emitted at snapshot-refresh time
        (once per runtime build), so it carries no per-run trace context; the per-turn hint
        adapter emits its own ``LEARNING_APPLIED`` inside the run when a hint is injected.
        """
        try:
            self._snapshot = await self._learning.get_tool_reputation(tool_names)
        except Exception:  # pragma: no cover - defensive: learning never breaks a run
            logger.debug("reputation snapshot refresh failed", exc_info=True)
            self._snapshot = {}
        await self._emit_if_reordering(tool_names)
        return dict(self._snapshot)

    async def _emit_if_reordering(self, tool_names: list[str]) -> None:
        """Emit ``LEARNING_APPLIED`` only when the stable sort actually moves a tool.

        Mirrors ``ToolService.bound_tools``: stable-sort the candidate order by negated
        score and compare it to the original order. An emit fires only on a genuine
        difference, so a snapshot whose sub-1.0 tools are already last (or where a single
        tool dips) does not falsely claim a reorder.
        """
        if self._event_sink is None:
            return
        ordered = [n for n in tool_names if n in self._snapshot]
        # ``sorted`` is stable — equal scores keep their original position, exactly as
        # ``bound_tools`` binds them, so a difference here means a real visible move.
        reordered = sorted(ordered, key=lambda n: -self._snapshot[n].score)
        if reordered == ordered:
            return  # stable sort is a no-op → nothing observable to audit
        deprioritised = sorted(
            rep.tool_name for rep in self._snapshot.values() if rep.score < 1.0
        )
        try:
            await self._event_sink.append_event(
                RunEvent(
                    event_type=EventType.LEARNING_APPLIED,
                    payload={
                        "tools_reordered": len(deprioritised),
                        "deprioritised_tools": deprioritised,
                    },
                )
            )
        except Exception:  # pragma: no cover - defensive: observability never breaks a run
            logger.debug("LEARNING_APPLIED emit failed", exc_info=True)

    @property
    def floor(self) -> float:
        """The unreliable-score floor used to annotate (not drop) flaky tools."""
        return self._floor

    def score_for(self, tool_name: str) -> float:
        """The cached score for a tool (neutral when unseen) — the sort key."""
        rep = self._snapshot.get(tool_name)
        return rep.score if rep is not None else NEUTRAL_SCORE

    def is_unreliable(self, tool_name: str) -> bool:
        """Whether a sufficiently-sampled tool's score is below the caution floor."""
        rep = self._snapshot.get(tool_name)
        return rep is not None and rep.has_min_samples and rep.score < self._floor


__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_WINDOW",
    "NEUTRAL_SCORE",
    "LearningService",
    "ToolReputation",
    "ToolReputationProvider",
]
