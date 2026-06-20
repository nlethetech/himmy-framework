"""Learned-hints context adapter: inject reliability notes into a prompt.

Registered on a :class:`~himmy.services.context.service.ContextService`, this adapter
turns recent tool failures into grounding context: when the runtime builds a snapshot it
asks the :class:`~himmy.services.learning.service.LearningService` which of the run's
tools have been unreliable lately and renders a SHORT system hint, so the model is told
to prefer alternatives or verify inputs before reaching for a flaky tool — no tool call.

PRIVACY: the rendered value is a **plain string** of tool names + aggregate counts only.
It is deliberately NOT a rich dict — the prompt mapper JSON-dumps a non-string field value
verbatim into the ``<context>`` block, so a dict could leak raw error text. A string value
is rendered as-is, and we only ever put tool names + counts in it (never user prompts /
PII), mirroring the P0-C scrub precedent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from himmy.core.events import EventType, RunEvent
from himmy.services.context.adapters import ContextAdapter
from himmy.services.context.models import ContextField

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.core.events import EventSink
    from himmy.services.learning.service import LearningService, ToolReputation
    from himmy.services.learning.trajectory import TrajectoryAdvisor

logger = logging.getLogger("himmy.services.learning")


class LearnedHintsContextAdapter(ContextAdapter):
    """A :class:`ContextAdapter` that injects reliability notes for unreliable tools."""

    name = "learned_hints"

    def __init__(
        self,
        learning: LearningService,
        *,
        tool_names: list[str] | None = None,
        floor: float = 0.2,
        max_hints: int = 5,
        event_sink: EventSink | None = None,
        trajectory_advisor: TrajectoryAdvisor | None = None,
    ) -> None:
        """Wrap a :class:`LearningService`; hint on tools scoring below ``floor``.

        ``tool_names`` pins the candidate tools to assess (the run's bound tools); when
        ``None`` the scope's ``tool_names`` is used. Only sufficiently-sampled tools whose
        score is below ``floor`` produce a hint, capped at ``max_hints`` (worst first), so
        an off-topic or all-healthy run injects nothing. ``event_sink`` (optional) receives
        a best-effort ``LEARNING_APPLIED`` event whenever a hint is actually injected.

        ``trajectory_advisor`` (P2, opt-in) adds run-sequence-aware hints (looping /
        don't-call-first) mined from the recent ``TOOL_CALLED`` order. When ``None`` (the
        default) the adapter renders exactly the P1 per-tool reliability note — byte-identical.
        """
        self._learning = learning
        self._tool_names = tool_names
        self._floor = floor
        self._max_hints = max_hints
        self._event_sink = event_sink
        self._trajectory_advisor = trajectory_advisor

    def bind_tools(self, tool_names: list[str]) -> None:
        """Pin the candidate tools to assess (the run's bound tools).

        The snapshot ``scope`` never carries the run's tool list, so the wiring primes the
        adapter here once the tool registry exists (mirroring the reputation snapshot
        refresh). Without it ``fetch`` has no candidates and silently injects nothing.
        """
        self._tool_names = tool_names

    async def fetch(self, key: str, scope: dict[str, Any]) -> ContextField | None:
        """Render a reliability note for the run's unreliable tools, or ``None``.

        Returns ``None`` (no block injected) when there is nothing useful — no candidate
        tools, no recent failures, or any error (swallowed to neutral). The field value is
        a plain string so the prompt mapper splices it verbatim without leaking structure.
        """
        names = self._tool_names or list(scope.get("tool_names") or [])
        if not names:
            return None
        try:
            reputation = await self._learning.get_tool_reputation(names)
        except Exception:  # pragma: no cover - defensive: learning never breaks a run
            logger.debug("learned-hints reputation read failed", exc_info=True)
            return None
        # A tool is hint-worthy when it is sufficiently sampled, scores below the floor, and
        # there is a CONCRETE reason to phrase: either recent operational failures (P1) OR a
        # low outcome grade (P2). The ``failed > 0 or _has_bad_outcome`` disjunction keeps the
        # P1 path byte-identical when the outcome signal is off — ``_has_bad_outcome`` is False
        # whenever ``outcome_score`` is None (off / cold), so the predicate collapses to the
        # original ``rep.failed > 0``.
        unreliable = [
            rep
            for rep in reputation.values()
            if rep.has_min_samples
            and rep.score < self._floor
            and (rep.failed > 0 or self._has_bad_outcome(rep))
        ]
        # Worst (lowest score) first, deterministic tiebreak on the tool name.
        unreliable.sort(key=lambda r: (r.score, r.tool_name))
        chosen = unreliable[: self._max_hints]
        reliability_note = self._render(chosen)

        # P2: trajectory-aware lines (looping / don't-call-first). Computed ONLY when an
        # advisor is wired, so the default path is byte-identical to P1. A loop on an
        # otherwise-healthy tool is still worth flagging, so these can fire even when the
        # per-tool reliability note above is empty.
        trajectory_lines = await self._trajectory_lines(reputation)

        if not reliability_note and not trajectory_lines:
            return None
        sections: list[str] = []
        if reliability_note:
            sections.append(reliability_note)
        if trajectory_lines:
            sections.append(
                "Run-sequence notes:\n" + "\n".join(trajectory_lines)
            )
        rendered = "\n".join(sections)
        await self._emit_applied(chosen, trajectory_hint_count=len(trajectory_lines))
        return ContextField(
            key=key,
            value=rendered,
            source="learning",
            confidence=1.0,
        )

    async def _trajectory_lines(
        self, reputation: dict[str, ToolReputation]
    ) -> list[str]:
        """Render P2 trajectory hints (or ``[]`` when no advisor / no signal).

        Cross-checks the advisor's first-call opener against the run's reputation so a
        don't-call-first line only fires for an opener that is ALSO below the floor —
        never for a healthy tool. Fail-open: any error yields no trajectory lines.
        """
        if self._trajectory_advisor is None:
            return []
        from himmy.services.learning.trajectory import render_trajectory_hints

        try:
            signal = await self._trajectory_advisor.analyze()
        except Exception:  # pragma: no cover - defensive: learning never breaks a run
            logger.debug("trajectory analyze failed; no trajectory hints", exc_info=True)
            return []
        unreliable_first = tuple(
            tool
            for tool in signal.first_call_tools
            if (rep := reputation.get(tool)) is not None
            and rep.has_min_samples
            and rep.score < self._floor
            and rep.failed > 0
        )
        return render_trajectory_hints(
            signal, unreliable_first_tools=unreliable_first
        )

    async def _emit_applied(
        self, chosen: list[ToolReputation], *, trajectory_hint_count: int = 0
    ) -> None:
        """Emit a best-effort ``LEARNING_APPLIED`` event for an injected hint.

        ``trajectory_hint_count`` (P2) is added to the payload only when non-zero so the
        P1 event shape is unchanged when no trajectory advisor is wired.
        """
        if self._event_sink is None:
            return
        payload: dict[str, Any] = {
            "hint_count": len(chosen),
            "hinted_tools": sorted(rep.tool_name for rep in chosen),
        }
        if trajectory_hint_count:
            payload["trajectory_hint_count"] = trajectory_hint_count
        try:
            await self._event_sink.append_event(
                RunEvent(
                    event_type=EventType.LEARNING_APPLIED,
                    payload=payload,
                )
            )
        except Exception:  # pragma: no cover - defensive: observability never breaks a run
            logger.debug("LEARNING_APPLIED hint emit failed", exc_info=True)

    def _render(self, reputations: list[ToolReputation]) -> str:
        """Render the unreliable tools as a short, privacy-safe plain-text note.

        Only tool names + aggregate counts/scores appear — never raw error text, args, or
        any user prompt / PII. A tool flagged by operational FAILURES (P1) keeps the exact
        legacy phrasing; one flagged ONLY by a low OUTCOME grade (ran fine but answered
        poorly — the P2-specific case) gets an outcome-phrased line instead.
        """
        lines = [self._render_line(rep) for rep in reputations]
        if not lines:
            return ""
        return "Reliability notes from past runs:\n" + "\n".join(lines)

    def _render_line(self, rep: ToolReputation) -> str:
        """One reliability line — failure-phrased (P1) or outcome-phrased (P2)."""
        if rep.failed > 0:
            # P1 phrasing, byte-identical to before (operational failures present).
            return (
                f'- tool "{rep.tool_name}" has failed {rep.failed} of its last '
                f"{rep.total} calls — prefer alternatives or verify inputs before using it."
            )
        # P2: the tool RAN fine but its answers graded poorly — phrase from the outcome.
        score = rep.outcome_score if rep.outcome_score is not None else rep.score
        return (
            f'- tool "{rep.tool_name}" ran without error but produced low-quality results '
            f"recently (outcome score {score:.2f} over {rep.outcome_samples} graded run(s)) "
            "— prefer alternatives or double-check its output before relying on it."
        )

    @staticmethod
    def _has_bad_outcome(rep: ToolReputation) -> bool:
        """Whether a tool's OUTCOME grade is what dragged it below the floor (P2).

        ``outcome_score`` is None whenever the outcome signal is off or cold, so this is
        always False on the P1 path — keeping the hint predicate byte-identical there.
        """
        return rep.outcome_score is not None


__all__ = ["LearnedHintsContextAdapter"]
