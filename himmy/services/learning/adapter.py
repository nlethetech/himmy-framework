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
    ) -> None:
        """Wrap a :class:`LearningService`; hint on tools scoring below ``floor``.

        ``tool_names`` pins the candidate tools to assess (the run's bound tools); when
        ``None`` the scope's ``tool_names`` is used. Only sufficiently-sampled tools whose
        score is below ``floor`` produce a hint, capped at ``max_hints`` (worst first), so
        an off-topic or all-healthy run injects nothing. ``event_sink`` (optional) receives
        a best-effort ``LEARNING_APPLIED`` event whenever a hint is actually injected.
        """
        self._learning = learning
        self._tool_names = tool_names
        self._floor = floor
        self._max_hints = max_hints
        self._event_sink = event_sink

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
        unreliable = [
            rep
            for rep in reputation.values()
            if rep.has_min_samples and rep.score < self._floor and rep.failed > 0
        ]
        if not unreliable:
            return None
        # Worst (lowest score) first, deterministic tiebreak on the tool name.
        unreliable.sort(key=lambda r: (r.score, r.tool_name))
        chosen = unreliable[: self._max_hints]
        rendered = self._render(chosen)
        if not rendered:
            return None
        await self._emit_applied(chosen)
        return ContextField(
            key=key,
            value=rendered,
            source="learning",
            confidence=1.0,
        )

    async def _emit_applied(self, chosen: list[ToolReputation]) -> None:
        """Emit a best-effort ``LEARNING_APPLIED`` event for an injected hint."""
        if self._event_sink is None:
            return
        try:
            await self._event_sink.append_event(
                RunEvent(
                    event_type=EventType.LEARNING_APPLIED,
                    payload={
                        "hint_count": len(chosen),
                        "hinted_tools": sorted(rep.tool_name for rep in chosen),
                    },
                )
            )
        except Exception:  # pragma: no cover - defensive: observability never breaks a run
            logger.debug("LEARNING_APPLIED hint emit failed", exc_info=True)

    @staticmethod
    def _render(reputations: list[ToolReputation]) -> str:
        """Render the unreliable tools as a short, privacy-safe plain-text note.

        Only tool names + aggregate counts appear — never raw error text, args, or any
        user prompt / PII.
        """
        lines = [
            f'- tool "{rep.tool_name}" has failed {rep.failed} of its last '
            f"{rep.total} calls — prefer alternatives or verify inputs before using it."
            for rep in reputations
        ]
        if not lines:
            return ""
        return "Reliability notes from past runs:\n" + "\n".join(lines)


__all__ = ["LearnedHintsContextAdapter"]
