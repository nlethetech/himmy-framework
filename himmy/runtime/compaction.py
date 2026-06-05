"""Automatic context compaction: summarize old turns when history outgrows a budget.

A multi-turn agent sends its ENTIRE conversation to the model every turn, so long runs
eventually overflow the context window (or just get slow + expensive). Compaction keeps
the run going by replacing the *middle* of the history with a short summary, the way a
long Claude Code session compacts — while holding three invariants that keep the message
list valid and the recent context intact:

1. the leading **system** message(s) are never touched (persona/instructions);
2. the most recent ``keep_recent`` messages are kept verbatim;
3. the summarized span never **splits a tool_call from its tool_return** — the tail
   boundary is snapped back so the kept tail starts at a non-``tool`` message.

This module is the pure *planner* (decide whether + what to compact); the runtime owns
the actual summarization inference call and applies the plan. Planning is fully testable
without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: ~4 characters per token — a deliberately cheap, provider-agnostic estimate, enough to
#: decide *when* to compact (the real token counts come back on each response).
_CHARS_PER_TOKEN = 4

#: Default summarization instruction; the runtime sends this with the span to compress.
SUMMARY_INSTRUCTION = (
    "Summarize the conversation excerpt below into a compact briefing that a teammate "
    "could use to continue the task. Preserve every fact, decision, tool result, name, "
    "and number; drop pleasantries and repetition. Write only the summary."
)


def estimate_tokens(text: str) -> int:
    """Cheap length-based token estimate (min 1 for non-empty)."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _role(message: Any) -> str:
    """The message role as a lowercase string (handles str or an Enum)."""
    role = message.role
    return (role.value if hasattr(role, "value") else str(role)).lower()


@dataclass(frozen=True)
class CompactionPlan:
    """The decision of what to compact. ``should_compact`` False means leave as-is."""

    should_compact: bool
    head_count: int = 0  # leading system messages kept untouched
    summarize_start: int = 0  # inclusive index of the first message to summarize
    summarize_end: int = 0  # exclusive index (tail starts here)
    before_tokens: int = 0
    reason: str = ""
    summarize: list[Any] = field(default_factory=list)

    @property
    def tail_start(self) -> int:
        return self.summarize_end


class ContextCompactor:
    """Plan compaction of a message history under a token budget."""

    def __init__(
        self,
        *,
        max_tokens: int = 3000,
        keep_recent: int = 4,
        min_summarize: int = 2,
    ) -> None:
        if keep_recent < 1:
            raise ValueError("keep_recent must be >= 1")
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.min_summarize = min_summarize

    def estimate(self, messages: Sequence[Any]) -> int:
        """Estimated token size of the whole message list."""
        return sum(estimate_tokens(m.content) for m in messages)

    def plan(self, messages: Sequence[Any]) -> CompactionPlan:
        """Decide whether and where to compact ``messages``.

        Returns a no-op plan when under budget, when there's too little to summarize, or
        when no tool-pairing-safe boundary exists.
        """
        before = self.estimate(messages)
        if before <= self.max_tokens:
            return CompactionPlan(False, before_tokens=before, reason="under budget")

        # 1. protect leading system messages.
        head_count = 0
        for m in messages:
            if _role(m) == "system":
                head_count += 1
            else:
                break

        # 2. keep the most recent `keep_recent`, but never less than the head.
        split = max(head_count, len(messages) - self.keep_recent)

        # 3. snap the tail boundary back so it doesn't start with an orphaned tool
        #    result (its assistant tool_call would otherwise be in the summarized span).
        while split > head_count and _role(messages[split]) == "tool":
            split -= 1

        summarize = list(messages[head_count:split])
        if len(summarize) < self.min_summarize:
            return CompactionPlan(
                False, before_tokens=before, reason="too little to summarize"
            )
        return CompactionPlan(
            should_compact=True,
            head_count=head_count,
            summarize_start=head_count,
            summarize_end=split,
            before_tokens=before,
            reason=f"{before} est. tokens over {self.max_tokens} budget",
            summarize=summarize,
        )

    def render_span(self, summarize: Sequence[Any]) -> str:
        """Flatten a span of messages into the text handed to the summarizer."""
        lines = []
        for m in summarize:
            content = m.content.strip()
            if content:
                lines.append(f"{_role(m)}: {content}")
        return "\n".join(lines)


__all__ = [
    "ContextCompactor",
    "CompactionPlan",
    "estimate_tokens",
    "SUMMARY_INSTRUCTION",
]
