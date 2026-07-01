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

Beyond the three structural invariants, the planner also refuses to summarize
**control-channel** and **security-boundary** messages. Any message carrying
``metadata['steer']`` (or another ``metadata['pin']`` marker) is an operator directive
that steers a running mission ("stop touching production", "do not email anyone"). A
lossy model-written summary could dilute or drop such a safety-relevant constraint, so
these messages are pinned to the kept tail and always ride verbatim — they are never
handed to the summarizer (sec-r1).

sec-r2 widens the pin set beyond operator steers to the run's own **security events**,
which carry no ``steer``/``pin`` marker and would otherwise fall inside the summarize
span and be lossily condensed away:

* a **guardrail-corrected** assistant turn (``metadata['guarded']`` — its output was
  rewritten/blocked by an output guardrail);
* a **HITL / policy denial** recorded as a TOOL message whose ``metadata['tool_outcome']``
  is ``rejected`` or ``denied`` (a human or policy refused a tool call).

Dropping such a boundary from the in-context history lets a later turn — now missing the
explicit "we already refused / this was denied" signal — be re-persuaded into the
previously-refused action. Pinning them keeps the refusal verbatim.

sec-r3 #4 changes HOW a pin is preserved. The original design snapped the tail boundary
back to just before the EARLIEST pinned message, keeping it and everything after it
verbatim. On a long run that hit one *early* refusal, that kept the whole rest of the run
un-summarized forever — compaction could no longer shrink the middle, restoring the
O(turns^2) re-send growth and risking a context-window overflow precisely on runs that
touched a security boundary (an availability/cost regression). Instead, each pinned
message is now LIFTED OUT of the summarize span into ``plan.carry`` and re-inserted
verbatim immediately after the summary, while the non-pinned middle around it is still
summarized. Tool-pairing is preserved: a pinned TOOL refusal carries its owning ASSISTANT
tool_call group forward with it, so no tool_result is orphaned after the summary.
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


#: Metadata keys that mark a message as a pinned control/safety directive which must
#: never be summarized away (it rides verbatim in the kept tail). ``steer`` is the
#: operator between-turns steering seam; ``pin`` is a general-purpose escape hatch;
#: ``guarded`` marks a guardrail-corrected assistant turn (a security boundary, sec-r2).
_PIN_METADATA_KEYS = ("steer", "pin", "guarded")

#: TOOL-message ``metadata['tool_outcome']`` values that record a refused tool call — a
#: human-in-the-loop rejection or a policy denial. These are security boundaries whose
#: "we did NOT run this" signal must survive compaction verbatim (sec-r2).
_PIN_TOOL_OUTCOMES = ("rejected", "denied")


def _role(message: Any) -> str:
    """The message role as a lowercase string (handles str or an Enum)."""
    role = message.role
    return (role.value if hasattr(role, "value") else str(role)).lower()


def _is_pinned(message: Any) -> bool:
    """True if ``message`` is a control/safety boundary that must not be summarized.

    Two families ride verbatim in the kept tail instead of being handed to the lossy
    summarizer:

    * **control directives** — an operator steer ("do not email anyone") or an explicit
      ``pin``, and a guardrail-corrected (``guarded``) assistant turn; and
    * **refusal boundaries** — a TOOL message recording a HITL/policy *rejection* or
      *denial* (``metadata['tool_outcome']`` in :data:`_PIN_TOOL_OUTCOMES`).

    A lossy summary could dilute or drop either signal, letting a later turn be
    re-persuaded into a previously-refused action, so both are pinned (sec-r1 + sec-r2).
    """
    metadata = getattr(message, "metadata", None) or {}
    if any(bool(metadata.get(key)) for key in _PIN_METADATA_KEYS):
        return True
    outcome = metadata.get("tool_outcome")
    return isinstance(outcome, str) and outcome.lower() in _PIN_TOOL_OUTCOMES


@dataclass(frozen=True)
class CompactionPlan:
    """The decision of what to compact. ``should_compact`` False means leave as-is.

    ``summarize`` is the (non-pinned) span the model condenses. ``carry`` (sec-r3 #4) is
    the list of pinned control/safety messages that fell INSIDE that span but must ride
    verbatim — they are lifted OUT of the summarize span and re-inserted, in original
    order, immediately after the summary message. This lets compaction still shrink the
    non-pinned middle of a long run even when an EARLY pin exists, instead of snapping the
    whole tail back to the earliest pin (which defeated compaction entirely on any long
    run that touched a security boundary — an availability/cost regression).
    """

    should_compact: bool
    head_count: int = 0  # leading system messages kept untouched
    summarize_start: int = 0  # inclusive index of the first message to summarize
    summarize_end: int = 0  # exclusive index (tail starts here)
    before_tokens: int = 0
    reason: str = ""
    summarize: list[Any] = field(default_factory=list)
    #: Pinned messages lifted out of the summarize span, carried verbatim after the summary.
    carry: list[Any] = field(default_factory=list)

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

        # 4. carry control/safety directives out of the summarize span (sec-r3 #4).
        #    A pinned message must ride VERBATIM, but — unlike the old design, which
        #    snapped the whole tail back to just before the EARLIEST pin (so a single
        #    early refusal kept the entire rest of the run un-summarized forever, an
        #    availability/cost regression) — we instead LIFT each pinned message out of
        #    the summarize span and carry it forward verbatim after the summary, while
        #    still summarizing the non-pinned middle around it. This keeps every
        #    security boundary intact AND lets compaction bound context growth on long
        #    runs that touched a boundary.
        #
        #    Tool-pairing: a pinned TOOL message (a HITL/policy rejection) is meaningless
        #    without the ASSISTANT tool_call that owns it, and a bare carried TOOL result
        #    after the USER summary would orphan a tool_result for strict providers. So
        #    when a pinned TOOL message is carried, the contiguous run of preceding
        #    ASSISTANT/TOOL messages (its call group) is carried with it.
        carry_indices: set[int] = set()
        for i in range(head_count, split):
            if not _is_pinned(messages[i]):
                continue
            carry_indices.add(i)
            if _role(messages[i]) == "tool":
                # walk back over the contiguous assistant/tool call-group owning it.
                j = i - 1
                while j >= head_count and _role(messages[j]) in ("assistant", "tool"):
                    carry_indices.add(j)
                    if _role(messages[j]) == "assistant":
                        break  # the owning tool_call — stop at the first assistant.
                    j -= 1

        carry = [messages[i] for i in sorted(carry_indices)]
        summarize = [
            messages[i] for i in range(head_count, split) if i not in carry_indices
        ]
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
            carry=carry,
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

#: Re-exported for the runtime + tests that need to reason about pinned messages.
__all__ += ["_is_pinned"]
