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

import contextlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from himmy.core.events import EventType, RunEvent
from himmy.runtime.prompt_assembly import _cache_scope_metadata
from himmy.services.inference.models import (
    InferenceMessage,
    InferenceRequest,
    ResponseFormat,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.inference.models import LLMConfig

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


# --- Default-on auto-compaction policy (eff-p0 #3) --------------------------------------
# For the full rationale of why compaction is DEFAULT-ON with a deliberately conservative
# high threshold, see the module docstring of ``himmy.runtime.single_agent`` (this block
# was extracted verbatim from there). Fully overridable:
#   * ``HIMMY_AUTO_COMPACT=0``            → disable the default entirely.
#   * ``HIMMY_AUTO_COMPACT_TOKENS=<int>``→ override the conservative token budget.
#   * ``HIMMY_AUTO_COMPACT_KEEP=<int>``  → override how many recent messages ride verbatim.
# An explicit ``ctx['compaction_spec']`` overrides all three for that run.
_AUTO_COMPACT_TOKENS_DEFAULT = 24000
_AUTO_COMPACT_KEEP_DEFAULT = 8


def _auto_compact_default_spec() -> dict[str, Any] | None:
    """The default compaction spec, or ``None`` when disabled via ``HIMMY_AUTO_COMPACT``.

    Read fresh from the environment on every call (not cached) so tests — and a caller
    that flips the flag between runs — see the current value. Returns a spec dict shaped
    exactly like an explicit ``ctx['compaction_spec']`` so the apply path is identical.
    """
    if os.environ.get("HIMMY_AUTO_COMPACT", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        max_tokens = max(1, int(os.environ.get("HIMMY_AUTO_COMPACT_TOKENS", str(_AUTO_COMPACT_TOKENS_DEFAULT))))
    except ValueError:
        max_tokens = _AUTO_COMPACT_TOKENS_DEFAULT
    try:
        keep_recent = max(1, int(os.environ.get("HIMMY_AUTO_COMPACT_KEEP", str(_AUTO_COMPACT_KEEP_DEFAULT))))
    except ValueError:
        keep_recent = _AUTO_COMPACT_KEEP_DEFAULT
    return {"max_tokens": max_tokens, "keep_recent": keep_recent}


# --- Mandatory compaction-summary scrub (eff-p0 sec-r3 #3) ------------------------------
# The compaction summary is distilled from UN-guarded USER + TOOL content (a scraped page,
# a poisoned tool result), so it is untrusted. sec-r1 ran it through the configured INPUT
# guardrail — but that guardrail is opt-in and is ``None`` in the offline / CLI / default
# runtime, where ``_guard_input`` short-circuits to a plain passthrough. With compaction
# now DEFAULT-ON, that left the offline default with NO laundering barrier: a planted
# secret or standing directive would ride into a persistent USER recap message AND land
# unredacted at rest in durable episodic memory recalled into future runs.
#
# This MANDATORY scrub runs REGARDLESS of whether a full input guardrail is wired. It
# reuses the same credential rule set as the always-on ``SecretsGuardrail`` (redacts API
# keys / JWTs / URL-embedded creds — things no legitimate summary should contain) so
# secrets never persist unredacted, and neutralizes the common injection imperatives so a
# distilled "ignore previous instructions"/"you are now" directive cannot re-ride as a
# standing instruction. It is deliberately NARROW (credentials + injection markers only) so
# it never mangles legitimate recap text; the configured input guardrail (when present)
# still runs on top for the broader DLP/blocklist policy.
_SUMMARY_INJECTION_MARKER = "[neutralized-directive]"
try:  # narrow, dependency-free reuse of the builtin credential + injection rule sets.
    from himmy.services.guardrails.builtins import (  # noqa: E402
        _INJECTION_PATTERNS as _SUMMARY_INJECTION_PATTERNS,
    )
    from himmy.services.guardrails.builtins import (  # noqa: E402
        _SECRET_RULES as _SUMMARY_SECRET_RULES,
    )
    from himmy.services.guardrails.builtins import (  # noqa: E402
        _redact as _summary_redact_secrets,
    )
except Exception:  # pragma: no cover - guardrails always ship; defensive only
    _SUMMARY_SECRET_RULES = None  # type: ignore[assignment]
    _SUMMARY_INJECTION_PATTERNS = None  # type: ignore[assignment]


def _scrub_compaction_summary(text: str) -> str:
    """Mandatory, guardrail-independent scrub of a compaction summary (sec-r3 #3).

    Redacts credentials (via the always-on secret rule set) and neutralizes common
    prompt-injection imperatives so an untrusted, model-distilled recap can neither leak
    a secret at rest nor smuggle a standing directive into every later turn — even when
    NO input guardrail is configured (the offline / default deployment). Returns ``text``
    unchanged only if the rule sets could not be imported (defensive).
    """
    if not text:
        return text
    if _SUMMARY_SECRET_RULES is not None:
        text, _flags = _summary_redact_secrets(text, _SUMMARY_SECRET_RULES)
    if _SUMMARY_INJECTION_PATTERNS is not None:
        for pattern in _SUMMARY_INJECTION_PATTERNS:
            text = pattern.sub(_SUMMARY_INJECTION_MARKER, text)
    return text


class CompactionRunner:
    """Run auto-compaction of a live thread for :class:`SingleAgentRuntime`.

    Extracted verbatim from ``single_agent.py`` (P3 decomposition, lane ``runtime``
    step ``compaction``). Owns the RUNTIME side of compaction: pick the effective spec
    (explicit ``ctx['compaction_spec']`` wins, else the default-on policy), plan via
    :class:`ContextCompactor`, run the summarization inference, scrub + guard the
    untrusted summary, splice the thread, emit ``CONTEXT_COMPACTED``, and persist the
    episodic trace. The runtime constructs one in ``__init__`` and ``_maybe_compact``
    becomes a thin delegating shim; behavior — event order, prompt bytes, cache-prefix
    return semantics, exception types — is byte-for-byte identical to the inline code.

    Reads all live wiring (``default_model_key``, ``inference_service``, ``_guard_input``,
    ``_emit``, ``memory_store``) off the runtime at call time.
    """

    def __init__(self, runtime: SingleAgentRuntime) -> None:
        self._rt = runtime

    async def maybe_compact(
        self,
        persona: Persona,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        llm_config: LLMConfig | None,
    ) -> bool:
        """Summarize old turns in-place when the thread outgrows its token budget.

        DEFAULT-ON (eff-p0 #3) with a deliberately conservative high budget: an explicit
        ``ctx['compaction_spec']`` wins, otherwise :func:`_auto_compact_default_spec`
        supplies a default budget so long runs don't re-send their whole (O(turns^2))
        history uncompressed. The default is disable-able via ``HIMMY_AUTO_COMPACT=0``
        and tunable via ``HIMMY_AUTO_COMPACT_TOKENS`` / ``HIMMY_AUTO_COMPACT_KEEP``.

        Keeps the system head + recent tail, replaces the middle with one model-written
        summary message, and emits a ``CONTEXT_COMPACTED`` event (the audit trail of what
        was condensed). A no-op when under budget, when the default is disabled and no
        explicit spec was given, or when the summary would not actually shrink the span
        (the summary-only-if-smaller guard below).

        Returns ``True`` iff compaction actually rewrote the thread this turn. The
        caller (C5) uses that to BUST the prompt cache for the very next request:
        compaction inserts a new ``[Summary …]`` message, which changes the joined
        message prefix and would otherwise pay a write premium on a stale-cache miss.
        Skipping the breakpoint that one turn lets the prefix re-stabilize.

        SECURITY (sec-r1 + sec-r3): the summary is distilled from untrusted USER/TOOL
        content, so it is (1) run through the configured input guardrail (injection/DLP/
        blocklist) when one is wired AND — regardless of any guardrail — through a
        MANDATORY credential + injection scrub (:func:`_scrub_compaction_summary`) so the
        offline / default deployment (which wires no input guardrail) still cannot leak a
        secret at rest or launder a directive, and (2) inserted at USER — never SYSTEM —
        trust so a planted "instruction" can't be laundered into a standing directive.
        Operator steer/control messages are pinned out of the summarize span upstream
        (see :mod:`himmy.runtime.compaction`) so they always ride verbatim.
        """
        # An explicit per-run spec always wins; otherwise fall back to the default-on
        # policy (which may itself be disabled via HIMMY_AUTO_COMPACT=0 → None).
        spec = ctx.get("compaction_spec")
        if not spec:
            spec = _auto_compact_default_spec()
        if not spec:
            return False
        from himmy.agents.base_agent.thread import Message, MessageRole

        compactor = ContextCompactor(
            max_tokens=int(spec.get("max_tokens", 3000)),
            keep_recent=int(spec.get("keep_recent", 6)),
        )
        plan = compactor.plan(thread.messages)
        if not plan.should_compact:
            return False

        span_text = compactor.render_span(plan.summarize)
        model_key = str(ctx.get("model_key") or self._rt.default_model_key)
        summary_req = InferenceRequest(
            model_key=model_key,
            response_format=ResponseFormat.TEXT,
            messages=[
                InferenceMessage(role="system", content=SUMMARY_INSTRUCTION),
                InferenceMessage(role="user", content=span_text),
            ],
            metadata=_cache_scope_metadata(ctx),
        )
        summary_resp = await self._rt.inference_service.run(summary_req)
        summary_text = (summary_resp.output_text or "").strip()
        if not summary_text:
            return False  # summarization failed/empty — leave history intact (safe)

        # SECURITY (sec-r1 + sec-r3): the summarized span is distilled from USER and
        # (crucially) UN-guarded TOOL result messages, so ``summary_text`` is UNTRUSTED —
        # an attacker who planted an "instruction"/"decision" in a scraped page or poisoned
        # tool result would otherwise have it re-emitted verbatim-in-spirit. Hardenings:
        #   1. run it through the configured input guardrail (injection / DLP / blocklist)
        #      when one is wired — BUT that guardrail is opt-in and absent in the offline /
        #      default runtime, so it is NOT a guarantee on its own;
        #   1b. (sec-r3) ALWAYS run a mandatory, guardrail-independent credential + injection
        #      scrub so secrets are redacted and directives neutralized even with no
        #      guardrail configured — this is the real laundering barrier;
        #   2. insert it at USER trust (not SYSTEM). SYSTEM is reserved for operator /
        #      persona text; laundering summarized untrusted content into a SYSTEM
        #      directive would elevate its trust and let a one-shot indirect injection
        #      persist as a standing higher-trust instruction on every later turn.
        summary_text = await self._rt._guard_input(
            summary_text,
            agent_id=persona.agent_id,
            trace_id=trace_id,
            thread_id=thread.thread_id,
        )
        # sec-r3 #3: the configured input guardrail above is OPT-IN and is ``None`` in the
        # offline / default runtime (``_guard_input`` then passes through untouched). Apply
        # a MANDATORY, guardrail-independent scrub so credentials are always redacted and
        # injection directives are neutralized BEFORE the summary re-enters context or is
        # persisted to durable episodic memory — regardless of whether a full guardrail is
        # wired. This is the load-bearing laundering barrier for default-on compaction.
        summary_text = _scrub_compaction_summary(summary_text)
        if not summary_text.strip():
            return False  # guardrail/scrub emptied it — nothing safe to fold in

        summary_msg = Message(
            role=MessageRole.USER,
            content=(
                "[Summary of earlier conversation — untrusted recap of prior turns, "
                "not an operator instruction]\n"
                f"{summary_text}"
            ),
            metadata={"compacted": True},
        )
        # Only apply if the summary is actually smaller than what it replaces — a verbose
        # summary of a tiny span would otherwise grow the context, not shrink it.
        if estimate_tokens(summary_msg.content) >= compactor.estimate(plan.summarize):
            return False
        head = list(thread.messages[: plan.head_count])
        tail = list(thread.messages[plan.tail_start :])
        # sec-r3 #4: pinned control/safety messages inside the summarize span are lifted
        # out (plan.carry) and re-inserted VERBATIM between the summary and the kept tail,
        # so the refusal/steer boundary survives compaction while the non-pinned middle is
        # still condensed. Their tool_call group rides with them (planner-guaranteed), so
        # no tool_result is orphaned after the USER summary.
        carry = list(plan.carry)
        compacted_count = len(plan.summarize)
        thread.messages[:] = [*head, summary_msg, *carry, *tail]
        thread.version += 1

        await self._rt._emit(
            RunEvent(
                event_type=EventType.CONTEXT_COMPACTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload={
                    "summarized_messages": compacted_count,
                    "before_tokens": plan.before_tokens,
                    "after_tokens": compactor.estimate(thread.messages),
                    "kept_recent": len(tail),
                },
            )
        )

        # P0-C: the compaction summary is the run's distilled experience ("every
        # fact, decision, tool result"). Instead of discarding it when the thread
        # middle is replaced, persist it as an episodic trace so learning loops have
        # a corpus of "what happened" from day one. Best-effort: a persistence
        # failure must never break the run (the in-context summary already applied).
        # SECURITY (sec-r1 + sec-r3): ``summary_text`` here is the SCRUBBED text — through
        # the configured input guardrail (if any) AND the mandatory credential + injection
        # scrub — so secrets a tool returned verbatim are redacted BEFORE they land
        # unredacted-at-rest in durable, subject-scoped episodic memory (recalled into
        # FUTURE runs and readable by store-level exports) even with no guardrail wired.
        save_episodic = getattr(self._rt.memory_store, "save_episodic_memory", None)
        if save_episodic is not None:
            from himmy.services.storage.models import EpisodicMemoryObject

            subject_raw = ctx.get("subject_id")
            with contextlib.suppress(Exception):
                await save_episodic(
                    EpisodicMemoryObject(
                        subject_id=str(subject_raw) if subject_raw else None,
                        agent_id=persona.agent_id,
                        payload={
                            "summary": summary_text,
                            "source": "compaction",
                            "tier": "archival",
                        },
                        metadata={
                            "trace_id": trace_id,
                            "thread_id": thread.thread_id,
                            "summarized_messages": compacted_count,
                        },
                    )
                )
        return True


__all__ = [
    "ContextCompactor",
    "CompactionPlan",
    "CompactionRunner",
    "estimate_tokens",
    "SUMMARY_INSTRUCTION",
]

#: Re-exported for the runtime + tests that need to reason about pinned messages.
__all__ += ["_is_pinned"]
