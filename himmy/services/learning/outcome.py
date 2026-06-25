"""Outcome scoring (P2): record whether a run's ANSWER was GOOD, not just that a tool RAN.

The P1 layer mines ``TOOL_COMPLETED`` / ``TOOL_FAILED`` into an OPERATIONAL reputation —
did the tool execute without erroring. That is silent on the question that actually
matters: when the tool ran, was the produced answer any *good*? This module closes that
gap by emitting a first-class :class:`~himmy.core.events.EventType.OUTCOME_SCORED` event
carrying a score in ``[0, 1]`` and its provenance, which the
:class:`~himmy.services.learning.service.LearningService` blends into reputation behind
the opt-in ``outcome_weight``.

Four provenance sources, all reduced to one event shape:

* ``judge`` — the existing same-model-guarded benchmark LLM judge
  (:mod:`himmy.benchmark.judge`) graded the answer against a rubric. The judge!=candidate
  guard is enforced UP THERE; this module never self-grades.
* ``deterministic_grader`` — a caller-supplied pure function scored the answer (exact
  match, schema/contract check, regression oracle …) with no model in the loop.
* ``user_feedback`` — an explicit human thumbs-up/down or rating, normalised to ``[0, 1]``.
* ``trajectory_grade`` — a structural grade of the run's tool-call SEQUENCE (e.g. it
  looped, or used a forbidden order), produced by :mod:`himmy.benchmark.trajectory`.

Everything here is opt-in and FAIL-OPEN: a recorder built without an event sink, or one
whose emit raises, degrades to a no-op and never breaks or slows a run. Recording an
outcome with ``outcome_weight == 0`` is harmless — the event lands in the audit stream but
:class:`LearningService` never reads it back (the weight short-circuits the outcome read),
so behaviour stays byte-identical until a non-zero weight opts in.

TENANCY: like the operational events, an :class:`OutcomeRecorder` built with a
``workspace_id`` stamps it onto every emitted event so the durable backends denormalise it
and the reputation miner reads outcomes back scoped to ONE tenant.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from himmy.core.events import EventType, RunEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.core.events import EventSink

logger = logging.getLogger("himmy.services.learning")


class OutcomeSource(str, Enum):
    """Where an outcome score came from — recorded on every ``OUTCOME_SCORED`` event."""

    JUDGE = "judge"
    USER_FEEDBACK = "user_feedback"
    DETERMINISTIC_GRADER = "deterministic_grader"
    TRAJECTORY_GRADE = "trajectory_grade"


def clamp_score(value: Any) -> float:
    """Coerce ``value`` into a score in ``[0, 1]``, raising on a non-numeric input.

    A score outside the unit interval is clamped (not rejected) so a caller that passes a
    1..5 star rating pre-normalised, or a tiny floating-point overshoot, still records
    cleanly. A non-numeric value is a programming error and raises ``ValueError`` — the
    recorder catches it so a bad score never breaks the run, but the contract is explicit.
    """
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"outcome score must be numeric, got {value!r}") from exc
    if f != f:  # NaN
        raise ValueError("outcome score must not be NaN")
    return min(1.0, max(0.0, f))


class OutcomeRecorder:
    """Emit :class:`~himmy.core.events.EventType.OUTCOME_SCORED` events, best-effort.

    Construct once per run (or per runtime) with the SAME event sink the runtime already
    writes to; call :meth:`record` whenever a quality signal lands. The recorder attributes
    an outcome to a single tool ONLY when a ``tool_name`` is supplied (e.g. a single-tool
    run, or a per-tool grade) — otherwise the outcome is recorded at the run level
    (``tool_name`` omitted) and contributes to no tool's blended score, by design, since
    attributing a multi-tool run's quality to every tool would be noise.
    """

    def __init__(
        self,
        event_sink: EventSink | None,
        *,
        workspace_id: str | None = None,
    ) -> None:
        self._sink = event_sink
        self._workspace_id = workspace_id

    @property
    def enabled(self) -> bool:
        """Whether a sink is wired (else :meth:`record` is a pure no-op)."""
        return self._sink is not None

    async def record(
        self,
        score: Any,
        *,
        source: OutcomeSource,
        tool_name: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
        thread_id: str | None = None,
        rationale: str | None = None,
    ) -> bool:
        """Emit one ``OUTCOME_SCORED`` event; return whether it was emitted (best-effort).

        ``score`` is clamped to ``[0, 1]`` (:func:`clamp_score`). ``tool_name`` (when given)
        is denormalised into ``payload['tool_name']`` — the SAME column the reputation miner
        filters on — so the score blends into that tool's reputation with no schema change.
        ``run_id`` / ``trace_id`` / ``thread_id`` thread the run reference for audit. Any
        failure (no sink, bad score, sink raises) is swallowed and returns ``False``; this
        never raises, so a learning signal can never break a run.
        """
        if self._sink is None:
            return False
        try:
            value = clamp_score(score)
        except ValueError:
            logger.debug("outcome record skipped: bad score %r", score, exc_info=True)
            return False
        payload: dict[str, Any] = {
            "outcome_score": value,
            "outcome_source": source.value,
        }
        if tool_name is not None:
            # Denormalised exactly like TOOL_COMPLETED/FAILED so list_events(tool_name=...)
            # finds it — this is what lets the outcome blend into ToolReputation.
            payload["tool_name"] = tool_name
        if run_id is not None:
            payload["run_id"] = run_id
        # Rationale is provenance, not PII: the judge rationale / grader detail. Kept short
        # and only when supplied; never a user prompt.
        if rationale:
            payload["rationale"] = rationale[:500]
        event = RunEvent(
            event_type=EventType.OUTCOME_SCORED,
            trace_id=trace_id,
            thread_id=thread_id,
            workspace_id=self._workspace_id,
            payload=payload,
        )
        try:
            await self._sink.append_event(event)
        except Exception:  # pragma: no cover - defensive: learning never breaks a run
            logger.debug("OUTCOME_SCORED emit failed", exc_info=True)
            return False
        return True

    async def record_judge_verdict(
        self,
        verdict: Any,
        *,
        tool_name: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
        thread_id: str | None = None,
    ) -> bool:
        """Record a :class:`~himmy.benchmark.judge.JudgeVerdict` as an outcome.

        An UNGRADED verdict (judge timeout / unparseable / same-model refusal upstream)
        carries no honest signal, so it is NOT recorded — mirroring the benchmark, which
        never counts an ungraded trial against a model. A graded verdict records its raw
        ``score`` (the judge!=candidate guard already enforced before the verdict exists).
        """
        if getattr(verdict, "ungraded", False):
            logger.debug("outcome record skipped: judge verdict ungraded")
            return False
        return await self.record(
            getattr(verdict, "score", 0.0),
            source=OutcomeSource.JUDGE,
            tool_name=tool_name,
            run_id=run_id,
            trace_id=trace_id,
            thread_id=thread_id,
            rationale=getattr(verdict, "rationale", None) or None,
        )

    async def record_user_feedback(
        self,
        score: Any,
        *,
        tool_names: list[str],
        run_id: str | None = None,
        trace_id: str | None = None,
        thread_id: str | None = None,
        rationale: str | None = None,
    ) -> int:
        """Attribute a run-level human verdict (👍/👎) to every tool the run used.

        A thumbs-up/down is a judgement of the whole answer, not one tool — but the tools a
        run actually invoked are the levers that produced it, so the score is recorded
        against each of them (deduplicated) as a :attr:`OutcomeSource.USER_FEEDBACK`
        outcome. This is the missing positive-signal producer: it only changes a tool's
        blended reputation for an agent that has opted into ``outcome_weight > 0`` — by
        default the events are recorded and visible but inert. Returns the number emitted.
        """
        emitted = 0
        for name in dict.fromkeys(tool_names):  # dedupe, preserve order
            if await self.record(
                score,
                source=OutcomeSource.USER_FEEDBACK,
                tool_name=name,
                run_id=run_id,
                trace_id=trace_id,
                thread_id=thread_id,
                rationale=rationale,
            ):
                emitted += 1
        return emitted


__all__ = ["OutcomeRecorder", "OutcomeSource", "clamp_score"]
