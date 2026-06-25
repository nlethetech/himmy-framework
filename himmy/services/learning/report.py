"""Read-only learning report — what self-learning has concluded, for humans + UI.

The reputation miner (:mod:`himmy.services.learning.service`) and its ``LEARNING_APPLIED``
audit events are powerful but invisible: nothing in the CLI or Studio ever surfaced a tool's
score, its failure counts, or the reorders/hints the loop applied. This module turns that
recorded signal into one structured :class:`LearningReport` that BOTH the ``himmy learned``
CLI command and the Studio "Learning" panel render, so an operator can finally SEE — and
therefore trust — what the agent learned about its tools.

It is pure read: it opens no new behaviour, mutates nothing, and (like every learning path)
is best-effort — a storage error yields an empty report rather than raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from himmy.core.events import EventType
from himmy.services.learning.overrides import OverrideKind
from himmy.services.learning.service import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_WINDOW,
    TRUST_FEEDBACK_DEFAULT_WEIGHT,
    LearningService,
)

if TYPE_CHECKING:
    from himmy.services.learning.overrides import OverrideSet
    from himmy.services.storage.protocols import EventLog

logger = logging.getLogger(__name__)

#: The reputation floor below which a sufficiently-sampled tool is flagged flaky. Mirrors
#: ``ToolReputationProvider``'s default so the report's "flaky" verdict matches the reorder
#: behaviour the runtime actually applies.
DEFAULT_FLOOR = 0.2


@dataclass(frozen=True)
class LearningToolRow:
    """One tool's learned reputation, as shown to an operator."""

    tool_name: str
    #: Blended reliability score in [0, 1] (the sort key the runtime reorders on).
    score: float
    #: Pure operational reliability: completed / (completed + failed).
    operational_score: float
    completed: int
    failed: int
    #: True once enough calls exist for the score to drop below neutral (else it is the
    #: neutral prior and ``flaky`` is always False — a new tool is never punished).
    has_min_samples: bool
    #: ``has_min_samples and score < floor`` — the tools the runtime demotes/​warns about.
    #: Recomputed AFTER any operator override is applied, so a pinned / muted tool is never
    #: flaky regardless of its raw failures.
    flaky: bool
    #: P2 outcome-quality signal (mean OUTCOME_SCORED), or None when off / under-sampled.
    outcome_score: float | None
    outcome_samples: int
    #: The active operator override verdict for this tool: ``"pinned"`` / ``"ignored"`` /
    #: ``"reset"`` (rebuilding), or ``None`` for pure auto-learning. Display-only label; the
    #: behavioural change already lives in the score the runtime acts on.
    override: str | None = None

    @property
    def total(self) -> int:
        """Scored calls observed in the window."""
        return self.completed + self.failed


@dataclass(frozen=True)
class ReorderEvent:
    """A recorded ``LEARNING_APPLIED`` reorder — learning visibly changed tool order."""

    when: str | None
    tools_reordered: int
    deprioritised_tools: list[str]


@dataclass(frozen=True)
class LearningReport:
    """Everything ``himmy learned`` / the Studio panel needs to render the learning state."""

    workspace_id: str | None
    floor: float
    #: The EFFECTIVE outcome_weight — resolved from ``trust_feedback`` so the displayed
    #: weight matches what the runtime actually blends (0.3 when the gate lifted it from 0).
    outcome_weight: float
    #: Whether the per-workspace 👍/👎 trust gate is ON (blending recorded outcomes).
    trust_feedback: bool
    #: Worst (lowest score / most-failed) tool first, so problems surface at the top.
    tools: list[LearningToolRow]
    #: Most-recent reorders first.
    recent_reorders: list[ReorderEvent]

    @property
    def flaky_count(self) -> int:
        """How many sufficiently-sampled tools are below the caution floor."""
        return sum(1 for t in self.tools if t.flaky)

    @property
    def has_data(self) -> bool:
        """Whether any tool reputation has been observed yet."""
        return bool(self.tools)


async def _discover_tool_names(
    event_log: EventLog, *, workspace_id: str | None, limit: int
) -> set[str]:
    """Collect distinct tool names from recent TOOL_COMPLETED / TOOL_FAILED events."""
    names: set[str] = set()
    for event_type in (EventType.TOOL_COMPLETED, EventType.TOOL_FAILED):
        try:
            events = await event_log.list_events(
                event_type=event_type,
                workspace_id=workspace_id,
                limit=limit,
                newest_first=True,
            )
        except Exception:  # pragma: no cover - defensive: report never breaks a CLI/UI
            logger.debug("tool discovery read failed", exc_info=True)
            continue
        for ev in events:
            name = ev.payload.get("tool_name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


async def _recent_reorders(
    event_log: EventLog, *, workspace_id: str | None, limit: int
) -> list[ReorderEvent]:
    """Read the most-recent ``LEARNING_APPLIED`` reorder events."""
    try:
        events = await event_log.list_events(
            event_type=EventType.LEARNING_APPLIED,
            workspace_id=workspace_id,
            limit=limit,
            newest_first=True,
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("reorder history read failed", exc_info=True)
        return []
    reorders: list[ReorderEvent] = []
    for ev in events:
        deprioritised = ev.payload.get("deprioritised_tools") or []
        if not isinstance(deprioritised, list):
            deprioritised = []
        reorders.append(
            ReorderEvent(
                when=getattr(ev, "timestamp", None),
                tools_reordered=int(ev.payload.get("tools_reordered") or 0),
                deprioritised_tools=[str(t) for t in deprioritised],
            )
        )
    return reorders


#: Map an override kind to its display verdict in the report / panel.
_OVERRIDE_LABEL = {
    OverrideKind.PIN: "pinned",
    OverrideKind.MUTE: "ignored",
    OverrideKind.RESET: "reset",
}


async def build_learning_report(
    event_log: EventLog,
    *,
    workspace_id: str | None = None,
    outcome_weight: float = 0.0,
    window: int = DEFAULT_WINDOW,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    floor: float = DEFAULT_FLOOR,
    discover_limit: int = 4000,
    reorder_limit: int = 20,
    override_set: OverrideSet | None = None,
) -> LearningReport:
    """Build a :class:`LearningReport` from a durable event log (best-effort, read-only).

    Discovers the tools that have recorded calls, computes each one's reputation through the
    SAME :class:`LearningService` the runtime uses (so the numbers match what the agent acts
    on), flags any below ``floor`` as flaky, and attaches the recent reorder history. Scope
    to a tenant with ``workspace_id`` (matching the runtime's per-tenant reputation read).

    Sprint 2: when an ``override_set`` is supplied it is threaded into the SAME
    :class:`LearningService` (so reps are post-apply — pinned / muted tools already report
    score 1.0 and ``has_min_samples=False``), each row is stamped with its override verdict,
    ``flaky`` is recomputed AFTER apply (so a pinned / muted tool is never flaky), and the
    effective ``outcome_weight`` is resolved from ``trust_feedback`` so the displayed weight
    matches what the runtime blends. With no override set and trust off the report is
    byte-identical to Sprint 1.
    """
    trust_feedback = bool(override_set.trust_feedback) if override_set else False
    requested_weight = min(1.0, max(0.0, float(outcome_weight)))
    # The trust gate lifts the weight from 0 -> a sensible default, never overriding an
    # explicit positive value — mirroring from_spec so the table matches the runtime.
    effective_weight = requested_weight
    if trust_feedback and effective_weight <= 0.0:
        effective_weight = TRUST_FEEDBACK_DEFAULT_WEIGHT

    service = LearningService(
        event_log,
        window=window,
        min_samples=min_samples,
        workspace_id=workspace_id,
        outcome_weight=effective_weight,
        override_set=override_set,
    )

    tool_names = await _discover_tool_names(
        event_log, workspace_id=workspace_id, limit=discover_limit
    )
    reps = (
        await service.get_tool_reputation(sorted(tool_names)) if tool_names else {}
    )

    rows = []
    for name, rep in reps.items():
        ov = override_set.for_tool(name) if override_set else None
        rows.append(
            LearningToolRow(
                tool_name=name,
                score=rep.score,
                operational_score=rep.operational_score,
                completed=rep.completed,
                failed=rep.failed,
                has_min_samples=rep.has_min_samples,
                # ``rep`` is already post-override, so a pinned/muted tool has
                # ``has_min_samples=False`` -> never flaky, exactly as the runtime sees it.
                flaky=rep.has_min_samples and rep.score < floor,
                outcome_score=rep.outcome_score,
                outcome_samples=rep.outcome_samples,
                override=_OVERRIDE_LABEL.get(ov.kind) if ov else None,
            )
        )
    # Worst first: lowest score, then most failures, then name — problems on top.
    rows.sort(key=lambda r: (r.score, -r.failed, r.tool_name))

    reorders = await _recent_reorders(
        event_log, workspace_id=workspace_id, limit=reorder_limit
    )

    return LearningReport(
        workspace_id=workspace_id,
        floor=floor,
        outcome_weight=effective_weight,
        trust_feedback=trust_feedback,
        tools=rows,
        recent_reorders=reorders,
    )


__all__ = [
    "DEFAULT_FLOOR",
    "LearningToolRow",
    "ReorderEvent",
    "LearningReport",
    "build_learning_report",
]
