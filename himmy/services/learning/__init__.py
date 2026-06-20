"""Self-learning (P1): turn recorded tool signals into behavioural change.

The P0 layer RECORDS ``TOOL_FAILED`` / ``TOOL_COMPLETED`` run-events into indexed
storage columns; this kernel is the *brain* that CONSUMES them. It mines a per-tool
reliability score from recent history and feeds two best-effort, opt-in hooks:

* :class:`LearnedHintsContextAdapter` injects a short, privacy-safe reliability note
  into the system prompt (e.g. "tool X failed 4 of its last 5 calls — verify inputs");
* a :class:`ToolReputationProvider` snapshot lets ``ToolService.bound_tools`` stable-sort
  flaky tools after reliable ones (annotate, never drop).

Everything here is wrapped so any storage/data error degrades to the neutral result —
learning must never break or noticeably slow a run.
"""

from __future__ import annotations

from himmy.services.learning.adapter import LearnedHintsContextAdapter
from himmy.services.learning.outcome import (
    OutcomeRecorder,
    OutcomeSource,
    clamp_score,
)
from himmy.services.learning.service import (
    LearningService,
    ToolReputation,
    ToolReputationProvider,
)
from himmy.services.learning.trajectory import (
    TrajectoryAdvisor,
    TrajectorySignal,
    render_trajectory_hints,
)

__all__ = [
    "LearnedHintsContextAdapter",
    "LearningService",
    "OutcomeRecorder",
    "OutcomeSource",
    "ToolReputation",
    "ToolReputationProvider",
    "TrajectoryAdvisor",
    "TrajectorySignal",
    "clamp_score",
    "render_trajectory_hints",
]
