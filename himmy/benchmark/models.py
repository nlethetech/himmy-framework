"""Benchmark data model: tasks, suites, model specs, trial results, and scorecards.

The flow: a :class:`BenchmarkSuite` (declarative tasks) is run for each :class:`ModelSpec`
over N trials, producing :class:`TrialResult`s; those aggregate into a :class:`TaskScore`
per task (pass rate + Wilson CI, tool-call rate, latency percentiles) and a
:class:`ModelScorecard` per model (overall accuracy + CI, tool-call accuracy, p50/p95
latency, cost, per-category breakdown) — so models/configs compare on hard numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from himmy.benchmark.stats import (
    McNemarResult,
    mcnemar_from_outcomes,
    mean,
    percentile,
    wilson_interval,
)


@dataclass(frozen=True)
class BenchmarkTask:
    """One graded task: a prompt, the tools it needs, and how to grade the answer."""

    id: str
    prompt: str
    grade: dict[str, Any]
    packs: list[str] = field(default_factory=list)
    skills: list[str] = field(
        default_factory=list
    )  # capabilities (imply packs+know-how)
    expect_tools: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    category: str = "general"
    # Self-contained fixtures (materialized in a temp workspace per trial) so file/SQL
    # tasks are reproducible without external setup.
    files: dict[str, str] = field(default_factory=dict)
    sqlite: list[str] = field(default_factory=list)  # init SQL statements
    # Optional trajectory grader (graded against the ordered tool-call sequence, not the
    # answer). Same declarative shape as `grade` but with trajectory predicate types
    # (first_tool/max_tool_calls/tool_called/tool_not_called/tool_sequence). When set,
    # BOTH `grade` (answer) and `trajectory` must pass for the trial to count correct.
    trajectory: dict[str, Any] = field(default_factory=dict)
    # Optional multi-agent team. When set, the task runs as a TEAM (handoff/delegation
    # via MultiAgentOrchestrator, or a GroupChatOrchestrator) instead of a single agent;
    # the synthetic collaboration tools (transfer_to_<peer>/ask_<worker>) appear in the
    # trial's ordered tool sequence so a `trajectory` grader can assert on the routing.
    # Empty ⇒ single-agent (the unchanged default path). See `himmy.benchmark.team`.
    team: dict[str, Any] = field(default_factory=dict)
    # Optional LLM-as-judge grader for genuinely open-ended tasks (no computable ground
    # truth — e.g. summarization). Shape: `{rubric: str, threshold: float}`. When set, the
    # trial is graded by a JUDGE model (configured per-run on the ModelSpec) against the
    # rubric instead of (or in addition to) the deterministic `grade`. Judge-tier results
    # are REPORTED, NEVER GATING — the baseline gate ignores them. Empty ⇒ no judging.
    # See `himmy.benchmark.judge`.
    judge: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSuite:
    """A named collection of benchmark tasks."""

    name: str
    tasks: list[BenchmarkTask]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSuite:
        """Build a suite from a parsed mapping (``{name, tasks: [...]}``).

        Every task must declare at least one way to be scored — a ``grade`` (answer),
        a ``trajectory`` (tool-path), or a ``judge`` (LLM-graded) block. A task with
        none of these would silently pass every trial (``correct=True`` with no check),
        inflating accuracy and masking a broken suite, so it is rejected here at load
        time (fail-loud) rather than scored as a vacuous 100%.
        """
        tasks: list[BenchmarkTask] = []
        for t in data.get("tasks", []):
            grade = dict(t.get("grade") or {})
            trajectory = dict(t.get("trajectory") or {})
            judge = dict(t.get("judge") or {})
            team = dict(t.get("team") or {})
            if not grade and not trajectory and not judge:
                raise ValueError(
                    f"benchmark task {str(t.get('id', '?'))!r} has no grade, trajectory, "
                    "or judge block — it would pass every trial unchecked; declare at "
                    "least one scoring block"
                )
            tasks.append(
                BenchmarkTask(
                    id=str(t["id"]),
                    prompt=str(t["prompt"]),
                    # `grade` is optional for judge-tier tasks (graded by an LLM judge
                    # against a `judge` block instead of a deterministic answer grader).
                    grade=grade,
                    packs=[str(p) for p in t.get("packs", [])],
                    skills=[str(s) for s in t.get("skills", [])],
                    expect_tools=[str(x) for x in t.get("expect_tools", [])],
                    instructions=[str(i) for i in t.get("instructions", [])],
                    category=str(t.get("category", "general")),
                    files={str(k): str(v) for k, v in (t.get("files") or {}).items()},
                    sqlite=[str(s) for s in (t.get("sqlite") or [])],
                    trajectory=trajectory,
                    team=team,
                    judge=judge,
                )
            )
        return cls(name=str(data.get("name", "suite")), tasks=tasks)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkSuite:
        """Load a suite from a YAML file."""
        import yaml

        raw = yaml.safe_load(Path(path).expanduser().read_text()) or {}
        return cls.from_dict(raw)

    @property
    def packs(self) -> list[str]:
        """The union of all tool packs any task needs (incl. those a skill implies)."""
        seen: list[str] = []
        registry: Any = None
        for task in self.tasks:
            packs = list(task.packs)
            if task.skills:
                from himmy.skills import build_skill_registry, resolve_skills

                if registry is None:
                    registry = build_skill_registry()
                packs.extend(resolve_skills(task.skills, registry).tool_packs)
            for p in packs:
                if p not in seen:
                    seen.append(p)
        return seen


@dataclass(frozen=True)
class ModelSpec:
    """A model + agent configuration to benchmark (one column of the scorecard)."""

    provider: str
    model: str
    label: str = ""
    tool_router: bool = False
    temperature: float | None = 0.0
    extra_packs: list[str] = field(default_factory=list)  # distractor tools
    max_turns: int = 6
    # Judge model for the LLM-as-judge grader tier (per-run configurable). When a suite
    # has judge-tier tasks, set `judge_model` (and optionally `judge_provider`, defaulting
    # to this spec's provider) to a model DISTINCT from this candidate — the runner refuses
    # to let a model grade its own output. Unset ⇒ judge-tier tasks for this candidate are
    # recorded ungraded. See `himmy.benchmark.judge`.
    judge_provider: str = ""
    judge_model: str = ""

    @property
    def name(self) -> str:
        """Display name (``label`` if set, else ``provider:model``)."""
        return self.label or f"{self.provider}:{self.model}"


@dataclass
class TrialResult:
    """The outcome of one task run (one trial).

    ``tools_called`` is the agent's tool calls **in call order, with repeats** — the
    canonical trajectory the trajectory graders run against. ``tool_ok`` is the
    set-membership "did it call the expected tools?" check (order-agnostic, kept for
    back-compat with the baseline gate and cache). ``answer_ok`` / ``trajectory_ok``
    split ``correct`` into its two halves so a report can show *why* a trial failed:
    ``correct = answer_ok and (trajectory_ok is not False)`` — i.e. the answer grader
    must pass, and the trajectory grader (when the task declares one) must too.
    ``trajectory_ok`` is ``None`` when the task declares no trajectory expectation.
    """

    task_id: str
    answer: str
    tools_called: list[str]
    correct: bool
    tool_ok: bool | None  # None when the task expects no specific tool
    latency_s: float
    turns: int
    input_tokens: int
    output_tokens: int
    cost: float
    error: str | None = None
    answer_ok: bool = True
    trajectory_ok: bool | None = None
    # LLM-as-judge tier (appended last so positional construction in existing tests stays
    # valid). ``judged`` marks a trial graded by a JUDGE model (a judge-tier task) — these
    # are reported separately and NEVER fold into the deterministic accuracy/gate.
    # ``judge_ok`` is the verdict (None when not judged or ungraded); ``judge_ungraded`` is
    # the third state (timeout / unparseable verdict) — distinct from a graded fail.
    judged: bool = False
    judge_ok: bool | None = None
    judge_score: float | None = None
    judge_ungraded: bool = False
    judge_model: str = ""


@dataclass
class TaskScore:
    """Aggregated trials for one task under one model."""

    task_id: str
    category: str
    trials: list[TrialResult]

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def successes(self) -> int:
        return sum(1 for t in self.trials if t.correct)

    @property
    def pass_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def pass_ci(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.n)

    @property
    def tool_trials(self) -> list[TrialResult]:
        return [t for t in self.trials if t.tool_ok is not None]

    @property
    def tool_call_rate(self) -> float | None:
        tt = self.tool_trials
        if not tt:
            return None
        return sum(1 for t in tt if t.tool_ok) / len(tt)

    @property
    def latencies(self) -> list[float]:
        return [t.latency_s for t in self.trials]

    @property
    def p50_latency(self) -> float:
        return percentile(self.latencies, 0.5)

    @property
    def errors(self) -> int:
        return sum(1 for t in self.trials if t.error)

    @property
    def trajectory_failures(self) -> int:
        """Trials whose trajectory grader explicitly failed (``trajectory_ok is False``).

        Distinct from ``errors`` and from a wrong answer: the agent produced a result but
        took a bad *path* (wrong first tool, too many tool calls, a forbidden/unknown
        tool, a missing required ordering). ``0`` when the task declares no trajectory
        expectation.
        """
        return sum(1 for t in self.trials if t.trajectory_ok is False)

    @property
    def is_judge_tier(self) -> bool:
        """Whether this task is graded by an LLM judge (any trial ``judged``)."""
        return any(t.judged for t in self.trials)

    @property
    def judge_passes(self) -> int:
        """Trials the judge graded as passing (``judge_ok is True``)."""
        return sum(1 for t in self.trials if t.judge_ok is True)

    @property
    def judge_graded(self) -> list[TrialResult]:
        """Judge-tier trials that received a real verdict (not ungraded)."""
        return [t for t in self.trials if t.judged and not t.judge_ungraded]

    @property
    def judge_ungraded(self) -> int:
        """Judge-tier trials the judge could not grade (timeout / unparseable verdict)."""
        return sum(1 for t in self.trials if t.judged and t.judge_ungraded)

    @property
    def judge_pass_rate(self) -> float | None:
        """Pass rate over *graded* judge trials (``None`` when none were graded)."""
        graded = self.judge_graded
        if not graded:
            return None
        return sum(1 for t in graded if t.judge_ok) / len(graded)


@dataclass
class ModelScorecard:
    """All task scores for one model, with model-level aggregates."""

    spec: ModelSpec
    task_scores: list[TaskScore]

    @property
    def deterministic_scores(self) -> list[TaskScore]:
        """Task scores graded deterministically (the gated tier — excludes judge tasks)."""
        return [s for s in self.task_scores if not s.is_judge_tier]

    @property
    def judge_scores(self) -> list[TaskScore]:
        """Task scores graded by an LLM judge (the reported-not-gated tier)."""
        return [s for s in self.task_scores if s.is_judge_tier]

    @property
    def _all(self) -> list[TrialResult]:
        # Deterministic-tier trials only: judge-tier results are reported separately and
        # NEVER fold into the gated accuracy/error/latency aggregates or the baseline gate.
        return [t for s in self.deterministic_scores for t in s.trials]

    @property
    def total_trials(self) -> int:
        return len(self._all)

    @property
    def accuracy(self) -> float:
        """Pooled correctness across every trial."""
        trials = self._all
        return sum(1 for t in trials if t.correct) / len(trials) if trials else 0.0

    @property
    def accuracy_ci(self) -> tuple[float, float]:
        trials = self._all
        return wilson_interval(sum(1 for t in trials if t.correct), len(trials))

    @property
    def tool_call_accuracy(self) -> float | None:
        tool_trials = [t for t in self._all if t.tool_ok is not None]
        if not tool_trials:
            return None
        return sum(1 for t in tool_trials if t.tool_ok) / len(tool_trials)

    @property
    def p50_latency(self) -> float:
        return percentile([t.latency_s for t in self._all], 0.5)

    @property
    def p95_latency(self) -> float:
        return percentile([t.latency_s for t in self._all], 0.95)

    @property
    def mean_cost(self) -> float:
        return mean([t.cost for t in self._all])

    @property
    def error_rate(self) -> float:
        trials = self._all
        return sum(1 for t in trials if t.error) / len(trials) if trials else 0.0

    @property
    def trajectory_failures(self) -> int:
        """Total trials across the suite whose trajectory grader explicitly failed."""
        return sum(1 for t in self._all if t.trajectory_ok is False)

    def by_category(self) -> dict[str, float]:
        """Accuracy per task category (deterministic tier only; feeds the gate)."""
        cats: dict[str, list[TrialResult]] = {}
        for score in self.deterministic_scores:
            cats.setdefault(score.category, []).extend(score.trials)
        return {
            cat: (sum(1 for t in trials if t.correct) / len(trials) if trials else 0.0)
            for cat, trials in cats.items()
        }

    def category_counts(self) -> dict[str, int]:
        """Number of deterministic *tasks* (not trials) in each category."""
        counts: dict[str, int] = {}
        for score in self.deterministic_scores:
            counts[score.category] = counts.get(score.category, 0) + 1
        return counts

    @property
    def has_judge_tier(self) -> bool:
        """Whether any task in this scorecard was graded by an LLM judge."""
        return bool(self.judge_scores)

    @property
    def judge_accuracy(self) -> float | None:
        """Pass rate over all *graded* judge-tier trials (``None`` when none graded)."""
        graded = [
            t
            for s in self.judge_scores
            for t in s.trials
            if t.judged and not t.judge_ungraded
        ]
        if not graded:
            return None
        return sum(1 for t in graded if t.judge_ok) / len(graded)

    @property
    def judge_ungraded(self) -> int:
        """Total judge-tier trials the judge could not grade (timeout / unparseable)."""
        return sum(s.judge_ungraded for s in self.judge_scores)

    @property
    def judge_total_trials(self) -> int:
        """Total judge-tier trials run (graded + ungraded)."""
        return sum(len(s.trials) for s in self.judge_scores)

    def trial_outcomes(self) -> dict[tuple[str, int], bool]:
        """Per-trial pass/fail grid keyed by ``(task_id, trial_index)``.

        The trial index is the position within a task's trial list. This is the plain
        data a paired model comparison (McNemar) needs — see
        :func:`compare_scorecards`.
        """
        grid: dict[tuple[str, int], bool] = {}
        for score in self.deterministic_scores:
            for i, trial in enumerate(score.trials):
                grid[(score.task_id, i)] = trial.correct
        return grid


def compare_scorecards(a: ModelScorecard, b: ModelScorecard) -> McNemarResult:
    """Paired McNemar comparison of two scorecards from the *same* benchmark run.

    Trials are paired by ``(task_id, trial_index)`` over the intersection of the two
    models' grids (trials one model ran but the other did not are ignored). Returns a
    :class:`~himmy.benchmark.stats.McNemarResult` — discordant counts, the exact
    two-sided p-value, the leading model name, and the discordant task ids. The math
    itself lives in :mod:`himmy.benchmark.stats` (pure functions over plain data).
    """
    return mcnemar_from_outcomes(
        a.spec.name, a.trial_outcomes(), b.spec.name, b.trial_outcomes()
    )


__all__ = [
    "BenchmarkTask",
    "BenchmarkSuite",
    "ModelSpec",
    "TrialResult",
    "TaskScore",
    "ModelScorecard",
    "compare_scorecards",
]
