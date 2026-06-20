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
from himmy.benchmark.trajectory import Trajectory

#: Task category that marks an irrelevance/abstention task — one that binds tools the
#: prompt does NOT need, where calling any tool is the failure (over-calling is a top
#: small/open-model failure, scored first-class by BFCL). The ``irrelevance_accuracy``
#: headline metric is computed over exactly these tasks, separate from tool-call accuracy.
ABSTENTION_CATEGORY = "irrelevance"

#: Task category that marks a MULTI-TURN tool task — one where a tool's RESULT must feed a
#: later tool call's argument (BFCL's multi-turn data-flow axis). The ``multi_turn``
#: leaderboard column is computed over exactly these tasks. A multi-turn task is scored
#: deterministically like any other (and gates ``correct``); the category is the headline
#: marker so chained-call competence reads as its own number, distinct from single-shot
#: tool accuracy.
MULTI_TURN_CATEGORY = "multi_turn"


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
    # The full structured tool-call trajectory (name + args + result, grouped by the turn
    # each call was emitted in) the arg-level / parallelism graders run against. The flat
    # name-only `tools_called` above is its `.names` view, kept for back-compat with the
    # baseline gate, cache, and the legacy trajectory graders. Empty when no tool was used.
    trajectory: Trajectory = field(default_factory=Trajectory)
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
    # Whether the task declared an ARGUMENT-/result-level trajectory grader (the BFCL
    # AST-check equivalent — "right tool, RIGHT args / right data-flow", not just the right
    # tool name). Set by the runner from the task's `trajectory` block. Feeds the
    # leaderboard's arg-accuracy column, which pools exactly these tasks so the dominant
    # small-model failure ("right tool, WRONG args") reads as its own headline. Defaults
    # False so existing positional construction (`TaskScore(id, cat, trials)`) is unchanged.
    uses_arg_grader: bool = False

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def is_multi_turn(self) -> bool:
        """Whether this is a multi-turn task (category ``multi_turn``).

        A multi-turn task chains tool calls so a tool's result feeds a later call's
        argument (the BFCL data-flow axis). The category is the first-class marker the
        leaderboard's ``multi_turn`` column keys off of.
        """
        return self.category == MULTI_TURN_CATEGORY

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
    def is_irrelevance(self) -> bool:
        """Whether this is an irrelevance/abstention task (category ``irrelevance``).

        These tasks bind tools the prompt does NOT need; the correct behaviour is to
        abstain (call no tool). The category is the first-class marker (BFCL scores
        irrelevance as its own suite), kept distinct from the gated accuracy/tool-call
        metrics so over-calling reads as its own headline number.
        """
        return self.category == ABSTENTION_CATEGORY

    @property
    def abstained(self) -> int:
        """Irrelevance trials where the model correctly abstained (called no tool).

        Excludes errored trials: a provider/schema error (e.g. a 400) is not an
        abstention DECISION — the model never got to choose — so it must not be
        counted as a correct abstention.
        """
        return sum(1 for t in self.trials if not t.tools_called and not t.error)

    @property
    def abstention_rate(self) -> float | None:
        """Fraction of (non-errored) trials that abstained (``None`` for non-irrelevance)."""
        if not self.is_irrelevance:
            return None
        clean = [t for t in self.trials if not t.error]
        if not clean:
            return None
        return self.abstained / len(clean)

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

    @property
    def irrelevance_scores(self) -> list[TaskScore]:
        """The irrelevance/abstention task scores (category ``irrelevance``)."""
        return [s for s in self.task_scores if s.is_irrelevance]

    @property
    def has_irrelevance_tier(self) -> bool:
        """Whether the suite has any irrelevance/abstention task."""
        return bool(self.irrelevance_scores)

    @property
    def irrelevance_accuracy(self) -> float | None:
        """Fraction of irrelevance trials the model correctly ABSTAINED on (no tool).

        The BFCL-style abstention headline: over tasks that bind tools the prompt does
        NOT need, the share of trials where the model used no tool at all. A model that
        always abstains scores ``1.0``; one that over-calls on every such task scores
        ``0.0``. ``None`` when the suite has no irrelevance task. Reported SEPARATELY
        from :attr:`tool_call_accuracy` (over-calling is its own failure mode).

        Errored trials (e.g. a provider 400) are EXCLUDED from both numerator and
        denominator — an error is not an abstention decision, so it must not inflate
        the score; surfacing errors is :attr:`error_rate`'s job.
        """
        trials = [
            t for s in self.irrelevance_scores for t in s.trials if not t.error
        ]
        if not trials:
            return None
        return sum(1 for t in trials if not t.tools_called) / len(trials)

    @property
    def irrelevance_total_trials(self) -> int:
        """Total irrelevance/abstention trials run."""
        return sum(len(s.trials) for s in self.irrelevance_scores)

    @property
    def arg_scores(self) -> list[TaskScore]:
        """Deterministic task scores that declared an argument-/result-level grader.

        These are the tasks whose `trajectory` block checks call ARGUMENTS or data-flow
        (the BFCL AST-check equivalent), not just the tool name — the dominant small-model
        failure is "right tool, WRONG args". Judge-tier tasks are excluded (their accuracy
        is reported separately and never gated).
        """
        return [s for s in self.deterministic_scores if s.uses_arg_grader]

    @property
    def has_arg_tier(self) -> bool:
        """Whether the suite has any argument-/result-level graded task."""
        return bool(self.arg_scores)

    @property
    def arg_accuracy(self) -> float | None:
        """Pooled correctness over tasks graded at the ARGUMENT/data-flow level.

        The BFCL "right tool, RIGHT args" headline: a model that calls the right tool with
        the wrong arguments scores 0 here even though a name-only tool-call check would pass.
        ``None`` when the suite has no argument-graded task.
        """
        trials = [t for s in self.arg_scores for t in s.trials]
        if not trials:
            return None
        return sum(1 for t in trials if t.correct) / len(trials)

    @property
    def arg_total_trials(self) -> int:
        """Total argument-/result-level graded trials run."""
        return sum(len(s.trials) for s in self.arg_scores)

    @property
    def multi_turn_scores(self) -> list[TaskScore]:
        """Deterministic task scores in the multi-turn category (chained tool calls)."""
        return [
            s
            for s in self.deterministic_scores
            if s.category == MULTI_TURN_CATEGORY
        ]

    @property
    def has_multi_turn_tier(self) -> bool:
        """Whether the suite has any multi-turn (chained-call) task."""
        return bool(self.multi_turn_scores)

    @property
    def multi_turn_accuracy(self) -> float | None:
        """Pooled correctness over multi-turn tasks (a tool result feeds a later call).

        The BFCL multi-turn headline: can the model carry one tool's output into the next
        call's argument? ``None`` when the suite has no multi-turn task.
        """
        trials = [t for s in self.multi_turn_scores for t in s.trials]
        if not trials:
            return None
        return sum(1 for t in trials if t.correct) / len(trials)

    @property
    def multi_turn_total_trials(self) -> int:
        """Total multi-turn trials run."""
        return sum(len(s.trials) for s in self.multi_turn_scores)

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
    "ABSTENTION_CATEGORY",
    "MULTI_TURN_CATEGORY",
]
