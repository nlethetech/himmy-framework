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

from himmy.benchmark.stats import mean, percentile, wilson_interval


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


@dataclass(frozen=True)
class BenchmarkSuite:
    """A named collection of benchmark tasks."""

    name: str
    tasks: list[BenchmarkTask]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSuite:
        """Build a suite from a parsed mapping (``{name, tasks: [...]}``)."""
        tasks = [
            BenchmarkTask(
                id=str(t["id"]),
                prompt=str(t["prompt"]),
                grade=dict(t["grade"]),
                packs=[str(p) for p in t.get("packs", [])],
                skills=[str(s) for s in t.get("skills", [])],
                expect_tools=[str(x) for x in t.get("expect_tools", [])],
                instructions=[str(i) for i in t.get("instructions", [])],
                category=str(t.get("category", "general")),
                files={str(k): str(v) for k, v in (t.get("files") or {}).items()},
                sqlite=[str(s) for s in (t.get("sqlite") or [])],
            )
            for t in data.get("tasks", [])
        ]
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

    @property
    def name(self) -> str:
        """Display name (``label`` if set, else ``provider:model``)."""
        return self.label or f"{self.provider}:{self.model}"


@dataclass
class TrialResult:
    """The outcome of one task run (one trial)."""

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


@dataclass
class ModelScorecard:
    """All task scores for one model, with model-level aggregates."""

    spec: ModelSpec
    task_scores: list[TaskScore]

    @property
    def _all(self) -> list[TrialResult]:
        return [t for s in self.task_scores for t in s.trials]

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

    def by_category(self) -> dict[str, float]:
        """Accuracy per task category."""
        cats: dict[str, list[TrialResult]] = {}
        for score in self.task_scores:
            cats.setdefault(score.category, []).extend(score.trials)
        return {
            cat: (sum(1 for t in trials if t.correct) / len(trials) if trials else 0.0)
            for cat, trials in cats.items()
        }


__all__ = [
    "BenchmarkTask",
    "BenchmarkSuite",
    "ModelSpec",
    "TrialResult",
    "TaskScore",
    "ModelScorecard",
]
