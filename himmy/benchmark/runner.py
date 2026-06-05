"""Benchmark runner: execute a suite against models over N trials, collect metrics.

For each :class:`ModelSpec` × task × trial it builds a fresh runtime (the task's tool
packs + any distractor ``extra_packs``, self-contained ``files``/``sqlite`` fixtures in a
throwaway workspace), runs the real agent loop, and records correctness (via the task's
grader), whether the expected tool was called, wall-clock latency, turns, tokens, and
cost. Every trial is isolated, so results are reproducible and a model can't leak state
between tasks.
"""

from __future__ import annotations

import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from himmy.benchmark.graders import grade
from himmy.benchmark.models import (
    ModelScorecard,
    ModelSpec,
    TaskScore,
    TrialResult,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Sequence

    from himmy.benchmark.models import BenchmarkSuite, BenchmarkTask

#: ``on_progress(spec, task, trial_index, total_trials)`` for live progress.
ProgressFn = "Callable[[ModelSpec, BenchmarkTask, int, int], None]"


class BenchmarkRunner:
    """Runs a suite against models and scores the results."""

    def __init__(
        self,
        *,
        trials: int = 3,
        on_progress: Any = None,
        clock: Any = None,
        inference_factory: Any = None,
    ) -> None:
        """``trials`` runs per task (more → tighter confidence intervals).

        ``inference_factory(spec) -> InferenceService`` overrides how a model's
        inference is built (default: the CLI provider builder); tests inject a
        deterministic one.
        """
        if trials < 1:
            raise ValueError("trials must be >= 1")
        self._trials = trials
        self._on_progress = on_progress
        self._clock = clock or time.perf_counter
        self._inference_factory = inference_factory

    async def run(
        self, suite: BenchmarkSuite, specs: Sequence[ModelSpec]
    ) -> list[ModelScorecard]:
        """Benchmark every spec against the suite; return one scorecard per model."""
        cards: list[ModelScorecard] = []
        for spec in specs:
            task_scores: list[TaskScore] = []
            for task in suite.tasks:
                trials: list[TrialResult] = []
                for i in range(self._trials):
                    trials.append(await self._run_trial(spec, task))
                    if self._on_progress is not None:
                        self._on_progress(spec, task, i + 1, self._trials)
                task_scores.append(TaskScore(task.id, task.category, trials))
            cards.append(ModelScorecard(spec=spec, task_scores=task_scores))
        return cards

    async def _run_trial(self, spec: ModelSpec, task: BenchmarkTask) -> TrialResult:
        """Run one task once and capture its metrics (errors are recorded, not raised)."""
        from himmy.agents.base_agent.task import Task as AgentTask
        from himmy.agents.personas.persona import Persona
        from himmy.services.inference.models import LLMConfig

        started = self._clock()
        try:
            with self._fixtures(task) as config:
                runtime = self._build_runtime(spec, task, config)
                llm = LLMConfig(model_key="default", temperature=spec.temperature)
                loop = await runtime.run_agent_loop(
                    Persona(name="bench", instructions=task.instructions),
                    AgentTask(title=task.id, prompt=task.prompt),
                    llm_config=llm,
                    max_turns=spec.max_turns,
                    route_tools=spec.tool_router,
                )
            latency = self._clock() - started
            final = loop.final
            called = [tc.tool_name for t in loop.turns for tc in t.tool_calls]
            answer = final.output_text or ""
            return TrialResult(
                task_id=task.id,
                answer=answer,
                tools_called=called,
                correct=grade(task.grade, answer),
                tool_ok=(set(task.expect_tools) <= set(called))
                if task.expect_tools
                else None,
                latency_s=latency,
                turns=loop.turn_count,
                input_tokens=final.input_tokens,
                output_tokens=final.output_tokens,
                cost=loop.total_cost,
                error=None if final.succeeded else (final.error or "run failed"),
            )
        except Exception as exc:  # noqa: BLE001 - a crash is a (recorded) trial failure
            return TrialResult(
                task_id=task.id,
                answer="",
                tools_called=[],
                correct=False,
                tool_ok=False if task.expect_tools else None,
                latency_s=self._clock() - started,
                turns=0,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                error=str(exc),
            )

    @contextmanager
    def _fixtures(self, task: BenchmarkTask) -> Iterator[Any]:
        """Materialize the task's files/sqlite in a temp workspace; yield a config."""
        from himmy.toolkit import ToolkitConfig

        config = ToolkitConfig.from_env()
        if not task.files and not task.sqlite:
            yield config
            return
        with tempfile.TemporaryDirectory(prefix="himmy-bench-") as workdir:
            work = Path(workdir)
            for name, content in task.files.items():
                (work / name).write_text(content, encoding="utf-8")
            updates: dict[str, Any] = {"fs_root": work, "fs_allow_write": True}
            if task.sqlite:
                import sqlite3

                db = work / "bench.db"
                conn = sqlite3.connect(db)
                for stmt in task.sqlite:
                    conn.executescript(stmt)
                conn.commit()
                conn.close()
                updates["sqlite_path"] = str(db)
            yield config.model_copy(update=updates)

    def _build_runtime(self, spec: ModelSpec, task: BenchmarkTask, config: Any) -> Any:
        """Build a fresh runtime with the task's packs + any distractor packs."""
        from himmy import build_runtime
        from himmy.services.tools.registry import ToolRegistry
        from himmy.toolkit import register_packs

        if self._inference_factory is not None:
            inference = self._inference_factory(spec)
        else:
            from himmy.cli.provider import build_inference_for

            inference = build_inference_for(spec.provider, spec.model)
        packs = list(dict.fromkeys([*task.packs, *spec.extra_packs]))
        overrides: dict[str, Any] = {"inference": inference}
        if packs:
            registry = ToolRegistry()
            register_packs(registry, packs, config)
            overrides["tool_registry"] = registry
        runtime, _inf, _tools = build_runtime(**overrides)
        return runtime


__all__ = ["BenchmarkRunner", "ProgressFn"]
