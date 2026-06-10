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

from himmy.benchmark.graders import grade, grade_trajectory
from himmy.benchmark.judge import build_judge, build_judge_spec, has_judge, judge_answer
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

#: A trial's normalized outcome, shared by the single-agent and team code paths:
#: ``(answer, ordered_tools_called, turns, (input_tokens, output_tokens), cost, error)``.
_RunOutcome = tuple[str, list[str], int, tuple[int, int], float, "str | None"]


class BenchmarkRunner:
    """Runs a suite against models and scores the results."""

    def __init__(
        self,
        *,
        trials: int = 3,
        on_progress: Any = None,
        clock: Any = None,
        inference_factory: Any = None,
        judge_inference_factory: Any = None,
        append_history: bool | None = None,
        history_path: Any = None,
    ) -> None:
        """``trials`` runs per task (more → tighter confidence intervals).

        ``inference_factory(spec) -> InferenceService`` overrides how a model's
        inference is built (default: the CLI provider builder); tests inject a
        deterministic one. ``judge_inference_factory(judge_spec) -> InferenceService``
        does the same for the LLM-judge tier (a distinct judge model); tests inject a
        deterministic judge stub, otherwise the CLI provider builder constructs it.

        ``append_history`` controls whether a scored :meth:`run` appends to the
        benchmark run history (``benchmarks/history.jsonl``). ``None`` (the default)
        defers to the ``HIMMY_BENCH_NO_HISTORY`` env switch (history on unless set);
        pass ``False`` to disable for this runner regardless. ``history_path``
        overrides the destination (tests point it at a tmp file).
        """
        if trials < 1:
            raise ValueError("trials must be >= 1")
        self._trials = trials
        self._on_progress = on_progress
        self._clock = clock or time.perf_counter
        self._inference_factory = inference_factory
        self._judge_inference_factory = judge_inference_factory
        self._append_history = append_history
        self._history_path = history_path

    async def run(
        self, suite: BenchmarkSuite, specs: Sequence[ModelSpec]
    ) -> list[ModelScorecard]:
        """Benchmark every spec against the suite; return one scorecard per model.

        When the suite has judge-tier tasks, the judge model for each candidate is
        validated up front (``judge != candidate``, judge model configured) so a
        misconfiguration fails loudly at setup rather than as a per-trial error.
        """
        self._preflight_judge(suite, specs)
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
        self._maybe_append_history(cards, suite)
        return cards

    def _maybe_append_history(
        self, cards: list[ModelScorecard], suite: BenchmarkSuite
    ) -> None:
        """Append the scored run to history (best-effort — never fails the run)."""
        from himmy.benchmark.history import append_run, history_enabled

        if not history_enabled(enabled=self._append_history):
            return
        try:
            append_run(
                cards,
                suite_name=suite.name,
                path=self._history_path,
                enabled=self._append_history,
            )
        except OSError as exc:  # noqa: BLE001 - history is best-effort, not the result
            import logging

            logging.getLogger(__name__).warning(
                "failed to append benchmark history: %s", exc
            )

    def _preflight_judge(
        self, suite: BenchmarkSuite, specs: Sequence[ModelSpec]
    ) -> None:
        """Validate judge config for every *configured* candidate when the suite judges.

        A spec that DOES configure a judge is checked up front: a judge equal to the
        candidate model raises :class:`SameModelJudgeError` before any trial runs (a
        misconfig fails loud, not as opaque per-trial errors). A spec with NO judge
        configured is left alone — its judge-tier trials are recorded *ungraded* (the
        documented degrade path, see :class:`~himmy.benchmark.models.ModelSpec`), so the
        suite's deterministic tasks still run instead of the whole run aborting.
        """
        from himmy.benchmark.judge import resolve_judge_model

        if not any(has_judge(t) for t in suite.tasks):
            return
        for spec in specs:
            if not spec.judge_model:
                continue  # no judge configured ⇒ judge-tier trials recorded ungraded
            # Configured: raises SameModelJudgeError if judge == candidate — fail loud.
            resolve_judge_model(spec, spec.provider)

    def _judge_model_id(self, spec: ModelSpec) -> str:
        """The configured judge model id for a candidate (``provider:model``)."""
        provider = spec.judge_provider or spec.provider
        return f"{provider}:{spec.judge_model}" if spec.judge_model else ""

    async def _run_judge(
        self, spec: ModelSpec, task: BenchmarkTask, answer: str
    ) -> Any:
        """Grade a judge-tier task's answer via the shared LLM-judge metric.

        Reuses :class:`himmy.services.evaluation.metrics.LLMJudgeMetric` through the
        :mod:`himmy.benchmark.judge` adapter (one judge implementation in the codebase).
        Returns a :class:`~himmy.benchmark.judge.JudgeVerdict` (``ungraded=True`` on a
        judge timeout / unparseable verdict).
        """
        rubric, threshold = build_judge_spec(task.judge)
        metric, judge_id = build_judge(
            spec, inference_factory=self._judge_inference_factory
        )
        return await judge_answer(
            metric,
            rubric=rubric,
            threshold=threshold,
            prompt=task.prompt,
            answer=answer,
            judge_model=judge_id,
        )

    async def _run_trial(self, spec: ModelSpec, task: BenchmarkTask) -> TrialResult:
        """Run one task once and capture its metrics (errors are recorded, not raised).

        A task with a ``team`` block runs as a multi-agent team (handoff/delegation or
        group chat); everything else runs as a single agent on the unchanged code path.
        Either way the trial records the same shape — including the ordered tool-call
        sequence (``tools_called``), which for a team includes the synthetic
        ``transfer_to_<peer>`` / ``ask_<worker>`` calls so a ``trajectory`` grader can
        assert on the routing.
        """
        started = self._clock()
        try:
            with self._fixtures(task) as config:
                if task.team:
                    outcome = await self._run_team(spec, task, config)
                else:
                    outcome = await self._run_single(spec, task, config)
            latency = self._clock() - started
            answer, called, turns, tokens, cost, run_error = outcome
            # A judge-tier task is graded by an LLM judge (reported, never gating). Its
            # deterministic `grade`/`trajectory` blocks are OPTIONAL, but when declared
            # they still gate `correct` *in addition to* the judge verdict — matching the
            # BenchmarkTask.judge docstring ("instead of OR IN ADDITION TO the
            # deterministic grade") and the TrialResult invariant. So always evaluate any
            # declared grade/trajectory block, judge-tier or not.
            judging = has_judge(task)
            answer_ok = grade(task.grade, answer) if task.grade else True
            # Trajectory grading is optional: when the task declares a `trajectory`
            # block, the ordered tool-call sequence must satisfy it too.
            trajectory_ok = (
                grade_trajectory(task.trajectory, called) if task.trajectory else None
            )
            # A judge-tier candidate with NO judge configured records the trial ungraded
            # (the documented ModelSpec degrade path) rather than crashing — so the
            # suite's deterministic tasks still run. A run error before judging also
            # leaves the trial ungraded (the judge never produced a verdict).
            judge = (
                await self._run_judge(spec, task, answer)
                if (judging and run_error is None and spec.judge_model)
                else None
            )
            # Deterministic correctness gates the (reported & gated) accuracy. A judge-tier
            # task's `correct` reflects the judge verdict AND any declared deterministic
            # gate (answer/trajectory): a wrong answer or a bad trajectory fails the trial
            # even if the judge passed. The judge only counts when it actually graded — an
            # ungraded judge leaves `correct=False` but is surfaced separately as ungraded.
            det_ok = answer_ok and trajectory_ok is not False
            if judging:
                correct = bool(judge and judge.passed) and det_ok
            else:
                correct = det_ok
            return TrialResult(
                task_id=task.id,
                answer=answer,
                tools_called=called,
                correct=correct,
                tool_ok=(set(task.expect_tools) <= set(called))
                if task.expect_tools
                else None,
                latency_s=latency,
                turns=turns,
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                cost=cost,
                error=run_error,
                answer_ok=answer_ok,
                trajectory_ok=trajectory_ok,
                judged=judging,
                judge_ok=(judge.passed if judge and not judge.ungraded else None),
                judge_score=(judge.score if judge and not judge.ungraded else None),
                judge_ungraded=bool(judge and judge.ungraded)
                or (judging and (run_error is not None or not spec.judge_model)),
                judge_model=(
                    judge.judge_model if judge else self._judge_model_id(spec)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a crash is a (recorded) trial failure
            judging = has_judge(task)
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
                answer_ok=False,
                trajectory_ok=False if task.trajectory else None,
                judged=judging,
                # A crash before/while judging leaves the trial ungraded (the judge never
                # produced a verdict), not a graded judge fail.
                judge_ungraded=judging,
            )

    async def _run_single(
        self, spec: ModelSpec, task: BenchmarkTask, config: Any
    ) -> _RunOutcome:
        """Run a single-agent task; return its normalized trial outcome."""
        from himmy.agents.base_agent.task import Task as AgentTask
        from himmy.agents.personas.persona import Persona
        from himmy.services.inference.models import LLMConfig

        runtime = self._build_runtime(spec, task, config)
        instructions = list(task.instructions)
        bundle = self._skill_bundle(task)
        if bundle is not None:
            from himmy.skills.resolve import format_examples

            instructions.extend(bundle.instruction_blocks)
            if bundle.examples:
                instructions.append(format_examples(bundle.examples))
        llm = LLMConfig(model_key="default", temperature=spec.temperature)
        loop = await runtime.run_agent_loop(
            Persona(name="bench", instructions=instructions),
            AgentTask(title=task.id, prompt=task.prompt),
            llm_config=llm,
            max_turns=spec.max_turns,
            route_tools=spec.tool_router,
        )
        final = loop.final
        called = [tc.tool_name for t in loop.turns for tc in t.tool_calls]
        return (
            final.output_text or "",
            called,
            loop.turn_count,
            (final.input_tokens, final.output_tokens),
            loop.total_cost,
            None if final.succeeded else (final.error or "run failed"),
        )

    async def _run_team(
        self, spec: ModelSpec, task: BenchmarkTask, config: Any
    ) -> _RunOutcome:
        """Run a multi-agent team task; return its normalized trial outcome.

        Constructs the orchestrator directly (not via ``AgentEvalHarness.evaluate_team``)
        because we need the per-turn ``RunResult`` list — the ordered tool sequence,
        token counts, and cost — that the eval harness collapses to a single answer
        string. ``MultiAgentOrchestrator`` covers handoff + delegation; group-chat tasks
        use ``GroupChatOrchestrator``. Both expose ``turns`` as ``(member, RunResult)``,
        so the synthetic ``transfer_to_*`` / ``ask_*`` calls flow straight into the
        trajectory.
        """
        from himmy.benchmark.team import build_team_plan
        from himmy.orchestrators import (
            GroupChatOrchestrator,
            MultiAgentOrchestrator,
            RoundRobinSelector,
        )

        plan = build_team_plan(task.team)
        registry, inference = self._team_registry(spec, task, config, plan.tool_names)
        runtime = self._runtime_over(inference, registry)

        # Both orchestrators expose the same triple we score from — a list of
        # ``(member, RunResult)`` turns, the final ``output_text``, and ``total_cost`` —
        # so we normalize to those here and grade the team exactly like a single agent.
        turns: list[tuple[str, Any]]
        if plan.mode == "group_chat":
            from himmy.orchestrators import LLMSelector

            selector = (
                LLMSelector(runtime) if plan.selector == "llm" else RoundRobinSelector()
            )
            chat = GroupChatOrchestrator(
                runtime,
                plan.team,
                registry,
                selector=selector,
                max_rounds=plan.max_rounds,
                default_temperature=spec.temperature or 0.0,
                # Honour the team's `entry` as the group-chat first speaker (documented in
                # team.py): the round-robin roster starts at speakers[0], so rotate the
                # roster so `entry` leads. Without this the first speaker was just the
                # members/speakers head — `entry` was validated then silently ignored.
                speakers=self._group_chat_roster(plan),
            )
            chat_result = await chat.run(task.prompt)
            turns = chat_result.turns
            output_text, total_cost = chat_result.output_text, chat_result.total_cost
            stop_reason = chat_result.stopped_reason
        else:
            orch = MultiAgentOrchestrator(
                runtime,
                plan.team,
                registry,
                max_turns=spec.max_turns,
                default_temperature=spec.temperature or 0.0,
            )
            team_result = await orch.run(task.prompt)
            turns = team_result.turns
            output_text, total_cost = team_result.output_text, team_result.total_cost
            stop_reason = team_result.stopped_reason

        called = [tc.tool_name for _, r in turns for tc in r.tool_calls]
        in_tokens = sum(r.input_tokens for _, r in turns)
        out_tokens = sum(r.output_tokens for _, r in turns)
        error = self._team_error(turns, stop_reason)
        return (
            output_text or "",
            called,
            len(turns),
            (in_tokens, out_tokens),
            total_cost,
            error,
        )

    #: Orchestrator ``stopped_reason`` values that mean the team finished cleanly (a real
    #: final answer was produced). Everything else (``no_progress``, ``max_turns``,
    #: ``max_rounds``, ``terminated``, ``error``) is a run failure for error-rate purposes.
    _CLEAN_TEAM_STOPS = frozenset({"final", "final_answer", "max_turns_synthesized"})

    def _team_error(self, turns: list[tuple[str, Any]], stop_reason: str) -> str | None:
        """Decide a team run's error like the single-agent path keys off the FINAL turn.

        The single-agent outcome is an error iff the *final* result didn't succeed. A team
        must mirror that — keying off ``any(succeeded)`` (the old rule) marked a run that
        failed on its last turn as a success just because an earlier turn passed,
        undercounting team failures in ``error_rate``. So we require the LAST turn to have
        succeeded AND the orchestrator to report a clean ``stopped_reason`` (a team that
        stalled on ``no_progress``/``max_turns``/``max_rounds`` produced no real answer).
        """
        if not turns:
            return "team run produced no turns"
        if not turns[-1][1].succeeded:
            return "team run failed (final turn errored)"
        if stop_reason not in self._CLEAN_TEAM_STOPS:
            return f"team run did not complete cleanly (stopped: {stop_reason})"
        return None

    def _group_chat_roster(self, plan: Any) -> list[str]:
        """Speaking roster for a group chat, rotated so ``plan.team.entry`` speaks first.

        ``plan.speakers`` (or, when unset, the member order) is the roster; the
        round-robin selector starts at ``roster[0]``. To honour the declared ``entry`` as
        the documented first speaker, rotate the roster so ``entry`` leads (preserving the
        cyclic order of the rest). When ``entry`` is not in the roster (a speakers list
        that excludes it), the roster is returned unchanged.
        """
        roster = (
            list(plan.speakers)
            if plan.speakers is not None
            else [m.name for m in plan.team.members]
        )
        entry = plan.team.entry
        if entry in roster:
            i = roster.index(entry)
            roster = roster[i:] + roster[:i]
        return roster

    def _team_registry(
        self, spec: ModelSpec, task: BenchmarkTask, config: Any, tool_names: list[str]
    ) -> tuple[Any, Any]:
        """Build a fresh registry with the team's pack tools + the trial's inference.

        The team's members reference plain tool names; those tools come from the task's
        ``packs`` (and any distractor ``extra_packs``). The orchestrator then registers
        its synthetic ``transfer_to_*`` / ``ask_*`` / ``final_answer`` tools on the SAME
        registry at construction time.
        """
        from himmy.services.tools.registry import ToolRegistry
        from himmy.toolkit import register_packs

        if self._inference_factory is not None:
            inference = self._inference_factory(spec)
        else:
            from himmy.cli.provider import build_inference_for

            inference = build_inference_for(spec.provider, spec.model)
        registry = ToolRegistry()
        packs = list(dict.fromkeys([*task.packs, *spec.extra_packs]))
        if packs:
            register_packs(registry, packs, config)
        return registry, inference

    def _runtime_over(self, inference: Any, registry: Any) -> Any:
        """Build a runtime sharing a pre-populated tool registry (team tools live on it)."""
        from himmy import build_runtime

        runtime, _inf, _tools = build_runtime(
            inference=inference, tool_registry=registry
        )
        return runtime

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

    def _skill_bundle(self, task: BenchmarkTask) -> Any | None:
        """Resolve a task's skills (if any) into their applied bundle, else ``None``."""
        if not task.skills:
            return None
        from himmy.skills import build_skill_registry, resolve_skills

        return resolve_skills(task.skills, build_skill_registry())

    def _build_runtime(self, spec: ModelSpec, task: BenchmarkTask, config: Any) -> Any:
        """Build a fresh runtime with the task's packs + skills' packs + distractors."""
        from himmy import build_runtime
        from himmy.services.tools.registry import ToolRegistry
        from himmy.toolkit import register_packs

        if self._inference_factory is not None:
            inference = self._inference_factory(spec)
        else:
            from himmy.cli.provider import build_inference_for

            inference = build_inference_for(spec.provider, spec.model)
        bundle = self._skill_bundle(task)
        skill_packs = list(bundle.tool_packs) if bundle is not None else []
        packs = list(dict.fromkeys([*task.packs, *skill_packs, *spec.extra_packs]))
        overrides: dict[str, Any] = {"inference": inference}
        if packs:
            registry = ToolRegistry()
            register_packs(registry, packs, config)
            overrides["tool_registry"] = registry
        runtime, _inf, _tools = build_runtime(**overrides)
        return runtime


__all__ = ["BenchmarkRunner", "ProgressFn"]
