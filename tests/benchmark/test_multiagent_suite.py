"""Offline integration tests for the multi-agent benchmark suite.

The default offline ``StubClientManager`` calls *every* bound tool, so it cannot model
*correct* routing (it would transfer to both specialists). These tests inject a scripted
inference stub — the same pattern as ``tests/benchmark/test_runner.py`` — that drives a
team turn-by-turn, choosing the right collaboration tool from what is bound on that turn.
We then assert the suite's graders return the right verdict BOTH ways: a competent
trajectory passes, a wrong-handoff trajectory fails.
"""

from __future__ import annotations

from typing import Any

from himmy.benchmark import BenchmarkRunner, ModelSpec, multiagent_suite
from himmy.benchmark.models import BenchmarkSuite, BenchmarkTask
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async


def _task(suite: BenchmarkSuite, task_id: str) -> BenchmarkTask:
    return next(t for t in suite.tasks if t.id == task_id)


def _one_task_suite(task: BenchmarkTask) -> BenchmarkSuite:
    return BenchmarkSuite(name="multiagent", tasks=[task])


class _ScriptedTeam:
    """A scripted "competent model" for a team run.

    Each turn it inspects ``request.bound_tools`` and calls exactly ONE tool — chosen by
    matching the bound names against a priority plan — executing it via the request's
    ``tool_executor`` so the synthetic handoff/delegate tools and ``final_answer`` flow
    through the real pipeline. ``answer`` is what it passes to ``final_answer``.

    ``plan`` maps a bound-tool name to the action: encountering it (highest priority
    first) drives the turn. A turn with no planned tool (or only ``final_answer``) calls
    ``final_answer`` with ``answer``.
    """

    provider_name = "stub"

    def __init__(self, priority: list[str], answer: str) -> None:
        self._priority = priority
        self._answer = answer

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:
        bound = {t.name: t for t in (request.bound_tools or [])}
        chosen = next((name for name in self._priority if name in bound), None)
        if chosen is None and "final_answer" in bound:
            chosen, args = "final_answer", {"answer": self._answer}
        elif chosen is not None:
            args = (
                {"answer": self._answer}
                if chosen == "final_answer"
                else _args_for(chosen)
            )
        else:
            # No tools bound at all (e.g. a synthesis turn) — answer in plain text.
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text=self._answer,
                input_tokens=1,
                output_tokens=1,
            )
        ret = await request.tool_executor(chosen, args)
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=self._answer,
            tool_calls=[
                ToolCallRecord(
                    tool_call_id=ret.tool_call_id, tool_name=chosen, args=args
                )
            ],
            tool_returns=[ret],
            input_tokens=1,
            output_tokens=1,
        )


def _args_for(tool: str) -> dict[str, Any]:
    """Minimal valid args for the synthetic / pack tools the scripted stub may call."""
    if tool.startswith("transfer_to_"):
        return {"reason": "routing"}
    if tool.startswith("ask_"):
        return {"task": "find the answer"}
    if tool == "read_file":
        return {"path": "codeword.txt"}
    if tool == "calculator":
        return {"expression": "6*7"}
    return {}


def _run(task: BenchmarkTask, manager: Any, trials: int = 1) -> Any:
    runner = BenchmarkRunner(
        trials=trials,
        inference_factory=lambda spec: InferenceService(manager),
        clock=lambda: 0.0,
    )
    cards = run_async(runner.run(_one_task_suite(task), [ModelSpec("stub", "m")]))
    return cards[0].task_scores[0].trials[0]


# --------------------------------------------------------------------------- handoff


def test_handoff_routing_correct_trajectory_passes() -> None:
    """Router transfers to math (not weather); math answers 42 — both graders pass."""
    task = _task(multiagent_suite(), "handoff_routing")
    # Router turn: transfer_to_math (preferred over weather). Math turn: calculator then
    # final_answer with the computed value.
    manager = _ScriptedTeam(
        priority=["transfer_to_math", "calculator", "final_answer"],
        answer="The answer is 42.",
    )
    trial = _run(task, manager)
    assert "transfer_to_math" in trial.tools_called
    assert "transfer_to_weather" not in trial.tools_called
    assert trial.answer_ok is True
    assert trial.trajectory_ok is True
    assert trial.correct is True


def test_handoff_routing_wrong_peer_fails_trajectory() -> None:
    """Transferring to the WRONG specialist fails the trajectory even with a right answer."""
    task = _task(multiagent_suite(), "handoff_routing")
    # The router goes to weather (wrong). Weather has no calculator, so the scripted
    # answer still says 42 — proving it is the TRAJECTORY, not the answer, that fails.
    manager = _ScriptedTeam(
        priority=["transfer_to_weather", "calculator", "final_answer"],
        answer="The answer is 42.",
    )
    trial = _run(task, manager)
    assert "transfer_to_weather" in trial.tools_called
    assert trial.answer_ok is True  # the answer text is right
    assert trial.trajectory_ok is False  # but it routed to the wrong peer
    assert trial.correct is False


# ------------------------------------------------------------------------ delegation


def test_delegation_synthesis_passes() -> None:
    """Manager delegates to the researcher and the worker's code word reaches the answer."""
    task = _task(multiagent_suite(), "delegation_synthesis")
    # Manager turn: ask_researcher (the worker reads codeword.txt and finds FALCON);
    # the manager then reports it. The worker's own turns read_file + final_answer.
    manager = _ScriptedTeam(
        priority=["ask_researcher", "read_file", "final_answer"],
        answer="The project code word is FALCON.",
    )
    trial = _run(task, manager)
    assert "ask_researcher" in trial.tools_called
    assert trial.answer_ok is True
    assert trial.trajectory_ok is True
    assert trial.correct is True


def test_delegation_without_asking_fails_trajectory() -> None:
    """A manager that answers WITHOUT delegating fails the ask_<worker> trajectory."""
    task = _task(multiagent_suite(), "delegation_synthesis")
    # Manager never calls ask_researcher (priority skips it) — straight to final_answer.
    manager = _ScriptedTeam(
        priority=["final_answer"],
        answer="The project code word is FALCON.",
    )
    trial = _run(task, manager)
    assert "ask_researcher" not in trial.tools_called
    assert trial.answer_ok is True
    assert trial.trajectory_ok is False
    assert trial.correct is False


# ------------------------------------------------------------------- no-handoff control


def test_no_handoff_control_passes_when_answered_directly() -> None:
    """The generalist answers Paris itself without transferring to a specialist."""
    task = _task(multiagent_suite(), "no_handoff_control")
    manager = _ScriptedTeam(priority=["final_answer"], answer="Paris")
    trial = _run(task, manager)
    assert "transfer_to_math" not in trial.tools_called
    assert "transfer_to_weather" not in trial.tools_called
    assert trial.answer_ok is True
    assert trial.trajectory_ok is True
    assert trial.correct is True


def test_no_handoff_control_fails_when_it_over_routes() -> None:
    """Transferring when none is warranted fails the tool_not_called control."""
    task = _task(multiagent_suite(), "no_handoff_control")
    manager = _ScriptedTeam(
        priority=["transfer_to_math", "final_answer"], answer="Paris"
    )
    trial = _run(task, manager)
    assert "transfer_to_math" in trial.tools_called
    assert trial.answer_ok is True
    assert trial.trajectory_ok is False
    assert trial.correct is False


# ------------------------------------------------------------------- group chat


def test_group_chat_selection_runs_and_grades_answer() -> None:
    """A round-robin group chat runs to a final answer carrying the agreed slogan."""
    task = _task(multiagent_suite(), "group_chat_selection")
    # Whoever speaks calls final_answer with the agreed slogan; round-robin starts with
    # the analyst (the entry / first candidate), so selection is observable.
    manager = _ScriptedTeam(priority=["final_answer"], answer="The slogan is Brew.")
    trial = _run(task, manager)
    assert trial.answer_ok is True  # the group chat synthesized the agreed slogan
    assert trial.trajectory_ok is None  # this task declares no trajectory expectation
    assert trial.correct is True
    assert trial.error is None


# ------------------------------------------------------------ team error attribution


class _FailLastTurn:
    """Succeeds on its first turn (transfer), then FAILS every later turn.

    Models a team that routes correctly but then stalls — an earlier turn succeeded but
    the run never produces a real final answer. The runner must key its error on the
    FINAL turn / stop reason, not on "any turn succeeded".
    """

    provider_name = "stub"

    def __init__(self) -> None:
        self._calls = 0

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:
        self._calls += 1
        bound = {t.name: t for t in (request.bound_tools or [])}
        if self._calls == 1 and "transfer_to_math" in bound:
            ret = await request.tool_executor("transfer_to_math", {"reason": "math"})
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="routing",
                tool_calls=[
                    ToolCallRecord(
                        tool_call_id=ret.tool_call_id,
                        tool_name="transfer_to_math",
                        args={"reason": "math"},
                    )
                ],
                tool_returns=[ret],
                input_tokens=1,
                output_tokens=1,
            )
        # Every subsequent turn fails: no tool calls, no final answer → the orchestrator
        # makes no progress and the LAST turn does not succeed.
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            output_text="",
            input_tokens=1,
            output_tokens=1,
        )


def test_team_error_keys_off_final_turn_not_any_success() -> None:
    # A team run that succeeds on turn 1 (transfer) but then stalls without a real final
    # answer must be recorded as a RUN ERROR — the old `any(succeeded)` rule marked it as
    # a non-error because an earlier turn passed, undercounting team failures. Finding 9.
    task = _task(multiagent_suite(), "handoff_routing")
    trial = _run(task, _FailLastTurn())
    assert trial.error is not None  # final turn failed / no clean stop → run failure


# ------------------------------------------------------------ group-chat entry speaker


def test_group_chat_entry_speaks_first_even_when_not_first_member() -> None:
    # team.py documents `entry` as the group-chat first speaker. Build a chat whose entry
    # is the SECOND member and confirm the runner rotates the roster so entry leads first
    # — the selector previously ignored `entry` and started at members[0]. Finding 11.
    from himmy.benchmark.team import build_team_plan

    plan = build_team_plan(
        {
            "mode": "group_chat",
            "selector": "round_robin",
            "max_rounds": 2,
            "entry": "editor",  # second member — must still speak first
            "members": [
                {"name": "analyst", "persona": {"instructions": ["propose"]}},
                {"name": "editor", "persona": {"instructions": ["finalize"]}},
            ],
        }
    )
    # The runner rotates the roster so the declared entry leads the round-robin order.
    runner = BenchmarkRunner(trials=1, clock=lambda: 0.0)
    assert runner._group_chat_roster(plan) == ["editor", "analyst"]

    # Unset entry defaults to the first member — roster unchanged.
    plan_default = build_team_plan(
        {
            "mode": "group_chat",
            "members": [
                {"name": "analyst", "persona": {}},
                {"name": "editor", "persona": {}},
            ],
        }
    )
    assert runner._group_chat_roster(plan_default) == ["analyst", "editor"]


# ------------------------------------------------------------------- whole suite smoke


def test_full_suite_runs_against_default_stub_without_crashing() -> None:
    """Every team task runs end-to-end on the zero-config stub (no errors, tools recorded).

    The default stub calls every bound tool, so routing graders won't all pass — but the
    suite must RUN (orchestrators wired, synthetic tools registered, outcomes recorded)
    with no crash, and the synthetic collaboration tools must appear in the trajectory.
    """
    runner = BenchmarkRunner(trials=1, clock=lambda: 0.0)
    cards = run_async(runner.run(multiagent_suite(), [ModelSpec("stub", "m")]))
    card = cards[0]
    assert card.total_trials == len(multiagent_suite().tasks)
    assert card.error_rate == 0.0
    by_id = {s.task_id: s.trials[0] for s in card.task_scores}
    assert "transfer_to_math" in by_id["handoff_routing"].tools_called
    assert "ask_researcher" in by_id["delegation_synthesis"].tools_called
