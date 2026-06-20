"""Tests for the BFCL-style per-model tool-use leaderboard render (FEATURE #4).

``render_leaderboard`` ranks models by accuracy with the headline tool-use axes as
columns (Accuracy + Wilson CI, Tool-call, Arg-acc, Multi-turn, Abstain), a per-category
breakdown, and the Holm-corrected pairwise McNemar comparison. The arg-/multi-turn axes
also surface in ``render_markdown`` and ``to_json``, and persist to history.
"""

from __future__ import annotations

from himmy.benchmark import render_leaderboard, render_markdown, to_json
from himmy.benchmark.models import (
    MULTI_TURN_CATEGORY,
    ModelScorecard,
    ModelSpec,
    TaskScore,
    TrialResult,
)


def _trial(task_id: str, correct: bool, *, tool_ok: bool | None = None) -> TrialResult:
    return TrialResult(
        task_id=task_id,
        answer="x",
        tools_called=["t"] if tool_ok else [],
        correct=correct,
        tool_ok=tool_ok,
        latency_s=1.0,
        turns=1,
        input_tokens=1,
        output_tokens=1,
        cost=0.0,
        trajectory_ok=correct if tool_ok is not None else None,
    )


def _card(
    name: str,
    *,
    arg_correct: tuple[bool, bool],
    mt_correct: tuple[bool, bool],
    irrel_abstain: bool,
) -> ModelScorecard:
    """A scorecard with one arg-graded task, one multi-turn task, one irrelevance task."""
    arg_score = TaskScore(
        "arg_task",
        "nested_args",
        [_trial("arg_task", c, tool_ok=c) for c in arg_correct],
        uses_arg_grader=True,
    )
    mt_score = TaskScore(
        "mt_task",
        MULTI_TURN_CATEGORY,
        [_trial("mt_task", c, tool_ok=c) for c in mt_correct],
        uses_arg_grader=False,  # graded by a name-only tool_sequence here, not arg-level
    )
    irrel_trial = TrialResult(
        "irr",
        "4",
        [] if irrel_abstain else ["calculator"],
        irrel_abstain,
        None,
        1.0,
        1,
        1,
        1,
        0.0,
        trajectory_ok=irrel_abstain,
    )
    irrel_score = TaskScore("irr", "irrelevance", [irrel_trial])
    return ModelScorecard(
        spec=ModelSpec("ollama", name, label=name),
        task_scores=[arg_score, mt_score, irrel_score],
    )


def test_leaderboard_ranks_by_accuracy_and_shows_axes() -> None:
    strong = _card("strong", arg_correct=(True, True), mt_correct=(True, True), irrel_abstain=True)
    weak = _card("weak", arg_correct=(False, False), mt_correct=(True, False), irrel_abstain=False)
    md = render_leaderboard([strong, weak], suite_name="bfcl")
    assert "Tool-use leaderboard: bfcl" in md
    # Headline axis columns are present.
    for col in ("Arg-acc", "Multi-turn", "Abstain"):
        assert col in md
    # The stronger model ranks first (rank 1 row precedes rank 2 row).
    assert md.index("| 1 | strong") < md.index("| 2 | weak")


def test_leaderboard_axis_percentages() -> None:
    card = _card("m", arg_correct=(True, False), mt_correct=(True, True), irrel_abstain=True)
    # arg: 1/2 = 50%, multi-turn: 2/2 = 100%, abstain: 1/1 = 100%.
    assert card.arg_accuracy == 0.5
    assert card.multi_turn_accuracy == 1.0
    assert card.irrelevance_accuracy == 1.0
    md = render_leaderboard([card])
    assert "50%" in md
    assert "100%" in md


def test_leaderboard_includes_category_and_pairwise_sections() -> None:
    a = _card("a", arg_correct=(True, True), mt_correct=(True, True), irrel_abstain=True)
    b = _card("b", arg_correct=(False, False), mt_correct=(False, False), irrel_abstain=False)
    md = render_leaderboard([a, b], suite_name="bfcl")
    assert "## By category" in md
    assert "## Pairwise comparison (McNemar, exact)" in md


def test_leaderboard_empty_is_handled() -> None:
    assert render_leaderboard([]) == "(no results)"


def test_render_markdown_surfaces_arg_and_multi_turn_columns() -> None:
    card = _card("m", arg_correct=(True, True), mt_correct=(True, False), irrel_abstain=True)
    md = render_markdown([card], suite_name="bfcl")
    assert "Arg-acc" in md
    assert "Multi-turn" in md


def test_render_markdown_omits_axes_when_suite_has_none() -> None:
    card = ModelScorecard(
        spec=ModelSpec("ollama", "m"),
        task_scores=[
            TaskScore("t", "arithmetic", [_trial("t", True, tool_ok=True)])
        ],
    )
    md = render_markdown([card], suite_name="core")
    assert "Arg-acc" not in md
    assert "Multi-turn" not in md


def test_to_json_carries_arg_and_multi_turn_axes() -> None:
    card = _card("m", arg_correct=(True, False), mt_correct=(True, True), irrel_abstain=True)
    data = to_json([card], suite_name="bfcl")
    model = data["models"][0]
    assert model["arg_accuracy"] == 0.5
    assert model["arg_total_trials"] == 2
    assert model["multi_turn_accuracy"] == 1.0
    assert model["multi_turn_total_trials"] == 2


def test_history_persists_arg_and_multi_turn(tmp_path) -> None:
    from himmy.benchmark.history import append_run, load_history

    card = _card("m", arg_correct=(True, False), mt_correct=(True, True), irrel_abstain=True)
    path = tmp_path / "history.jsonl"
    append_run([card], suite_name="bfcl", path=path, enabled=True, sha="abc")
    records = load_history(path)
    assert len(records) == 1
    metrics = records[0]["metrics"]
    assert metrics["arg_accuracy"] == 0.5
    assert metrics["multi_turn_accuracy"] == 1.0
