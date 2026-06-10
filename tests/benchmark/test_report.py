"""Tests for benchmark report rendering (Markdown + JSON)."""

from __future__ import annotations

from himmy.benchmark import default_suite, render_markdown, to_json
from himmy.benchmark.models import ModelScorecard, ModelSpec, TaskScore, TrialResult


def _card(name: str, accuracy_correct: int) -> ModelScorecard:
    trials = [
        TrialResult(
            "a", "x", ["calculator"], i < accuracy_correct, True, 1.0 + i, 1, 1, 1, 0.0
        )
        for i in range(4)
    ]
    return ModelScorecard(
        spec=ModelSpec("ollama", name),
        task_scores=[TaskScore("a", "arithmetic", trials)],
    )


def test_render_markdown_has_table_and_categories() -> None:
    cards = [_card("m-good", 4), _card("m-bad", 1)]
    md = render_markdown(cards, suite_name="core")
    assert "# Benchmark: core" in md
    assert "Accuracy (95% CI)" in md
    assert "m-good" in md and "m-bad" in md
    assert "arithmetic" in md  # category breakdown
    # Best model is sorted first in the table body.
    assert md.index("m-good") < md.index("m-bad")


def test_to_json_structure() -> None:
    data = to_json([_card("m", 3)], suite_name="core")
    assert data["suite"] == "core"
    assert data["trials"] == 4
    model = data["models"][0]
    assert model["name"] == "ollama:m"
    assert 0.0 <= model["accuracy"] <= 1.0
    assert len(model["accuracy_ci"]) == 2
    assert model["tasks"][0]["id"] == "a"
    assert "pass_ci" in model["tasks"][0]


def _multi_cat_card(name: str, *, big_correct: int) -> ModelScorecard:
    """A card with one >=5-task category ('big') and one small category ('small')."""
    scores: list[TaskScore] = []
    for i in range(6):  # 6 tasks → 'big' gets a percentage
        trials = [
            TrialResult(f"big{i}", "x", [], i < big_correct, None, 1.0, 1, 1, 1, 0.0)
        ]
        scores.append(TaskScore(f"big{i}", "big", trials))
    # 'small' has 2 tasks → per-task counts, not a percentage.
    scores.append(
        TaskScore(
            "sql_count",
            "small",
            [
                TrialResult("sql_count", "x", [], True, None, 1, 1, 1, 1, 0.0)
                for _ in range(3)
            ],
        )
    )
    scores.append(
        TaskScore(
            "sql_aggregate",
            "small",
            [
                TrialResult("sql_aggregate", "x", [], False, None, 1, 1, 1, 1, 0.0)
                for _ in range(3)
            ],
        )
    )
    return ModelScorecard(spec=ModelSpec("ollama", name), task_scores=scores)


def test_small_category_shows_per_task_counts_not_percentage() -> None:
    md = render_markdown([_multi_cat_card("m", big_correct=6)], suite_name="core")
    # Small category: per-task pass counts, not a rate.
    assert "sql_count 3/3" in md
    assert "sql_aggregate 0/3" in md
    # The honesty note is present.
    assert "per-task pass counts" in md


def test_large_category_keeps_percentage() -> None:
    md = render_markdown([_multi_cat_card("m", big_correct=3)], suite_name="core")
    # 'big' has 6 tasks, 3 correct → 50%.
    assert "50%" in md


def test_json_carries_category_counts() -> None:
    data = to_json([_multi_cat_card("m", big_correct=6)], suite_name="core")
    model = data["models"][0]
    assert model["category_counts"]["big"] == 6
    assert model["category_counts"]["small"] == 2
    assert data["min_category_tasks"] == 5


def test_pairwise_section_rendered_for_two_models() -> None:
    a = _card("winner", 4)
    b = _card("loser", 0)
    md = render_markdown([a, b], suite_name="core")
    assert "Pairwise comparison (McNemar" in md
    assert "ollama:winner vs ollama:loser" in md
    # winner beats loser on all 4 paired trials → b=4,c=0,p≈0.125 (not significant).
    # Assert the EXACT not-significant rendering, and that no "leads (significant" string
    # leaked — a disjunction over "leads" would pass even if the verdict regressed to a
    # spurious "significant" (the bug finding 20 flags).
    assert "not significant at 0.05" in md
    assert "leads (significant" not in md


def test_pairwise_significant_verdict() -> None:
    # 8 trials, winner sweeps → p ≈ 0.0078 < 0.05.
    def _card8(name: str, correct: int) -> ModelScorecard:
        trials = [
            TrialResult("a", "x", [], i < correct, None, 1.0, 1, 1, 1, 0.0)
            for i in range(8)
        ]
        return ModelScorecard(
            spec=ModelSpec("ollama", name),
            task_scores=[TaskScore("a", "arithmetic", trials)],
        )

    md = render_markdown([_card8("winner", 8), _card8("loser", 0)], suite_name="core")
    assert "**ollama:winner** leads (significant" in md


def test_pairwise_json_present_for_two_models() -> None:
    data = to_json([_card("winner", 4), _card("loser", 0)], suite_name="core")
    assert "pairwise" in data
    pair = data["pairwise"][0]
    assert pair["a"] == "ollama:winner" and pair["b"] == "ollama:loser"
    assert pair["b_wins"] == 4 and pair["c_wins"] == 0
    assert "discordant_task_ids" in pair


def test_pairwise_tie_verdict_rendered() -> None:
    # b == c == 0: the two models pass and fail the exact same trials → a tie verdict,
    # which has its own rendering branch (report.py) never exercised before.
    def _card_pattern(name: str, pattern: list[bool]) -> ModelScorecard:
        trials = [
            TrialResult("a", "x", [], ok, None, 1.0, 1, 1, 1, 0.0) for ok in pattern
        ]
        return ModelScorecard(
            spec=ModelSpec("ollama", name),
            task_scores=[TaskScore("a", "arithmetic", trials)],
        )

    # Identical pass/fail grids → no discordant pairs → b = c = 0 → tie.
    pat = [True, False, True, False]
    md = render_markdown(
        [_card_pattern("m1", pat), _card_pattern("m2", pat)], suite_name="core"
    )
    assert "tie (b = c = 0)" in md
    assert "leads (significant" not in md


def test_pairwise_holm_correction_curbs_family_false_positives() -> None:
    # Three models, one clearly best. With many pairs the per-comparison alpha would
    # over-claim significance; Holm correction tightens it. Verify the corrected family
    # alpha is named and the JSON exposes the comparison count + correction method.
    def _card8(name: str, correct: int) -> ModelScorecard:
        trials = [
            TrialResult("a", "x", [], i < correct, None, 1.0, 1, 1, 1, 0.0)
            for i in range(8)
        ]
        return ModelScorecard(
            spec=ModelSpec("ollama", name),
            task_scores=[TaskScore("a", "arithmetic", trials)],
        )

    cards = [_card8("a", 8), _card8("b", 4), _card8("c", 0)]
    md = render_markdown(cards, suite_name="core")
    assert "Holm-corrected across 3" in md
    data = to_json(cards, suite_name="core")
    assert data["pairwise_correction"] == "holm"
    assert data["pairwise_n_comparisons"] == 3
    assert data["pairwise_alpha"] == 0.05


def test_holm_step_down_unit() -> None:
    from himmy.benchmark.report import _holm_significant

    # p = [0.01, 0.04, 0.2] at alpha 0.05, m=3: thresholds 0.05/3, 0.05/2, 0.05/1.
    # 0.01 < 0.0167 ✓; 0.04 < 0.025? no → step-down stops, the rest are non-significant.
    assert _holm_significant([0.01, 0.04, 0.2], 0.05) == [True, False, False]
    # Order-independent: same set permuted yields flags in input order.
    assert _holm_significant([0.2, 0.01, 0.04], 0.05) == [False, True, False]
    # Single comparison: Holm reduces to the raw per-pair test.
    assert _holm_significant([0.04], 0.05) == [True]
    assert _holm_significant([], 0.05) == []


def test_pairwise_absent_for_single_model() -> None:
    assert "pairwise" not in to_json([_card("m", 4)], suite_name="core")
    assert "Pairwise" not in render_markdown([_card("m", 4)], suite_name="core")


def test_default_suite_loads() -> None:
    suite = default_suite()
    assert suite.name == "core"
    assert len(suite.tasks) >= 8
    # Every task has a grader and the file/sql tasks carry fixtures.
    assert all(t.grade for t in suite.tasks)
    sql = next(t for t in suite.tasks if t.id == "sql_count")
    assert sql.sqlite and "animals" in sql.sqlite[0]
