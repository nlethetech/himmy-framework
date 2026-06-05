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


def test_default_suite_loads() -> None:
    suite = default_suite()
    assert suite.name == "core"
    assert len(suite.tasks) >= 8
    # Every task has a grader and the file/sql tasks carry fixtures.
    assert all(t.grade for t in suite.tasks)
    sql = next(t for t in suite.tasks if t.id == "sql_count")
    assert sql.sqlite and "animals" in sql.sqlite[0]
