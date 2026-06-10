"""Tests for the exact McNemar paired comparison (stats + scorecard pairing)."""

from __future__ import annotations

import pytest

from himmy.benchmark.models import (
    ModelScorecard,
    ModelSpec,
    TaskScore,
    TrialResult,
    compare_scorecards,
)
from himmy.benchmark.stats import mcnemar_exact, mcnemar_from_outcomes


def test_mcnemar_no_discordant_pairs_is_one() -> None:
    # b + c == 0: no signal either way → p == 1.0.
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_symmetric_is_one() -> None:
    # b == c: perfectly balanced disagreement → no evidence of a difference.
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(1, 1) == 1.0


def test_mcnemar_hand_computed_values() -> None:
    # b=0, c=1: n=1, k=0, P(X<=0)=0.5 → two-sided = min(1, 2*0.5) = 1.0.
    assert mcnemar_exact(0, 1) == pytest.approx(1.0)
    # b=0, c=2: n=2, k=0, P(X<=0)=1/4 → 2*0.25 = 0.5.
    assert mcnemar_exact(0, 2) == pytest.approx(0.5)
    # b=0, c=5: n=5, k=0, P(X<=0)=1/32 → 2/32 = 0.0625.
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    # b=1, c=9: n=10, k=1, P(X<=1) = (C(10,0)+C(10,1))/1024 = 11/1024 → 22/1024.
    assert mcnemar_exact(1, 9) == pytest.approx(22.0 / 1024.0)


def test_mcnemar_asymmetric_significant_case() -> None:
    # b=0, c=8: n=8, k=0, P(X<=0)=1/256 → 2/256 ≈ 0.0078 < 0.05 (significant).
    p = mcnemar_exact(0, 8)
    assert p < 0.05
    assert p == pytest.approx(2.0 / 256.0)


def test_mcnemar_negative_counts_raise() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_from_outcomes_pairs_and_reports_leader() -> None:
    # A wins t1 (A pass, B fail), B wins t2; t3 concordant (both pass).
    a = {("t1", 0): True, ("t2", 0): False, ("t3", 0): True}
    b = {("t1", 0): False, ("t2", 0): True, ("t3", 0): True}
    res = mcnemar_from_outcomes("A", a, "B", b)
    assert res.n_pairs == 3
    assert res.b == 1 and res.c == 1  # one discordant each way
    assert res.leader is None  # tie
    assert res.p_value == 1.0
    assert sorted(res.discordant_task_ids) == ["t1", "t2"]


def test_from_outcomes_leader_is_winning_model() -> None:
    a = {("t1", 0): True, ("t2", 0): True, ("t3", 0): True}
    b = {("t1", 0): False, ("t2", 0): False, ("t3", 0): False}
    res = mcnemar_from_outcomes("A", a, "B", b)
    assert res.b == 3 and res.c == 0
    assert res.leader == "A"


def test_from_outcomes_only_pairs_intersection() -> None:
    # B did not run t3; only t1/t2 are shared and paired.
    a = {("t1", 0): True, ("t2", 0): False, ("t3", 0): True}
    b = {("t1", 0): False, ("t2", 0): False}
    res = mcnemar_from_outcomes("A", a, "B", b)
    assert res.n_pairs == 2  # t3 ignored (no pair)
    assert res.b == 1 and res.c == 0
    assert res.discordant_task_ids == ["t1"]


def test_from_outcomes_dedups_discordant_task_ids() -> None:
    # Same task discordant on two trials → listed once.
    a = {("t1", 0): True, ("t1", 1): True}
    b = {("t1", 0): False, ("t1", 1): False}
    res = mcnemar_from_outcomes("A", a, "B", b)
    assert res.b == 2
    assert res.discordant_task_ids == ["t1"]


def _card(name: str, results: list[bool]) -> ModelScorecard:
    trials = [TrialResult("t1", "x", [], ok, None, 1.0, 1, 1, 1, 0.0) for ok in results]
    return ModelScorecard(
        spec=ModelSpec("ollama", name),
        task_scores=[TaskScore("t1", "arithmetic", trials)],
    )


def test_compare_scorecards_pairs_by_task_and_trial() -> None:
    a = _card("good", [True, True, True])
    b = _card("bad", [False, False, False])
    res = compare_scorecards(a, b)
    assert res.n_pairs == 3
    assert res.b == 3 and res.c == 0
    assert res.leader == "ollama:good"
    assert res.p_value == pytest.approx(2.0 / 8.0)  # b=3,c=0 → n=3,k=0 → 2/8
