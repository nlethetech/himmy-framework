"""Render benchmark scorecards as a comparative Markdown table and as JSON.

Markdown is the human scorecard (models side by side on accuracy + confidence interval,
tool-call accuracy, latency percentiles, cost, error rate, and a per-category
breakdown). JSON is the machine record — persist it to track regressions over time and
to diff configs (e.g. router on vs off).
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from himmy.benchmark.models import ModelScorecard

#: Below this many tasks, a category percentage is too noisy to report honestly — we
#: show the per-task pass counts instead. Surfaces anywhere a category rate is rendered.
MIN_CATEGORY_TASKS = 5
#: Significance level for the pairwise McNemar verdict line.
PAIRWISE_ALPHA = 0.05


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _category_breakdown(card: ModelScorecard) -> dict[str, str]:
    """Render each category honestly: a percentage only when it has >= 5 tasks.

    Small-n categories show per-task pass counts (e.g. ``sql_count 3/3,
    sql_aggregate 0/3``) rather than a misleading single percentage off a handful of
    tasks. Keyed by category, in the card's task order.
    """
    counts = card.category_counts()
    accuracy = card.by_category()
    # Deterministic tier only: judge-tier tasks are reported in their own section, never
    # folded into the (gated) deterministic per-category breakdown.
    by_cat: dict[str, list[str]] = {}
    for score in card.deterministic_scores:
        by_cat.setdefault(score.category, []).append(
            f"{score.task_id} {score.successes}/{score.n}"
        )
    out: dict[str, str] = {}
    for cat in by_cat:
        if counts.get(cat, 0) >= MIN_CATEGORY_TASKS:
            out[cat] = _pct(accuracy.get(cat))
        else:
            out[cat] = ", ".join(by_cat[cat])
    return out


def render_markdown(cards: Sequence[ModelScorecard], *, suite_name: str = "") -> str:
    """A comparative Markdown scorecard across models.

    When any model has a trajectory failure (a trial that produced a result but took a
    bad tool *path*), a ``Traj✗`` column is added so a process failure reads distinctly
    from a wrong answer or a crash, and a per-task footnote lists the offending tasks.
    """
    if not cards:
        return "(no results)"
    trials = cards[0].task_scores[0].n if cards[0].task_scores else 0
    n_tasks = len(cards[0].task_scores)
    show_traj = any(card.trajectory_failures for card in cards)
    header = (
        "| Model | Accuracy (95% CI) | Tool-call | p50 | p95 | Cost/trial | Errors |"
    )
    rule = "|---|---|---|---|---|---|---|"
    if show_traj:
        header += " Traj✗ |"
        rule += "---|"
    lines = [
        f"# Benchmark{f': {suite_name}' if suite_name else ''} "
        f"({n_tasks} tasks × {trials} trials)",
        "",
        header,
        rule,
    ]
    for card in sorted(cards, key=lambda c: c.accuracy, reverse=True):
        lo, hi = card.accuracy_ci
        row = (
            f"| {card.spec.name} "
            f"| {_pct(card.accuracy)} ({_pct(lo)}–{_pct(hi)}) "
            f"| {_pct(card.tool_call_accuracy)} "
            f"| {card.p50_latency:.1f}s "
            f"| {card.p95_latency:.1f}s "
            f"| ${card.mean_cost:.4f} "
            f"| {_pct(card.error_rate)} |"
        )
        if show_traj:
            row += f" {card.trajectory_failures} |"
        lines.append(row)
    if show_traj:
        lines += _trajectory_section(cards)

    categories = sorted({c for card in cards for c in card.by_category()})
    if categories:
        lines += [
            "",
            "## By category",
            "",
            f"_Categories with < {MIN_CATEGORY_TASKS} tasks show per-task pass "
            "counts (a percentage off a handful of tasks is noise), not a rate._",
            "",
        ]
        lines.append("| Model | " + " | ".join(categories) + " |")
        lines.append("|---|" + "|".join(["---"] * len(categories)) + "|")
        for card in cards:
            cells = _category_breakdown(card)
            row = " | ".join(cells.get(cat, "—") for cat in categories)
            lines.append(f"| {card.spec.name} | {row} |")

    lines += _pairwise_section(cards)
    lines += _judge_section(cards)
    return "\n".join(lines)


def _judge_section(cards: Sequence[ModelScorecard]) -> list[str]:
    """The LLM-judge tier, visually separated from the deterministic tier.

    Judge-tier tasks have no computable ground truth and are graded by a *judge model*
    (named here) — they are REPORTED, NEVER GATING (the baseline gate ignores them). The
    table reports the judge pass rate over *graded* trials plus a separate ungraded count
    (judge timeout / unparseable verdict — distinct from a graded fail). Only rendered when
    the run has judge-tier tasks.
    """
    if not any(card.has_judge_tier for card in cards):
        return []
    judge_models = sorted(
        {
            t.judge_model
            for card in cards
            for s in card.judge_scores
            for t in s.trials
            if t.judge_model
        }
    )
    judged_by = f" — judged by {', '.join(judge_models)}" if judge_models else ""
    lines = [
        "",
        "## Judge tier (LLM-graded — reported, NOT gated)",
        "",
        "Open-ended tasks with no computable ground truth, graded by a judge model"
        f"{judged_by}. These results are **excluded from the baseline gate** and from the "
        "deterministic accuracy above. `Ungraded` = the judge timed out or returned an "
        "unparseable verdict (distinct from a graded fail).",
        "",
        "| Model | Judge pass rate | Graded | Ungraded |",
        "|---|---|---|---|",
    ]
    for card in cards:
        if not card.has_judge_tier:
            continue
        graded = card.judge_total_trials - card.judge_ungraded
        lines.append(
            f"| {card.spec.name} "
            f"| {_pct(card.judge_accuracy)} "
            f"| {graded}/{card.judge_total_trials} "
            f"| {card.judge_ungraded} |"
        )
    # Per-task judge breakdown so a reader sees which open-ended tasks passed.
    lines += [
        "",
        "| Model | Task | Judge pass (graded) | Ungraded |",
        "|---|---|---|---|",
    ]
    for card in cards:
        for s in card.judge_scores:
            rate = _pct(s.judge_pass_rate)
            graded_n = len(s.judge_graded)
            lines.append(
                f"| {card.spec.name} | {s.task_id} | {rate} ({s.judge_passes}/{graded_n}) "
                f"| {s.judge_ungraded} |"
            )
    return lines


def _trajectory_section(cards: Sequence[ModelScorecard]) -> list[str]:
    """List which tasks had a trajectory (bad tool-path) failure, per model.

    A trajectory failure is a trial that produced a result but took a wrong path (wrong
    first tool, a runaway tool loop, a forbidden/unknown tool, or a missing required
    ordering) — distinct from a wrong answer or a crash. Only rendered when at least one
    model had one (the caller gates on that).
    """
    lines = [
        "",
        "## Trajectory failures",
        "",
        "Trials that returned a result but took a bad tool *path* "
        "(graded against the ordered tool-call sequence, not the answer).",
        "",
        "| Model | Tasks (failed trials / n) |",
        "|---|---|",
    ]
    for card in cards:
        offenders = [
            f"{s.task_id} {s.trajectory_failures}/{s.n}"
            for s in card.task_scores
            if s.trajectory_failures
        ]
        cell = ", ".join(offenders) if offenders else "—"
        lines.append(f"| {card.spec.name} | {cell} |")
    return lines


def _holm_significant(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Holm–Bonferroni step-down: which of ``p_values`` are significant at family ``alpha``.

    Sorts the ``m`` p-values ascending and compares the ``k``-th smallest (0-indexed) to
    ``alpha / (m - k)``; once one fails, every larger p-value is non-significant too
    (step-down monotonicity). Controls the family-wise error rate across the C(k,2) pairs
    a multi-model run emits — the uncorrected per-pair 0.05 would let ~26% of 4-model
    runs show a spurious "significant" with no real difference. Returns a flag per input
    in the ORIGINAL order.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    significant = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] < alpha / (m - rank):
            significant[idx] = True
        else:
            break  # step-down: no larger p-value can be significant
    return significant


def _pairwise_section(cards: Sequence[ModelScorecard]) -> list[str]:
    """Pairwise McNemar comparison for every model pair (only when 2+ models).

    The per-pair exact McNemar test is sound, but running it for all ``C(k,2)`` pairs at a
    raw 0.05 inflates the family-wise false-positive rate (≈26% at 4 models). We therefore
    apply a **Holm–Bonferroni** correction across the emitted pairs before calling a
    leader "significant", and the verdict line names the corrected family alpha.
    """
    if len(cards) < 2:
        return []
    from himmy.benchmark.models import compare_scorecards

    pairs = list(combinations(cards, 2))
    results = [compare_scorecards(a, b) for a, b in pairs]
    holm = _holm_significant([r.p_value for r in results], PAIRWISE_ALPHA)
    n_comparisons = len(pairs)
    lines = [
        "",
        "## Pairwise comparison (McNemar, exact)",
        "",
        "Paired by `(task_id, trial)`. `b` = first model wins, `c` = second wins; "
        "only discordant pairs carry signal. Significance is **Holm-corrected across "
        f"the {n_comparisons} pair(s)** at family alpha {PAIRWISE_ALPHA:g} (the raw "
        "per-comparison alpha would inflate the family-wise false-positive rate).",
        "",
        "| A vs B | b / c | n pairs | p-value | Verdict |",
        "|---|---|---|---|---|",
    ]
    for (a, b), res, sig in zip(pairs, results, holm, strict=True):
        if sig and res.leader is not None:
            verdict = (
                f"**{res.leader}** leads (significant at {PAIRWISE_ALPHA:g}, "
                f"Holm-corrected across {n_comparisons})"
            )
        elif res.b == res.c:
            verdict = f"tie (b = c = {res.b})"
        else:
            verdict = (
                f"not significant at {PAIRWISE_ALPHA:g} (Holm, {n_comparisons} pairs)"
            )
        lines.append(
            f"| {a.spec.name} vs {b.spec.name} "
            f"| {res.b} / {res.c} "
            f"| {res.n_pairs} "
            f"| {res.p_value:.3f} "
            f"| {verdict} |"
        )
    return lines


def _judge_json(card: ModelScorecard) -> dict[str, Any]:
    """The judge-tier sub-record for one model (reported, never gated)."""
    judge_models = sorted(
        {t.judge_model for s in card.judge_scores for t in s.trials if t.judge_model}
    )
    return {
        "judge_models": judge_models,
        "pass_rate": card.judge_accuracy,
        "total_trials": card.judge_total_trials,
        "ungraded": card.judge_ungraded,
        "tasks": [
            {
                "id": s.task_id,
                "category": s.category,
                "judge_pass_rate": s.judge_pass_rate,
                "judge_passes": s.judge_passes,
                "graded": len(s.judge_graded),
                "ungraded": s.judge_ungraded,
            }
            for s in card.judge_scores
        ],
    }


def to_json(cards: Sequence[ModelScorecard], *, suite_name: str = "") -> dict[str, Any]:
    """A structured, persistable record of the benchmark run.

    ``by_category`` keeps the raw rate, but ``category_counts`` ships the per-category
    task count alongside it so consumers can apply the
    "no percentage below :data:`MIN_CATEGORY_TASKS` tasks" rule themselves. When the
    run has 2+ models, ``pairwise`` carries the McNemar comparisons.
    """
    out: dict[str, Any] = {
        "suite": suite_name,
        "trials": cards[0].task_scores[0].n if cards and cards[0].task_scores else 0,
        "min_category_tasks": MIN_CATEGORY_TASKS,
        "models": [
            {
                "name": card.spec.name,
                "provider": card.spec.provider,
                "model": card.spec.model,
                "tool_router": card.spec.tool_router,
                "accuracy": card.accuracy,
                "accuracy_ci": list(card.accuracy_ci),
                "tool_call_accuracy": card.tool_call_accuracy,
                "p50_latency_s": card.p50_latency,
                "p95_latency_s": card.p95_latency,
                "mean_cost": card.mean_cost,
                "error_rate": card.error_rate,
                "by_category": card.by_category(),
                "category_counts": card.category_counts(),
                "trajectory_failures": card.trajectory_failures,
                # Judge tier kept under its own key so a consumer never mistakes a judge
                # pass rate for the (gated) deterministic accuracy. None when no judge task.
                "judge": _judge_json(card) if card.has_judge_tier else None,
                "tasks": [
                    {
                        "id": s.task_id,
                        "category": s.category,
                        "pass_rate": s.pass_rate,
                        "pass_ci": list(s.pass_ci),
                        "tool_call_rate": s.tool_call_rate,
                        "p50_latency_s": s.p50_latency,
                        "errors": s.errors,
                        "trajectory_failures": s.trajectory_failures,
                    }
                    for s in card.task_scores
                ],
            }
            for card in cards
        ],
    }
    if len(cards) >= 2:
        from himmy.benchmark.models import compare_scorecards

        pairs = list(combinations(cards, 2))
        results = [compare_scorecards(a, b) for a, b in pairs]
        # `significant` is Holm-corrected across the family of pairs (not the raw per-pair
        # 0.05), so a consumer does not over-read multiplicity-inflated wins. We also ship
        # `n_comparisons` + the raw `p_value` so a consumer can apply its own correction.
        holm = _holm_significant([r.p_value for r in results], PAIRWISE_ALPHA)
        out["pairwise_alpha"] = PAIRWISE_ALPHA
        out["pairwise_correction"] = "holm"
        out["pairwise_n_comparisons"] = len(pairs)
        out["pairwise"] = [
            {
                "a": a.spec.name,
                "b": b.spec.name,
                "b_wins": res.b,
                "c_wins": res.c,
                "n_pairs": res.n_pairs,
                "p_value": res.p_value,
                "leader": res.leader,
                # Holm-corrected across all `n_comparisons` pairs at `pairwise_alpha`.
                "significant": sig and res.leader is not None,
                "discordant_task_ids": res.discordant_task_ids,
            }
            for (a, b), res, sig in zip(pairs, results, holm, strict=True)
        ]
    return out


__all__ = [
    "render_markdown",
    "to_json",
    "MIN_CATEGORY_TASKS",
    "PAIRWISE_ALPHA",
]
