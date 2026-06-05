"""Render benchmark scorecards as a comparative Markdown table and as JSON.

Markdown is the human scorecard (models side by side on accuracy + confidence interval,
tool-call accuracy, latency percentiles, cost, error rate, and a per-category
breakdown). JSON is the machine record — persist it to track regressions over time and
to diff configs (e.g. router on vs off).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from himmy.benchmark.models import ModelScorecard


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def render_markdown(cards: Sequence[ModelScorecard], *, suite_name: str = "") -> str:
    """A comparative Markdown scorecard across models."""
    if not cards:
        return "(no results)"
    trials = cards[0].task_scores[0].n if cards[0].task_scores else 0
    n_tasks = len(cards[0].task_scores)
    lines = [
        f"# Benchmark{f': {suite_name}' if suite_name else ''} "
        f"({n_tasks} tasks × {trials} trials)",
        "",
        "| Model | Accuracy (95% CI) | Tool-call | p50 | p95 | Cost/trial | Errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for card in sorted(cards, key=lambda c: c.accuracy, reverse=True):
        lo, hi = card.accuracy_ci
        lines.append(
            f"| {card.spec.name} "
            f"| {_pct(card.accuracy)} ({_pct(lo)}–{_pct(hi)}) "
            f"| {_pct(card.tool_call_accuracy)} "
            f"| {card.p50_latency:.1f}s "
            f"| {card.p95_latency:.1f}s "
            f"| ${card.mean_cost:.4f} "
            f"| {_pct(card.error_rate)} |"
        )

    categories = sorted({c for card in cards for c in card.by_category()})
    if categories:
        lines += ["", "## Accuracy by category", ""]
        lines.append("| Model | " + " | ".join(categories) + " |")
        lines.append("|---|" + "|".join(["---"] * len(categories)) + "|")
        for card in cards:
            cats = card.by_category()
            row = " | ".join(_pct(cats.get(cat)) for cat in categories)
            lines.append(f"| {card.spec.name} | {row} |")
    return "\n".join(lines)


def to_json(cards: Sequence[ModelScorecard], *, suite_name: str = "") -> dict[str, Any]:
    """A structured, persistable record of the benchmark run."""
    return {
        "suite": suite_name,
        "trials": cards[0].task_scores[0].n if cards and cards[0].task_scores else 0,
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
                "tasks": [
                    {
                        "id": s.task_id,
                        "category": s.category,
                        "pass_rate": s.pass_rate,
                        "pass_ci": list(s.pass_ci),
                        "tool_call_rate": s.tool_call_rate,
                        "p50_latency_s": s.p50_latency,
                        "errors": s.errors,
                    }
                    for s in card.task_scores
                ],
            }
            for card in cards
        ],
    }


__all__ = ["render_markdown", "to_json"]
