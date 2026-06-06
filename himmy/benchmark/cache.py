"""A small on-disk cache of benchmark scorecards, shared by the CLI and Studio.

``himmy bench`` writes its per-model results here so Studio's Doctor can answer
"is this local model good enough?" with a number (tool-call accuracy, overall
accuracy, latency) right where a user picks a provider — without re-running a slow
benchmark on every page load. Keyed by ``model + suite`` so a re-run replaces the
prior entry. Stored at ``~/.himmy/benchmarks.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from himmy.benchmark.models import ModelScorecard


def cache_path() -> Path:
    """Path to the benchmark cache (``~/.himmy/benchmarks.json``), dir created."""
    d = Path.home() / ".himmy"
    d.mkdir(exist_ok=True)
    return d / "benchmarks.json"


def summarize(card: ModelScorecard, *, suite_name: str, when: str) -> dict[str, Any]:
    """A compact, JSON-able summary of one model's scorecard for the GUI."""
    return {
        "model": card.spec.name,
        "provider": card.spec.provider,
        "model_id": card.spec.model,
        "suite": suite_name,
        "when": when,
        "accuracy": card.accuracy,
        "accuracy_ci": list(card.accuracy_ci),
        "tool_call_accuracy": card.tool_call_accuracy,
        "p50_latency_s": card.p50_latency,
        "p95_latency_s": card.p95_latency,
        "error_rate": card.error_rate,
        "total_trials": card.total_trials,
    }


def load_entries() -> list[dict[str, Any]]:
    """Load cached scorecard summaries (newest-written last), or [] if none/corrupt."""
    path = cache_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    return entries if isinstance(entries, list) else []


def save_scorecards(
    cards: Sequence[ModelScorecard], *, suite_name: str, when: str
) -> None:
    """Upsert each model's summary into the cache, replacing the same model+suite."""
    existing = load_entries()
    fresh = {(c.spec.name, suite_name) for c in cards}
    kept = [
        e for e in existing if (e.get("model"), e.get("suite")) not in fresh
    ]
    new = [summarize(c, suite_name=suite_name, when=when) for c in cards]
    cache_path().write_text(
        json.dumps({"entries": kept + new}, indent=2), encoding="utf-8"
    )


__all__ = ["cache_path", "summarize", "load_entries", "save_scorecards"]
