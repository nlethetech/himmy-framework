"""Studio backend for model reliability: read cached benchmarks + run a quick probe.

Surfaces ``himmy bench`` numbers (tool-call accuracy, overall accuracy, latency) in
Doctor, so "is this local model good enough?" is a number on the screen where a user
picks a provider. The quick probe runs a small tool-focused suite against the local
models it can detect (Ollama tags + the Claude CLI), caches the result, and returns it.
"""

from __future__ import annotations

from typing import Any

from himmy.benchmark.cache import load_entries, save_scorecards, summarize
from himmy.core.ids import utc_now_iso

# A small, tool-focused subset of the core suite — fast enough to run on demand.
_QUICK_TASK_IDS = {"arithmetic", "file_read", "sql_count", "nepal_date"}
# Caps so an on-demand probe stays bounded.
_MAX_OLLAMA_MODELS = 2


def list_cached() -> list[dict[str, Any]]:
    """Cached per-model scorecard summaries, newest first."""
    entries = load_entries()
    return sorted(entries, key=lambda e: e.get("when", ""), reverse=True)


def _quick_suite() -> Any:
    from himmy.benchmark import default_suite
    from himmy.benchmark.models import BenchmarkSuite

    tasks = [t for t in default_suite().tasks if t.id in _QUICK_TASK_IDS]
    return BenchmarkSuite(name="quick", tasks=tasks)


async def _ollama_models(cap: int = _MAX_OLLAMA_MODELS) -> list[str]:
    """Models a local Ollama server reports (empty if it isn't running)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            models = resp.json().get("models", [])
    except Exception:  # noqa: BLE001 - ollama not running / unreachable
        return []
    return [m["name"] for m in models if m.get("name")][:cap]


async def detect_probe_specs() -> list[Any]:
    """Build :class:`ModelSpec`s for the local models we can detect."""
    import shutil

    from himmy.benchmark.models import ModelSpec

    specs: list[Any] = []
    for name in await _ollama_models():
        specs.append(ModelSpec(provider="ollama", model=name))
    if shutil.which("claude"):
        specs.append(ModelSpec(provider="claude-cli", model="haiku"))
    return specs


async def run_probe(specs: list[Any] | None = None) -> dict[str, Any]:
    """Run the quick suite against detected (or given) models; cache + return results."""
    from himmy.benchmark.runner import BenchmarkRunner

    if specs is None:
        specs = await detect_probe_specs()
    if not specs:
        return {
            "ran": False,
            "reason": "No local models detected. Start Ollama (`ollama serve`) or "
            "install the Claude CLI, then try again.",
            "results": [],
        }
    suite = _quick_suite()
    # The Doctor probe is a freshness check, not a tracked benchmark run — keep it out
    # of the committed run history.
    cards = await BenchmarkRunner(trials=1, append_history=False).run(suite, specs)
    when = utc_now_iso()
    try:
        save_scorecards(cards, suite_name=suite.name, when=when)
    except Exception:  # noqa: BLE001 - caching is best-effort
        pass
    return {
        "ran": True,
        "suite": suite.name,
        "results": [summarize(c, suite_name=suite.name, when=when) for c in cards],
    }


__all__ = ["list_cached", "detect_probe_specs", "run_probe"]
