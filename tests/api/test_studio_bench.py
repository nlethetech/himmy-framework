"""Tests for the Studio model-reliability surface (cache + endpoints + probe guard)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # The cache lives at ~/.himmy/benchmarks.json — point HOME at a tmp dir.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _write_cache(home: Path, entries: list[dict]) -> None:
    d = home / ".himmy"
    d.mkdir(exist_ok=True)
    (d / "benchmarks.json").write_text(json.dumps({"entries": entries}))


def test_benchmarks_endpoint_reads_cache(home: Path) -> None:
    _write_cache(
        home,
        [
            {
                "model": "ollama:qwen2.5:3b-instruct",
                "provider": "ollama",
                "suite": "quick",
                "when": "2026-06-06T00:00:00Z",
                "accuracy": 0.64,
                "tool_call_accuracy": 0.58,
                "p50_latency_s": 1.2,
                "p95_latency_s": 3.1,
                "error_rate": 0.0,
                "total_trials": 4,
            }
        ],
    )
    client = TestClient(create_app())
    body = client.get("/api/studio/benchmarks").json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["tool_call_accuracy"] == 0.58


def test_benchmarks_empty_when_no_cache(home: Path) -> None:
    client = TestClient(create_app())
    assert client.get("/api/studio/benchmarks").json() == {"entries": []}


def test_probe_with_no_models_is_graceful() -> None:
    from himmy.api.studio_bench import run_probe

    result = asyncio.run(run_probe(specs=[]))
    assert result["ran"] is False
    assert "No local models" in result["reason"]
    assert result["results"] == []


def test_cache_upsert_replaces_same_model(home: Path) -> None:
    """A re-run for the same model+suite replaces the prior entry, not appends."""
    from himmy.benchmark.cache import load_entries, save_scorecards
    from himmy.benchmark.models import ModelScorecard, ModelSpec, TaskScore

    spec = ModelSpec(provider="ollama", model="qwen2.5:3b-instruct")
    card = ModelScorecard(spec=spec, task_scores=[TaskScore("t", "general", [])])
    save_scorecards([card], suite_name="quick", when="2026-06-06T00:00:00Z")
    save_scorecards([card], suite_name="quick", when="2026-06-06T01:00:00Z")
    entries = load_entries()
    matching = [e for e in entries if e["model"] == "ollama:qwen2.5:3b-instruct"]
    assert len(matching) == 1
    assert matching[0]["when"] == "2026-06-06T01:00:00Z"
