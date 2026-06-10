"""Tests for the Studio Evaluation surface: discovery, run streaming, baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_eval as se
from himmy.api.app import create_app

# ---- fixtures -------------------------------------------------------------

_SUITE_YAML = (
    "name: demo\n"
    "cases:\n"
    "  - input: { prompt: say hello }\n"
    "    expected_output: { contains: hello }\n"
    "    metric_weights: { accuracy: 1.0, safety: 1.0 }\n"
    "  - input: { prompt: say goodbye }\n"
    "    expected_output: { contains: goodbye }\n"
    "    metric_weights: { accuracy: 1.0 }\n"
)

_BASELINE = {
    "schema": 1,
    "suite": "core-gate",
    "baselined_at": {"sha": "c1ce40987f16", "date": "2026-06-09", "trials": 5},
    "gate": {"tasks": ["arithmetic", "file_read"], "trials": 5, "min_trials": 5},
    "models": {
        "ollama:qwen2.5:3b-instruct": {
            "measured": {
                "accuracy": 0.66,
                "tool_call_accuracy": 1.0,
                "error_rate": 0.0,
                "by_category": {"arithmetic": 1.0, "files": 1.0},
            },
            "floors": {"accuracy": 0.517, "tool_call_accuracy": 0.85},
            "ceilings": {"error_rate": 0.15},
        }
    },
}


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_suite(root: Path) -> None:
    (root / "demo.eval.yaml").write_text(_SUITE_YAML)


def _write_baseline(root: Path, data: dict[str, Any] | None = None) -> None:
    bench = root / "benchmarks"
    bench.mkdir(exist_ok=True)
    (bench / "baseline.json").write_text(json.dumps(data or _BASELINE))


def _client() -> TestClient:
    return TestClient(create_app())


def _frames(text: str) -> list[dict[str, Any]]:
    """Decode the SSE body into its `data:` event dicts."""
    out: list[dict[str, Any]] = []
    for frame in text.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: ") :]))
    return out


# ---- suite discovery (module) ----------------------------------------------


def test_discover_suites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "demo.eval.yaml").write_text(
        "name: demo\n"
        "cases:\n"
        "  - input: { prompt: hi }\n"
        "    expected_output: { contains: hello }\n"
        "    metric_weights: { accuracy: 1.0 }\n"
    )
    (tmp_path / "not-a-suite.yaml").write_text("name: x\n")  # ignored (wrong glob)
    monkeypatch.chdir(tmp_path)
    suites = se.discover_suites()
    assert [s.name for s in suites] == ["demo"]
    assert suites[0].cases == 1
    assert suites[0].path == "demo.eval.yaml"


# ---- GET /api/studio/eval/suites --------------------------------------------


def test_suites_lists_eval_and_bench_gate(project: Path) -> None:
    _write_suite(project)
    _write_baseline(project)
    body = _client().get("/api/studio/eval/suites").json()
    by_kind = {s["kind"]: s for s in body["suites"]}
    assert by_kind["eval"]["name"] == "demo"
    assert by_kind["eval"]["cases"] == 2
    assert by_kind["eval"]["id"] == "demo.eval.yaml"
    assert by_kind["bench"]["id"] == "bench:gate"
    assert by_kind["bench"]["cases"] == 2  # the gate's task subset
    assert by_kind["bench"]["name"] == "core-gate"


def test_suites_empty_project(project: Path) -> None:
    assert _client().get("/api/studio/eval/suites").json() == {"suites": []}


def test_suites_skips_malformed_baseline(project: Path) -> None:
    (project / "benchmarks").mkdir()
    (project / "benchmarks" / "baseline.json").write_text("{not json")
    assert _client().get("/api/studio/eval/suites").json() == {"suites": []}


# ---- GET /api/studio/eval/baseline -------------------------------------------


def test_baseline_parsed_for_display(project: Path) -> None:
    _write_baseline(project)
    body = _client().get("/api/studio/eval/baseline").json()
    assert body["exists"] is True
    assert body["sha"] == "c1ce40987f16"
    assert body["date"] == "2026-06-09"
    assert body["trials"] == 5 and body["min_trials"] == 5
    assert body["gate_tasks"] == ["arithmetic", "file_read"]
    (model,) = body["models"]
    assert model["name"] == "ollama:qwen2.5:3b-instruct"
    rows = {r["metric"]: r for r in model["rows"]}
    assert rows["accuracy"]["bound"] == "floor"
    assert rows["accuracy"]["limit"] == 0.517
    assert rows["accuracy"]["measured"] == 0.66
    assert rows["error_rate"]["bound"] == "ceiling"
    assert rows["error_rate"]["limit"] == 0.15
    assert rows["cat:arithmetic"]["measured"] == 1.0
    assert rows["cat:arithmetic"]["limit"] is None  # no per-category floor set


def test_baseline_missing_is_a_display_state_not_an_error(project: Path) -> None:
    body = _client().get("/api/studio/eval/baseline").json()
    assert body["exists"] is False
    assert "baseline" in body["reason"]


def test_baseline_malformed_reports_reason(project: Path) -> None:
    (project / "benchmarks").mkdir()
    (project / "benchmarks" / "baseline.json").write_text('{"models": []}')
    body = _client().get("/api/studio/eval/baseline").json()
    assert body["exists"] is False
    assert "parsed" in body["reason"]


# ---- POST /api/studio/eval/run — eval suites (offline stub) ------------------


def test_run_eval_streams_cases_then_summary(project: Path) -> None:
    _write_suite(project)
    resp = _client().post("/api/studio/eval/run", json={"suite": "demo.eval.yaml"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _frames(resp.text)
    assert events[0]["type"] == "start"
    assert events[0]["kind"] == "eval"
    assert events[0]["cases"] == 2
    assert events[0]["provider"] == "stub"  # offline by default
    cases = [e for e in events if e["type"] == "case"]
    assert len(cases) == 2
    assert [c["index"] for c in cases] == [0, 1]
    for case in cases:
        assert 0.0 <= case["score"] <= 1.0
        assert isinstance(case["passed"], bool)
        names = {m["metric"] for m in case["metrics"]}
        assert "accuracy" in names
    summary = events[-1]
    assert summary["type"] == "summary"
    assert summary["cases"] == 2
    assert summary["passed"] + summary["failed"] == 2
    assert 0.0 <= summary["aggregate_score"] <= 1.0
    assert summary["duration_s"] >= 0.0


def test_run_eval_suite_not_found_404(project: Path) -> None:
    resp = _client().post("/api/studio/eval/run", json={"suite": "missing.eval.yaml"})
    assert resp.status_code == 404


def test_run_eval_path_traversal_rejected(project: Path) -> None:
    resp = _client().post("/api/studio/eval/run", json={"suite": "../evil.eval.yaml"})
    assert resp.status_code == 400
    assert "escapes" in resp.json()["detail"]


def test_run_eval_empty_suite_400(project: Path) -> None:
    (project / "empty.eval.yaml").write_text("name: empty\ncases: []\n")
    resp = _client().post("/api/studio/eval/run", json={"suite": "empty.eval.yaml"})
    assert resp.status_code == 400
    assert "no cases" in resp.json()["detail"]


def test_run_eval_unparseable_suite_400(project: Path) -> None:
    (project / "bad.eval.yaml").write_text("cases: {not: [a, list}\n")
    resp = _client().post("/api/studio/eval/run", json={"suite": "bad.eval.yaml"})
    assert resp.status_code == 400


def test_run_unknown_provider_400(project: Path) -> None:
    _write_suite(project)
    resp = _client().post(
        "/api/studio/eval/run", json={"suite": "demo.eval.yaml", "provider": "nope"}
    )
    assert resp.status_code == 400
    assert "provider" in resp.json()["detail"]


def test_run_request_bounds_422(project: Path) -> None:
    resp = _client().post("/api/studio/eval/run", json={"suite": "x" * 600})
    assert resp.status_code == 422


# ---- POST /api/studio/eval/run — bench gate (offline stub) -------------------


def test_run_bench_gate_offline(project: Path) -> None:
    _write_baseline(project)
    resp = _client().post("/api/studio/eval/run", json={"suite": "bench:gate"})
    assert resp.status_code == 200
    events = _frames(resp.text)
    assert events[0]["type"] == "start"
    assert events[0]["kind"] == "bench"
    assert events[0]["provider"] == "stub"
    cases = [e for e in events if e["type"] == "case"]
    assert [c["id"] for c in cases] == ["arithmetic", "file_read"]
    for case in cases:
        assert any(m["metric"] == "grade" for m in case["metrics"])
    summary = events[-1]
    assert summary["type"] == "summary"
    assert summary["kind"] == "bench"
    assert summary["model"] == "stub:stub"
    metrics = summary["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["error_rate"] <= 1.0


def test_run_bench_gate_without_baseline_404(project: Path) -> None:
    resp = _client().post("/api/studio/eval/run", json={"suite": "bench:gate"})
    assert resp.status_code == 404
    assert "baseline" in resp.json()["detail"]


def test_run_bench_gate_unknown_gate_task_400(project: Path) -> None:
    data = json.loads(json.dumps(_BASELINE))
    data["gate"]["tasks"] = ["no_such_task"]
    _write_baseline(project, data)
    resp = _client().post("/api/studio/eval/run", json={"suite": "bench:gate"})
    assert resp.status_code == 400
    assert "unknown task" in resp.json()["detail"]
