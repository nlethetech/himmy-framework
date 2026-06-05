"""Tests for the agent evaluation harness (offline, deterministic metrics)."""

from __future__ import annotations

from pathlib import Path

from himmy import build_runtime
from himmy.agents.personas.persona import Persona
from himmy.config.eval_spec import load_eval_suite
from himmy.orchestrators import AgentTeam, TeamMember
from himmy.services.evaluation.agent_harness import AgentEvalHarness
from himmy.services.evaluation.models import EvaluationCase, EvaluationSuite
from himmy.services.tools.registry import ToolRegistry
from tests.conftest import run_async


def _suite() -> EvaluationSuite:
    return EvaluationSuite(
        name="demo",
        cases=[
            EvaluationCase(
                input={"prompt": "say hello"},
                expected_output={"answer": "hello"},
                metric_weights={"relevance": 1.0},
            ),
            EvaluationCase(
                input={"prompt": "summarize"},
                expected_output={"answer": "summary"},
                metric_weights={"relevance": 1.0},
            ),
        ],
    )


def test_evaluate_agent_produces_run() -> None:
    runtime, _i, _t = build_runtime()
    harness = AgentEvalHarness(runtime)
    run = run_async(harness.evaluate_agent(_suite(), Persona(name="a")))
    assert run.suite_name == "demo"
    assert len(run.case_results) == 2
    for case in run.case_results:
        assert any(m.metric == "relevance" for m in case.metric_scores)


def test_evaluate_team_produces_run() -> None:
    registry = ToolRegistry()
    runtime, _i, _t = build_runtime(tool_registry=registry)
    team = AgentTeam(
        members=[TeamMember(name="solo", persona=Persona(name="solo"))], entry="solo"
    )
    run = run_async(AgentEvalHarness(runtime).evaluate_team(_suite(), team, registry))
    assert len(run.case_results) == 2


def test_load_eval_suite(tmp_path: Path) -> None:
    (tmp_path / "suite.yaml").write_text(
        "name: faq\n"
        "cases:\n"
        "  - input: {prompt: 'q1'}\n"
        "    expected_output: {answer: 'a1'}\n"
        "    metric_weights: {relevance: 1.0}\n"
    )
    suite = load_eval_suite(tmp_path / "suite.yaml")
    assert suite.name == "faq"
    assert len(suite.cases) == 1
    assert suite.cases[0].input["prompt"] == "q1"


def test_load_eval_suite_defaults_name(tmp_path: Path) -> None:
    (tmp_path / "mysuite.yaml").write_text("cases: []\n")
    assert load_eval_suite(tmp_path / "mysuite.yaml").name == "mysuite"
