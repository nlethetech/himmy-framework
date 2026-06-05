"""Declarative evaluation suites: define test cases in YAML, run them with `himmy eval`.

A ``suite.yaml`` lists cases (an input, an expected output, and per-metric weights). It
loads straight into an :class:`~himmy.services.evaluation.models.EvaluationSuite` that the
:class:`~himmy.services.evaluation.agent_harness.AgentEvalHarness` runs an agent/team
against::

    name: faq-suite
    cases:
      - input: {prompt: "What is the capital of Nepal?"}
        expected_output: {answer: "Kathmandu"}
        metric_weights: {relevance: 1.0}
"""

from __future__ import annotations

from pathlib import Path

import yaml

from himmy.services.evaluation.models import EvaluationSuite


def load_eval_suite(path: str | Path) -> EvaluationSuite:
    """Load an :class:`EvaluationSuite` from a YAML file."""
    raw = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"eval suite {path} must be a YAML mapping")
    raw.setdefault("name", Path(path).stem)
    return EvaluationSuite.model_validate(raw)


__all__ = ["load_eval_suite"]
