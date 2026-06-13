"""Evaluation kernel: typed shapes for cases, suites, scores, and runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso


class EvaluationCase(BaseModel):
    """One input → expected-behavior pair with per-metric weights."""

    case_id: str = Field(default_factory=new_uuid)
    input: dict[str, Any] = {}
    expected_output: dict[str, Any] = {}
    metric_weights: dict[str, float] = {}
    metadata: dict[str, Any] = {}


class EvaluationSuite(BaseModel):
    """A named collection of evaluation cases."""

    suite_id: str = Field(default_factory=new_uuid)
    name: str
    cases: list[EvaluationCase] = []
    metadata: dict[str, Any] = {}


class MetricScore(BaseModel):
    """A single metric's verdict for one case (score in ``0.0..1.0``)."""

    metric: str
    score: float = 0.0
    passed: bool = False
    detail: str = ""


class EvaluationCaseResult(BaseModel):
    """Per-case roll-up: every metric score plus the weighted aggregate."""

    case_id: str
    metric_scores: list[MetricScore] = []
    aggregate: float = 0.0
    passed: bool = False
    actual_output: Any = None


class EvaluationRun(BaseModel):
    """One execution of a suite against a set of actual agent outputs.

    ``metadata`` carries derived diagnostics (e.g. ``calibration_ece``, AAEO-5)
    and a ``persisted`` flag when storage write fails (AAEO-15). ``workspace_id``
    carries the tenant the run was scored under so reads can be tenant-scoped
    (AAEO-4) — ``None`` on the offline/single-tenant path and for legacy rows
    written before this field existed.
    """

    run_id: str = Field(default_factory=new_uuid)
    suite_id: str
    suite_name: str
    workspace_id: str | None = None
    aggregate_score: float = 0.0
    case_results: list[EvaluationCaseResult] = []
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
