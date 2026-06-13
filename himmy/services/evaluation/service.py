"""Evaluation kernel: runs a suite over agent outputs into a scored run.

Async-capable (AAEO-10): metrics may score synchronously OR return a coroutine
(LLM-judge / embedding similarity); the service awaits coroutine results and fans
case/metric scoring out with a bounded ``asyncio.gather``. Veto/must-pass metrics
force a case to fail regardless of the weighted aggregate (AAEO-14), the per-case
pass threshold is configurable, and persistence failures are logged rather than
silently swallowed (AAEO-15).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any

from himmy.services.evaluation.metrics import (
    EvaluationMetricRegistry,
    bucketed_ece,
    build_registry,
    default_metric_registry,
)
from himmy.services.evaluation.models import (
    EvaluationCaseResult,
    EvaluationRun,
    EvaluationSuite,
    MetricScore,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.services.storage.service import StorageService

logger = logging.getLogger("himmy.evaluation")

# Default per-case pass threshold for the weighted aggregate (overridable).
_CASE_PASS_THRESHOLD = 0.5

# Bound on concurrent case scoring so a judge-backed suite isn't O(N) serial.
_DEFAULT_CONCURRENCY = 8

# Metrics treated as veto/must-pass by default (AAEO-14): a hard fail on any of
# these forces the case to fail regardless of the weighted aggregate.
_DEFAULT_VETO_METRICS = ("safety",)


class EvaluationService:
    """Executes an :class:`EvaluationSuite` against actual outputs.

    Each case's metrics are scored (awaiting coroutine results), weighted (per
    ``case.metric_weights``, falling back to equal weights), and aggregated into
    an overall run score. A failed veto/must-pass metric forces the case to fail.
    When a storage service is supplied the run is persisted for the dashboard.
    """

    def __init__(
        self,
        *,
        metric_registry: EvaluationMetricRegistry | None = None,
        storage_service: StorageService | None = None,
        inference_service: Any = None,
        embedder: Any = None,
        concurrency: int = _DEFAULT_CONCURRENCY,
        case_pass_threshold: float = _CASE_PASS_THRESHOLD,
        veto_metrics: tuple[str, ...] = _DEFAULT_VETO_METRICS,
    ) -> None:
        """Wire the metric registry, store, optional judge/embedding metrics.

        When ``metric_registry`` is omitted but ``inference_service`` or
        ``embedder`` is given, a registry with the baselines PLUS the LLM-judge /
        embedding-similarity metrics is built (AAEO-10). ``case_pass_threshold``
        and ``veto_metrics`` configure the pass verdict (AAEO-14).
        """
        if metric_registry is not None:
            self._metrics = metric_registry
        elif inference_service is not None or embedder is not None:
            self._metrics = build_registry(
                inference_service=inference_service, embedder=embedder
            )
        else:
            self._metrics = default_metric_registry
        self._storage = storage_service
        self._concurrency = max(1, concurrency)
        self._case_pass_threshold = case_pass_threshold
        self._veto_metrics = tuple(veto_metrics)

    @property
    def storage(self) -> StorageService | None:
        """The wired storage service, if any (the read-side a privacy audit prefetches).

        Exposed read-only so a downstream auditor (the WS4.7
        :class:`~himmy.services.evaluation.privacy.service.PrivacyAuditService`) can
        ``await`` the same recorded-data surface this kernel persists into, without
        reaching past the encapsulation boundary.
        """
        return self._storage

    async def run_suite(
        self,
        *,
        suite: EvaluationSuite,
        actual_outputs: dict[str, Any],
        workspace_id: str | None = None,
        actor: dict[str, Any] | None = None,
    ) -> EvaluationRun:
        """Score ``suite`` against ``actual_outputs`` (keyed by ``case_id``).

        ``workspace_id`` (when supplied) stamps the run with its owning tenant so
        the persisted row — and every later read of it — is tenant-scoped (AAEO-4).
        It is ``None`` on the offline/single-tenant path, preserving legacy behavior.
        ``actor`` (a compact, log-safe principal descriptor) is folded into the run
        metadata for the audit spine when this run is created by an authenticated
        caller.
        """
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _run_case(case: Any) -> EvaluationCaseResult:
            async with semaphore:
                actual = actual_outputs.get(case.case_id)
                metric_scores = await self._score_case(case, actual)
                aggregate = self._weighted_aggregate(case.metric_weights, metric_scores)
                threshold = self._case_threshold(case)
                vetoed = self._vetoed(case, metric_scores)
                passed = (aggregate >= threshold) and not vetoed
                return EvaluationCaseResult(
                    case_id=case.case_id,
                    metric_scores=metric_scores,
                    aggregate=aggregate,
                    passed=passed,
                    actual_output=actual,
                )

        case_results: list[EvaluationCaseResult] = list(
            await asyncio.gather(*(_run_case(case) for case in suite.cases))
        )

        overall = (
            sum(r.aggregate for r in case_results) / len(case_results)
            if case_results
            else 0.0
        )
        run = EvaluationRun(
            suite_id=suite.suite_id,
            suite_name=suite.name,
            workspace_id=workspace_id,
            aggregate_score=overall,
            case_results=case_results,
        )
        # AAEO-5: fold a suite-level bucketed ECE into the run metadata when
        # calibration samples are available.
        ece = self._suite_ece(case_results)
        if ece is not None:
            run.metadata = {**(run.metadata or {}), "calibration_ece": ece}
        # T1.1: stamp the creating actor into the run metadata for the audit spine.
        if actor:
            run.metadata = {**(run.metadata or {}), "actor": dict(actor)}

        if self._storage is not None:
            try:
                await self._storage.save_evaluation_run(run)
            except Exception:  # noqa: BLE001 - AAEO-15: log, don't swallow
                logger.warning(
                    "failed to persist evaluation run %s", run.run_id, exc_info=True
                )
                run.metadata = {**(run.metadata or {}), "persisted": False}
        return run

    async def _score_case(self, case: Any, actual: Any) -> list[MetricScore]:
        """Score every metric named in the case's weights (or all built-ins).

        Each metric's ``score`` may return synchronously or as a coroutine; the
        latter is awaited (AAEO-10). Metrics within a case are scored concurrently.
        """
        metric_names = list(case.metric_weights.keys()) or [
            "accuracy",
            "relevance",
            "groundedness",
            "safety",
            "calibration",
        ]

        async def _one(name: str) -> MetricScore:
            evaluator = self._metrics.get(name)
            if evaluator is None:
                return MetricScore(
                    metric=name,
                    score=0.0,
                    passed=False,
                    detail="unknown metric",
                )
            try:
                result = evaluator.score(case, actual)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as exc:  # noqa: BLE001 - a broken metric fails closed
                logger.warning("metric %s raised: %s", name, exc)
                return MetricScore(
                    metric=name,
                    score=0.0,
                    passed=False,
                    detail=f"metric error: {exc}",
                )

        return list(await asyncio.gather(*(_one(name) for name in metric_names)))

    def _case_threshold(self, case: Any) -> float:
        """Per-case pass threshold (case metadata overrides the service default)."""
        meta = getattr(case, "metadata", {}) or {}
        override = meta.get("pass_threshold")
        if isinstance(override, (int, float)):
            return float(override)
        return self._case_pass_threshold

    def _vetoed(self, case: Any, scores: list[MetricScore]) -> bool:
        """True when a must-pass/veto metric failed (AAEO-14).

        The veto set is the service default unioned with any
        ``case.metadata['veto_metrics']`` list. A failed metric in that set forces
        the case to fail regardless of the weighted aggregate.
        """
        meta = getattr(case, "metadata", {}) or {}
        case_veto = meta.get("veto_metrics")
        veto = set(self._veto_metrics)
        if isinstance(case_veto, (list, tuple, set)):
            veto |= {str(m) for m in case_veto}
        if not veto:
            return False
        return any((s.metric in veto and not s.passed) for s in scores)

    @staticmethod
    def _suite_ece(case_results: list[EvaluationCaseResult]) -> float | None:
        """Compute a suite-level bucketed ECE from calibration samples (AAEO-5).

        Pairs each case's stated confidence (from the calibration metric detail)
        with its accuracy. Returns None when no calibration samples exist.
        """
        samples: list[tuple[float, float]] = []
        for result in case_results:
            conf = None
            acc = None
            for score in result.metric_scores:
                if score.metric == "calibration":
                    conf = _parse_detail_value(score.detail, "confidence")
                    acc = _parse_detail_value(score.detail, "accuracy")
            if conf is not None and acc is not None:
                samples.append((conf, acc))
        if not samples:
            return None
        return bucketed_ece(samples)

    @staticmethod
    def _weighted_aggregate(
        weights: dict[str, float], scores: list[MetricScore]
    ) -> float:
        """Combine metric scores by ``weights`` (equal weights when unspecified)."""
        if not scores:
            return 0.0
        if weights:
            total_weight = sum(weights.get(s.metric, 0.0) for s in scores)
            if total_weight <= 0.0:
                return sum(s.score for s in scores) / len(scores)
            return (
                sum(s.score * weights.get(s.metric, 0.0) for s in scores) / total_weight
            )
        return sum(s.score for s in scores) / len(scores)


def _parse_detail_value(detail: str, key: str) -> float | None:
    """Pull a ``key=<float>`` value out of a calibration metric's detail string."""
    import re

    match = re.search(rf"{re.escape(key)}=([0-9]*\.?[0-9]+)", detail)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:  # pragma: no cover - defensive
        return None
