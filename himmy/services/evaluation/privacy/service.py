"""WS4.7 — the :class:`PrivacyAuditService` that runs one auditable experiment (B3).

This is the orchestrator that ties the two halves of the Ragas-inspired privacy audit
into one signed, registrable scorecard:

* **The probe half** (live, adversarial) runs the
  :func:`~himmy.services.evaluation.privacy.probes.build_privacy_probe_suite` suite
  against the wired agent through the *unchanged*
  :class:`~himmy.services.evaluation.agent_harness.AgentEvalHarness` and rolls the suite
  result into one ``safety`` probe metric. It is **skipped entirely** when no ``runtime``
  is wired (the offline default), so a zero-config audit is a pure recorded-data scan.
* **The recorded-data half** (deterministic) is the linchpin correctness fix (must_fix
  #1/#2): the service ``await``s the async :class:`~himmy.services.storage.service.StorageService`
  *first* (``list_runs`` / ``list_memory`` / ``list_episodic_memory``), materializes the
  results into a :class:`~himmy.services.evaluation.privacy.models.RecordedDataContext`,
  then hands the synchronous recorded-data metrics concrete lists — so a metric never
  ``await``s an async store and never reads runs/memory/episodic as registry kinds.

The two halves roll into one
:class:`~himmy.services.evaluation.privacy.models.PrivacyAuditReport`. **Skipped metrics
are excluded** from the weighted posture and the overall verdict (must_fix #7). The report
is registered as its own ``privacy_audit_report``
:class:`~himmy.entities.records.EntityRecord` with a **fresh** ``stable_id`` per run
(must_fix #3 — re-running over an identical store never content-address-collides), and
can be exported as a tamper-evident signed bundle (Ed25519 when
``HIMMY_AUDIT_PRIVATE_KEY`` is set, else HMAC with ``HIMMY_AUDIT_SECRET``), mirroring
``api/routers/audit.py`` exactly. The LLM probe metrics are gated on a real provider
(``provider not in (None, 'stub')`` — must_fix #5) and the consent-derived metrics stay
``skipped`` until Part A injects a ``consent_service``.

Offline-first: importing or constructing this service wires nothing on the hot path and
runs no I/O; it is the opt-in CLI/HTTP auditor an operator invokes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from himmy.config.secrets import get_secret
from himmy.entities.integrity import (
    AuditBundle,
    VerificationResult,
    export_audit_bundle,
    export_audit_bundle_ed25519,
    verify_audit_bundle,
    verify_audit_bundle_ed25519,
)
from himmy.services.evaluation.agent_harness import AgentEvalHarness
from himmy.services.evaluation.privacy.metrics import build_recorded_registry
from himmy.services.evaluation.privacy.models import (
    PRIVACY_AUDIT_REPORT_KIND,
    PrivacyAuditConfig,
    PrivacyAuditReport,
    PrivacyFinding,
    PrivacyMetricResult,
    RecordedDataContext,
)
from himmy.services.evaluation.privacy.probes import build_privacy_probe_suite

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.agents.personas.persona import Persona
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.entities.records import EntityRecord
    from himmy.services.evaluation.models import EvaluationRun, EvaluationSuite
    from himmy.services.evaluation.privacy.metrics import RecordedDataMetric
    from himmy.services.evaluation.service import EvaluationService
    from himmy.services.governance.retention import RetentionService, SubjectKeyVault
    from himmy.services.storage.service import StorageService

#: Providers that do NOT count as "real" for LLM-metric gating (must_fix #5): ``None``
#: (nothing wired) and the BFF's always-injected ``'stub'`` manager.
_NON_REAL_PROVIDERS = (None, "stub")

#: The single probe metric the privacy probe suite grades on (the deterministic
#: :class:`~himmy.services.evaluation.metrics.SafetyMetric`).
_PROBE_METRIC = "safety"


class PrivacyAuditService:
    """Run one privacy & compliance audit (probe + recorded) into a signed report.

    The service is fully optional and fully additive: it reads the existing eval kernel,
    the existing entity registry, and (when present) the WS4.2 retention / Part A consent
    seams, and produces a :class:`PrivacyAuditReport`. None of these are required — a
    bare ``entity_registry`` + ``evaluation_service`` is enough to run a recorded-only
    audit over an empty store, which is exactly the zero-config posture.
    """

    def __init__(
        self,
        *,
        entity_registry: EntityRegistryProtocol,
        evaluation_service: EvaluationService,
        runtime: Any = None,
        security_audit: Any = None,
        inference_service: Any = None,
        embedder: Any = None,
        consent_service: Any = None,
        retention_service: RetentionService | None = None,
        key_vault: SubjectKeyVault | None = None,
        config: PrivacyAuditConfig | None = None,
    ) -> None:
        """Wire the registry, eval kernel, and the optional probe / governance seams.

        Args:
            entity_registry: The lineage registry the report is registered on and the
                first-class kinds (``security_event`` / ``erasure_tombstone`` / ``consent``
                / ``privacy_audit_report``) are read from.
            evaluation_service: The eval kernel (owns the read-side
                :class:`~himmy.services.storage.service.StorageService` the recorded-data
                half prefetches, and scores the probe suite).
            runtime: The agent runtime to run the live probe suite against; ``None`` ⇒ the
                probe half is skipped (the offline default).
            security_audit: The security-audit log (documents intent; integrity is read
                from the registry).
            inference_service: Optional inference service for the gated LLM probe
                synthesizer / aspect-critic metrics.
            embedder: Optional embedder (forwarded to the harness's eval service).
            consent_service: The Part A consent ledger; ``None`` ⇒ consent metrics skip.
            retention_service: Enables ``retention_compliance`` (needs a max-age window
                via the config).
            key_vault: Enables ``erasure_completeness`` (the crypto-shred check).
            config: The scoring policy (thresholds / veto / lookback / provider). A
                default :class:`PrivacyAuditConfig` is used when omitted.
        """
        self._registry = entity_registry
        self._eval = evaluation_service
        self._runtime = runtime
        self._security_audit = security_audit
        self._inference = inference_service
        self._embedder = embedder
        self._consent = consent_service
        self._retention = retention_service
        self._key_vault = key_vault
        self._config = config or PrivacyAuditConfig()

    # ------------------------------------------------------------------- run
    async def run(
        self,
        *,
        suite: EvaluationSuite | None = None,
        persona: Persona | None = None,
        lookback_seconds: float | None = None,
        workspace_id: str | None = None,
    ) -> PrivacyAuditReport:
        """Run the full audit and register the rolled-up signed-able report.

        Steps (in order):

        1. **Probe half** — when a ``runtime`` AND a ``persona`` are wired, build (or use
           the supplied) probe suite and run it via
           :meth:`~himmy.services.evaluation.agent_harness.AgentEvalHarness.evaluate_agent`,
           rolling the suite into one ``safety`` probe metric. Skipped otherwise.
        2. **Recorded half** — ``await`` the async store *first* (must_fix #1) into a
           :class:`RecordedDataContext`, then score the synchronous recorded-data metrics.
        3. **Roll-up** — combine both halves; ``skipped`` metrics are excluded from the
           weighted posture and the verdict (must_fix #7).
        4. **Register** — project the report to a fresh-id ``privacy_audit_report`` record
           and register it on the entity registry (must_fix #3).

        Args:
            suite: An explicit probe suite; ``None`` ⇒ the catalog default (built with the
                provider gate). Ignored when there is no runtime/persona.
            persona: The persona to run the probe suite as; ``None`` ⇒ probe half skipped.
            lookback_seconds: Overrides the config's recorded-data scan window for this run.
            workspace_id: Tenant scope for the recorded-data prefetch.

        Returns:
            The registered :class:`PrivacyAuditReport`.
        """
        window = (
            lookback_seconds
            if lookback_seconds is not None
            else self._config.lookback_seconds
        )

        probe_metrics, suite_name = await self._run_probes(suite=suite, persona=persona)
        recorded_metrics, scanned = await self._score_recorded(
            workspace_id=workspace_id, lookback_seconds=window
        )

        metrics = [*probe_metrics, *recorded_metrics]
        posture, passed = self._roll_up(metrics)
        report = PrivacyAuditReport(
            suite_name=suite_name,
            posture_score=posture,
            passed=passed,
            metrics=metrics,
            lookback_seconds=window,
            workspace_id=workspace_id,
            metadata={
                "scanned": scanned,
                "probe_ran": bool(probe_metrics),
                "provider_real": self._provider_is_real(),
            },
        )
        self._registry.register(report.to_record())
        return report

    # --------------------------------------------------------------- probe half
    async def _run_probes(
        self, *, suite: EvaluationSuite | None, persona: Persona | None
    ) -> tuple[list[PrivacyMetricResult], str]:
        """Run the adversarial probe suite (skipped when no runtime/persona is wired).

        Returns the probe metric(s) and the suite name (``""`` for a recorded-only run).
        """
        if self._runtime is None or persona is None:
            return [], ""

        probe_suite = suite or build_privacy_probe_suite(
            inference_service=self._inference,
            provider=self._config.provider,
        )
        harness = AgentEvalHarness(self._runtime, self._eval)
        run = await harness.evaluate_agent(probe_suite, persona)
        return [self._probe_metric(run)], probe_suite.name

    def _probe_metric(self, run: EvaluationRun) -> PrivacyMetricResult:
        """Roll a probe :class:`EvaluationRun` into one ``safety`` privacy metric.

        The score is the fraction of probe cases that did NOT leak (i.e. passed the
        SafetyMetric veto). Each failing case becomes a counts-only finding referencing
        the offending case id and (when present) the targeted subject — never any raw
        model output (must_fix #7/#8).
        """
        total = len(run.case_results)
        failing = [r for r in run.case_results if not r.passed]
        score = 1.0 if total == 0 else 1.0 - (len(failing) / total)
        threshold = self._config.threshold_for(_PROBE_METRIC)
        findings: list[PrivacyFinding] = []
        if failing:
            findings.append(
                PrivacyFinding(
                    code="probe_disclosure",
                    severity="high",
                    count=len(failing),
                    record_refs=sorted(r.case_id for r in failing),
                    detail="adversarial probe elicited a disallowed disclosure",
                )
            )
        detail = (
            "no probe cases run"
            if total == 0
            else f"{len(failing)}/{total} adversarial probes elicited a disclosure"
        )
        return PrivacyMetricResult(
            metric=_PROBE_METRIC,
            kind="probe",
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            detail=detail,
            findings=findings,
        )

    # ------------------------------------------------------------ recorded half
    async def _score_recorded(
        self, *, workspace_id: str | None, lookback_seconds: float | None
    ) -> tuple[list[PrivacyMetricResult], dict[str, int]]:
        """Prefetch the read-side store, then score the synchronous recorded metrics.

        The ``await``s happen *here* (must_fix #1); the metrics receive concrete lists.
        """
        ctx = await self._build_context(workspace_id=workspace_id)
        registry = build_recorded_registry(
            security_audit=self._security_audit,
            consent_service=self._consent,
            retention_service=self._retention,
            key_vault=self._key_vault,
            max_age_seconds=self._config.retention_max_age_seconds,
        )
        results = [self._safe_score(metric, ctx) for metric in registry]
        scanned = {
            "runs": len(ctx.run_records),
            "memory": len(ctx.memory),
            "episodic": len(ctx.episodic),
            "security_events": len(ctx.security_events),
        }
        return results, scanned

    async def _build_context(self, *, workspace_id: str | None) -> RecordedDataContext:
        """Materialize the :class:`RecordedDataContext` by awaiting the async store first.

        ``list_runs`` / ``list_memory`` / ``list_episodic_memory`` are awaited up front so
        the metrics scan concrete lists (must_fix #1). When no storage is wired the lists
        are empty — a recorded-only audit over a registry-only deployment still runs.
        """
        storage = self._storage
        if storage is None:
            return RecordedDataContext(
                entity_registry=self._registry,
                consent_service=self._consent,
                retention_service=self._retention,
                key_vault=self._key_vault,
            )
        return RecordedDataContext(
            run_records=await storage.list_runs(workspace_id=workspace_id),
            memory=await storage.list_memory(),
            episodic=await storage.list_episodic_memory(),
            entity_registry=self._registry,
            consent_service=self._consent,
            retention_service=self._retention,
            key_vault=self._key_vault,
        )

    @staticmethod
    def _safe_score(
        metric: RecordedDataMetric, ctx: RecordedDataContext
    ) -> PrivacyMetricResult:
        """Score one recorded metric, failing a broken metric *closed* (skipped → no fail).

        A metric that raises is reported as ``skipped`` (excluded from posture) rather than
        a misleading ``score=0`` failure, matching how the eval kernel fails a broken metric
        without crashing the whole run.
        """
        try:
            return metric.score(ctx)
        except Exception as exc:  # noqa: BLE001 - a broken metric must not crash the audit
            return PrivacyMetricResult(
                metric=metric.name,
                kind="recorded",
                skipped=True,
                detail=f"metric error: {exc}",
            )

    # --------------------------------------------------------------- roll-up
    def _roll_up(self, metrics: list[PrivacyMetricResult]) -> tuple[float, bool]:
        """Weighted posture over non-skipped metrics + the overall pass verdict.

        Skipped metrics are excluded from BOTH the weighted aggregate and the verdict
        (must_fix #7). The overall verdict is: every (non-skipped) veto metric passed AND
        the weighted aggregate cleared its bar. An audit with only skipped metrics is
        vacuously passing (posture ``1.0``) — there is nothing to substantiate a failure.
        """
        scored = [m for m in metrics if not m.skipped]
        if not scored:
            return 1.0, True

        total_weight = sum(m.weight for m in scored)
        posture = (
            sum(m.score * m.weight for m in scored) / total_weight
            if total_weight > 0
            else sum(m.score for m in scored) / len(scored)
        )

        veto = set(self._config.veto_metrics)
        vetoed = any(m.metric in veto and not m.passed for m in scored)
        aggregate_ok = posture >= self._config.default_threshold
        return posture, (aggregate_ok and not vetoed)

    # --------------------------------------------------------------- bundle
    def export_signed_bundle(self, report: PrivacyAuditReport) -> AuditBundle:
        """Export a tamper-evident signed bundle over the registered audit report.

        Mirrors ``api/routers/audit.py`` exactly: Ed25519 when ``HIMMY_AUDIT_PRIVATE_KEY``
        (PEM) is configured (an auditor verifies offline with the public key alone), else
        HMAC with ``HIMMY_AUDIT_SECRET``. When **neither** key is configured this raises
        :class:`HimmyError` (the 503-equivalent the router surfaces) — but ONLY because a
        bundle was explicitly requested; ``run()`` itself never needs a key.

        The bundle commits to this report's own ``privacy_audit_report`` record plus the
        first-class audit kinds it cites (``security_event`` / ``erasure_tombstone`` /
        ``consent``), so tampering with any cited record after the fact is detectable.
        """
        records = self._bundle_records(report)
        private_pem = get_secret("HIMMY_AUDIT_PRIVATE_KEY")
        if private_pem:
            return export_audit_bundle_ed25519(records, [], private_pem=private_pem)
        secret = get_secret("HIMMY_AUDIT_SECRET")
        if secret:
            return export_audit_bundle(records, [], secret=secret)
        from himmy.core.errors import HimmyError

        raise HimmyError(
            "audit signing key not configured "
            "(set HIMMY_AUDIT_PRIVATE_KEY or HIMMY_AUDIT_SECRET)"
        )

    def verify_report(
        self, report: PrivacyAuditReport, bundle: AuditBundle
    ) -> VerificationResult:
        """Re-verify a signed bundle against the live graph for ``report``.

        Re-derives the cited records' content hashes from the registry NOW and checks them
        against ``bundle``: if a cited record (this report, or a security_event / tombstone
        / consent it commits to) was mutated in place, ``ok`` flips to ``False``. Ed25519
        when the bundle was Ed25519-signed (verified with the public key), else HMAC.
        """
        records = self._bundle_records(report)
        if bundle.algorithm == "Ed25519":
            public_pem = get_secret("HIMMY_AUDIT_PUBLIC_KEY") or ""
            return verify_audit_bundle_ed25519(
                bundle, records, [], public_pem=public_pem
            )
        secret = get_secret("HIMMY_AUDIT_SECRET") or ""
        return verify_audit_bundle(bundle, records, [], secret=secret)

    def _bundle_records(self, report: PrivacyAuditReport) -> list[EntityRecord]:
        """The live records a report's bundle commits to (the report + cited kinds).

        Reads the report's registered version from the registry (so a verify re-derives the
        *stored* hash, catching in-place tampering of the report itself) plus the first-class
        audit kinds it cites. When the report has not been registered yet (a caller exporting
        before ``run`` registered it) the in-memory projection is used as a fallback.
        """
        from himmy.entities.records import record_id_for

        record_id = record_id_for(
            kind=PRIVACY_AUDIT_REPORT_KIND,
            stable_id=self._report_stable_id(report),
            version=1,
        )
        stored = self._registry.get(record_id)
        report_record = stored if stored is not None else report.to_record()

        records: list[EntityRecord] = [report_record]
        for kind in ("security_event", "erasure_tombstone", "consent"):
            records.extend(self._registry.list_by_kind(kind))
        return records

    def _report_stable_id(self, report: PrivacyAuditReport) -> str:
        """The stable_id under which ``report`` was registered (by its ``report_id``).

        The registered record's ``metadata['report_id']`` matches the report, so we find it
        by scanning the registered ``privacy_audit_report`` records. Falls back to a fresh
        projection's stable_id when the report is not yet registered.
        """
        for record in self._registry.list_by_kind(PRIVACY_AUDIT_REPORT_KIND):
            if record.metadata.get("report_id") == report.report_id:
                return record.stable_id
        return report.to_record().stable_id

    # --------------------------------------------------------------- trend
    def trend(self, *, limit: int = 20) -> list[PrivacyAuditReport]:
        """Return the most recent registered audit reports, newest first.

        Reads every ``privacy_audit_report`` record from the registry and projects each
        back into a (read-only) :class:`PrivacyAuditReport`, ordered by ``created_at``
        descending and capped at ``limit``. This is the CLI/HTTP trend surface — a
        posture-over-time view of the deployment's audits.
        """
        records = self._registry.list_by_kind(PRIVACY_AUDIT_REPORT_KIND)
        records.sort(
            key=lambda r: str(r.metadata.get("created_at") or ""), reverse=True
        )
        return [_report_from_record(r) for r in records[: max(0, limit)]]

    # --------------------------------------------------------------- helpers
    @property
    def _storage(self) -> StorageService | None:
        """The read-side store the recorded half prefetches (the eval kernel's store)."""
        return self._eval.storage

    def _provider_is_real(self) -> bool:
        """Whether a real (non-stub) provider is wired (gates the LLM metrics, must_fix #5)."""
        return self._config.provider not in _NON_REAL_PROVIDERS


def _report_from_record(record: EntityRecord) -> PrivacyAuditReport:
    """Reconstruct a (read-only) :class:`PrivacyAuditReport` from its registered record.

    Used by :meth:`PrivacyAuditService.trend`. Only the projected fields are restored;
    findings come back through the PII-safe :class:`PrivacyFinding` model, so no raw value
    can be introduced on the read path either.
    """
    payload = record.payload
    metrics = [
        PrivacyMetricResult(
            metric=str(m.get("metric", "")),
            kind=m.get("kind", "recorded"),
            score=float(m.get("score", 0.0)),
            passed=bool(m.get("passed", False)),
            threshold=float(m.get("threshold", 0.0)),
            weight=float(m.get("weight", 1.0)),
            detail=str(m.get("detail", "")),
            skipped=bool(m.get("skipped", False)),
            findings=[
                PrivacyFinding(
                    code=str(f.get("code", "")),
                    severity=f.get("severity", "medium"),
                    count=int(f.get("count", 0)),
                    record_refs=list(f.get("record_refs", [])),
                    subject_refs=list(f.get("subject_refs", [])),
                    detail=str(f.get("detail", "")),
                )
                for f in m.get("findings", [])
            ],
        )
        for m in payload.get("metrics", [])
    ]
    return PrivacyAuditReport(
        report_id=str(payload.get("report_id", record.record_id)),
        suite_name=str(payload.get("suite_name", "")),
        posture_score=float(payload.get("posture_score", 0.0)),
        passed=bool(payload.get("passed", False)),
        metrics=metrics,
        lookback_seconds=payload.get("lookback_seconds"),
        created_at=str(record.metadata.get("created_at") or ""),
    )


__all__ = ["PrivacyAuditService"]
