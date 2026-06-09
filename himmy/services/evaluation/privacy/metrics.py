"""WS4.7 — deterministic recorded-data privacy metrics (Part B, B1).

The recorded-data half of the privacy audit scans what is **already persisted** and
scores it for compliance. Every metric here is a pure, **synchronous** function over a
pre-materialized :class:`~himmy.services.evaluation.privacy.models.RecordedDataContext`
— it never ``await``s the async :class:`~himmy.services.storage.service.StorageService`
(must_fix #1) and never tries to read runs / memory / episodic via
``entity_registry.list_by_kind`` (those kinds are *not* projected as first-class
registry records — must_fix #2). The service ``await``s storage *first*, then hands the
metrics concrete lists; the registry is consulted **only** for the kinds that genuinely
are first-class records (``security_event`` / ``erasure_tombstone`` / ``consent`` /
``privacy_audit_report``).

Five deterministic metrics ship here:

* :class:`PiiLeakageMetric` — DLP scan of recorded outputs/payloads using the combined
  ``_PII_RULES + _NEPAL_PII_RULES`` rule set (must_fix #6), reporting **counts-only**
  findings (no raw value ever — must_fix #7/#8).
* :class:`AuditIntegrityMetric` — re-derives each scanned record's
  :func:`~himmy.entities.integrity.content_hash` against a caller-supplied baseline; the
  **first run has no baseline ⇒ the metric is skipped** (no false fail).
* :class:`RetentionComplianceMetric` — flags any record older than ``max_age_seconds``
  via :meth:`~himmy.services.governance.retention.RetentionService.expired` (its args are
  keyword-only).
* :class:`ErasureCompletenessMetric` — for every ``erasure_tombstone`` checks the
  subject's key is shredded (``key_vault.has(subject) is False``), the tombstone is
  present, **no** post-erasure record still references the subject, and (optionally) a
  signed bundle still verifies.
* The consent-derived metrics ``consent_coverage`` / ``purpose_limitation`` /
  ``denial_enforcement`` are **defined but None-guarded ⇒ ``skipped=True``** until Part
  A's ``ConsentService`` / ``denied_*`` events exist (must_fix #7); they are excluded
  from the posture roll-up rather than scored ``0``.

:func:`build_recorded_registry` registers only the metrics whose data source is actually
present, so the harness can run with whatever seams a deployment has wired.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from himmy.entities.integrity import content_hash
from himmy.entities.records import EntityRecord
from himmy.services.evaluation.privacy.models import (
    PrivacyFinding,
    PrivacyMetricResult,
    RecordedDataContext,
)
from himmy.services.governance.retention import ERASURE_KIND, RetentionService
from himmy.services.guardrails.builtins import _NEPAL_PII_RULES, _PII_RULES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable

    from himmy.entities.integrity import AuditBundle
    from himmy.services.governance.retention import SubjectKeyVault
    from himmy.services.guardrails.builtins import PIIRule
    from himmy.services.storage.models import RunRecord

#: The combined DLP rule set the leakage scan uses: the generic rules AND the
#: Nepal-specific rules are SEPARATE lists in :mod:`himmy.services.guardrails.builtins`,
#: so they must be concatenated for full coverage (must_fix #6). The Nepal-specific rules
#: are placed FIRST so a domestic mobile (e.g. ``9812345678``) is labelled ``np_mobile``
#: rather than being swallowed by the generic, looser ``phone`` rule (the scan redacts
#: each validated match in place, so whichever rule fires first claims the digits).
_LEAKAGE_RULES: list[PIIRule] = [*_NEPAL_PII_RULES, *_PII_RULES]

#: Default pass bar shared by the recorded-data metrics (the audit config / suite can
#: override per-metric; this is only used when a metric is asked to self-evaluate).
_DEFAULT_THRESHOLD = 0.9


@runtime_checkable
class RecordedDataMetric(Protocol):
    """A deterministic, synchronous privacy metric over already-recorded data.

    A recorded-data metric is fed the pre-materialized
    :class:`~himmy.services.evaluation.privacy.models.RecordedDataContext` and returns a
    :class:`~himmy.services.evaluation.privacy.models.PrivacyMetricResult` *synchronously*
    — it must never ``await`` an async store (must_fix #1). ``name`` matches the metric
    key used in the suite weighting / config thresholds.
    """

    name: str

    def score(self, ctx: RecordedDataContext) -> PrivacyMetricResult:
        """Score ``ctx`` and return a PII-safe metric result (no raw values)."""
        ...


# --------------------------------------------------------------------------- helpers


def _now_epoch() -> float:
    """Current wall-clock time as a UTC epoch (the default retention clock)."""
    return datetime.now(UTC).timestamp()


def _texts_of_run(run: RunRecord) -> list[str]:
    """Every free-text surface of a run that could leak PII (output + error)."""
    out: list[str] = []
    if run.output_text:
        out.append(run.output_text)
    if run.error:
        out.append(run.error)
    if run.output_structured is not None:
        out.append(_flatten(run.output_structured))
    return out


def _flatten(value: object) -> str:
    """Render an arbitrary structured value into one searchable string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _count_pii(text: str) -> dict[str, int]:
    """Return per-label match counts for ``text`` — **counts only, never values**.

    Mirrors :class:`~himmy.services.guardrails.dlp.DlpGuardrail`'s detection loop
    (validator-gated, no overlap double-counting because the rules are ordered most-
    specific → loosest) but discards the matched strings entirely so nothing leaks into
    a finding (must_fix #7). Each validated match is replaced with its rule placeholder
    before the next (looser) rule scans, so a broad phone rule can't re-count digits a
    specific card / IP rule already claimed.
    """
    counts: dict[str, int] = {}
    redacted = text
    for rule in _LEAKAGE_RULES:
        hit_count = 0

        def _replace(match: re.Match[str], r: PIIRule = rule) -> str:
            nonlocal hit_count
            if r.validator is not None and not r.validator(match.group(0)):
                return match.group(0)
            hit_count += 1
            return r.placeholder

        redacted = rule.pattern.sub(_replace, redacted)
        if hit_count:
            counts[rule.label] = counts.get(rule.label, 0) + hit_count
    return counts


# --------------------------------------------------------------------- pii_leakage


class PiiLeakageMetric:
    """1 − leak-rate over recorded outputs (DLP over ``_PII_RULES + _NEPAL_PII_RULES``).

    Scans every recorded run's output/error/structured surface for PII using the
    combined rule set. The score is ``1 − (leaking outputs / scanned outputs)`` so a
    clean store scores ``1.0`` and every leak drags it down proportionally. Findings
    carry the offending labels and ``run_id`` reference + match counts **only** — the
    raw matched string is never put in the finding (must_fix #7/#8).
    """

    name = "pii_leakage"

    def __init__(self, *, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold

    def score(self, ctx: RecordedDataContext) -> PrivacyMetricResult:
        """Score recorded outputs; emit a counts-only finding per leaking run."""
        scanned = 0
        leaking = 0
        findings: list[PrivacyFinding] = []
        total_hits = 0

        for run in ctx.run_records:
            for text in _texts_of_run(run):
                scanned += 1
                counts = _count_pii(text)
                if not counts:
                    continue
                leaking += 1
                hits = sum(counts.values())
                total_hits += hits
                findings.append(
                    PrivacyFinding(
                        code="pii_in_output",
                        severity="high",
                        count=hits,
                        record_refs=[run.run_id],
                        subject_refs=[run.subject_id] if run.subject_id else [],
                        # Labels only (category), never the matched value.
                        detail=",".join(sorted(counts)),
                    )
                )

        score = 1.0 if scanned == 0 else 1.0 - (leaking / scanned)
        detail = (
            "no recorded outputs to scan"
            if scanned == 0
            else f"{leaking}/{scanned} outputs leaked PII ({total_hits} matches)"
        )
        return PrivacyMetricResult(
            metric=self.name,
            kind="recorded",
            score=score,
            passed=score >= self._threshold,
            threshold=self._threshold,
            detail=detail,
            findings=findings,
        )


# ------------------------------------------------------------------ audit_integrity


class AuditIntegrityMetric:
    """Re-derive ``content_hash`` for scanned records against a signed baseline.

    The metric needs a trusted "as-of" hash baseline (``record_id -> content_hash``,
    typically the last signed :class:`~himmy.entities.integrity.AuditBundle`). The
    **first ever run has no baseline ⇒ the metric reports ``skipped=True``** so a fresh
    deployment is not falsely failed (must_fix: first run ⇒ skipped). When a baseline is
    present, any record whose live content hash differs (in-place mutation) drags the
    score and raises a tamper finding by ``record_id`` only.
    """

    name = "audit_integrity"

    def __init__(
        self,
        *,
        baseline: AuditBundle | dict[str, str] | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._baseline = _baseline_hashes(baseline)
        self._threshold = threshold

    def score(self, ctx: RecordedDataContext) -> PrivacyMetricResult:
        """Compare live registry hashes to the baseline; skip when no baseline."""
        if self._baseline is None:
            return PrivacyMetricResult(
                metric=self.name,
                kind="recorded",
                skipped=True,
                threshold=self._threshold,
                detail="no signed baseline yet (first run) — integrity check skipped",
            )

        records = _registry_records(ctx)
        live = {r.record_id: content_hash(r) for r in records}
        checked = 0
        tampered: list[str] = []
        for rid, expected in self._baseline.items():
            actual = live.get(rid)
            if actual is None:
                # A baselined record that is no longer present (covered by retention /
                # erasure metrics); not a content tamper, so don't penalise here.
                continue
            checked += 1
            if actual != expected:
                tampered.append(rid)

        score = 1.0 if checked == 0 else 1.0 - (len(tampered) / checked)
        findings = (
            [
                PrivacyFinding(
                    code="audit_record_tampered",
                    severity="critical",
                    count=len(tampered),
                    record_refs=sorted(tampered),
                    detail="content_hash diverged from signed baseline",
                )
            ]
            if tampered
            else []
        )
        detail = (
            "no baselined records present to verify"
            if checked == 0
            else f"{len(tampered)}/{checked} records diverged from baseline"
        )
        return PrivacyMetricResult(
            metric=self.name,
            kind="recorded",
            score=score,
            passed=score >= self._threshold,
            threshold=self._threshold,
            detail=detail,
            findings=findings,
        )


def _baseline_hashes(
    baseline: AuditBundle | dict[str, str] | None,
) -> dict[str, str] | None:
    """Normalise a bundle / mapping / None into a ``record_id -> hash`` map or None."""
    if baseline is None:
        return None
    if isinstance(baseline, dict):
        return dict(baseline)
    return dict(baseline.records)


def _registry_records(ctx: RecordedDataContext) -> list[EntityRecord]:
    """The first-class registry records in scope for integrity / erasure checks.

    Only the kinds that ARE projected as first-class records are read from the registry
    (``security_event`` / ``erasure_tombstone`` / ``consent`` / ``privacy_audit_report``)
    — runs / memory / episodic are never read this way (must_fix #2).
    """
    if ctx.entity_registry is None:
        return []
    kinds = (
        "security_event",
        ERASURE_KIND,
        "consent",
        "privacy_audit_report",
    )
    out: list[EntityRecord] = []
    for kind in kinds:
        out.extend(ctx.entity_registry.list_by_kind(kind))
    return out


# ------------------------------------------------------------ retention_compliance


class RetentionComplianceMetric:
    """No recorded operational record may live past ``max_age_seconds``.

    Uses :meth:`~himmy.services.governance.retention.RetentionService.expired`
    (keyword-only ``max_age_seconds`` / ``now_epoch``) over the pre-fetched runs +
    memory + episodic. The score is ``1 − (expired / scanned)``; an over-age record
    raises a finding by id only. ``now_epoch`` defaults to wall-clock time but is
    injectable for deterministic tests.
    """

    name = "retention_compliance"

    def __init__(
        self,
        *,
        retention_service: RetentionService,
        max_age_seconds: float,
        now_epoch: Callable[[], float] | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._retention = retention_service
        self._max_age = max_age_seconds
        self._now = now_epoch or _now_epoch
        self._threshold = threshold

    def score(self, ctx: RecordedDataContext) -> PrivacyMetricResult:
        """Flag any pre-fetched record older than the retention window."""
        now = self._now()
        scanned: list[object] = [*ctx.run_records, *ctx.memory, *ctx.episodic]
        expired = self._retention.expired(
            scanned, max_age_seconds=self._max_age, now_epoch=now
        )
        total = len(scanned)
        score = 1.0 if total == 0 else 1.0 - (len(expired) / total)
        findings = (
            [
                PrivacyFinding(
                    code="record_past_retention",
                    severity="medium",
                    count=len(expired),
                    record_refs=sorted(_ref_ids(expired)),
                    subject_refs=sorted(_subject_ids(expired)),
                    detail=f"older than {self._max_age:.0f}s retention window",
                )
            ]
            if expired
            else []
        )
        detail = (
            "no records in scope"
            if total == 0
            else f"{len(expired)}/{total} records past retention"
        )
        return PrivacyMetricResult(
            metric=self.name,
            kind="recorded",
            score=score,
            passed=score >= self._threshold,
            threshold=self._threshold,
            detail=detail,
            findings=findings,
        )


def _ref_ids(records: Iterable[object]) -> list[str]:
    """Best-effort id of each record (``run_id`` / ``memory_id`` / ``episode_id``)."""
    ids: list[str] = []
    for record in records:
        for attr in ("run_id", "memory_id", "episode_id", "record_id"):
            value = getattr(record, attr, None)
            if value:
                ids.append(str(value))
                break
    return ids


def _subject_ids(records: Iterable[object]) -> list[str]:
    """The distinct (truthy) subject ids attached to ``records``."""
    seen: set[str] = set()
    for record in records:
        subject = getattr(record, "subject_id", None)
        if subject:
            seen.add(str(subject))
    return list(seen)


# ----------------------------------------------------------- erasure_completeness


class ErasureCompletenessMetric:
    """Every erased subject is provably gone: key shredded + no surviving references.

    For each ``erasure_tombstone`` in the registry the metric checks that (1) the
    subject's key is shredded (``key_vault.has(subject) is False``), (2) the tombstone
    record is present (it is, by construction), (3) **no** post-erasure record still
    references the subject in cleartext (pre-fetched runs/memory/episodic + the
    first-class registry records), and (optionally) (4) a supplied signed bundle still
    verifies. Each incomplete erasure raises a finding referencing the subject id only.
    """

    name = "erasure_completeness"

    def __init__(
        self,
        *,
        key_vault: SubjectKeyVault,
        verify_bundle: Callable[[], bool] | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._vault = key_vault
        self._verify_bundle = verify_bundle
        self._threshold = threshold

    def score(self, ctx: RecordedDataContext) -> PrivacyMetricResult:
        """Verify each tombstone's erasure is complete; skip when no tombstones."""
        tombstones = (
            ctx.entity_registry.list_by_kind(ERASURE_KIND)
            if ctx.entity_registry is not None
            else []
        )
        if not tombstones:
            return PrivacyMetricResult(
                metric=self.name,
                kind="recorded",
                skipped=True,
                threshold=self._threshold,
                detail="no erasure tombstones to verify",
            )

        referencing = _referencing_subjects(ctx)
        bundle_ok = self._verify_bundle() if self._verify_bundle is not None else True

        incomplete: list[PrivacyFinding] = []
        verified = 0
        for tombstone in tombstones:
            subject = str(tombstone.payload.get("subject_id", ""))
            if not subject:
                continue
            verified += 1
            problems: list[str] = []
            if self._vault.has(subject):
                problems.append("key_not_shredded")
            survivors = referencing.get(subject, [])
            if survivors:
                problems.append("post_erasure_reference")
            if not bundle_ok:
                problems.append("bundle_verify_failed")
            if problems:
                incomplete.append(
                    PrivacyFinding(
                        code="incomplete_erasure",
                        severity="critical",
                        count=len(problems),
                        subject_refs=[subject],
                        record_refs=sorted(survivors),
                        detail=",".join(problems),
                    )
                )

        score = 1.0 if verified == 0 else 1.0 - (len(incomplete) / verified)
        detail = (
            "no subjects erased"
            if verified == 0
            else f"{len(incomplete)}/{verified} erasures incomplete"
        )
        return PrivacyMetricResult(
            metric=self.name,
            kind="recorded",
            score=score,
            passed=score >= self._threshold,
            threshold=self._threshold,
            detail=detail,
            findings=incomplete,
        )


def _referencing_subjects(ctx: RecordedDataContext) -> dict[str, list[str]]:
    """Map ``subject_id -> [record ids]`` for every record still referencing a subject.

    Spans both the pre-fetched operational records (runs/memory/episodic) and the
    first-class registry records (whose ``payload.subject_id`` is checked) — so a
    surviving cleartext reference anywhere is caught.
    """
    refs: dict[str, list[str]] = {}

    def _add(subject: object, ref: object) -> None:
        if subject and ref:
            refs.setdefault(str(subject), []).append(str(ref))

    for run in ctx.run_records:
        _add(run.subject_id, run.run_id)
    for memory in ctx.memory:
        _add(memory.subject_id, memory.memory_id)
    for episode in ctx.episodic:
        _add(episode.subject_id, episode.episode_id)

    if ctx.entity_registry is not None:
        for kind in ("security_event", "consent", "privacy_audit_report"):
            for record in ctx.entity_registry.list_by_kind(kind):
                _add(record.payload.get("subject_id"), record.record_id)
    return refs


# ------------------------------------------------ consent-derived (None-guarded)


class _ConsentMetric:
    """A consent-derived metric that stays SKIPPED until Part A is wired (must_fix #7).

    ``consent_coverage`` / ``purpose_limitation`` / ``denial_enforcement`` all need
    Part A's ``ConsentService`` (the ledger seam) and the ``denied_*`` security events
    it emits. Until that seam is present (``ctx.consent_service is None``) the metric
    reports ``skipped=True`` — it is excluded from the posture roll-up rather than
    scored a misleading ``0`` (must_fix #7). The scoring body is intentionally a
    placeholder: it is only reached once Part A injects a non-None ``consent_service``,
    at which point B-follow-up work fills it in. Today it conservatively returns a
    skipped result even with a service present, so it can never silently pass.
    """

    def __init__(self, name: str, *, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self.name = name
        self._threshold = threshold

    def score(self, ctx: RecordedDataContext) -> PrivacyMetricResult:
        """Skip when no consent seam; otherwise (Part A) defer to the follow-up impl."""
        reason = (
            "consent service not wired (Part A) — metric skipped"
            if ctx.consent_service is None
            else "consent-derived scoring lands with Part A wiring"
        )
        return PrivacyMetricResult(
            metric=self.name,
            kind="recorded",
            skipped=True,
            threshold=self._threshold,
            detail=reason,
        )


def consent_coverage_metric() -> _ConsentMetric:
    """Fraction of persisted subjects with an explicit consent record (Part A)."""
    return _ConsentMetric("consent_coverage")


def purpose_limitation_metric() -> _ConsentMetric:
    """No subject's data used beyond its consented purposes (Part A)."""
    return _ConsentMetric("purpose_limitation")


def denial_enforcement_metric() -> _ConsentMetric:
    """Every denied purpose has a matching ``denied_*`` enforcement event (Part A)."""
    return _ConsentMetric("denial_enforcement")


# --------------------------------------------------------------------- registry


def build_recorded_registry(
    *,
    security_audit: object | None = None,
    consent_service: object | None = None,
    retention_service: RetentionService | None = None,
    dlp: object | None = None,
    key_vault: SubjectKeyVault | None = None,
    audit_baseline: AuditBundle | dict[str, str] | None = None,
    verify_bundle: Callable[[], bool] | None = None,
    max_age_seconds: float | None = None,
    now_epoch: Callable[[], float] | None = None,
) -> list[RecordedDataMetric]:
    """Register only the recorded-data metrics whose data source is present.

    Mirrors how the rest of himmy turns capabilities off when nothing is configured:
    a metric is included **only** when its required seam is wired, so an audit run
    scores exactly what the deployment can actually substantiate. ``pii_leakage`` and
    ``audit_integrity`` always apply (they read only pre-fetched outputs / the
    registry); the others are gated on their service.

    Args:
        security_audit: The security-audit sink/log (present ⇒ enable
            ``audit_integrity``; even absent, integrity still runs over the registry,
            so this only documents intent — the real gate is ``audit_baseline``).
        consent_service: The Part A consent ledger; ``None`` ⇒ the consent-derived
            metrics report ``skipped`` (must_fix #7).
        retention_service: Enables ``retention_compliance`` (needs ``max_age_seconds``).
        dlp: The DLP guardrail (informational; ``pii_leakage`` uses the rule set
            directly so it needs no live guardrail instance).
        key_vault: Enables ``erasure_completeness`` (the shred check).
        audit_baseline: The last signed hash baseline; ``None`` ⇒ ``audit_integrity``
            self-skips on first run.
        verify_bundle: Optional bundle re-verification callable for erasure-completeness.
        max_age_seconds: The retention window for ``retention_compliance``.
        now_epoch: Injectable clock for deterministic retention tests.

    Returns:
        The ordered list of registered :class:`RecordedDataMetric`s.
    """
    metrics: list[RecordedDataMetric] = [
        PiiLeakageMetric(),
        AuditIntegrityMetric(baseline=audit_baseline),
    ]

    if retention_service is not None and max_age_seconds is not None:
        metrics.append(
            RetentionComplianceMetric(
                retention_service=retention_service,
                max_age_seconds=max_age_seconds,
                now_epoch=now_epoch,
            )
        )

    if key_vault is not None:
        metrics.append(
            ErasureCompletenessMetric(key_vault=key_vault, verify_bundle=verify_bundle)
        )

    # Consent-derived metrics are always *defined*; they self-skip until Part A wires
    # ``consent_service`` (must_fix #7). They are registered so the scorecard records
    # them as explicitly skipped rather than silently absent.
    metrics.extend(
        (
            consent_coverage_metric(),
            purpose_limitation_metric(),
            denial_enforcement_metric(),
        )
    )

    # ``security_audit`` / ``consent_service`` / ``dlp`` are accepted to match the
    # documented seam shape and to gate follow-up scoring; referencing them keeps the
    # signature honest without changing today's behaviour.
    _ = (security_audit, consent_service, dlp)
    return metrics


__all__ = [
    "RecordedDataMetric",
    "PiiLeakageMetric",
    "AuditIntegrityMetric",
    "RetentionComplianceMetric",
    "ErasureCompletenessMetric",
    "consent_coverage_metric",
    "purpose_limitation_metric",
    "denial_enforcement_metric",
    "build_recorded_registry",
]
