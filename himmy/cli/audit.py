"""``himmy audit privacy`` — run the WS4.7 privacy & compliance audit from the CLI (B4).

Builds an OFFLINE-FIRST :class:`~himmy.services.evaluation.privacy.service.PrivacyAuditService`
over an in-memory eval kernel + registry and runs a **recorded-data scan** of what is persisted,
printing a posture scorecard (or ``--json``). It is a CI gate: it **exits non-zero when the
posture fails** (``report.passed`` is False), so a planted PII leak / retention gap / erasure gap
fails the build. With ``--export-bundle PATH`` it writes a tamper-evident signed bundle over the
registered report (Ed25519 when ``HIMMY_AUDIT_PRIVATE_KEY`` is set, else HMAC with
``HIMMY_AUDIT_SECRET``) and prints how to verify it.

What a *standalone* CLI run scans: himmy's default registry/storage are process-local and
in-memory (``ApiContainer`` builds an in-memory ``EntityRegistry``), so a bare
``himmy audit privacy`` scans an EMPTY store — that is the *clean* posture by construction. To
audit a deployment's REAL persisted posture, run the audit where that data lives: the HTTP
``POST /v1/audit/privacy`` route scans the live container's registry/storage, and an embedder can
construct ``PrivacyAuditService`` with its own wired registry/store. (A durable on-disk spine scan
is a future enhancement, gated on himmy persisting the entity registry durably by default.)

Offline-first: the command runs no model and needs no keys; the consent-derived metrics stay
``skipped`` until Part A is wired, and the LLM probe metrics stay off for the stub provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def _eprint(*args: Any) -> None:
    """Print diagnostics to stderr (machine-readable JSON stays on stdout)."""
    print(*args, file=sys.stderr)


def _build_service() -> Any:
    """Construct an offline :class:`PrivacyAuditService` for a recorded-data CI scan.

    Wires an in-memory :class:`EntityRegistry` (the spine the recorded metrics scan for the
    ``security_event`` / ``consent`` / ``erasure_tombstone`` / ``privacy_audit_report``
    kinds) + an in-memory :class:`StorageService` (the run/memory read side). An empty store
    is the clean posture; a deployment plants real data by handing the auditor its own wired
    registry/store (the BFF does exactly this — see :class:`~himmy.api.deps.ApiContainer`).
    No provider is wired, so the LLM probe metrics stay skipped (must_fix #5).

    NOTE: the registry is in-memory rather than the durable on-disk
    :class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry`. The old blocker — that
    :class:`~himmy.services.evaluation.privacy.models.RecordedDataContext` validated the
    registry against the concrete :class:`EntityRegistry` type — is GONE (that field is now
    typed against :class:`~himmy.entities.protocol.EntityRegistryProtocol`, so a durable
    registry validates in fine). Pointing this CLI scan at the durable shared spine is a
    deliberate follow-on (plan item T2.1, SpineFactory); the HTTP surface already scans the
    live container's registry.
    """
    from himmy.entities.registry import EntityRegistry
    from himmy.services.audit.log import SecurityAuditLog
    from himmy.services.evaluation.privacy import PrivacyAuditService
    from himmy.services.evaluation.service import EvaluationService
    from himmy.services.storage.service import StorageService

    registry = EntityRegistry()
    eval_service = EvaluationService(storage_service=StorageService())
    return PrivacyAuditService(
        entity_registry=registry,
        evaluation_service=eval_service,
        security_audit=SecurityAuditLog(registry),
    )


def _print_scorecard(report: Any) -> None:
    """Emit a human-readable posture scorecard for one report (stdout)."""
    verdict = "PASS" if report.passed else "FAIL"
    suite = report.suite_name or "(recorded-only)"
    print(f"privacy audit: {verdict}   posture={report.posture_score:.3f}   {suite}")
    for m in report.metrics:
        if m.skipped:
            mark = "SKIP"
        elif m.passed:
            mark = "ok  "
        else:
            mark = "FAIL"
        print(
            f"  [{mark}] {m.metric:<22} {m.score:.2f} (>= {m.threshold:.2f}) {m.detail}"
        )
        for f in m.findings:
            refs = f"refs={len(f.record_refs)}" if f.record_refs else ""
            subs = f"subjects={len(f.subject_refs)}" if f.subject_refs else ""
            print(f"        - {f.severity}:{f.code} x{f.count} {refs} {subs}".rstrip())


def cmd_audit_privacy(args: argparse.Namespace) -> int:
    """Run the privacy audit; exit non-zero when the posture fails (a CI gate).

    Builds the offline service, runs a recorded-data scan (``--lookback`` bounds the
    window), prints the scorecard (or ``--json``), optionally writes a signed bundle on
    ``--export-bundle``, and returns ``0`` only when ``report.passed`` is True.
    """
    service = _build_service()
    report = asyncio.run(service.run(lookback_seconds=args.lookback))

    if args.export_bundle:
        rc = _export_bundle(service, report, args.export_bundle)
        if rc != 0:
            return rc

    if args.json:
        print(json.dumps(report.model_dump(mode="json")))
    else:
        _print_scorecard(report)

    return 0 if report.passed else 1


def _export_bundle(service: Any, report: Any, path: str) -> int:
    """Write a signed bundle over ``report`` to ``path`` (1 on a missing signing key)."""
    from himmy.core.errors import HimmyError

    try:
        bundle = service.export_signed_bundle(report)
    except HimmyError as exc:
        _eprint(f"error: {exc}")
        return 1
    Path(path).write_text(json.dumps(bundle.model_dump(mode="json"), indent=2))
    _eprint(
        f"signed audit bundle written to {path} "
        f"(algorithm={bundle.algorithm}, records={len(bundle.records)})"
    )
    return 0


def add_audit_parser(sub: Any) -> None:
    """Register the ``audit`` subcommand tree on the CLI's subparsers (WS4.7 B4)."""
    p = sub.add_parser("audit", help="run himmy's privacy & compliance audit (WS4.7)")
    p.set_defaults(func=lambda _args: (p.print_help(), 1)[1])
    asub = p.add_subparsers(dest="audit_command")

    p_priv = asub.add_parser(
        "privacy",
        help="run a privacy & compliance audit; exits non-zero on a failing posture",
    )
    p_priv.add_argument(
        "--lookback",
        type=float,
        default=None,
        help="recorded-data scan window in seconds (default: unbounded)",
    )
    p_priv.add_argument(
        "--export-bundle",
        metavar="PATH",
        help="write a verifiable signed bundle over the registered report to PATH",
    )
    p_priv.add_argument(
        "--json", action="store_true", help="machine-readable JSON scorecard"
    )
    p_priv.set_defaults(func=cmd_audit_privacy)


__all__ = ["add_audit_parser", "cmd_audit_privacy"]
