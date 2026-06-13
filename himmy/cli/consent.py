"""``himmy consent`` — record + inspect consent from the CLI (WS4.6 A3, offline parity).

Talks to a :class:`~himmy.services.governance.consent_ledger.ConsentLedger` directly over a
durable on-disk :class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry`
(``.himmy/consent.db``), so an operator gets the same grant/deny/status/history/revoke
surface the ``/v1/consent`` API exposes without running the server. The CLI always builds a
**governed** policy (consent governance is the whole point of the command) regardless of
``HIMMY_CONSENT``; the rest of himmy stays offline-first because the CLI ledger is isolated
to its own registry and nothing else reads it.

``revoke`` withdraws consent and routes through :class:`RetentionService` to write a signed
erasure tombstone. The per-subject key vault is in-process, so a CLI ``revoke`` records the
withdrawal + tombstone durably but reports ``crypto_shredded=False`` when the key was minted
in a different process (durable key vaults live behind the Postgres backend); the consent
*state* and the tombstone are what persist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from himmy.services.governance.consent import (
    ConsentPolicy,
    ConsentState,
    Purpose,
    build_consent_policy,
)


def _eprint(*args: Any) -> None:
    """Print diagnostics to stderr (machine-readable JSON stays on stdout)."""
    print(*args, file=sys.stderr)


def _consent_db() -> str:
    """Path to the durable consent registry (``.himmy/consent.db``), dir created."""
    path = Path(".himmy")
    path.mkdir(exist_ok=True)
    return str(path / "consent.db")


def _policy() -> ConsentPolicy:
    """A governed policy: honour ``HIMMY_CONSENT_FILE`` defaults but force governed on."""
    built = build_consent_policy()
    if built.governed:
        return built
    return ConsentPolicy(governed=True, defaults=built.defaults)


def _build_ledger() -> Any:
    """Construct a durable :class:`ConsentLedger` + erasure service for the CLI.

    Uses the on-disk :class:`SqliteEntityRegistry`, which satisfies
    :class:`EntityRegistryProtocol` — the structural surface the ledger/retention services
    accept (``new_version``/``get_latest``/``get_history``/``query`` …) — so the durable
    spine drops in with no cast (the registry Protocol that closes the old gap).
    """
    from himmy.entities.sqlite_registry import SqliteEntityRegistry
    from himmy.services.governance.consent_ledger import ConsentLedger
    from himmy.services.governance.retention import RetentionService, SubjectKeyVault

    registry = SqliteEntityRegistry(_consent_db())
    retention = RetentionService(registry, key_vault=SubjectKeyVault())
    return ConsentLedger(registry, policy=_policy(), retention_service=retention)


def _purpose(value: str) -> Purpose:
    """Parse a ``--purpose`` token into a :class:`Purpose` (clear error otherwise)."""
    try:
        return Purpose(value.lower())
    except ValueError as exc:
        choices = ", ".join(p.value for p in Purpose)
        raise SystemExit(
            f"error: unknown purpose {value!r} (choose: {choices})"
        ) from exc


def _print_record(record: Any, *, as_json: bool) -> None:
    """Emit a :class:`ConsentRecord` as JSON or a one-line human summary."""
    data = record.model_dump(mode="json")
    if as_json:
        print(json.dumps(data))
    else:
        print(
            f"{data['subject_id']} | {data['purpose']} | {data['state']}"
            f" | {data['recorded_at']}"
        )


# --------------------------------------------------------------------------- commands
def cmd_consent_grant(args: argparse.Namespace) -> int:
    """Record a GRANTED consent for ``(subject, purpose)``."""
    ledger = _build_ledger()
    purpose = _purpose(args.purpose)
    ledger.grant(
        args.subject,
        purpose,
        actor=args.actor or "",
        source="cli",
        basis=args.basis,
    )
    _print_record(ledger.latest(args.subject, purpose), as_json=args.json)
    return 0


def cmd_consent_deny(args: argparse.Namespace) -> int:
    """Record a DENIED consent for ``(subject, purpose)``."""
    ledger = _build_ledger()
    purpose = _purpose(args.purpose)
    ledger.deny(
        args.subject,
        purpose,
        actor=args.actor or "",
        source="cli",
        basis=args.basis,
    )
    _print_record(ledger.latest(args.subject, purpose), as_json=args.json)
    return 0


def cmd_consent_status(args: argparse.Namespace) -> int:
    """Print the PDP decision for ``(subject, purpose)`` (its state + effect)."""
    ledger = _build_ledger()
    purpose = _purpose(args.purpose)
    state: ConsentState = ledger.state(args.subject, purpose)
    decision = ledger.decision(args.subject, purpose)
    payload = {
        "subject_id": args.subject,
        "purpose": purpose.value,
        "state": state.value,
        "effect": decision.effect.value,
        "allowed": decision.allowed,
        "reason": decision.reason,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(
            f"{args.subject} | {purpose.value} | state={state.value}"
            f" | effect={decision.effect.value} | allowed={decision.allowed}"
        )
    return 0


def cmd_consent_history(args: argparse.Namespace) -> int:
    """Print the full append-only version chain for ``(subject, purpose)``."""
    ledger = _build_ledger()
    records = ledger.history(args.subject, _purpose(args.purpose))
    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in records]))
    else:
        if not records:
            _eprint("(no consent records)")
        for record in records:
            _print_record(record, as_json=False)
    return 0


def cmd_consent_revoke(args: argparse.Namespace) -> int:
    """Withdraw consent (all purposes by default) and write an erasure tombstone."""
    from himmy.services.governance.consent import ConsentRecord

    ledger = _build_ledger()
    purpose = _purpose(args.purpose) if args.purpose else None
    # ``withdraw`` returns the raw EntityRecord versions; project them back to the typed
    # ConsentRecord (its payload schema) for a uniform output with the other commands.
    records = [
        ConsentRecord.model_validate(r.payload)
        for r in ledger.withdraw(
            args.subject,
            purpose,
            reason=args.reason or "",
            actor=args.actor or "",
            source="cli",
        )
    ]
    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in records]))
    else:
        if not records:
            _eprint(f"(no consent on file for {args.subject!r}; nothing to withdraw)")
        for record in records:
            _print_record(record, as_json=False)
        _eprint(f"erasure tombstone written for {args.subject!r}")
    return 0


def add_consent_parser(sub: Any) -> None:
    """Register the ``consent`` subcommand tree on the CLI's subparsers."""
    p = sub.add_parser("consent", help="record + inspect data-subject consent (WS4.6)")
    p.set_defaults(func=lambda _args: (p.print_help(), 1)[1])
    csub = p.add_subparsers(dest="consent_command")

    def _common(parser: argparse.ArgumentParser, *, with_purpose: bool = True) -> None:
        parser.add_argument("subject", help="the data subject id")
        if with_purpose:
            parser.add_argument(
                "purpose",
                help="purpose: " + ", ".join(pp.value for pp in Purpose),
            )
        parser.add_argument("--json", action="store_true", help="machine-readable JSON")

    p_grant = csub.add_parser("grant", help="record a GRANTED consent")
    _common(p_grant)
    p_grant.add_argument("--actor", help="who recorded this decision")
    p_grant.add_argument("--basis", help="legal basis / free note")
    p_grant.set_defaults(func=cmd_consent_grant)

    p_deny = csub.add_parser("deny", help="record a DENIED consent")
    _common(p_deny)
    p_deny.add_argument("--actor", help="who recorded this decision")
    p_deny.add_argument("--basis", help="legal basis / free note")
    p_deny.set_defaults(func=cmd_consent_deny)

    p_status = csub.add_parser("status", help="print the PDP decision for a pair")
    _common(p_status)
    p_status.set_defaults(func=cmd_consent_status)

    p_history = csub.add_parser("history", help="print the consent version chain")
    _common(p_history)
    p_history.set_defaults(func=cmd_consent_history)

    p_revoke = csub.add_parser(
        "revoke", help="withdraw consent + write an erasure tombstone"
    )
    p_revoke.add_argument("subject", help="the data subject id")
    p_revoke.add_argument(
        "purpose",
        nargs="?",
        help="purpose to withdraw (default: every recorded purpose)",
    )
    p_revoke.add_argument("--reason", help="why consent is being withdrawn")
    p_revoke.add_argument("--actor", help="who recorded this withdrawal")
    p_revoke.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_revoke.set_defaults(func=cmd_consent_revoke)


__all__ = ["add_consent_parser"]
