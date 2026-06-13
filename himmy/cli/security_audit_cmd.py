"""``himmy seclog`` — surface the SECURITY audit log from a durable entity spine.

Distinct from the privacy ``audit`` command (which scores a posture): this lists the
recent :class:`~himmy.services.audit.models.SecurityEvent` records the CLI emits —
denied tools, approvals, policy blocks, auth/authz decisions — straight off the
tamper-evident entity registry (kind ``security_event``). Each row is who/what/when:
``time · actor · action · resource · outcome``.

The CLI keeps a **durable** security spine at ``.himmy/spine.db`` (a
:class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry`, mirroring the
``himmy consent`` durable-db pattern). The real decision points record into it via
:func:`cli_security_log`: an RBAC deny in ``enforce_role_on_registry`` /
``prompt_approval`` and an approval grant/reject at the HITL prompt. ``himmy seclog``
reads that SAME on-disk spine, so after a real denial it shows the ``authz_denied``
row — the audit trail persists across processes.

To inspect a live in-process session's events, the ``/seclog`` REPL slash also reads
the durable spine; an embedder may hand :func:`render_seclog` its own wired
:class:`SecurityAuditLog`. Recording NEVER gates a run: every record path fails open on
a logging error (logging is the audit, not the access control).

Offline-first: no model, no keys, no network. Output is decoration-free on non-TTY/pipes
(``--json`` stays byte-identical for scripts), so CI and shell pipelines never hang or get ANSI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from himmy.cli.ui import styles

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.audit.log import SecurityAuditLog
    from himmy.services.audit.models import SecurityEvent

#: Column width caps so a long resource/actor cannot smear the table.
_ACTOR_CAP = 22
_ACTION_CAP = 18
_RESOURCE_CAP = 30


def _eprint(*args: Any) -> None:
    """Print diagnostics to stderr (machine-readable JSON stays on stdout)."""
    print(*args, file=sys.stderr)


def _spine_db() -> str:
    """Path to the durable CLI security spine (``.himmy/spine.db``), dir created.

    Mirrors :func:`himmy.cli.consent._consent_db`: a per-project on-disk registry so
    security events persist across CLI invocations (a deny recorded by ``himmy run``
    is visible to a later ``himmy seclog``).
    """
    path = Path(".himmy")
    path.mkdir(exist_ok=True)
    return str(path / "spine.db")


def cli_security_log() -> SecurityAuditLog:
    """A :class:`SecurityAuditLog` over the durable CLI spine (``.himmy/spine.db``).

    The single durable security log the CLI both WRITES (at RBAC deny / approval
    decision points) and READS (``himmy seclog`` / ``/seclog``). Built over the on-disk
    :class:`SqliteEntityRegistry`, which satisfies :class:`EntityRegistryProtocol` —
    the structural surface :class:`SecurityAuditLog` accepts — so the durable spine drops
    in with no cast (the registry Protocol that closes the old concrete-vs-Sqlite gap).
    """
    from himmy.entities.sqlite_registry import SqliteEntityRegistry
    from himmy.services.audit.log import SecurityAuditLog

    return SecurityAuditLog(SqliteEntityRegistry(_spine_db()))


def record_security_event(
    *,
    event_type: str,
    outcome: str,
    actor: dict[str, Any] | None = None,
    resource: str | None = None,
    action: str | None = None,
    detail: str = "",
) -> None:
    """Record one :class:`SecurityEvent` to the durable CLI spine — fail open on error.

    The recording hook the RBAC deny sites and the approval prompt call. It NEVER
    raises into a run: any failure (a locked/odd db, an import hiccup) is swallowed
    so logging can never become the gate. A best-effort audit write, not access control.
    """
    try:
        from himmy.services.audit.models import SecurityEvent

        cli_security_log().record(
            SecurityEvent(
                event_type=event_type,
                outcome=outcome,
                actor=actor or {},
                resource=resource,
                action=action,
                detail=detail,
            )
        )
    except Exception:  # noqa: BLE001 - logging must never break a run (fail open)
        pass


def _truncate(text: str, cap: int) -> str:
    """Clip ``text`` to ``cap`` chars with an ellipsis (never raises on None)."""
    text = text or ""
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _short_time(created_at: str) -> str:
    """``2026-06-11T14:03:22.5Z`` → ``2026-06-11 14:03:22`` (best-effort, plain on odd input)."""
    t = created_at or ""
    if "T" in t:
        date, _, rest = t.partition("T")
        clock = rest.split(".")[0].split("+")[0].rstrip("Z")
        return f"{date} {clock}".rstrip()
    return t


def _actor_label(event: SecurityEvent) -> str:
    """The principal that acted: the actor ``subject`` (or ``-`` when unknown)."""
    actor = event.actor or {}
    subject = actor.get("subject") or actor.get("id") or actor.get("name")
    return str(subject) if subject else "-"


def _outcome_style(c: dict[str, str], outcome: str) -> str:
    """Green for ``allow``, crimson for ``deny``, gold for anything else."""
    if outcome == "allow":
        return c["green"]
    if outcome == "deny":
        return c["crimson"]
    return c["gold"]


def _event_dict(event: SecurityEvent) -> dict[str, Any]:
    """A flat, machine-stable row for ``--json`` (full event, not the clipped table)."""
    return event.model_dump(mode="json")


def render_seclog(
    audit: SecurityAuditLog,
    *,
    limit: int = 20,
    event_type: str | None = None,
    workspace_id: str | None = None,
    as_json: bool = False,
    stream: Any = None,
) -> int:
    """Render the recent security events from ``audit`` to stdout; return an exit code.

    Reusable by both the ``seclog`` subcommand and the ``/seclog`` REPL slash. ``--json``
    emits the full events (no styling, no clipping); the human table is styled to the
    stream (plain on a pipe). Always returns ``0`` — reading an empty trail is success,
    not failure (that is the clean posture, like the privacy audit's empty store).
    """
    out = stream if stream is not None else sys.stdout
    events = audit.recent(
        limit=max(0, limit), event_type=event_type, workspace_id=workspace_id
    )

    if as_json:
        print(json.dumps([_event_dict(e) for e in events]), file=out)
        return 0

    c = styles(out)
    if not events:
        print(
            f"{c['dim']}no security events recorded "
            f"(clean trail — .himmy/spine.db){c['reset']}",
            file=out,
        )
        return 0

    header = (
        f"{c['faint']}{'time':<19}  {'actor':<{_ACTOR_CAP}}  "
        f"{'action':<{_ACTION_CAP}}  {'resource':<{_RESOURCE_CAP}}  "
        f"outcome{c['reset']}"
    )
    print(header, file=out)
    for e in events:
        action = _truncate(e.action or e.event_type or "-", _ACTION_CAP)
        resource = _truncate(e.resource or e.path or "-", _RESOURCE_CAP)
        actor = _truncate(_actor_label(e), _ACTOR_CAP)
        oc = e.outcome or "-"
        print(
            f"{c['dim']}{_short_time(e.created_at):<19}{c['reset']}  "
            f"{c['ink']}{actor:<{_ACTOR_CAP}}{c['reset']}  "
            f"{c['gold']}{action:<{_ACTION_CAP}}{c['reset']}  "
            f"{c['snow']}{resource:<{_RESOURCE_CAP}}{c['reset']}  "
            f"{_outcome_style(c, oc)}{oc}{c['reset']}",
            file=out,
        )
    return 0


def cmd_seclog(args: argparse.Namespace) -> int:
    """``himmy seclog`` — list recent security events from the spine.

    Reads the durable CLI spine (``.himmy/spine.db``), then defers to :func:`render_seclog`
    so the table/JSON rendering is shared with the ``/seclog`` REPL slash. ``--limit``
    bounds the rows (newest first); ``--type`` filters by event type
    (``auth_failure``/``authz_denied``/``access``/``admin``); ``--json`` is machine-readable.
    """
    audit = cli_security_log()
    return render_seclog(
        audit,
        limit=args.limit,
        event_type=getattr(args, "type", None),
        workspace_id=getattr(args, "workspace", None),
        as_json=args.json,
    )


def add_seclog_parser(sub: Any) -> None:
    """Register the ``seclog`` subcommand on the CLI's subparsers."""
    p = sub.add_parser(
        "seclog",
        help="list recent SECURITY audit events (denied tools, approvals, policy blocks)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="max events to show, newest first (default: 20)",
    )
    p.add_argument(
        "--type",
        dest="type",
        default=None,
        help="filter by event_type (auth_failure|authz_denied|access|admin)",
    )
    p.add_argument(
        "--workspace",
        dest="workspace",
        default=None,
        help="filter to one workspace_id",
    )
    p.add_argument("--json", action="store_true", help="machine-readable JSON events")
    p.set_defaults(func=cmd_seclog)


__all__ = [
    "add_seclog_parser",
    "cmd_seclog",
    "cli_security_log",
    "record_security_event",
    "render_seclog",
]
