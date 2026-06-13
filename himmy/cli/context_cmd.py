"""``himmy context`` — upsert/list context fields and build/show snapshots (T2h).

The CLI front door to the SAME :class:`~himmy.application.services.ContextAppService` that
backs the ``/v1/context`` API and the Studio context UI, built over the durable local store
by :func:`himmy.cli.app_services.build_app_container`. Every field/snapshot is stamped with
the pinned ``__local__`` workspace/subject so the CLI's view is internally consistent with
``himmy dashboard`` and what ``--persist`` writes:

* ``himmy context fields set <key> <value> [--confidence C] [--source S]`` — upsert one
  typed context field (a numeric value is stored as a number; otherwise as a string).
* ``himmy context fields list`` — the subject's stored fields, scoped to the local workspace.
* ``himmy context snapshot`` — build + persist an evidenced :class:`ContextSnapshot` over the
  subject's stored field keys (``STORAGE_FIRST``), stamping the workspace, and print its id.
* ``himmy context snapshot --show <id>`` — load + render a stored snapshot (workspace-scoped).

``--subject`` overrides the pinned local subject for multi-subject local books. All
machine-readable output goes to **stdout**; diagnostics to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


def _eprint(*args: Any) -> None:
    """Print diagnostics to stderr (rows/JSON stay on stdout)."""
    print(*args, file=sys.stderr)


def _coerce_value(raw: str) -> Any:
    """Parse a CLI ``value`` token: a JSON scalar when it parses, else the raw string."""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _subject(args: argparse.Namespace, container: Any) -> str:
    """The subject to read/write: ``--subject`` override or the pinned local subject."""
    return str(getattr(args, "subject", None) or container.subject_id)


def cmd_context(args: argparse.Namespace) -> int:
    """Dispatch ``himmy context <fields|snapshot>``."""
    from himmy.cli.app_services import build_app_container

    action = getattr(args, "action", None)
    if action is None:
        _eprint("error: choose a context action — fields | snapshot")
        return 2
    container = build_app_container()
    try:
        if action == "fields":
            return _cmd_fields(args, container)
        if action == "snapshot":
            return _cmd_snapshot(args, container)
        _eprint(f"error: unknown action {action!r}")
        return 2
    finally:
        container.close()


def _cmd_fields(args: argparse.Namespace, container: Any) -> int:
    """``himmy context fields <set|list>`` over the local workspace/subject."""
    sub = getattr(args, "fields_action", None) or "list"
    subject = _subject(args, container)
    if sub == "set":
        from himmy.services.context.models import ContextField

        field = ContextField(
            key=args.key,
            value=_coerce_value(args.value),
            confidence=float(getattr(args, "confidence", None) or 1.0),
            source=getattr(args, "source", None) or "cli",
        )
        saved = asyncio.run(
            container.context.upsert_fields(
                container.workspace_id, subject, [field]
            )
        )
        out = saved[0]
        if getattr(args, "json", False):
            print(json.dumps(out.model_dump(mode="json")))
        else:
            print(f"{out.key} = {out.value!r}  (confidence {out.confidence})")
        return 0

    # list
    fields = asyncio.run(
        container.context.list_fields(subject, workspace_id=container.workspace_id)
    )
    if getattr(args, "json", False):
        print(json.dumps([f.model_dump(mode="json") for f in fields]))
        return 0
    if not fields:
        _eprint(f"(no context fields for {subject!r})")
        return 0
    for f in fields:
        print(f"{f.key} = {f.value!r}  (confidence {f.confidence}, source {f.source})")
    return 0


def _cmd_snapshot(args: argparse.Namespace, container: Any) -> int:
    """``himmy context snapshot [--show <id>]`` — build over stored keys, or render one."""
    subject = _subject(args, container)
    show = getattr(args, "show", None)
    if show:
        snapshot = asyncio.run(
            container.context.get_snapshot(show, workspace_id=container.workspace_id)
        )
        if snapshot is None:
            _eprint(f"error: no snapshot {show!r} in this workspace")
            return 1
        print(json.dumps(snapshot.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    from himmy.services.context.models import ContextBuildSpec, ContextSpecKey

    fields = asyncio.run(
        container.context.list_fields(subject, workspace_id=container.workspace_id)
    )
    if not fields:
        _eprint(
            f"(no context fields for {subject!r} — set some with "
            "`himmy context fields set` first)"
        )
        return 1
    build_spec = ContextBuildSpec(
        keys=[ContextSpecKey(key=f.key) for f in fields]
    )
    snapshot = asyncio.run(
        container.context.build_snapshot(
            subject_id=subject,
            build_spec=build_spec,
            metadata={"workspace_id": container.workspace_id},
        )
    )
    if getattr(args, "json", False):
        print(json.dumps(snapshot.model_dump(mode="json")))
    else:
        print(snapshot.snapshot_id)
        _eprint(f"built snapshot over {len(snapshot.fields)} field(s)")
    return 0


def add_context_parser(sub: Any) -> None:
    """Register the ``context`` subcommand tree on the CLI's subparsers."""
    p = sub.add_parser(
        "context",
        help="upsert/list context fields + build snapshots (shared with /v1 + Studio)",
    )
    p.set_defaults(func=cmd_context, action=None)
    csub = p.add_subparsers(dest="action")

    p_fields = csub.add_parser("fields", help="upsert/list context fields")
    p_fields.add_argument("--subject", help="subject id (default: the local subject)")
    p_fields.set_defaults(func=cmd_context, action="fields", fields_action=None)
    fsub = p_fields.add_subparsers(dest="fields_action")

    p_set = fsub.add_parser("set", help="upsert one context field")
    p_set.add_argument("--subject", help="subject id (default: the local subject)")
    p_set.add_argument("key", help="the field key")
    p_set.add_argument("value", help="the field value (JSON scalar or raw string)")
    p_set.add_argument(
        "--confidence", type=float, help="confidence in [0,1] (default 1.0)"
    )
    p_set.add_argument("--source", help="provenance source label (default: cli)")
    p_set.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_set.set_defaults(func=cmd_context, action="fields", fields_action="set")

    p_flist = fsub.add_parser("list", help="list the subject's stored fields")
    p_flist.add_argument("--subject", help="subject id (default: the local subject)")
    p_flist.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_flist.set_defaults(func=cmd_context, action="fields", fields_action="list")

    p_snap = csub.add_parser(
        "snapshot", help="build an evidenced snapshot over stored fields (or --show one)"
    )
    p_snap.add_argument("--subject", help="subject id (default: the local subject)")
    p_snap.add_argument("--show", metavar="ID", help="render a stored snapshot by id")
    p_snap.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_snap.set_defaults(func=cmd_context, action="snapshot")


__all__ = ["add_context_parser", "cmd_context"]
