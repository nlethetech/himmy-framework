"""``himmy runs`` — browse the CANONICAL run-record store from the terminal (T2b).

This reads the ONE canonical :class:`~himmy.services.storage.models.RunRecord` store every
surface shares (``.himmy/storage.db`` via :class:`RunAppService` readers) — NOT
``.himmy/studio.db`` (a demoted presentation cache) and NOT just ``.himmy/trace.db`` (the
event-only log ``himmy trace``/``himmy lineage`` read). So a run written by ``/v1``, by the
single-user-local Studio, or by ``himmy run --persist`` all appear in one list:

* ``himmy runs list`` — the local workspace's runs, newest first, filterable by ``--status``.
* ``himmy runs show <id>`` — one run record in full (workspace-scoped to the local scope).

The container pins the ``__local__`` workspace so ``himmy runs list`` shows exactly the runs
``--persist`` writes (and the local Studio browses), without leaking another tenant's runs.
``--json`` prints full JSON; everything machine-readable goes to **stdout**.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from himmy.services.storage.models import RunStatus


def _eprint(*args: Any) -> None:
    """Print diagnostics to stderr (rows/JSON stay on stdout)."""
    print(*args, file=sys.stderr)


def _status(value: str) -> RunStatus:
    """Parse a ``--status`` token into a :class:`RunStatus` (clear error otherwise)."""
    try:
        return RunStatus(value.upper())
    except ValueError as exc:
        choices = ", ".join(s.value for s in RunStatus)
        raise SystemExit(
            f"error: unknown status {value!r} (choose: {choices})"
        ) from exc


def _print_run_row(run: Any) -> None:
    """One-line human summary of a run: id, status, persona/model, created-at."""
    label = run.persona_name or run.model_key or "—"
    print(f"{run.run_id} | {run.status.value} | {label} | {run.created_at}")


def cmd_runs(args: argparse.Namespace) -> int:
    """Dispatch ``himmy runs <list|show>`` (default: list)."""
    from himmy.cli.app_services import build_app_container

    action = getattr(args, "action", None) or "list"
    container = build_app_container()
    try:
        if action == "list":
            return _cmd_list(args, container)
        if action == "show":
            return _cmd_show(args, container)
        _eprint(f"error: unknown action {action!r}")
        return 2
    finally:
        container.close()


def _cmd_list(args: argparse.Namespace, container: Any) -> int:
    """``himmy runs list`` — the local workspace's runs, newest first."""
    status = _status(args.status) if getattr(args, "status", None) else None
    runs = asyncio.run(
        container.runs.list_runs(workspace_id=container.workspace_id, status=status)
    )
    if getattr(args, "json", False):
        print(json.dumps([r.model_dump(mode="json") for r in runs]))
        return 0
    if not runs:
        _eprint("(no persisted runs — try `himmy run --persist ...`)")
        return 0
    for run in runs:
        _print_run_row(run)
    return 0


def _cmd_show(args: argparse.Namespace, container: Any) -> int:
    """``himmy runs show <id>`` — one canonical run record, workspace-scoped."""
    run = asyncio.run(
        container.runs.get_run(args.run_id, workspace_id=container.workspace_id)
    )
    if run is None:
        _eprint(f"error: no run {args.run_id!r} in this workspace")
        return 1
    print(json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def add_runs_parser(sub: Any) -> None:
    """Register the ``runs`` subcommand tree on the CLI's subparsers."""
    p = sub.add_parser(
        "runs",
        help="browse the canonical run store (shared with /v1 + Studio)",
    )
    p.set_defaults(func=cmd_runs, action="list")
    rsub = p.add_subparsers(dest="action")

    p_list = rsub.add_parser("list", help="list persisted runs (newest first)")
    p_list.add_argument(
        "--status",
        help="filter by status: " + ", ".join(s.value for s in RunStatus),
    )
    p_list.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_list.set_defaults(func=cmd_runs, action="list")

    p_show = rsub.add_parser("show", help="show one run record by id")
    p_show.add_argument("run_id", help="the run id")
    p_show.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_show.set_defaults(func=cmd_runs, action="show")


__all__ = ["add_runs_parser", "cmd_runs"]
