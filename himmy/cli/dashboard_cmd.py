"""``himmy dashboard`` — the operator overview tile from the terminal (T2h).

The CLI front door to the SAME :class:`~himmy.application.services.DashboardQueryService`
that backs ``/v1/dashboard/summary`` and the Studio dashboard, built over the durable local
store by :func:`himmy.cli.app_services.build_app_container`. ``himmy dashboard summary``
returns ONE object: context field stats, run counts by status, recommendation counts by
status, and the latest evaluation aggregate — all scoped to the pinned ``__local__``
workspace/subject so the numbers match ``himmy runs list`` / ``himmy recommendations list``
for the same scope (the dashboard service REQUIRES a concrete ``workspace_id`` and filters
strictly, which is exactly why the container pins one scope for both reads and writes).

``--json`` prints the full summary object; the default prints a compact human view. All
machine-readable output goes to **stdout**.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


def _eprint(*args: Any) -> None:
    """Print diagnostics to stderr (the summary stays on stdout)."""
    print(*args, file=sys.stderr)


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Dispatch ``himmy dashboard summary`` (the only action today; the default)."""
    from himmy.cli.app_services import build_app_container

    action = getattr(args, "action", None) or "summary"
    if action != "summary":
        _eprint(f"error: unknown action {action!r}")
        return 2
    container = build_app_container()
    try:
        subject = getattr(args, "subject", None) or container.subject_id
        summary = asyncio.run(
            container.dashboard.summary(
                subject_id=subject, workspace_id=container.workspace_id
            )
        )
    finally:
        container.close()

    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    _render(summary)
    return 0


def _render(summary: dict[str, Any]) -> None:
    """Print a compact human-readable view of the summary object."""
    ctx = summary.get("context", {})
    runs = summary.get("runs", {})
    recs = summary.get("recommendations", {})
    ev = summary.get("evaluation") or {}
    print(f"workspace: {summary.get('workspace_id')}  subject: {summary.get('subject_id')}")
    print(
        f"context:   {ctx.get('field_count', 0)} field(s), "
        f"avg confidence {ctx.get('avg_confidence', 0.0):.2f}"
    )
    print(f"runs:      {runs.get('total', 0)} total  {_by_status(runs.get('by_status'))}")
    print(
        f"recs:      {recs.get('total', 0)} total  {_by_status(recs.get('by_status'))}"
    )
    print(f"eval:      {ev.get('total', 0)} run(s)")


def _by_status(by_status: dict[str, int] | None) -> str:
    """Render a non-zero ``{status: count}`` breakdown compactly (empty when all zero)."""
    if not by_status:
        return ""
    parts = [f"{k}={v}" for k, v in by_status.items() if v]
    return "(" + ", ".join(parts) + ")" if parts else ""


def add_dashboard_parser(sub: Any) -> None:
    """Register the ``dashboard`` subcommand on the CLI's subparsers."""
    p = sub.add_parser(
        "dashboard",
        help="operator overview: context/run/recommendation/eval counts (like /v1)",
    )
    p.set_defaults(func=cmd_dashboard, action="summary")
    dsub = p.add_subparsers(dest="action")

    p_sum = dsub.add_parser("summary", help="print the overview summary object")
    p_sum.add_argument("--subject", help="subject id (default: the local subject)")
    p_sum.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_sum.set_defaults(func=cmd_dashboard, action="summary")


__all__ = ["add_dashboard_parser", "cmd_dashboard"]
