"""``himmy recommendations`` — list, show, and transition advisory recommendations (T2h).

The CLI front door to the SAME :class:`~himmy.application.services.RecommendationAppService`
that backs ``/v1/recommendations`` and the Studio recs screen, built over the durable local
store by :func:`himmy.cli.app_services.build_app_container` (so a recommendation extracted
from a ``himmy run --persist`` run — or one written by ``/v1`` / Studio into the shared
``.himmy/storage.db`` — is listed here):

* ``himmy recommendations list`` — the workspace's recommendations (newest first), filterable
  by ``--status``/``--kind``/``--run``.
* ``himmy recommendations show <id>`` — one recommendation in full. Routed through the
  app-service getter :meth:`RecommendationAppService.get` (added for T2h — the service had
  no single-item read before), tenant-scoped to the pinned local workspace.
* ``himmy recommendations transition <id> --status <S>`` — move a recommendation's lifecycle
  state (PROPOSED → ACCEPTED/DISMISSED/SCHEDULED), the same write ``/v1`` exposes.

Everything machine-readable goes to **stdout** (``--json`` for full JSON); diagnostics go to
stderr so the command scripts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from himmy.services.storage.models import RecommendationStatus


def _eprint(*args: Any) -> None:
    """Print diagnostics to stderr (machine-readable rows/JSON stay on stdout)."""
    print(*args, file=sys.stderr)


def _status(value: str) -> RecommendationStatus:
    """Parse a ``--status`` token into a :class:`RecommendationStatus` (clear error)."""
    try:
        return RecommendationStatus(value.upper())
    except ValueError as exc:
        choices = ", ".join(s.value for s in RecommendationStatus)
        raise SystemExit(
            f"error: unknown status {value!r} (choose: {choices})"
        ) from exc


def _print_item(item: Any, *, as_json: bool) -> None:
    """Emit one recommendation as JSON or a one-line human summary."""
    data = item.model_dump(mode="json")
    if as_json:
        print(json.dumps(data))
    else:
        print(
            f"{data['recommendation_id']} | {data['status']} | {data['kind']}"
            f" | {data['title']}"
        )


def cmd_recommendations(args: argparse.Namespace) -> int:
    """Dispatch ``himmy recommendations <list|show|transition>`` (default: list)."""
    from himmy.cli.app_services import build_app_container

    action = getattr(args, "action", None) or "list"
    container = build_app_container()
    try:
        if action == "list":
            return _cmd_list(args, container)
        if action == "show":
            return _cmd_show(args, container)
        if action == "transition":
            return _cmd_transition(args, container)
        _eprint(f"error: unknown action {action!r}")
        return 2
    finally:
        container.close()


def _cmd_list(args: argparse.Namespace, container: Any) -> int:
    """``himmy recommendations list`` — newest-first, filtered, over the local workspace."""
    status = _status(args.status) if getattr(args, "status", None) else None
    items = asyncio.run(
        container.recommendations.list(
            workspace_id=container.workspace_id,
            status=status,
            kind=getattr(args, "kind", None),
            run_id=getattr(args, "run", None),
        )
    )
    if getattr(args, "json", False):
        print(json.dumps([i.model_dump(mode="json") for i in items]))
        return 0
    if not items:
        _eprint("(no recommendations — extract one from a `himmy run --persist` run)")
        return 0
    for item in items:
        _print_item(item, as_json=False)
    return 0


def _cmd_show(args: argparse.Namespace, container: Any) -> int:
    """``himmy recommendations show <id>`` — one recommendation via the app-service getter."""
    item = asyncio.run(
        container.recommendations.get(
            args.recommendation_id, workspace_id=container.workspace_id
        )
    )
    if item is None:
        _eprint(f"error: no recommendation {args.recommendation_id!r}")
        return 1
    _print_item(item, as_json=getattr(args, "json", False))
    return 0


def _cmd_transition(args: argparse.Namespace, container: Any) -> int:
    """``himmy recommendations transition <id> --status <S>`` — the /v1 status write."""
    status = _status(args.status)
    item = asyncio.run(
        container.recommendations.update_status(
            args.recommendation_id,
            status=status,
            notes=getattr(args, "notes", None),
            workspace_id=container.workspace_id,
        )
    )
    if item is None:
        _eprint(f"error: no recommendation {args.recommendation_id!r}")
        return 1
    _print_item(item, as_json=getattr(args, "json", False))
    return 0


def add_recommendations_parser(sub: Any) -> None:
    """Register the ``recommendations`` subcommand tree on the CLI's subparsers."""
    p = sub.add_parser(
        "recommendations",
        help="list/show/transition advisory recommendations (shared with /v1 + Studio)",
    )
    p.set_defaults(func=cmd_recommendations, action="list")
    rsub = p.add_subparsers(dest="action")

    p_list = rsub.add_parser("list", help="list recommendations (newest first)")
    p_list.add_argument(
        "--status",
        help="filter by status: " + ", ".join(s.value for s in RecommendationStatus),
    )
    p_list.add_argument("--kind", help="filter by recommendation kind")
    p_list.add_argument("--run", help="filter to one run's recommendations")
    p_list.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_list.set_defaults(func=cmd_recommendations, action="list")

    p_show = rsub.add_parser("show", help="show one recommendation by id")
    p_show.add_argument("recommendation_id", help="the recommendation id")
    p_show.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_show.set_defaults(func=cmd_recommendations, action="show")

    p_tr = rsub.add_parser("transition", help="move a recommendation's lifecycle state")
    p_tr.add_argument("recommendation_id", help="the recommendation id")
    p_tr.add_argument(
        "--status",
        required=True,
        help="new status: " + ", ".join(s.value for s in RecommendationStatus),
    )
    p_tr.add_argument("--notes", help="optional note to attach to the transition")
    p_tr.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_tr.set_defaults(func=cmd_recommendations, action="transition")


__all__ = ["add_recommendations_parser", "cmd_recommendations"]
