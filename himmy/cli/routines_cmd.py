"""``himmy routines`` — schedule unattended agent runs from the terminal (T3c).

A *routine* is a saved prompt against a saved agent that runs on a simple schedule
(``daily at HH:MM`` or ``every N hours``) without anyone asking. This command drives the
SAME store (``.himmy/routines.db``) and scheduler the Studio routines screen uses — so a
routine added here shows in Studio and vice-versa, and a ``run-now`` run lands in the ONE
canonical run store (visible in ``himmy runs`` + ``GET /v1/runs`` + Studio):

* ``himmy routines list`` — every local routine, with its schedule + last-run status.
* ``himmy routines add --name N -f agent.yaml -p "..." --daily HH:MM | --every N`` — create
  a routine bound to a project-local ``agent.yaml`` (the single-user-local seam).
* ``himmy routines show <id>`` — one routine in full.
* ``himmy routines enable|disable <id>`` — flip whether the scheduler fires it.
* ``himmy routines remove <id>`` — delete it.
* ``himmy routines run-now <id>`` — run it immediately through the unattended rails (an
  approval-gated tool PAUSES the run; the scheduler never auto-approves). A
  cross-process ``flock`` refuses a second concurrent run of the same routine (so a CLI
  run-now and the Studio scheduler can never double-execute it).

The CLI surface is single-user-local: routines created here live in the ``__local__``
workspace and bind their agent by a filesystem ``agent.yaml`` path (the multi-tenant,
``agent_id``-bound resource is ``/v1/routines``). Decoration goes to stderr; machine-
readable JSON (``--json``) goes to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from himmy.api import routines as svc


def _eprint(*args: Any) -> None:
    """Print diagnostics to stderr (rows / JSON stay on stdout)."""
    print(*args, file=sys.stderr)


def _routine_json(routine: svc.Routine) -> dict[str, Any]:
    """A stable machine-readable projection of a routine (drops internal-only churn)."""
    return {
        "id": routine.id,
        "name": routine.name,
        "agent_path": routine.agent_path,
        "prompt": routine.prompt,
        "schedule": routine.schedule.model_dump(),
        "provider": routine.provider,
        "model": routine.model,
        "deliver": routine.deliver,
        "enabled": routine.enabled,
        "last_run_at": routine.last_run_at,
        "last_status": routine.last_status,
        "last_preview": routine.last_preview,
        "last_error": routine.last_error,
        "last_delivery": routine.last_delivery,
    }


def _print_row(routine: svc.Routine) -> None:
    """One-line human summary: id, name, schedule, enabled, last status."""
    flag = "on " if routine.enabled else "off"
    status = routine.last_status or "—"
    print(
        f"{routine.id} | {flag} | {routine.schedule.describe():<10} | "
        f"{status:<17} | {routine.name}"
    )


def _build_schedule(args: argparse.Namespace) -> svc.Schedule:
    """Build a validated :class:`Schedule` from ``--daily HH:MM`` / ``--every N``."""
    daily: str | None = getattr(args, "daily", None)
    every: int | None = getattr(args, "every", None)
    if (daily is None) == (every is None):
        raise SystemExit(
            "error: choose exactly one of --daily HH:MM or --every N (hours)"
        )
    if daily is not None:
        return svc.Schedule(kind="daily", at=daily)
    return svc.Schedule(kind="every", hours=every)


def cmd_routines(args: argparse.Namespace) -> int:
    """Dispatch ``himmy routines <list|add|show|enable|disable|remove|run-now>``."""
    action = getattr(args, "action", None) or "list"
    if action == "list":
        return _cmd_list(args)
    if action == "add":
        return _cmd_add(args)
    if action == "show":
        return _cmd_show(args)
    if action in ("enable", "disable"):
        return _cmd_toggle(args, enabled=action == "enable")
    if action == "remove":
        return _cmd_remove(args)
    if action == "run-now":
        return _cmd_run_now(args)
    _eprint(f"error: unknown action {action!r}")
    return 2


def _cmd_list(args: argparse.Namespace) -> int:
    """``himmy routines list`` — the local routines, newest first."""
    routines = svc.get_routines_store().list(workspace_id=svc.LOCAL_WORKSPACE)
    if getattr(args, "json", False):
        print(json.dumps([_routine_json(r) for r in routines]))
        return 0
    if not routines:
        _eprint("(no routines — add one with `himmy routines add`)")
        return 0
    for routine in routines:
        _print_row(routine)
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    """``himmy routines add`` — create a routine bound to a local agent.yaml."""
    from himmy.api.studio_service import resolve_spec_path

    agent_path = args.file
    try:
        resolve_spec_path(agent_path)
    except FileNotFoundError as exc:
        _eprint(f"error: {exc}")
        return 1
    except ValueError as exc:
        _eprint(f"error: {exc}")
        return 1
    try:
        schedule = _build_schedule(args)
    except ValueError as exc:
        _eprint(f"error: {exc}")
        return 1
    routine = svc.Routine(
        name=args.name,
        agent_path=agent_path,
        prompt=args.prompt,
        schedule=schedule,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        deliver=getattr(args, "deliver", None) or "none",
        enabled=not getattr(args, "disabled", False),
    )
    stored = svc.get_routines_store().upsert(routine)
    _eprint(f"created routine {stored.id} ({schedule.describe()})")
    if getattr(args, "json", False):
        print(json.dumps(_routine_json(stored)))
    else:
        print(stored.id)
    return 0


def _resolve_local(routine_id: str) -> svc.Routine | None:
    """Fetch a routine in the local workspace (None when unknown / out-of-workspace)."""
    routine: svc.Routine | None = svc.get_routines_store().get(
        routine_id, workspace_id=svc.LOCAL_WORKSPACE
    )
    return routine


def _cmd_show(args: argparse.Namespace) -> int:
    routine = _resolve_local(args.routine_id)
    if routine is None:
        _eprint(f"error: no routine {args.routine_id!r}")
        return 1
    print(json.dumps(_routine_json(routine), indent=2, ensure_ascii=False))
    return 0


def _cmd_toggle(args: argparse.Namespace, *, enabled: bool) -> int:
    store = svc.get_routines_store()
    routine = _resolve_local(args.routine_id)
    if routine is None:
        _eprint(f"error: no routine {args.routine_id!r}")
        return 1
    routine.enabled = enabled
    store.upsert(routine)
    _eprint(f"routine {routine.id} {'enabled' if enabled else 'disabled'}")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    if not svc.get_routines_store().delete(
        args.routine_id, workspace_id=svc.LOCAL_WORKSPACE
    ):
        _eprint(f"error: no routine {args.routine_id!r}")
        return 1
    _eprint(f"removed routine {args.routine_id}")
    return 0


def _cmd_run_now(args: argparse.Namespace) -> int:
    """``himmy routines run-now`` — run once through the unattended rails (flock-guarded).

    Point the run pipeline's canonical run store at the SAME durable ``.himmy/storage.db``
    the co-located server + ``himmy runs`` read (via the CLI app container), so a CLI
    run-now is visible in ``himmy runs`` AND ``GET /v1/runs`` AND Studio — not lost to a
    throwaway in-RAM store. The provider is cleared afterwards so a later in-process run
    does not mirror into this (closed) container's store.
    """
    if _resolve_local(args.routine_id) is None:
        _eprint(f"error: no routine {args.routine_id!r}")
        return 1
    from himmy.api.studio_canonical import set_canonical_storage_provider
    from himmy.cli.app_services import build_app_container

    container = build_app_container()
    set_canonical_storage_provider(lambda: container.storage)
    try:
        result = asyncio.run(svc.execute_routine(args.routine_id))
    except svc.RoutineBusyError:
        _eprint("error: routine is already running (in this or another process)")
        return 1
    finally:
        set_canonical_storage_provider(None)
        container.close()
    if result is None:
        _eprint(f"error: routine {args.routine_id!r} disappeared mid-run")
        return 1
    _eprint(f"ran routine {result.id}: {result.last_status}")
    if getattr(args, "json", False):
        print(json.dumps(_routine_json(result)))
    else:
        print(result.last_preview or result.last_error or "")
    return 0 if result.last_status not in ("error", "timeout") else 1


def add_routines_parser(sub: Any) -> None:
    """Register the ``routines`` subcommand tree on the CLI's subparsers."""
    p = sub.add_parser(
        "routines",
        help="schedule unattended agent runs (shared with Studio + /v1)",
    )
    p.set_defaults(func=cmd_routines, action="list")
    rsub = p.add_subparsers(dest="action")

    p_list = rsub.add_parser("list", help="list routines (newest first)")
    p_list.add_argument("--json", action="store_true", help="machine-readable JSON")
    p_list.set_defaults(func=cmd_routines, action="list")

    p_add = rsub.add_parser("add", help="create a routine bound to a local agent.yaml")
    p_add.add_argument("--name", required=True, help="a name for the routine")
    p_add.add_argument(
        "-f", "--file", required=True, help="path to the agent.yaml to run"
    )
    p_add.add_argument(
        "-p", "--prompt", required=True, help="the prompt to run each time"
    )
    sched = p_add.add_mutually_exclusive_group(required=True)
    sched.add_argument("--daily", metavar="HH:MM", help="run daily at this 24h time")
    sched.add_argument(
        "--every", metavar="HOURS", type=int, help="run every N hours (1..168)"
    )
    p_add.add_argument("--provider", help="inference provider (default: auto)")
    p_add.add_argument("--model", help="model key/name for the provider")
    p_add.add_argument(
        "--deliver",
        choices=["none", "telegram", "email"],
        help="deliver the result through a configured connection (default: none)",
    )
    p_add.add_argument(
        "--disabled", action="store_true", help="create it disabled (don't schedule yet)"
    )
    p_add.add_argument("--json", action="store_true", help="print the routine as JSON")
    p_add.set_defaults(func=cmd_routines, action="add")

    p_show = rsub.add_parser("show", help="show one routine in full")
    p_show.add_argument("routine_id", help="the routine id")
    p_show.set_defaults(func=cmd_routines, action="show")

    p_enable = rsub.add_parser("enable", help="enable a routine (the scheduler fires it)")
    p_enable.add_argument("routine_id", help="the routine id")
    p_enable.set_defaults(func=cmd_routines, action="enable")

    p_disable = rsub.add_parser("disable", help="disable a routine (stop scheduling it)")
    p_disable.add_argument("routine_id", help="the routine id")
    p_disable.set_defaults(func=cmd_routines, action="disable")

    p_remove = rsub.add_parser("remove", help="delete a routine")
    p_remove.add_argument("routine_id", help="the routine id")
    p_remove.set_defaults(func=cmd_routines, action="remove")

    p_run = rsub.add_parser(
        "run-now", help="run a routine immediately through the unattended rails"
    )
    p_run.add_argument("routine_id", help="the routine id")
    p_run.add_argument("--json", action="store_true", help="print the result as JSON")
    p_run.set_defaults(func=cmd_routines, action="run-now")


__all__ = ["add_routines_parser", "cmd_routines"]
