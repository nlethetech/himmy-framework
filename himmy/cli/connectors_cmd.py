"""``himmy connectors`` — manage the connectors that bridge agents to Slack/Discord/etc.

A *connector* lets an agent act on an external system (post to Slack, open a GitHub
issue, search the web) or receive a message from one (an HMAC-signed webhook). The
connector classes are built and hardened in :mod:`himmy.connectors`; this command is the
terminal front door to the unifying :class:`~himmy.connectors.manage.ConnectorService`:

* ``himmy connectors list`` — the catalog: each connector's kind, whether its required
  secrets are present, whether it is enabled, and whether it is live-available.
* ``himmy connectors set <name> --secret NAME=VALUE [--allowlist a,b]`` — write the
  connector's credentials/allow-list through the *writable* secrets backend (keychain or
  file). On the bare env-only backend this FAILS LOUDLY — connectors can't be configured
  without a writable store.
* ``himmy connectors enable <name> [--surface outbound|inbound]`` /
  ``himmy connectors disable <name>`` — flip the non-secret enable flag (default OFF).
* ``himmy connectors test <name>`` — reuse the connector's capability probe, then (when
  the connector declares one) a live read-only ping that proves the credential works.

Secrets are NEVER printed: ``list`` shows only presence, ``set`` echoes only the names it
wrote. All live decoration goes to stderr; stdout stays plain so the command scripts.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from himmy.cli.ui import styles
from himmy.connectors.manage import (
    ConnectorService,
    ConnectorStatus,
    ReadOnlyBackendError,
)


def _eprint(*args: Any) -> None:
    """Print to stderr (decoration / status; stdout stays data-only)."""
    print(*args, file=sys.stderr)


def _service() -> ConnectorService:
    return ConnectorService()


def _parse_secret_pairs(pairs: list[str]) -> dict[str, str]:
    """Parse ``--secret NAME=VALUE`` pairs; raise ``ValueError`` on a malformed one."""
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--secret must be NAME=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--secret name is empty in {pair!r}")
        out[key] = value
    return out


# --------------------------------------------------------------------------- list


def _enabled_label(status: ConnectorStatus, c: dict[str, str]) -> str:
    surfaces = []
    if status.enabled_outbound:
        surfaces.append("outbound")
    if status.enabled_inbound:
        surfaces.append("inbound")
    if surfaces:
        return f"{c['green']}on:{','.join(surfaces)}{c['reset']}"
    return f"{c['dim']}off{c['reset']}"


def _cmd_list(args: argparse.Namespace, c: dict[str, str]) -> int:
    svc = _service()
    catalog = svc.catalog()
    _eprint(f"{c['dim']}connectors:{c['reset']}")
    for status in catalog:
        cfg = (
            f"{c['green']}configured{c['reset']}"
            if status.configured
            else f"{c['dim']}not configured{c['reset']}"
        )
        enabled = _enabled_label(status, c)
        # stdout: stable, script-friendly tab-separated rows (no secrets).
        print(
            f"{status.name}\t{status.kind}\t"
            f"{'configured' if status.configured else 'unconfigured'}\t"
            f"{'available' if status.available else 'unavailable'}"
        )
        line = (
            f"  {c['green']}{status.name}{c['reset']} "
            f"{c['faint']}({status.kind}){c['reset']}  {cfg}  {enabled}"
        )
        _eprint(line)
        if not status.configured and status.capability_reason:
            _eprint(f"    {c['dim']}{status.capability_reason}{c['reset']}")
    if not any(s.writable for s in catalog):
        _eprint(
            f"{c['dim']}note: secrets backend is read-only — set HIMMY_SECRETS=keychain "
            f"or file to configure connectors{c['reset']}"
        )
    return 0


# ---------------------------------------------------------------------------- set


def _cmd_set(args: argparse.Namespace, c: dict[str, str]) -> int:
    name = getattr(args, "name", None)
    if not name:
        _eprint(
            f"{c['crimson']}usage: himmy connectors set <name> "
            f"--secret NAME=VALUE{c['reset']}"
        )
        return 2
    svc = _service()
    try:
        descriptor = svc.descriptor(name)
    except KeyError:
        _eprint(f"{c['crimson']}unknown connector {name!r}{c['reset']}")
        return 1
    try:
        values: dict[str, Any] = dict(_parse_secret_pairs(getattr(args, "secret", None) or []))
    except ValueError as exc:
        _eprint(f"{c['crimson']}{exc}{c['reset']}")
        return 2
    allowlist = getattr(args, "allowlist", None)
    if allowlist is not None and descriptor.allowlist_env:
        values[descriptor.allowlist_env] = allowlist
    if not values:
        _eprint(
            f"{c['crimson']}nothing to set — pass --secret NAME=VALUE "
            f"(or --allowlist){c['reset']}"
        )
        return 2
    # Validate keys against the descriptor so a typo'd secret name is a clean error,
    # not a silently-ignored write.
    known = {s.name for s in descriptor.secrets}
    if descriptor.allowlist_env:
        known.add(descriptor.allowlist_env)
    unknown = [k for k in values if k not in known]
    if unknown:
        _eprint(
            f"{c['crimson']}unknown field(s) for {name!r}: {', '.join(unknown)}"
            f"{c['reset']} {c['dim']}(known: {', '.join(sorted(known))}){c['reset']}"
        )
        return 1
    try:
        status = svc.configure(name, values)
    except ReadOnlyBackendError as exc:
        _eprint(f"{c['crimson']}{exc}{c['reset']}")
        return 1
    # Echo only the NAMES written, never the values.
    wrote = ", ".join(sorted(values))
    print(f"{name}\t{'configured' if status.configured else 'partial'}")
    _eprint(f"{c['green']}✓ wrote {wrote} for {name!r}{c['reset']}")
    return 0


# --------------------------------------------------------------- enable / disable


def _cmd_enable(args: argparse.Namespace, c: dict[str, str]) -> int:
    return _toggle(args, c, on=True)


def _cmd_disable(args: argparse.Namespace, c: dict[str, str]) -> int:
    return _toggle(args, c, on=False)


def _toggle(args: argparse.Namespace, c: dict[str, str], *, on: bool) -> int:
    verb = "enable" if on else "disable"
    name = getattr(args, "name", None)
    if not name:
        _eprint(f"{c['crimson']}usage: himmy connectors {verb} <name>{c['reset']}")
        return 2
    surface = getattr(args, "surface", None) or "outbound"
    svc = _service()
    try:
        if on:
            svc.enable(name, surface)
        else:
            svc.disable(name, surface)
    except KeyError:
        _eprint(f"{c['crimson']}unknown connector {name!r}{c['reset']}")
        return 1
    except ValueError as exc:
        _eprint(f"{c['crimson']}{exc}{c['reset']}")
        return 2
    except ReadOnlyBackendError as exc:
        _eprint(f"{c['crimson']}{exc}{c['reset']}")
        return 1
    print(f"{name}\t{surface}\t{'on' if on else 'off'}")
    _eprint(f"{c['green']}✓ {verb}d {name!r} on {surface}{c['reset']}")
    return 0


# --------------------------------------------------------------------------- test


def _cmd_test(args: argparse.Namespace, c: dict[str, str]) -> int:
    name = getattr(args, "name", None)
    if not name:
        _eprint(f"{c['crimson']}usage: himmy connectors test <name>{c['reset']}")
        return 2
    svc = _service()
    try:
        svc.descriptor(name)
    except KeyError:
        _eprint(f"{c['crimson']}unknown connector {name!r}{c['reset']}")
        return 1
    _eprint(f"{c['dim']}testing {name!r}…{c['reset']}")
    result = asyncio.run(svc.test(name))
    print(f"{name}\t{'ok' if result.ok else 'fail'}\t{result.checked}")
    if result.ok:
        _eprint(
            f"{c['green']}✓ {name!r} {result.checked}: {result.detail}{c['reset']}"
        )
        return 0
    _eprint(f"{c['crimson']}✗ {name!r} {result.checked}: {result.detail}{c['reset']}")
    return 1


# ------------------------------------------------------------------------ dispatch

_ACTIONS = {
    "list": _cmd_list,
    "set": _cmd_set,
    "enable": _cmd_enable,
    "disable": _cmd_disable,
    "test": _cmd_test,
}


def cmd_connectors(args: argparse.Namespace) -> int:
    """Dispatch ``himmy connectors <action>`` to its handler (default: ``list``)."""
    c = styles(sys.stdout)
    action = getattr(args, "action", None) or "list"
    handler = _ACTIONS.get(action)
    if handler is None:
        _eprint(
            f"{c['crimson']}unknown connectors action {action!r} — "
            f"use list | set | enable | disable | test{c['reset']}"
        )
        return 2
    return handler(args, c)


__all__ = ["cmd_connectors"]
