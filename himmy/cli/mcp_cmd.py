"""``himmy mcp`` — manage the stdio MCP servers wired into an agent's ``agent.yaml``.

Declarative MCP servers (:mod:`himmy.config.mcp_spec`) let an ``agent.yaml`` pull in
the whole Model Context Protocol ecosystem — GitHub, Slack, a filesystem server, a
browser — as ordinary agent tools without writing code. This command surfaces that
list on the terminal and edits it in place:

* ``himmy mcp list`` — show the MCP servers configured in the nearest ``agent.yaml``.
* ``himmy mcp add <name> --command npx --arg=-y --arg @scope/server`` — append an
  :class:`~himmy.config.mcp_spec.MCPServerConfig` to ``mcp_servers``, preserving the
  rest of the file (and validating the result through the real spec loader). Note the
  ``--arg=-y`` form: a dash-leading value must use ``=`` so argparse keeps it as the
  arg value rather than treating ``-y`` as an option flag.
* ``himmy mcp test <name>`` — actually CONNECT to that server via
  :func:`~himmy.config.mcp_spec.attach_mcp_servers`, list the tools it advertises,
  then disconnect; reports the tool count or the connection error.
* ``himmy mcp remove <name>`` — drop the matching server from ``mcp_servers``.

A server has no name field of its own in :class:`MCPServerConfig`, so ``<name>`` here
is a friendly label stored on the entry's ``prefix`` (``<name>_``) when none is given,
and matched against either the prefix or the command for ``test``/``remove`` — the same
structural identity the Studio router uses. All live decoration goes to stderr only on a
real TTY (see :mod:`himmy.cli.ui`); stdout stays plain so the command scripts cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml

from himmy.cli.ui import styles
from himmy.config.mcp_spec import (
    MCPServerConfig,
    attach_mcp_servers,
    close_mcp_clients,
)


def _eprint(*args: Any) -> None:
    """Print to stderr (decoration / status; stdout stays data-only)."""
    print(*args, file=sys.stderr)


def _discover_agent_yaml() -> Path | None:
    """git-style upward search from cwd for the nearest ``agent.yaml``."""
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / "agent.yaml"
        if candidate.is_file():
            return candidate
    return None


def _resolve_path(args: argparse.Namespace) -> Path | None:
    """The agent.yaml to act on: ``-f/--file`` if given, else the nearest one."""
    explicit = getattr(args, "file", None)
    if explicit:
        return Path(explicit).expanduser()
    return _discover_agent_yaml()


def _load_raw(path: Path) -> dict[str, Any]:
    """Load ``agent.yaml`` as a raw mapping (so we preserve every other key).

    Raises ``ValueError`` with a clear message on a non-mapping / invalid YAML.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    if raw.get("entry") and isinstance(raw.get("members"), list):
        raise ValueError(f"{path} is a team spec — attach MCP servers to single agents")
    return raw


def _entries(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``mcp_servers`` list as plain dicts (drops malformed items)."""
    existing = raw.get("mcp_servers")
    if not isinstance(existing, list):
        return []
    return [e for e in existing if isinstance(e, dict)]


def _label(entry: dict[str, Any]) -> str:
    """A friendly name for one entry: its prefix (trailing ``_`` stripped), else
    the command. Mirrors how :func:`_make_entry` stores the name on ``prefix``."""
    prefix = str(entry.get("prefix") or "").rstrip("_")
    if prefix:
        return prefix
    return str(entry.get("command") or "?")


def _matches(entry: dict[str, Any], name: str) -> bool:
    """Does ``entry`` correspond to the friendly ``name`` (label or command)?"""
    return _label(entry) == name or str(entry.get("command") or "") == name


def _make_entry(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Build (and validate) one ``mcp_servers`` entry from the CLI flags.

    The entry is validated through the real :class:`MCPServerConfig` so what lands
    in the file is exactly what the runtime will launch. Raises ``ValueError`` on a
    bad config (surfacing the pydantic message).
    """
    raw: dict[str, Any] = {"command": getattr(args, "command", None) or ""}
    cmd_args = list(getattr(args, "arg", None) or [])
    if cmd_args:
        raw["args"] = cmd_args
    env = _parse_env(getattr(args, "env", None) or [])
    if env:
        raw["env"] = env
    cwd = getattr(args, "cwd", None)
    if cwd:
        raw["cwd"] = cwd
    prefix = getattr(args, "prefix", None)
    raw["prefix"] = prefix if prefix else f"{name}_"
    if getattr(args, "requires_approval", False):
        raw["requires_approval"] = True
    tools = list(getattr(args, "tool", None) or [])
    if tools:
        raw["tools"] = tools
    try:
        MCPServerConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - surface pydantic message verbatim
        raise ValueError(f"invalid MCP server config: {exc}") from exc
    return raw


def _parse_env(pairs: list[str]) -> dict[str, str]:
    """Parse ``--env KEY=VALUE`` pairs; raises ``ValueError`` on a malformed one."""
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--env must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        if not key:
            raise ValueError(f"--env key is empty in {pair!r}")
        env[key] = value
    return env


def _write_back(path: Path, raw: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    """Write ``entries`` back into ``raw['mcp_servers']`` and validate the result.

    On any loader rejection the original bytes are restored, so a hand-authored
    ``agent.yaml`` can never be corrupted by this command.
    """
    from himmy.config.agent_spec import load_agent_spec

    if entries:
        raw["mcp_servers"] = entries
    else:
        raw.pop("mcp_servers", None)
    original = path.read_text(encoding="utf-8") if path.is_file() else None
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    try:
        load_agent_spec(str(path))  # the authoritative round-trip check
    except Exception as exc:  # noqa: BLE001 - restore + report, never corrupt
        if original is not None:
            path.write_text(original, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        raise ValueError(f"refused to write a spec the loader rejects: {exc}") from exc


# --------------------------------------------------------------------------- list


def _cmd_list(args: argparse.Namespace, c: dict[str, str]) -> int:
    path = _resolve_path(args)
    if path is None:
        _eprint(
            f"{c['dim']}no agent.yaml found (searched upward from cwd) — "
            f"create one with `himmy init`{c['reset']}"
        )
        return 1
    if not path.is_file():
        _eprint(f"{c['crimson']}no such file: {path}{c['reset']}")
        return 1
    try:
        raw = _load_raw(path)
    except ValueError as exc:
        _eprint(f"{c['crimson']}{exc}{c['reset']}")
        return 1
    entries = _entries(raw)
    if not entries:
        _eprint(f"{c['dim']}no MCP servers configured in {path}{c['reset']}")
        return 0
    _eprint(f"{c['dim']}MCP servers in {path}:{c['reset']}")
    for entry in entries:
        name = _label(entry)
        command = str(entry.get("command") or "")
        cmd_args = " ".join(str(a) for a in entry.get("args") or [])
        gate = (
            f" {c['gold']}[approval]{c['reset']}"
            if entry.get("requires_approval")
            else ""
        )
        line = f"  {c['green']}{name}{c['reset']}  {command}"
        if cmd_args:
            line += f" {c['faint']}{cmd_args}{c['reset']}"
        print(f"{name}\t{command} {cmd_args}".rstrip())
        _eprint(line + gate)
    return 0


# ---------------------------------------------------------------------------- add


def _cmd_add(args: argparse.Namespace, c: dict[str, str]) -> int:
    name = getattr(args, "name", None)
    if not name:
        _eprint(
            f"{c['crimson']}usage: himmy mcp add <name> --command <cmd>{c['reset']}"
        )
        return 2
    if not getattr(args, "command", None):
        _eprint(f"{c['crimson']}error: --command is required{c['reset']}")
        return 2
    path = _resolve_path(args)
    if path is None:
        _eprint(
            f"{c['crimson']}no agent.yaml found — pass -f <file> or run "
            f"`himmy init`{c['reset']}"
        )
        return 1
    try:
        raw = _load_raw(path) if path.is_file() else {}
        entries = _entries(raw)
        if any(_matches(e, name) for e in entries):
            _eprint(
                f"{c['crimson']}an MCP server named {name!r} already exists in "
                f"{path}{c['reset']}"
            )
            return 1
        entry = _make_entry(name, args)
        entries.append(entry)
        _write_back(path, raw, entries)
    except ValueError as exc:
        _eprint(f"{c['crimson']}{exc}{c['reset']}")
        return 1
    _eprint(
        f"{c['green']}✓ added MCP server {name!r}{c['reset']} "
        f"{c['faint']}({entry.get('command')}){c['reset']} → {path}"
    )
    return 0


# ------------------------------------------------------------------------- remove


def _cmd_remove(args: argparse.Namespace, c: dict[str, str]) -> int:
    name = getattr(args, "name", None)
    if not name:
        _eprint(f"{c['crimson']}usage: himmy mcp remove <name>{c['reset']}")
        return 2
    path = _resolve_path(args)
    if path is None or not path.is_file():
        _eprint(f"{c['crimson']}no agent.yaml found{c['reset']}")
        return 1
    try:
        raw = _load_raw(path)
        entries = _entries(raw)
        kept = [e for e in entries if not _matches(e, name)]
        if len(kept) == len(entries):
            _eprint(f"{c['crimson']}no MCP server named {name!r} in {path}{c['reset']}")
            return 1
        _write_back(path, raw, kept)
    except ValueError as exc:
        _eprint(f"{c['crimson']}{exc}{c['reset']}")
        return 1
    _eprint(f"{c['green']}✓ removed MCP server {name!r}{c['reset']} → {path}")
    return 0


# --------------------------------------------------------------------------- test


async def _connect_and_count(config: MCPServerConfig) -> tuple[int, list[str]]:
    """CONNECT to the server, list its tools, disconnect; return (count, names).

    Connect, list, and close all happen inside one event loop (the clients' reader
    tasks are bound to the running loop — see :func:`attach_mcp_servers`).
    """
    from himmy.services.tools.registry import ToolRegistry

    registry = ToolRegistry()
    clients = await attach_mcp_servers(registry, [config])
    try:
        names = sorted(d.name for d in registry.list())
        return len(names), names
    finally:
        await close_mcp_clients(clients)


def _cmd_test(args: argparse.Namespace, c: dict[str, str]) -> int:
    name = getattr(args, "name", None)
    if not name:
        _eprint(f"{c['crimson']}usage: himmy mcp test <name>{c['reset']}")
        return 2
    path = _resolve_path(args)
    if path is None or not path.is_file():
        _eprint(f"{c['crimson']}no agent.yaml found{c['reset']}")
        return 1
    try:
        raw = _load_raw(path)
    except ValueError as exc:
        _eprint(f"{c['crimson']}{exc}{c['reset']}")
        return 1
    match = next((e for e in _entries(raw) if _matches(e, name)), None)
    if match is None:
        _eprint(f"{c['crimson']}no MCP server named {name!r} in {path}{c['reset']}")
        return 1
    try:
        config = MCPServerConfig.model_validate(match)
    except Exception as exc:  # noqa: BLE001 - bad stored entry is a clean failure
        _eprint(
            f"{c['crimson']}invalid MCP server config for {name!r}: {exc}{c['reset']}"
        )
        return 1

    _eprint(f"{c['dim']}connecting to {name!r} ({config.command})…{c['reset']}")
    try:
        count, names = asyncio.run(_connect_and_count(config))
    except Exception as exc:  # noqa: BLE001 - a connection failure is a clean report
        _eprint(f"{c['crimson']}✗ {name!r} failed to connect: {exc}{c['reset']}")
        return 1
    print(f"{name}\t{count}")
    _eprint(
        f"{c['green']}✓ {name!r} connected — {count} tool(s){c['reset']}"
        + (f"{c['faint']}: {', '.join(names)}{c['reset']}" if names else "")
    )
    return 0


# ------------------------------------------------------------------------ dispatch

_ACTIONS = {
    "list": _cmd_list,
    "add": _cmd_add,
    "remove": _cmd_remove,
    "test": _cmd_test,
}


def cmd_mcp(args: argparse.Namespace) -> int:
    """Dispatch ``himmy mcp <action>`` to its handler (default: ``list``)."""
    c = styles(sys.stdout)
    action = getattr(args, "action", None) or "list"
    handler = _ACTIONS.get(action)
    if handler is None:
        _eprint(
            f"{c['crimson']}unknown mcp action {action!r} — "
            f"use list | add | test | remove{c['reset']}"
        )
        return 2
    return handler(args, c)


__all__ = ["cmd_mcp"]
