#!/usr/bin/env python3
"""Operational health probe for a running himmy deployment.

Runs a handful of cheap, offline-first checks an operator (or a container
HEALTHCHECK / a monitoring cron) can use to answer "is this deployment healthy?"
without pulling in any third-party dependency: the HTTP probes use stdlib
``urllib``, the SQLite check uses stdlib ``sqlite3``, and the Postgres check is
skipped gracefully when ``asyncpg`` isn't installed.

Each check yields one ``ok | warn | fail | skip`` result. The process exit code is
the worst outcome: ``2`` if anything FAILED, ``1`` if anything only WARNED, ``0``
when everything is ok/skip/info — so it slots straight into shell ``&&`` chains and
HEALTHCHECK directives.

Usage:
    python scripts/ops_health.py                       # text report, env defaults
    python scripts/ops_health.py --json                # machine-readable results
    python scripts/ops_health.py --url http://h:8765 --timeout 5
    python scripts/ops_health.py --min-free-mb 2000    # stricter disk floor

Environment fallbacks: ``HIMMY_HEALTH_URL`` (default http://127.0.0.1:8765),
``HIMMY_OLLAMA_URL`` (no default — the Ollama check is skipped unless ``--ollama-url``
or this is explicitly set), ``HIMMY_STORE_PATH`` (default .himmy/storage.db),
``HIMMY_DATABASE_URL`` (selects the Postgres check).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DEFAULT_HEALTH_URL = "http://127.0.0.1:8765"
_DEFAULT_STORE_PATH = ".himmy/storage.db"

# disk thresholds (bytes)
_WARN_FREE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
_FAIL_FREE_BYTES = 100 * 1024 * 1024  # 100 MiB


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


@dataclass
class Check:
    """One health probe result."""

    name: str
    status: str  # "ok" | "warn" | "fail" | "skip" | "info"
    detail: str


HttpGetter = Callable[[str, float], "tuple[int, bytes]"]


def _http_get(url: str, timeout: float) -> tuple[int, bytes]:
    """GET ``url`` and return ``(status_code, body)``; raises on transport error."""
    req = urllib.request.Request(url, method="GET")  # noqa: S310 - operator-supplied
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return int(resp.status), resp.read()


def check_health(url: str, timeout: float, http: HttpGetter) -> Check:
    endpoint = url.rstrip("/") + "/health"
    try:
        status, body = http(endpoint, timeout)
    except Exception as exc:  # noqa: BLE001 - any transport error is a fail here
        return Check("health", "fail", f"{endpoint} unreachable ({exc})")
    if status != 200:
        return Check("health", "fail", f"{endpoint} returned HTTP {status}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return Check("health", "warn", f"{endpoint} returned non-JSON body")
    if payload.get("status") == "ok":
        return Check("health", "ok", f"{endpoint} status=ok")
    return Check("health", "warn", f"{endpoint} status={payload.get('status')!r}")


def _classify_disk(free_bytes: int, warn_below: int, fail_below: int) -> str:
    if free_bytes < fail_below:
        return "fail"
    if free_bytes < warn_below:
        return "warn"
    return "ok"


def check_disk(store_path: str, warn_below: int, fail_below: int) -> Check:
    target = Path(store_path).resolve().parent
    # Walk up to the nearest existing ancestor so a not-yet-created store dir still
    # reports the filesystem it will live on.
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return Check("disk", "warn", f"could not stat {probe} ({exc})")
    free_gb = usage.free / (1024**3)
    status = _classify_disk(usage.free, warn_below, fail_below)
    return Check("disk", status, f"{free_gb:.2f} GiB free at {probe}")


def check_sqlite(store_path: str, dsn: str | None) -> Check:
    if _is_postgres_dsn(dsn):
        return Check("sqlite_quick_check", "skip", "Postgres backend in use")
    path = Path(store_path)
    if not path.exists():
        return Check("sqlite_quick_check", "skip", f"{path} does not exist yet")
    try:
        conn = sqlite3.connect(path)
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return Check("sqlite_quick_check", "fail", f"{path}: {exc}")
    result = str(row[0]) if row else ""
    if result == "ok":
        return Check("sqlite_quick_check", "ok", f"{path}: ok")
    return Check("sqlite_quick_check", "fail", f"{path}: {result}")


def _is_postgres_dsn(dsn: str | None) -> bool:
    if not dsn:
        return False
    lowered = dsn.strip().lower()
    return lowered.startswith(("postgres://", "postgresql://", "postgresql+"))


def check_postgres(dsn: str | None, timeout: float) -> Check:
    if not _is_postgres_dsn(dsn):
        return Check("postgres", "skip", "no Postgres DSN configured")
    try:
        import asyncpg  # noqa: F401, PLC0415 - optional dependency
    except ImportError:
        return Check("postgres", "skip", "asyncpg not installed")
    try:
        from himmy.services.storage.postgres import STORAGE_MIGRATIONS

        code_max: int | None = max(v for v, _, _ in STORAGE_MIGRATIONS)
    except Exception:  # noqa: BLE001 - himmy not importable → still probe connectivity
        code_max = None

    import asyncio

    async def _probe() -> tuple[int, int | None]:
        import asyncpg  # noqa: PLC0415

        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout)
        try:
            await conn.fetchval("SELECT 1")
            applied = await conn.fetchval("SELECT max(version) FROM schema_migrations")
        finally:
            await conn.close()
        return 1, (int(applied) if applied is not None else None)

    try:
        _, applied_max = asyncio.run(_probe())
    except Exception as exc:  # noqa: BLE001 - unreachable DB is a fail
        return Check("postgres", "fail", f"connect/query failed ({exc})")
    if code_max is None:
        return Check("postgres", "ok", f"SELECT 1 ok; applied max={applied_max}")
    if applied_max is None or applied_max < code_max:
        return Check(
            "postgres",
            "warn",
            f"pending migrations: applied={applied_max} code max={code_max}",
        )
    return Check("postgres", "ok", f"up to date (version {applied_max})")


def check_ollama(url: str | None, timeout: float, http: HttpGetter) -> Check:
    # Ollama is opt-in: if neither --ollama-url nor $HIMMY_OLLAMA_URL was explicitly
    # provided, skip rather than probe a default localhost:11434 — otherwise every
    # non-Ollama deployment would WARN forever and a container HEALTHCHECK that maps
    # warn→exit 1 would report the container permanently unhealthy.
    if not url:
        return Check(
            "ollama",
            "skip",
            "no Ollama URL configured (set --ollama-url or $HIMMY_OLLAMA_URL)",
        )
    endpoint = url.rstrip("/") + "/api/tags"
    try:
        status, body = http(endpoint, timeout)
    except Exception as exc:  # noqa: BLE001 - Ollama is optional: unreachable = warn
        return Check("ollama", "warn", f"{endpoint} unreachable ({exc})")
    if status != 200:
        return Check("ollama", "warn", f"{endpoint} returned HTTP {status}")
    try:
        models = json.loads(body.decode("utf-8")).get("models", [])
        return Check("ollama", "ok", f"{len(models)} model(s) available")
    except (ValueError, UnicodeDecodeError):
        return Check("ollama", "warn", f"{endpoint} returned non-JSON body")


def check_versions() -> Check:
    py = ".".join(str(n) for n in sys.version_info[:3])
    try:
        from himmy import __version__ as himmy_version
    except Exception:  # noqa: BLE001 - himmy may not be importable from here
        return Check("versions", "info", f"python {py}; himmy version unknown")
    return Check("versions", "info", f"himmy {himmy_version}; python {py}")


def _overall_exit(checks: list[Check]) -> int:
    statuses = {c.status for c in checks}
    if "fail" in statuses:
        return 2
    if "warn" in statuses:
        return 1
    return 0


def run_checks(args: argparse.Namespace, http: HttpGetter = _http_get) -> list[Check]:
    """Run every probe and return the result list (pure aside from I/O)."""
    timeout = float(args.timeout)
    dsn = args.dsn
    return [
        check_health(args.url, timeout, http),
        check_disk(args.store_path, args.warn_free_bytes, args.fail_free_bytes),
        check_sqlite(args.store_path, dsn),
        check_postgres(dsn, timeout),
        check_ollama(args.ollama_url, timeout, http),
        check_versions(),
    ]


_BADGE = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "SKIP", "info": "INFO"}


def _render_text(checks: list[Check]) -> None:
    width = max((len(c.name) for c in checks), default=0)
    for c in checks:
        print(f"[{_BADGE.get(c.status, c.status)}] {c.name:<{width}}  {c.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=os.environ.get("HIMMY_HEALTH_URL", _DEFAULT_HEALTH_URL),
        help="base URL of the himmy server (health probed at <url>/health)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("HIMMY_OLLAMA_URL"),
        help="base URL of the Ollama server (tags probed at <url>/api/tags); "
        "Ollama check is skipped unless this or $HIMMY_OLLAMA_URL is set",
    )
    parser.add_argument(
        "--store-path",
        default=os.environ.get("HIMMY_STORE_PATH", _DEFAULT_STORE_PATH),
        help="path to the SQLite store (for disk + quick_check)",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("HIMMY_DATABASE_URL"),
        help="Postgres DSN (default $HIMMY_DATABASE_URL); enables the Postgres check",
    )
    parser.add_argument("--timeout", type=float, default=3.0, help="per-probe seconds")
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=None,
        help="warn when free disk drops below this many MiB (default 1024)",
    )
    parser.add_argument("--json", action="store_true", help="emit results as JSON")
    args = parser.parse_args(argv)

    if args.min_free_mb is not None:
        args.warn_free_bytes = int(args.min_free_mb) * 1024 * 1024
    else:
        args.warn_free_bytes = _WARN_FREE_BYTES
    args.fail_free_bytes = _FAIL_FREE_BYTES

    checks = run_checks(args)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        _render_text(checks)
    code = _overall_exit(checks)
    if code:
        _eprint(f"ops_health: exit {code} (worst status among {len(checks)} checks)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
