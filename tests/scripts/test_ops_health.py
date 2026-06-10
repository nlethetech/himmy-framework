"""Tests for ``scripts/ops_health.py`` (loaded by file path; no real network)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = _REPO_ROOT / "scripts" / "ops_health.py"
    spec = importlib.util.spec_from_file_location("ops_health", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so dataclass annotation resolution works
    spec.loader.exec_module(mod)
    return mod


ops = _load()


def _args(**over: object) -> argparse.Namespace:
    base = {
        "url": "http://127.0.0.1:8765",
        "ollama_url": "http://localhost:11434",
        "store_path": ".himmy/storage.db",
        "dsn": None,
        "timeout": 1.0,
        "warn_free_bytes": 1 << 30,
        "fail_free_bytes": 100 << 20,
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_overall_exit_severity() -> None:
    Check = ops.Check
    assert ops._overall_exit([Check("a", "ok", "")]) == 0
    assert ops._overall_exit([Check("a", "skip", ""), Check("b", "info", "")]) == 0
    assert ops._overall_exit([Check("a", "ok", ""), Check("b", "warn", "")]) == 1
    assert ops._overall_exit([Check("a", "warn", ""), Check("b", "fail", "")]) == 2


def test_disk_classifier() -> None:
    warn, fail = 1 << 30, 100 << 20
    assert ops._classify_disk(2 << 30, warn, fail) == "ok"
    assert ops._classify_disk(500 << 20, warn, fail) == "warn"
    assert ops._classify_disk(10 << 20, warn, fail) == "fail"


def test_sqlite_quick_check_ok(tmp_path: Path) -> None:
    db = tmp_path / "storage.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    result = ops.check_sqlite(str(db), dsn=None)
    assert result.status == "ok"


def test_sqlite_quick_check_skips_when_absent(tmp_path: Path) -> None:
    result = ops.check_sqlite(str(tmp_path / "nope.db"), dsn=None)
    assert result.status == "skip"


def test_sqlite_check_skips_for_postgres() -> None:
    result = ops.check_sqlite(".himmy/storage.db", dsn="postgresql://u@h/db")
    assert result.status == "skip"


def test_check_health_ok_with_stubbed_http() -> None:
    def fake(url: str, timeout: float) -> tuple[int, bytes]:
        assert url.endswith("/health")
        return 200, json.dumps({"status": "ok"}).encode()

    result = ops.check_health("http://h:8765", 1.0, fake)
    assert result.status == "ok"


def test_check_health_fail_on_exception() -> None:
    def fake(url: str, timeout: float) -> tuple[int, bytes]:
        raise OSError("connection refused")

    result = ops.check_health("http://h:8765", 1.0, fake)
    assert result.status == "fail"


def test_check_ollama_warn_when_unreachable() -> None:
    def fake(url: str, timeout: float) -> tuple[int, bytes]:
        raise OSError("no ollama")

    result = ops.check_ollama("http://h:11434", 1.0, fake)
    assert result.status == "warn"  # offline-first: not a hard fail


def test_check_ollama_skips_when_url_not_configured() -> None:
    def fake(url: str, timeout: float) -> tuple[int, bytes]:  # pragma: no cover
        raise AssertionError("must not probe when no Ollama URL is configured")

    # No --ollama-url / $HIMMY_OLLAMA_URL → skip, never warn (so a non-Ollama
    # deployment's HEALTHCHECK stays exit-0 instead of permanently unhealthy).
    result = ops.check_ollama(None, 1.0, fake)
    assert result.status == "skip"


def test_run_checks_exit_zero_when_ollama_unconfigured(tmp_path: Path) -> None:
    """With no Ollama URL, the unconfigured Ollama check is skip — so a healthy
    non-Ollama deployment exits 0, not 1."""
    db = tmp_path / "storage.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()

    def fake(url: str, timeout: float) -> tuple[int, bytes]:
        # Only the health endpoint is probed; Ollama is skipped.
        assert url.endswith("/health")
        return 200, json.dumps({"status": "ok"}).encode()

    args = _args(store_path=str(db), ollama_url=None)
    checks = ops.run_checks(args, http=fake)
    by_name = {c.name: c for c in checks}
    assert by_name["ollama"].status == "skip"
    assert ops._overall_exit(checks) == 0


def test_run_checks_clean_exit_zero(tmp_path: Path) -> None:
    db = tmp_path / "storage.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()

    def fake(url: str, timeout: float) -> tuple[int, bytes]:
        if url.endswith("/health"):
            return 200, json.dumps({"status": "ok"}).encode()
        return 200, json.dumps({"models": [{"name": "x"}]}).encode()

    args = _args(store_path=str(db))
    checks = ops.run_checks(args, http=fake)
    # health ok, sqlite ok, ollama ok, disk ok (real fs has space), pg skip, versions info
    assert ops._overall_exit(checks) == 0
    names = {c.name for c in checks}
    assert {
        "health",
        "disk",
        "sqlite_quick_check",
        "postgres",
        "ollama",
        "versions",
    } <= names
