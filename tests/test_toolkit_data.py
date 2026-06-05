"""Tests for the data pack: read-only sql_query against SQLite. Offline."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.data import register_data_pack
from tests.conftest import run_async


def _db(tmp_path: Path) -> str:
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'apple'), (2, 'banana')")
    conn.commit()
    conn.close()
    return str(path)


def _query(db: str):
    registry = ToolRegistry()
    register_data_pack(registry, ToolkitConfig(sqlite_path=db))
    return registry.handler_for("sql_query")


def test_select_returns_rows(tmp_path: Path) -> None:
    handler = _query(_db(tmp_path))
    out = run_async(handler({"sql": "SELECT id, name FROM items ORDER BY id"}))
    assert out["columns"] == ["id", "name"]
    assert out["rows"] == [{"id": 1, "name": "apple"}, {"id": 2, "name": "banana"}]
    assert out["row_count"] == 2


def test_select_with_params_and_limit(tmp_path: Path) -> None:
    handler = _query(_db(tmp_path))
    out = run_async(
        handler({"sql": "SELECT name FROM items WHERE id = ?", "params": [2]})
    )
    assert out["rows"] == [{"name": "banana"}]


def test_insert_is_denied(tmp_path: Path) -> None:
    handler = _query(_db(tmp_path))
    with pytest.raises(sqlite3.DatabaseError):
        run_async(handler({"sql": "INSERT INTO items VALUES (3, 'cherry')"}))


def test_ddl_is_denied(tmp_path: Path) -> None:
    handler = _query(_db(tmp_path))
    with pytest.raises(sqlite3.DatabaseError):
        run_async(handler({"sql": "DROP TABLE items"}))


def test_multiple_statements_rejected(tmp_path: Path) -> None:
    handler = _query(_db(tmp_path))
    with pytest.raises(ToolSecurityError):
        run_async(handler({"sql": "SELECT 1; DROP TABLE items"}))


def test_no_database_configured() -> None:
    registry = ToolRegistry()
    register_data_pack(registry, ToolkitConfig())
    with pytest.raises(ToolSecurityError):
        run_async(registry.handler_for("sql_query")({"sql": "SELECT 1"}))
