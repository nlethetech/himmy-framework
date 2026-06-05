"""The data pack's empty-result value hint + sql_schema (self-correcting SQL)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from himmy.toolkit import ToolkitConfig, register_packs
from tests.conftest import run_async


def _service(tmp_path: Path) -> ToolService:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE animals(name TEXT, count INTEGER);"
        "INSERT INTO animals VALUES ('ducks',12),('chickens',30),('bees',40000);"
    )
    conn.commit()
    conn.close()
    cfg = ToolkitConfig.from_env().model_copy(update={"sqlite_path": str(db)})
    reg = ToolRegistry()
    register_packs(reg, ["data"], cfg)
    return ToolService(reg)


def _q(svc: ToolService, sql: str) -> dict:
    res = run_async(
        svc.execute(ToolInvocation(tool_name="sql_query", args={"sql": sql}))
    )
    assert res.outcome == "success", res
    return res.result


def test_empty_result_hints_real_values(tmp_path: Path) -> None:
    # The exact systematic failure: 'chicken' (singular) vs the stored 'chickens'.
    result = _q(
        svc := _service(tmp_path), "SELECT count FROM animals WHERE name='chicken'"
    )
    assert result["row_count"] == 0
    assert "hint" in result
    assert "chickens" in result["hint"]  # the real value is surfaced
    assert "animals.name" in result["hint"]
    # And the corrected query (which the model would now run) returns the answer.
    fixed = _q(svc, "SELECT count FROM animals WHERE name='chickens'")
    assert fixed["rows"] == [{"count": 30}]
    assert "hint" not in fixed  # a matching result carries no hint


def test_hint_suggests_closest_value(tmp_path: Path) -> None:
    # A near-miss literal gets an explicit, copy-pasteable correction.
    result = _q(_service(tmp_path), "SELECT count FROM animals WHERE name='chicken'")
    assert "Did you mean" in result["hint"]
    assert "name = 'chickens'" in result["hint"]  # the exact corrected predicate
    assert "you wrote 'chicken'" in result["hint"]


def test_hint_only_lists_text_columns(tmp_path: Path) -> None:
    result = _q(_service(tmp_path), "SELECT * FROM animals WHERE name='nope'")
    assert "animals.name" in result["hint"]
    assert "animals.count" not in result["hint"]  # integer column not dumped


def test_sql_schema_lists_tables_and_columns(tmp_path: Path) -> None:
    res = run_async(
        _service(tmp_path).execute(ToolInvocation(tool_name="sql_schema", args={}))
    )
    assert res.outcome == "success"
    tables = res.result["tables"]
    assert "animals" in tables
    cols = {c["column"]: c["type"] for c in tables["animals"]}
    assert cols == {"name": "TEXT", "count": "INTEGER"}
