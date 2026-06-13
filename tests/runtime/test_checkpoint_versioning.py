"""Tests for AgentCheckpoint schema versioning and stepwise migration."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.runtime import SingleAgentRuntime, SqliteCheckpointStore
from himmy.runtime.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    AgentCheckpoint,
    _migrate_checkpoint,
)
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


class _ApprovalGateManager:
    """Calls the (approval-gated) bound tool until a tool result exists, then finals."""

    def resolve(self, model_key: str) -> str:
        return model_key

    async def generate(self, request: Any) -> InferenceResponse:
        tool_in_history = any(
            getattr(m, "role", "") == "tool" for m in request.messages
        )
        if not tool_in_history and request.bound_tools:
            ret = await request.tool_executor(request.bound_tools[0].name, {})
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="invoking the tool",
                tool_calls=[
                    ToolCallRecord(
                        tool_call_id=ret.tool_call_id, tool_name=ret.tool_name, args={}
                    )
                ],
                tool_returns=[ret],
                input_tokens=1,
                output_tokens=1,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="done with the result",
            input_tokens=1,
            output_tokens=1,
        )


def _hitl_runtime(store: SqliteCheckpointStore) -> SingleAgentRuntime:
    storage = StorageService()
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="wire_money",
        handler=lambda args: {"sent": True},
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=True,
    )
    return SingleAgentRuntime(
        inference_service=InferenceService(_ApprovalGateManager()),
        memory_store=storage,
        tool_service=ToolService(registry, event_sink=storage),
        checkpoint_store=store,
    )


def _task() -> tuple[Persona, Task]:
    return Persona(name="treasurer"), Task(
        title="payment",
        prompt="wire 1000 to the vendor",
        context={"tool_names": ["wire_money"], "response_format": "AUTO_TOOLS"},
    )


def _rewrite_row(path: str, checkpoint_id: str, data: dict[str, Any]) -> None:
    """Overwrite the persisted JSON blob for a checkpoint row in place."""
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE agent_checkpoints SET data = ? WHERE checkpoint_id = ?",
        (json.dumps(data), checkpoint_id),
    )
    conn.commit()
    conn.close()


def _read_row(path: str, checkpoint_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT data FROM agent_checkpoints WHERE checkpoint_id = ?",
        (checkpoint_id,),
    ).fetchone()
    conn.close()
    return dict(json.loads(row[0]))


def test_current_checkpoints_round_trip_with_version(tmp_path) -> None:
    """A freshly saved checkpoint persists and loads at the current schema version."""
    path = str(tmp_path / "checkpoints.db")
    store = SqliteCheckpointStore(path)
    original = AgentCheckpoint(persona={"name": "p"}, task={"title": "t"})
    assert original.schema_version == CHECKPOINT_SCHEMA_VERSION
    store.save(original)

    loaded = store.load(original.checkpoint_id)
    assert loaded is not None
    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.model_dump() == original.model_dump()
    assert "schema_version" in _read_row(path, original.checkpoint_id)
    store.close()


def test_legacy_versionless_checkpoint_migrates_on_load(tmp_path) -> None:
    """A pre-versioning (no schema_version) row loads as version 0 -> current."""
    path = str(tmp_path / "checkpoints.db")
    store = SqliteCheckpointStore(path)
    checkpoint = AgentCheckpoint(persona={"name": "p"}, task={"title": "t"})
    store.save(checkpoint)

    # Hand-build the legacy on-disk shape: the same row without schema_version.
    legacy = _read_row(path, checkpoint.checkpoint_id)
    del legacy["schema_version"]
    _rewrite_row(path, checkpoint.checkpoint_id, legacy)

    loaded = store.load(checkpoint.checkpoint_id)
    assert loaded is not None
    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.persona == {"name": "p"}
    assert store.list_by_status(loaded.status)[0].schema_version == (
        CHECKPOINT_SCHEMA_VERSION
    )
    store.close()


def test_legacy_checkpoint_resumes_end_to_end(tmp_path) -> None:
    """A real paused run stripped of its version stamp still migrates and resumes."""
    path = str(tmp_path / "checkpoints.db")
    store = SqliteCheckpointStore(path)
    rt = _hitl_runtime(store)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))
    assert paused.stopped_reason == "awaiting_approval"
    store.close()

    # Simulate a checkpoint written by an older himmy (no schema_version field).
    legacy = _read_row(path, paused.checkpoint_id)
    del legacy["schema_version"]
    _rewrite_row(path, paused.checkpoint_id, legacy)

    rt2 = _hitl_runtime(SqliteCheckpointStore(path))
    final = run_async(rt2.resume_agent_loop(paused.checkpoint_id, approved=True))
    assert final.stopped_reason == "final"
    tool_msgs = [m for m in final.thread.messages if m.role == MessageRole.TOOL]
    assert any('"sent"' in (m.content or "") for m in tool_msgs)  # the tool really ran


def test_future_version_is_rejected_with_a_clear_error(tmp_path) -> None:
    """A checkpoint from a newer himmy fails loudly, not with a validation error."""
    path = str(tmp_path / "checkpoints.db")
    store = SqliteCheckpointStore(path)
    checkpoint = AgentCheckpoint(persona={"name": "p"}, task={"title": "t"})
    store.save(checkpoint)

    future = _read_row(path, checkpoint.checkpoint_id)
    future["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
    _rewrite_row(path, checkpoint.checkpoint_id, future)

    with pytest.raises(HimmyError, match="newer than this himmy build"):
        store.load(checkpoint.checkpoint_id)
    with pytest.raises(HimmyError, match="newer than this himmy build"):
        store.list_by_status(checkpoint.status)
    store.close()


def test_v2_checkpoint_migrates_to_final_output_none() -> None:
    """A v2 record (no ``final_output``) migrates forward with the field defaulted None.

    The v2->v3 step (added with the crash-recovery anchor, #2) must leave the not-yet-
    recorded sentinel as ``None`` — distinct from a legitimately-recorded empty answer.
    """
    migrated = _migrate_checkpoint(
        {"schema_version": 2, "persona": {"name": "p"}, "executed_tool_results": {}}
    )
    assert migrated["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert migrated["final_output"] is None
    # A v2 row that somehow already carried the field keeps its value (idempotent).
    kept = _migrate_checkpoint(
        {"schema_version": 2, "persona": {"name": "p"}, "final_output": "answer"}
    )
    assert kept["final_output"] == "answer"


def test_migrate_checkpoint_unit_behaviors() -> None:
    """_migrate_checkpoint stamps legacy dicts and rejects bad versions."""
    migrated = _migrate_checkpoint({"persona": {"name": "p"}})
    assert migrated["schema_version"] == CHECKPOINT_SCHEMA_VERSION

    current = _migrate_checkpoint({"schema_version": CHECKPOINT_SCHEMA_VERSION})
    assert current["schema_version"] == CHECKPOINT_SCHEMA_VERSION

    with pytest.raises(HimmyError, match="newer than this himmy build"):
        _migrate_checkpoint({"schema_version": CHECKPOINT_SCHEMA_VERSION + 5})
    with pytest.raises(HimmyError, match="Invalid checkpoint schema_version"):
        _migrate_checkpoint({"schema_version": -1})
    with pytest.raises(HimmyError, match="Invalid checkpoint schema_version"):
        _migrate_checkpoint({"schema_version": "two"})
