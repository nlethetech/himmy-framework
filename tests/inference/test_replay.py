"""Deterministic record-and-replay: a recorded run replays exactly, no provider, no
tool side effects."""

from __future__ import annotations

import pytest

from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    BoundTool,
    InferenceMessage,
    InferenceRequest,
    ResponseFormat,
    ToolReturnRecord,
)
from himmy.services.inference.replay import (
    InferenceCassette,
    RecordingClientManager,
    ReplayClientManager,
    ReplayError,
)
from tests.conftest import run_async


def _req(text: str, tools: list[BoundTool] | None = None) -> InferenceRequest:
    return InferenceRequest(
        messages=[InferenceMessage(role="user", content=text)],
        bound_tools=tools or [],
    )


def test_record_then_replay_is_identical() -> None:
    rec = RecordingClientManager(StubClientManager())
    r1 = run_async(rec.generate(_req("hello")))
    r2 = run_async(rec.generate(_req("world")))
    assert len(rec.cassette.entries) == 2

    replay = ReplayClientManager(rec.cassette)
    # Fresh requests (new request_ids) with the same content replay the recorded outputs.
    assert run_async(replay.generate(_req("hello"))).output_text == r1.output_text
    assert run_async(replay.generate(_req("world"))).output_text == r2.output_text
    assert replay.remaining() == 0


def test_replay_preserves_caller_request_id() -> None:
    rec = RecordingClientManager(StubClientManager())
    run_async(rec.generate(_req("x")))
    replay = ReplayClientManager(rec.cassette)
    req = _req("x")
    out = run_async(replay.generate(req))
    assert out.request_id == req.request_id  # stamped with the live id, body recorded


def test_replay_fifo_for_duplicate_requests() -> None:
    # Two identical requests (same cache key) must replay in recorded order.
    rec = RecordingClientManager(StubClientManager())
    run_async(rec.generate(_req("same")))
    run_async(rec.generate(_req("same")))
    assert rec.cassette.entries[0].cache_key == rec.cassette.entries[1].cache_key
    replay = ReplayClientManager(rec.cassette)
    run_async(replay.generate(_req("same")))
    assert replay.remaining() == 1  # one consumed, one left


def test_replay_miss_is_strict() -> None:
    replay = ReplayClientManager(InferenceCassette())
    with pytest.raises(ReplayError):
        run_async(replay.generate(_req("never recorded")))


def test_replay_miss_falls_back_when_configured() -> None:
    replay = ReplayClientManager(InferenceCassette(), fallback=StubClientManager())
    out = run_async(replay.generate(_req("uncorded")))
    assert out.status.value == "SUCCESS"  # served by the fallback


def test_replay_does_not_reexecute_tools() -> None:
    calls: list[dict] = []

    async def handler(args: dict) -> ToolReturnRecord:
        calls.append(args)
        return ToolReturnRecord(tool_call_id="", tool_name="t", content={"ran": True})

    tool = BoundTool(
        name="t",
        description="d",
        args_json_schema={"type": "object", "properties": {}},
        handler=handler,
    )
    rec = RecordingClientManager(StubClientManager())
    run_async(
        rec.generate(_req("go", [tool]))
    )  # records; stub executes the handler once
    ran_during_record = len(calls)

    replay = ReplayClientManager(rec.cassette)
    run_async(replay.generate(_req("go", [tool])))
    # The handler did NOT fire again on replay — the recorded return is used.
    assert len(calls) == ran_during_record


def test_cassette_round_trips_through_disk(tmp_path) -> None:
    rec = RecordingClientManager(StubClientManager(), label="run-42")
    run_async(rec.generate(_req("persist me")))
    path = rec.dump(tmp_path / "cassette.json")
    loaded = InferenceCassette.load(path)
    assert loaded.label == "run-42"
    assert loaded.entries[0].cache_key == rec.cassette.entries[0].cache_key
    # And a replay from the file works.
    replay = ReplayClientManager.from_file(path)
    assert run_async(replay.generate(_req("persist me"))).status.value == "SUCCESS"


def test_structured_response_replays() -> None:
    rec = RecordingClientManager(StubClientManager())
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="extract")],
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    recorded = run_async(rec.generate(req))
    replay = ReplayClientManager(rec.cassette)
    out = run_async(replay.generate(req))
    assert out.output_structured == recorded.output_structured
