"""Tests for Studio's live cognition stream (think → act → observe).

Covers the event → frame mapping (:class:`_Cognition`), read/write intent tagging,
agent/delegate tracking, and round-trip persistence of the recorded steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_service as ss
from himmy.api.studio_runs import reset_run_store
from himmy.core.events import EventType, RunEvent


def _tool_event(name: str, *, outcome: str = "success", **payload) -> RunEvent:
    et = EventType.TOOL_COMPLETED if outcome == "success" else EventType.TOOL_FAILED
    return RunEvent(
        event_type=et,
        payload={"tool_name": name, "tool_outcome": outcome, **payload},
    )


def _inference_event(
    *, model: str, in_tok: int, out_tok: int, cost: float, latency: float
) -> RunEvent:
    return RunEvent(
        event_type=EventType.INFERENCE_SUCCEEDED,
        latency_ms=latency,
        cost=cost,
        payload={
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "io": {"model": model},
        },
    )


def test_tool_frame_carries_args_result_intent_latency() -> None:
    cog = ss._Cognition({"egg_totals": True}, "livestock")
    frames = cog.frames(
        _tool_event(
            "egg_totals",
            tool_args={"days": 7},
            result="245 eggs",
            latency_ms=12.5,
        )
    )
    assert len(frames) == 1
    f = frames[0]
    assert f["type"] == "tool"
    assert f["name"] == "egg_totals"
    assert f["args"] == {"days": 7}
    assert f["result"] == "245 eggs"
    assert f["read_only"] is True
    assert f["latency_ms"] == 12.5
    # the step is recorded for persistence too
    assert [s["kind"] for s in cog.steps] == ["tool"]
    assert cog.tools_used == ["egg_totals"]


def test_write_intent_falls_back_to_name_heuristic() -> None:
    # Registry has no read_only flag → classify by name (log_* is a write).
    cog = ss._Cognition({}, "livestock")
    frame = cog.frames(_tool_event("log_eggs", tool_args={"count": 12}))[0]
    assert frame["read_only"] is False


def test_failed_tool_marks_outcome() -> None:
    cog = ss._Cognition({"pond_status": True}, "pond")
    frame = cog.frames(_tool_event("pond_status", outcome="error"))[0]
    assert frame["outcome"] == "error"


def test_synthetic_orchestration_tools_are_hidden() -> None:
    cog = ss._Cognition({}, "manager")
    assert cog.frames(_tool_event("ask_livestock")) == []
    assert cog.frames(_tool_event("transfer_to_pond")) == []
    assert cog.frames(_tool_event("final_answer")) == []
    assert cog.steps == []


def test_agent_switch_and_delegate_are_tracked() -> None:
    cog = ss._Cognition({}, "manager")
    # a worker starts running
    fr = cog.frames(
        RunEvent(
            event_type=EventType.AGENT_RUN_STARTED,
            payload={"persona_name": "livestock", "model_key": "ollama:qwen"},
        )
    )
    assert fr == [{"type": "agent", "name": "livestock", "model": "ollama:qwen"}]
    assert cog.active == "livestock"
    # the manager delegates to that worker
    fr = cog.frames(
        RunEvent(
            event_type=EventType.AGENT_DELEGATED,
            payload={"worker": "livestock", "task": "count eggs", "answer": "245"},
        )
    )
    assert fr[0]["type"] == "delegate"
    assert fr[0]["worker"] == "livestock"
    assert cog.delegate_answers == [("livestock", "245")]
    kinds = [s["kind"] for s in cog.steps]
    assert kinds == ["agent", "delegate"]


def test_memory_recall_emits_grounding_with_citations() -> None:
    import json

    cog = ss._Cognition({"recall": True}, "agent")
    result = json.dumps(
        {
            "query": "duck breed",
            "results": [
                {"text": "We keep Khaki Campbell ducks.", "similarity": 0.91},
                {"text": "The flock has 12 birds.", "similarity": 0.55},
            ],
        }
    )
    frames = cog.frames(_tool_event("recall", result=result))
    grounding = [f for f in frames if f["type"] == "grounding"]
    assert grounding, "recall should emit a grounding frame"
    g = grounding[0]
    assert g["source"] == "memory"
    assert g["query"] == "duck breed"
    assert len(g["citations"]) == 2
    assert g["citations"][0]["similarity"] == 0.91
    # also recorded as a grounding step for persistence
    assert any(s["kind"] == "grounding" for s in cog.steps)


def test_kb_search_tool_emits_knowledge_grounding() -> None:
    import json

    cog = ss._Cognition({"kb_search": True}, "agent")
    result = json.dumps(
        {
            "chunks": [
                {
                    "text": "Bardaghat lies in Nawalparasi.",
                    "similarity": 0.8,
                    "source_uri": "docs/geo.md",
                }
            ],
        }
    )
    g = [
        f
        for f in cog.frames(_tool_event("kb_search", result=result))
        if f["type"] == "grounding"
    ]
    assert g and g[0]["source"] == "knowledge"
    assert g[0]["citations"][0]["source_uri"] == "docs/geo.md"


def test_context_snapshot_grounding_maps_to_frames() -> None:
    cog = ss._Cognition({}, "agent")
    ev = RunEvent(
        event_type=EventType.CONTEXT_SNAPSHOT_BUILT,
        payload={
            "snapshot_id": "s1",
            "grounding": [
                {
                    "source": "knowledge",
                    "key": "farm_facts",
                    "query": "what breed",
                    "citations": [
                        {
                            "text": "Khaki Campbell",
                            "similarity": 0.7,
                            "source_uri": "kb://1",
                        }
                    ],
                }
            ],
        },
    )
    frames = cog.frames(ev)
    assert frames and frames[0]["type"] == "grounding"
    assert frames[0]["citations"][0]["source_uri"] == "kb://1"
    assert any(s["kind"] == "grounding" for s in cog.steps)


def test_non_grounding_tool_has_no_grounding_frame() -> None:
    import json

    cog = ss._Cognition({"egg_totals": True}, "agent")
    frames = cog.frames(_tool_event("egg_totals", result=json.dumps({"total": 245})))
    assert not any(f["type"] == "grounding" for f in frames)


def test_inference_events_emit_usage_frames_and_accumulate() -> None:
    cog = ss._Cognition({}, "manager")
    f1 = cog.frames(
        _inference_event(
            model="claude:sonnet", in_tok=100, out_tok=20, cost=0.01, latency=900.0
        )
    )
    usage = [f for f in f1 if f["type"] == "usage"]
    assert usage and usage[0]["total_cost"] == 0.01
    assert usage[0]["total_input_tokens"] == 100
    # a second inference on another model accumulates + splits per model
    cog.frames(
        _inference_event(
            model="ollama:qwen", in_tok=50, out_tok=10, cost=0.0, latency=300.0
        )
    )
    assert cog.input_tokens == 150
    assert cog.output_tokens == 30
    assert cog.cost == 0.01
    by_model = {u["model"]: u for u in cog.usage_by_model}
    assert set(by_model) == {"claude:sonnet", "ollama:qwen"}
    assert by_model["claude:sonnet"]["cost"] == 0.01
    assert by_model["ollama:qwen"]["inferences"] == 1


def test_analytics_rolls_up_cost_latency_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    from himmy.api.studio_runs import (
        ModelUsage,
        StudioRun,
        get_run_store,
    )

    store = get_run_store()
    store.save(
        StudioRun(
            id="r1",
            created_at="2026-06-01T10:00:00+00:00",
            status="ok",
            model="claude:sonnet",
            duration_ms=1000.0,
            input_tokens=100,
            output_tokens=20,
            cost=0.01,
            usage_by_model=[
                ModelUsage(
                    model="claude:sonnet",
                    input_tokens=100,
                    output_tokens=20,
                    cost=0.01,
                    inferences=1,
                )
            ],
        )
    )
    store.save(
        StudioRun(
            id="r2",
            created_at="2026-06-02T10:00:00+00:00",
            status="error",
            model="ollama:qwen",
            duration_ms=3000.0,
            input_tokens=50,
            output_tokens=5,
            cost=0.0,
            usage_by_model=[
                ModelUsage(
                    model="ollama:qwen",
                    input_tokens=50,
                    output_tokens=5,
                    cost=0.0,
                    inferences=1,
                )
            ],
        )
    )

    a = store.analytics()
    assert a.total_runs == 2
    assert a.ok_runs == 1 and a.error_runs == 1
    assert a.success_rate == 0.5
    assert abs(a.total_cost - 0.01) < 1e-9
    assert a.total_input_tokens == 150
    assert a.p50_latency_ms in (1000.0, 3000.0)
    assert a.p95_latency_ms == 3000.0
    models = {m.model: m for m in a.by_model}
    assert models["claude:sonnet"].cost == 0.01
    assert {d.day for d in a.by_day} == {"2026-06-01", "2026-06-02"}


def test_recorded_steps_round_trip_through_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    cog = ss._Cognition({"egg_totals": True}, "livestock")
    cog.frames(_tool_event("egg_totals", tool_args={"days": 7}, result="245"))

    spec = type(
        "S", (), {"name": "livestock", "provider": "ollama", "model": "default"}
    )()
    ss._record_run(
        run_id="run-cog-1",
        spec=spec,
        agent_path="livestock.yaml",
        provider="ollama",
        model=None,
        prompt="how many eggs?",
        history=[],
        output="245 eggs",
        tools=cog.tools_used,
        events=[],
        steps=cog.steps,
        status="ok",
        error=None,
        duration_ms=1.0,
        thread_id="t1",
    )

    from himmy.api.studio_runs import get_run_store

    run = get_run_store().get("run-cog-1")
    assert run is not None
    assert len(run.steps) == 1
    step = run.steps[0]
    assert step.kind == "tool"
    assert step.name == "egg_totals"
    assert step.args == {"days": 7}
    assert step.read_only is True
