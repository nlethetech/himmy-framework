"""Tier-3 visibility & autonomy in the chat REPL: /why, budgets, /spawn+/jobs, /compact.

Each feature is exercised against the offline stub (no network) through the real
``ChatRepl`` code paths — the recorder/budget/job/compaction machinery — rather than a
parallel mock. Output is captured by swapping ``_eprint`` for a list-appender (the same
pattern the approval/plan tests use).
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Any

import pytest

from himmy.cli import permissions
from himmy.cli.repl import ChatRepl, _Job
from himmy.cli.ui import LiveRunUI, TurnRecorder
from tests.conftest import run_async


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "file": None,
        "name": "t",
        "instruction": None,
        "provider": "stub",
        "model": None,
        "message": None,
        "session": None,
        "continue_last": False,
        "yolo": False,
        "safe": False,
        "budget": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _stub_repl(
    inputs: list[str] | None = None, *, tty: bool | None = None, **kw: Any
) -> tuple[ChatRepl, list[str]]:
    """A real stub-backed ChatRepl with captured stderr lines."""
    from himmy.config.agent_spec import AgentSpec

    it = iter(inputs or [])
    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(**kw), input_fn=lambda _p: next(it), tty=tty)
    repl.rebuild()
    lines: list[str] = []
    repl._eprint = staticmethod(  # type: ignore[method-assign,assignment]
        lambda *a: lines.append(" ".join(str(x) for x in a))
    )
    return repl, lines


def _event(kind: str, **kw: Any) -> Any:
    """A minimal RunEvent for recorder-driven tests."""
    from himmy.core.events import EventType, RunEvent

    return RunEvent(event_type=EventType(kind), **kw)


# ------------------------------------------------------------------ /why


def test_why_nothing_yet() -> None:
    repl, lines = _stub_repl()
    repl._why()
    assert any("nothing to explain yet" in line for line in lines)


def test_why_after_a_turn_shows_timeline_and_args() -> None:
    """After a real stub turn, /why renders the timeline + untruncated tool args."""
    repl, lines = _stub_repl()
    # Seed the recorder with a turn's worth of events including a tool call with
    # long args that must NOT be truncated by /why.
    long_args = {"q": "x" * 300}
    repl.recorder.start_turn()
    run_async(repl.recorder.handle(_event("AGENT_RUN_STARTED")))
    run_async(
        repl.recorder.handle(
            _event(
                "TOOL_CALLED", payload={"tool_name": "search", "tool_args": long_args}
            )
        )
    )
    run_async(
        repl.recorder.handle(
            _event(
                "INFERENCE_SUCCEEDED",
                latency_ms=42.0,
                cost=0.0123,
                payload={"input_tokens": 10, "output_tokens": 5},
            )
        )
    )
    repl._why()
    blob = "\n".join(lines)
    assert "why — last turn" in blob
    assert "run started" in blob  # timeline line
    assert "x" * 300 in blob  # full, untruncated args
    assert "search" in blob
    assert "42ms" in blob and "$0.0123" in blob  # inference latency + cost


def test_why_decodes_unicode_result_and_truncates() -> None:
    """BUG FIX A: a JSON-string tool result renders decoded unicode, truncated.

    A tool result is commonly an ALREADY-serialized JSON string. The old /why path
    re-json.dumps'd it with ensure_ascii=True, double-escaping Devanagari into
    \\uXXXX sprawl over many lines. The fix decodes it (ensure_ascii=False) and caps
    the block at ~6 lines with a '… +N more lines' marker.
    """
    import json

    repl, lines = _stub_repl()
    # A serialized JSON string with Devanagari + enough keys to overflow 6 lines.
    payload_obj = {
        "city": "काठमाडौं",
        **{f"k{i}": f"v{i}" for i in range(12)},
    }
    serialized = json.dumps(payload_obj)  # ensure_ascii=True by default → \uXXXX
    assert "\\u0915" in serialized  # the raw string IS escaped going in
    repl.recorder.start_turn()
    run_async(
        repl.recorder.handle(
            _event(
                "TOOL_COMPLETED",
                payload={"tool_name": "geocode", "result": serialized},
            )
        )
    )
    repl._why()
    blob = "\n".join(lines)
    assert "काठमाडौं" in blob  # decoded, not क
    assert "\\u0915" not in blob  # no double-escaped escapes leaked through
    assert "more line" in blob  # truncated with the marker


# -------------------------------------------------------------- budgets


def test_budget_flag_parses_and_wins_over_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "himmy.toml").write_text("[limits]\nsession_budget = 0.50\n")
    # No flag → toml value.
    repl, _ = _stub_repl()
    assert repl.budget == 0.50
    # Flag wins.
    repl2, _ = _stub_repl(budget=2.0)
    assert repl2.budget == 2.0


def test_budget_toml_loader(tmp_path: Path) -> None:
    (tmp_path / "himmy.toml").write_text("[limits]\nsession_budget = 1.25\n")
    assert permissions.load_session_budget(tmp_path) == 1.25
    assert permissions.load_session_budget(tmp_path / "nope") is None


def test_budget_warns_once_at_80pct() -> None:
    repl, lines = _stub_repl(budget=1.0)
    repl.recorder.total_cost = 0.85
    repl._maybe_warn_budget()
    repl._maybe_warn_budget()  # second call must NOT re-warn
    warnings = [line for line in lines if "budget" in line and "≥80%" in line]
    assert len(warnings) == 1


def test_budget_gate_blocks_turn_on_no() -> None:
    """At/over budget, a scripted 'n' blocks the turn; 'y' allows it."""
    repl_n, _ = _stub_repl(["n"], budget=1.0)
    repl_n.recorder.total_cost = 1.5
    assert repl_n._budget_blocks_turn() is True

    repl_y, _ = _stub_repl(["y"], budget=1.0)
    repl_y.recorder.total_cost = 1.5
    assert repl_y._budget_blocks_turn() is False

    # Under budget: never prompts (would raise StopIteration if it tried to read).
    repl_u, _ = _stub_repl([], budget=1.0)
    repl_u.recorder.total_cost = 0.1
    assert repl_u._budget_blocks_turn() is False


def test_run_threads_cost_budget_to_loop() -> None:
    """cmd_run's _answer passes cost_budget into run_agent_loop and reports the stop."""
    from himmy.cli import commands

    seen: dict[str, Any] = {}

    class _FakeRuntime:
        async def run_agent_loop(self, *a: Any, **kw: Any) -> Any:
            seen["cost_budget"] = kw.get("cost_budget")

            class _Loop:
                stopped_reason = "budget"
                final = object()

            return _Loop()

    stops: list[str] = []
    run_async(
        commands._answer(
            _FakeRuntime(),
            object(),
            object(),
            has_tools=True,
            cost_budget=0.25,
            on_stop=stops.append,
        )
    )
    assert seen["cost_budget"] == 0.25
    assert stops == ["budget"]


# ---------------------------------------------------------- spawn / jobs


def test_spawn_completes_and_jobs_lists_and_reads() -> None:
    """/spawn runs a stub sub-agent turn; /jobs lists it done; /jobs id shows answer."""
    repl, lines = _stub_repl()
    repl._spawn("summarize the news")
    # Wait for the daemon thread to finish (bounded).
    for _ in range(100):
        with repl._jobs_lock:
            done = repl.jobs and repl.jobs[0].status != "running"
        if done:
            break
        time.sleep(0.02)
    with repl._jobs_lock:
        assert repl.jobs[0].status == "done"
        assert repl.jobs[0].result  # the stub produced some final text

    lines.clear()
    repl._jobs([])  # list
    assert any("j1" in line and "done" in line for line in lines)

    lines.clear()
    import sys

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        repl._jobs(["j1"])  # read the answer
    finally:
        sys.stdout = old
    assert buf.getvalue().strip()  # the job's answer was printed


def test_notify_finished_jobs_announces_once() -> None:
    repl, lines = _stub_repl()
    job = _Job(job_id="j1", task="x", status="done", result="r")
    repl.jobs.append(job)
    repl._notify_finished_jobs()
    repl._notify_finished_jobs()  # must not re-announce
    notes = [line for line in lines if "j1" in line and "/jobs j1 to read" in line]
    assert len(notes) == 1


def test_jobs_missing_id() -> None:
    repl, lines = _stub_repl()
    repl._jobs(["jX"])
    assert any("no such job" in line for line in lines)


def test_exit_with_running_jobs_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exiting the REPL while a job is still running prints an abandon warning."""
    monkeypatch.chdir(tmp_path)
    repl, lines = _stub_repl(["/exit"], tty=False)
    # Inject a still-running job WITHOUT a real worker thread (deterministic).
    repl.jobs.append(_Job(job_id="j1", task="slow", status="running"))
    repl.run()
    assert any("still running" in line and "abandoned" in line for line in lines)


# --------------------------------------------------------- context meter


def test_context_meter_string_after_turn() -> None:
    repl, _ = _stub_repl()
    from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole

    thread = ChatThread(agent_id=repl.persona.agent_id)
    thread.messages.append(Message(role=MessageRole.USER, content="hello there"))
    gauge = repl._context_meter(thread)
    assert "tokens" in gauge and gauge.startswith("~")


def test_compact_reduces_messages_on_long_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/compact summarizes a long thread in place (real CONTEXT_COMPACTED machinery)."""
    monkeypatch.chdir(tmp_path)
    from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole

    repl, lines = _stub_repl()
    repl.rt["session_store"] = None  # _persist no-op
    thread = ChatThread(agent_id=repl.persona.agent_id)
    thread.messages.append(Message(role=MessageRole.SYSTEM, content="you are t"))
    # Many fat user/assistant messages so the thread blows past the budget.
    for i in range(40):
        thread.messages.append(
            Message(role=MessageRole.USER, content=f"question {i} " + "z" * 200)
        )
        thread.messages.append(
            Message(role=MessageRole.ASSISTANT, content=f"answer {i} " + "y" * 200)
        )
    before = len(thread.messages)
    repl._compact(thread)
    after = len(thread.messages)
    assert after < before  # the middle was summarized away
    assert any("compacted:" in line for line in lines)


# ---------------------------------------------------------- recorder unit


def test_turn_recorder_accumulates_cost_across_turns() -> None:
    rec = TurnRecorder()
    rec.start_turn()
    run_async(rec.handle(_event("INFERENCE_SUCCEEDED", cost=0.10)))
    rec.start_turn()  # new turn clears per-turn events but keeps cumulative cost
    run_async(rec.handle(_event("INFERENCE_SUCCEEDED", cost=0.05)))
    assert len(rec.events) == 1  # only the current turn
    assert abs(rec.total_cost - 0.15) < 1e-9  # cumulative


def test_live_run_ui_silent_off_tty_but_recorder_counts() -> None:
    """Off-TTY: LiveRunUI counts nothing, but the recorder still tracks cost."""
    live = LiveRunUI(model_label="", stream=io.StringIO())
    assert live.enabled is False
    rec = TurnRecorder()
    run_async(rec.handle(_event("INFERENCE_SUCCEEDED", cost=0.2)))
    assert rec.total_cost == 0.2
