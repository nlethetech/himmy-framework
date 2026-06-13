"""P0-A: the REPL ``/good`` / ``/bad`` commands rate the last turn onto the spine."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from himmy.api.studio_entities import get_entity_registry, reset_entity_registry
from himmy.api.studio_feedback import feedback_history
from himmy.api.studio_runs import reset_run_store
from himmy.cli.repl import ChatRepl
from himmy.config.agent_spec import AgentSpec
from himmy.core.events import EventType, RunEvent


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "file": None,
        "name": "t",
        "instruction": None,
        "provider": "stub",
        "model": None,
        "message": None,
        "session": None,
        "yolo": False,
        "safe": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _repl() -> ChatRepl:
    spec = AgentSpec(name="t", description="d", provider="stub")
    return ChatRepl(spec, _args(), input_fn=lambda _p: "")


def _turn(repl: ChatRepl, trace_id: str = "th1:task1", thread_id: str = "th1") -> None:
    """Seed the recorder with a finished turn so feedback has a run to attach to."""
    repl.recorder.events.append(
        RunEvent(
            event_type=EventType.AGENT_RUN_FINISHED,
            trace_id=trace_id,
            thread_id=thread_id,
        )
    )


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    reset_entity_registry()
    reset_run_store()
    yield
    reset_entity_registry()
    reset_run_store()


def test_good_records_feedback_to_spine() -> None:
    repl = _repl()
    _turn(repl)
    repl._feedback("up", "nailed it")

    recs = get_entity_registry().list_by_kind("run_feedback")
    assert len(recs) == 1
    assert recs[0].payload["verdict"] == "up"
    assert recs[0].payload["note"] == "nailed it"
    assert recs[0].metadata["source"] == "cli"
    assert recs[0].metadata["thread_id"] == "th1"
    assert recs[0].metadata["run_id"] == "th1:task1"


def test_slash_good_then_bad_appends_versions() -> None:
    repl = _repl()
    _turn(repl, trace_id="t:1")
    repl._slash("/good", thread=None)
    repl._slash("/bad too verbose", thread=None)

    hist = feedback_history("t:1")
    assert [f.verdict for f in hist] == ["up", "down"]
    assert hist[-1].note == "too verbose"
    assert [f.version for f in hist] == [1, 2]


def test_feedback_without_a_turn_is_a_no_op() -> None:
    repl = _repl()  # empty recorder — nothing rated yet
    repl._feedback("up", None)
    assert get_entity_registry().list_by_kind("run_feedback") == []
