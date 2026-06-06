"""Tests for the Studio run-lineage (provenance) transform."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_lineage as sl
from himmy.api.studio_runs import (
    CognitionStep,
    ModelUsage,
    StudioRun,
    get_run_store,
    reset_run_store,
)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    yield
    reset_run_store()


def test_run_lineage_reshapes_trace(store: None) -> None:
    get_run_store().save(
        StudioRun(
            id="r1",
            created_at="t",
            agent_name="mailer",
            provider="ollama",
            prompt="email bob",
            output="Sent to bob.",
            input_tokens=100,
            output_tokens=20,
            usage_by_model=[
                ModelUsage(model="ollama:qwen", input_tokens=100, output_tokens=20)
            ],
            steps=[
                CognitionStep(
                    seq=1, kind="tool", name="send_email", read_only=False, result="ok"
                ),
                CognitionStep(
                    seq=2,
                    kind="grounding",
                    source="memory",
                    query="bob",
                    citations=[{"text": "Bob is the farm vet", "similarity": 0.8}],
                ),
                CognitionStep(seq=3, kind="delegate", worker="ops", task="schedule"),
            ],
        )
    )
    view = sl.run_lineage("r1")
    assert view is not None
    assert view.agent == "mailer"
    assert view.answer == "Sent to bob."
    assert view.models == ["ollama:qwen"]
    assert view.tokens == 120
    assert [t.name for t in view.tools] == ["send_email"]
    assert view.tools[0].read_only is False
    assert view.evidence[0].source == "memory"
    assert view.evidence[0].text == "Bob is the farm vet"
    assert view.delegates[0].worker == "ops"


def test_run_lineage_unknown(store: None) -> None:
    assert sl.run_lineage("nope") is None
