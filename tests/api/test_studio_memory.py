"""Tests for the Studio Memory browser (durable store wrap)."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_memory as sm
from tests.conftest import run_async


@pytest.fixture
def mem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIMMY_MEMORY_PATH", str(tmp_path / "memory.db"))
    sm.reset_memory_service()
    yield
    sm.reset_memory_service()


def test_add_list_forget(mem: None) -> None:
    a = sm.add_memory("We keep ducks", subject_id="farm")
    sm.add_memory("Pond has carp", subject_id="farm")
    items = sm.list_memories("farm")
    assert {i.text for i in items} == {"We keep ducks", "Pond has carp"}
    assert "farm" in sm.list_subjects()
    assert sm.forget(a.memory_id) is True
    assert len(sm.list_memories("farm")) == 1


def test_recall_ranks_by_similarity(mem: None) -> None:
    sm.add_memory("We keep Khaki Campbell ducks", subject_id="farm")
    sm.add_memory("The tractor needs an oil change", subject_id="farm")
    hits = run_async(sm.recall("what ducks do we have", subject_id="farm", top_k=2))
    assert hits
    assert "duck" in hits[0].text.lower()
