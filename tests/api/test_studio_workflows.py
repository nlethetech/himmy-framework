"""Tests for the workflow YAML loader + Studio discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_workflows as sw
from himmy.config.workflow_spec import load_workflow_spec


def test_load_workflow_spec(tmp_path: Path) -> None:
    p = tmp_path / "brief.workflow.yaml"
    p.write_text(
        "name: brief\n"
        "description: two steps\n"
        "steps:\n"
        "  - name: find\n"
        "    subtask: 'Find facts about {topic}'\n"
        "    tool_names: [web_search]\n"
        "    output_key: facts\n"
        "  - name: write\n"
        "    subtask: 'Summarize {facts}'\n"
    )
    wf = load_workflow_spec(str(p))
    assert wf.name == "brief"
    assert [s.name for s in wf.steps] == ["find", "write"]
    assert wf.steps[0].tool_names == ["web_search"]
    assert wf.steps[0].output_key == "facts"


def test_discover_workflows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.workflow.yaml").write_text(
        "name: a\nsteps:\n  - name: s\n    subtask: do it\n"
    )
    monkeypatch.chdir(tmp_path)
    found = sw.discover_workflows()
    assert [w.name for w in found] == ["a"]
    assert found[0].steps[0].subtask == "do it"
