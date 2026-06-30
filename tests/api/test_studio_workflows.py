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


# ---- rbac-harden(mopup-r1): run_workflow threads the within-tenant subject_scope axis


class _ScopeCaptured(Exception):
    """Sentinel raised by the build spy after recording the scope (short-circuit)."""


def test_run_workflow_threads_subject_scope_into_runtime_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_workflow`` must forward the within-tenant USER axis into the runtime build.

    Regression for the dropped subject_scope: workflow_run discarded the principal's subject
    and run_workflow did not forward subject_scope, so two subject_scoped users of one tenant
    pooled their workflow agent's memory/KB onto one namespace. Assert subject_scope reaches
    ``build_runtime_for_spec``.
    """
    from himmy.runtime import from_spec
    from tests.conftest import run_async

    monkeypatch.chdir(tmp_path)
    (tmp_path / "brief.workflow.yaml").write_text(
        "name: brief\nsteps:\n  - name: s\n    subtask: do it\n"
    )
    (tmp_path / "agent.yaml").write_text("name: a\nprovider: stub\n")

    captured: dict[str, object] = {}

    def _spy(spec, **kw):
        captured["subject"] = kw.get("subject")
        captured["subject_scope"] = kw.get("subject_scope")
        raise _ScopeCaptured()

    monkeypatch.setattr(from_spec, "build_runtime_for_spec", _spy)

    with pytest.raises(_ScopeCaptured):
        run_async(
            sw.run_workflow(
                "brief.workflow.yaml",
                "agent.yaml",
                subject="tenant-A",
                subject_scope="userA",
            )
        )
    assert captured == {"subject": "tenant-A", "subject_scope": "userA"}
