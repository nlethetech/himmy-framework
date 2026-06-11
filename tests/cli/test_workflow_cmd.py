"""Tests for ``himmy workflow`` — the CLI surface over the WORKFLOW orchestrator.

The headline test (`test_cmd_workflow_runs_real_orchestrator`) drives a tiny
2-step workflow through the ACTUAL :class:`WorkflowOrchestrator` on the offline
stub provider and asserts both steps ran — no mocking of the orchestrator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from himmy.cli import workflow_cmd

_TWO_STEP_YAML = """\
name: tiny
description: A two-step smoke workflow.
steps:
  - name: draft
    subtask: "Draft a note about {topic}."
    output_key: draft
  - name: brief
    subtask: "Tighten this: {draft}"
    output_key: brief
"""


def _args(**kw: Any) -> argparse.Namespace:
    """A Namespace with the attrs cmd_workflow reads, defaulting to the stub."""
    base = {
        "action": "run",
        "file": None,
        "scaffold": False,
        "directory": None,
        "force": False,
        "state": None,
        "timeout": None,
        "json": False,
        "provider": "stub",
        "model": None,
        "name": None,
        "instruction": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------- happy path


def test_cmd_workflow_runs_real_orchestrator(tmp_path: Path, capsys: Any) -> None:
    """A 2-step workflow runs through the REAL orchestrator; both steps complete."""
    wf = tmp_path / "tiny.workflow.yaml"
    wf.write_text(_TWO_STEP_YAML, encoding="utf-8")

    code = workflow_cmd.cmd_workflow(
        _args(file=str(wf), state=["topic=ACME"], json=True)
    )
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    # Both steps actually executed (not mocked) and completed on the stub.
    assert payload["status"] == "completed"
    assert [s["step_name"] for s in payload["step_results"]] == ["draft", "brief"]
    assert all(s["status"] == "completed" for s in payload["step_results"])
    # State threaded: the first step's output_key landed in final_state.
    assert "draft" in payload["final_state"]


def test_cmd_workflow_enforces_rbac_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Finding 2: cmd_workflow calls enforce_role_on_registry with the --role so a
    viewer is enforced consistently with run/chat (not bypassable via workflow)."""
    wf = tmp_path / "tiny.workflow.yaml"
    wf.write_text(_TWO_STEP_YAML, encoding="utf-8")
    seen: dict[str, Any] = {}

    import himmy.cli.rbac_cmd as rbac_cmd

    real = rbac_cmd.enforce_role_on_registry

    def _spy(registry: Any, *, role: Any = None, **kw: Any) -> Any:
        seen["role"] = role
        seen["called"] = True
        return real(registry, role=role, **kw)

    monkeypatch.setattr(rbac_cmd, "enforce_role_on_registry", _spy)
    code = workflow_cmd.cmd_workflow(
        _args(file=str(wf), state=["topic=ACME"], role="viewer")
    )
    assert code == 0
    assert seen.get("called") is True
    assert seen["role"] == "viewer"


def test_core_runner_executes_both_steps_on_stub() -> None:
    """_run_workflow_core drives a real WorkflowOrchestrator end-to-end (no mocks)."""
    from himmy.agents.personas.persona import Persona
    from himmy.entities.registry import EntityRegistry
    from himmy.orchestrators import Workflow, WorkflowStep
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.inference.client_manager import StubClientManager
    from himmy.services.inference.service import InferenceService
    from himmy.services.storage.service import StorageService

    storage = StorageService()
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        entity_registry=EntityRegistry(),
    )
    workflow = Workflow(
        name="tiny",
        steps=[
            WorkflowStep(name="a", subtask="Do A on {topic}", output_key="a"),
            WorkflowStep(name="b", subtask="Do B with {a}", output_key="b"),
        ],
    )
    seen: list[str] = []

    async def _on_event(ev: Any) -> None:
        kind = getattr(ev.event_type, "value", str(ev.event_type))
        if kind == "WORKFLOW_STEP_COMPLETED":
            seen.append((ev.payload or {}).get("step_name", ""))

    result = workflow_cmd._run_workflow_core(
        workflow,
        Persona(name="W"),
        runtime,
        initial_state={"topic": "X"},
        on_event=_on_event,
        total_timeout_seconds=None,
    )
    assert result.status == "completed"
    assert seen == ["a", "b"]
    assert len(result.step_results) == 2


# ------------------------------------------------------------------ scaffold


def test_scaffold_writes_runnable_workflow(tmp_path: Path, capsys: Any) -> None:
    """`himmy workflow init` writes a workflow.yaml that itself loads + runs."""
    code = workflow_cmd.cmd_workflow(_args(action="init", directory=str(tmp_path)))
    assert code == 0
    dest = tmp_path / "workflow.yaml"
    assert dest.is_file()

    # The scaffold is a valid spec the loader accepts and the runner can run.
    from himmy.config.workflow_spec import load_workflow_spec

    spec = load_workflow_spec(str(dest))
    assert spec.steps
    capsys.readouterr()
    code = workflow_cmd.cmd_workflow(_args(file=str(dest), state=["topic=soil"]))
    assert code == 0


def test_scaffold_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """A second init without --force fails cleanly rather than clobbering."""
    assert workflow_cmd.cmd_workflow(_args(scaffold=True, directory=str(tmp_path))) == 0
    # Second run, no force → non-zero, file untouched.
    assert workflow_cmd.cmd_workflow(_args(scaffold=True, directory=str(tmp_path))) == 1
    # With force it succeeds.
    assert (
        workflow_cmd.cmd_workflow(
            _args(scaffold=True, directory=str(tmp_path), force=True)
        )
        == 0
    )


# ------------------------------------------------------------------- errors


def test_missing_file_flag_returns_2(capsys: Any) -> None:
    """`himmy workflow run` with no -f reports the error and exits 2."""
    assert workflow_cmd.cmd_workflow(_args(file=None)) == 2
    assert "needs -f" in capsys.readouterr().err


def test_nonexistent_file_returns_2(tmp_path: Path, capsys: Any) -> None:
    """A -f path that doesn't exist is reported, not raised."""
    missing = tmp_path / "nope.yaml"
    assert workflow_cmd.cmd_workflow(_args(file=str(missing))) == 2
    assert "not found" in capsys.readouterr().err


def test_empty_workflow_returns_2(tmp_path: Path, capsys: Any) -> None:
    """A spec with zero steps is rejected with a clean message."""
    wf = tmp_path / "empty.workflow.yaml"
    wf.write_text("name: empty\nsteps: []\n", encoding="utf-8")
    assert workflow_cmd.cmd_workflow(_args(file=str(wf))) == 2
    assert "no steps" in capsys.readouterr().err


def test_bad_state_pair_returns_2(tmp_path: Path, capsys: Any) -> None:
    """A malformed --state (no '=') is reported as a usage error, not a crash."""
    wf = tmp_path / "tiny.workflow.yaml"
    wf.write_text(_TWO_STEP_YAML, encoding="utf-8")
    assert workflow_cmd.cmd_workflow(_args(file=str(wf), state=["bogus"])) == 2
    assert "key=value" in capsys.readouterr().err


# -------------------------------------------------------------- state parse


def test_parse_state_json_and_raw() -> None:
    """JSON values are decoded; bare strings stay strings."""
    state = workflow_cmd._parse_state(["topic=ACME", "count=3", 'tags=["a","b"]'])
    assert state == {"topic": "ACME", "count": 3, "tags": ["a", "b"]}


def test_parse_state_rejects_missing_eq() -> None:
    with pytest.raises(ValueError):
        workflow_cmd._parse_state(["nokey"])


def test_parse_state_empty() -> None:
    assert workflow_cmd._parse_state(None) == {}


# ------------------------------------------------------------------ /slash


class _FakeRepl:
    """A minimal stand-in for ChatRepl so slash_workflow is unit-testable."""

    def __init__(self) -> None:
        from himmy.agents.personas.persona import Persona
        from himmy.entities.registry import EntityRegistry
        from himmy.runtime.single_agent import SingleAgentRuntime
        from himmy.services.inference.client_manager import StubClientManager
        from himmy.services.inference.service import InferenceService
        from himmy.services.storage.service import StorageService

        storage = StorageService()
        runtime = SingleAgentRuntime(
            inference_service=InferenceService(StubClientManager(), event_sink=storage),
            memory_store=storage,
            entity_registry=EntityRegistry(),
        )
        self.persona = Persona(name="ReplAgent")
        self.rt: dict[str, Any] = {"runtime": runtime, "live": None}
        self.loop = None
        self._c = {k: "" for k in ("dim", "reset", "snow", "faint", "green", "gold")}
        self.errs: list[str] = []

    def _eprint(self, *a: Any) -> None:
        self.errs.append(" ".join(str(x) for x in a))


_NO_STATE_YAML = """\
name: standalone
description: Two steps that need no injected state.
steps:
  - name: one
    subtask: "Say hello."
    output_key: greeting
  - name: two
    subtask: "Now expand on it: {greeting}"
    output_key: expanded
"""


def test_slash_workflow_runs_on_repl_runtime(tmp_path: Path, capsys: Any) -> None:
    """/workflow <file> runs the workflow on the REPL's live runtime."""
    wf = tmp_path / "standalone.workflow.yaml"
    wf.write_text(_NO_STATE_YAML, encoding="utf-8")
    repl = _FakeRepl()
    workflow_cmd.slash_workflow(repl, str(wf))
    joined = "\n".join(repl.errs)
    assert "workflow completed" in joined


def test_slash_workflow_usage_when_blank() -> None:
    """/workflow with no argument prints usage, runs nothing."""
    repl = _FakeRepl()
    workflow_cmd.slash_workflow(repl, "")
    assert any("usage:" in e for e in repl.errs)


def test_slash_workflow_missing_file() -> None:
    """/workflow pointing at a missing file reports it cleanly."""
    repl = _FakeRepl()
    workflow_cmd.slash_workflow(repl, "/no/such/file.yaml")
    assert any("not found" in e for e in repl.errs)
