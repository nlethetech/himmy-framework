"""Studio Workflows: discover *.workflow.yaml and run one through an agent.

A workflow is an ordered set of steps sharing one thread + accumulating state. This
discovers declarative workflow specs and runs them with the WorkflowOrchestrator,
using a chosen agent for the persona/tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from himmy.api.studio_service import project_root


class WorkflowStepInfo(BaseModel):
    name: str
    subtask: str
    tools: list[str] = []
    output_key: str | None = None


class WorkflowInfo(BaseModel):
    path: str
    name: str
    description: str = ""
    steps: list[WorkflowStepInfo] = []


def _info(path: Path, root: Path) -> WorkflowInfo | None:
    from himmy.config.workflow_spec import load_workflow_spec

    try:
        wf = load_workflow_spec(str(path))
    except Exception:  # noqa: BLE001 - skip an unparseable spec
        return None
    return WorkflowInfo(
        path=str(path.relative_to(root)),
        name=wf.name,
        description=wf.description,
        steps=[
            WorkflowStepInfo(
                name=s.name,
                subtask=s.subtask,
                tools=list(s.tool_names),
                output_key=s.output_key,
            )
            for s in wf.steps
        ],
    )


def discover_workflows(root: Path | None = None) -> list[WorkflowInfo]:
    """Find ``*.workflow.yaml`` (and ``workflows/*.yaml``) under the project root."""
    root = root or project_root()
    found: set[Path] = set()
    for pat in ("*.workflow.yaml", "workflows/*.yaml"):
        found.update(root.glob(pat))
    out = [_info(f, root) for f in sorted(found) if f.is_file()]
    return [w for w in out if w is not None]


async def run_workflow(
    workflow_path: str,
    agent_path: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    initial_state: dict[str, Any] | None = None,
) -> Any:
    """Run the workflow with the chosen agent's persona; return the WorkflowResult."""
    import asyncio

    from himmy.api.studio_service import load_studio_spec, resolve_spec_path
    from himmy.config.workflow_spec import load_workflow_spec
    from himmy.orchestrators.workflow import WorkflowOrchestrator
    from himmy.runtime import from_spec

    wf = load_workflow_spec(str(resolve_spec_path(workflow_path)))
    spec = load_studio_spec(agent_path, provider=provider, model=model)
    runtime, _registry = await asyncio.to_thread(
        lambda: from_spec.build_runtime_for_spec(spec, provider=provider, model=model)
    )
    orch = WorkflowOrchestrator(runtime)
    return await orch.run(wf, spec.to_persona(), initial_state=initial_state or {})


__all__ = [
    "WorkflowInfo",
    "WorkflowStepInfo",
    "discover_workflows",
    "run_workflow",
]
