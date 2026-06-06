"""Load a :class:`~himmy.orchestrators.workflow.Workflow` from YAML.

Workflows were Python-only; this lets them be authored declaratively (so Studio can
discover + run them). Shape mirrors the model:

    name: research-brief
    description: Search then summarize.
    steps:
      - name: search
        subtask: "Find recent facts about {topic}."
        tool_names: [web_search]
        output_key: findings
      - name: write
        subtask: "Write a 3-bullet brief from: {findings}"
"""

from __future__ import annotations

from pathlib import Path

import yaml

from himmy.orchestrators.workflow import Workflow


def load_workflow_spec(path: str | Path) -> Workflow:
    """Parse a workflow YAML file into a :class:`Workflow` (validated)."""
    raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"workflow {path} must be a YAML mapping")
    raw.setdefault("name", Path(path).stem.replace(".workflow", ""))
    return Workflow.model_validate(raw)


__all__ = ["load_workflow_spec"]
