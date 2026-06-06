"""Studio Lineage: trace a run's answer back to what produced it.

Studio runs aren't projected into the entity-registry lineage graph, but each run
already persists its full cognition trace. This reshapes that trace into a provenance
view — answer ← agent (on a model) ← prompt, plus the tools it called, the evidence it
grounded on, and any specialists it delegated to — so you can see *why it said that*.
"""

from __future__ import annotations

from pydantic import BaseModel


class LineageTool(BaseModel):
    name: str
    args: dict | None = None
    result: str | None = None
    outcome: str | None = None
    read_only: bool | None = None
    agent: str | None = None


class LineageEvidence(BaseModel):
    source: str  # memory | knowledge
    query: str | None = None
    text: str
    similarity: float | None = None
    source_uri: str | None = None


class LineageDelegate(BaseModel):
    worker: str
    task: str | None = None


class LineageView(BaseModel):
    run_id: str
    agent: str | None = None
    provider: str | None = None
    prompt: str = ""
    answer: str = ""
    models: list[str] = []
    tools: list[LineageTool] = []
    evidence: list[LineageEvidence] = []
    delegates: list[LineageDelegate] = []
    cost: float = 0.0
    tokens: int = 0


def run_lineage(run_id: str) -> LineageView | None:
    """Build a provenance view for a persisted Studio run, or None if unknown."""
    from himmy.api.studio_runs import get_run_store

    run = get_run_store().get(run_id)
    if run is None:
        return None

    tools: list[LineageTool] = []
    evidence: list[LineageEvidence] = []
    delegates: list[LineageDelegate] = []
    for s in run.steps:
        if s.kind == "tool":
            tools.append(
                LineageTool(
                    name=s.name or "tool",
                    args=s.args,
                    result=s.result,
                    outcome=s.outcome,
                    read_only=s.read_only,
                    agent=s.agent,
                )
            )
        elif s.kind == "grounding":
            for c in s.citations or []:
                evidence.append(
                    LineageEvidence(
                        source=s.source or "knowledge",
                        query=s.query,
                        text=str(c.get("text", "")),
                        similarity=c.get("similarity"),
                        source_uri=c.get("source_uri"),
                    )
                )
        elif s.kind == "delegate":
            delegates.append(LineageDelegate(worker=s.worker or "?", task=s.task))

    models = [u.model for u in run.usage_by_model] or ([run.model] if run.model else [])
    return LineageView(
        run_id=run.id,
        agent=run.agent_name,
        provider=run.provider,
        prompt=run.prompt,
        answer=run.output,
        models=models,
        tools=tools,
        evidence=evidence,
        delegates=delegates,
        cost=run.cost,
        tokens=run.input_tokens + run.output_tokens,
    )


__all__ = [
    "LineageView",
    "LineageTool",
    "LineageEvidence",
    "LineageDelegate",
    "run_lineage",
]
