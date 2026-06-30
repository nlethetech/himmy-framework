"""Studio Evaluation: discover eval suites and run one against an agent.

Discovers ``*.eval.yaml`` suites in the project, then runs a chosen agent over every
case and scores it with the evaluation harness — returning the per-metric scorecard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from himmy.api.studio_service import project_root


class EvalSuiteInfo(BaseModel):
    path: str  # relative to the project root
    name: str
    cases: int


def discover_suites(root: Path | None = None) -> list[EvalSuiteInfo]:
    """Find ``*.eval.yaml`` (and ``evals/*.yaml``) suites under the project root."""
    from himmy.config.eval_spec import load_eval_suite

    root = root or project_root()
    found: set[Path] = set()
    for pat in ("*.eval.yaml", "evals/*.yaml", "eval/*.yaml"):
        found.update(root.glob(pat))
    out: list[EvalSuiteInfo] = []
    for f in sorted(found):
        if not f.is_file():
            continue
        try:
            suite = load_eval_suite(str(f))
        except Exception:  # noqa: BLE001 - skip an unparseable suite
            continue
        out.append(
            EvalSuiteInfo(
                path=str(f.relative_to(root)), name=suite.name, cases=len(suite.cases)
            )
        )
    return out


async def run_eval(
    suite_path: str,
    agent_path: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    tool_authorizer: Any = None,
    subject: str | None = None,
) -> object:
    """Run the agent over every case in the suite and return the EvaluationRun.

    ``tool_authorizer`` (centralize-tool-gate) is the request principal's tool-capability
    gate, bound AMBIENTLY for the runtime build + every eval case so the agent's tool
    dispatch is enforced (this surface builds the runtime without threading the authorizer
    explicitly). ``None`` offline → inert, byte-unchanged.

    ``subject`` (rbac-harden tenancy) is the LAUNCHING tenant's ``workspace_id`` resolved
    from the verified principal by the router (``_run_owner(request)[0]``). It scopes the
    eval agent's memory + knowledge tool packs to this tenant on a shared durable store
    instead of the static ``default`` subject / ``(local, local)`` KB scope — symmetric to
    the ``/v1`` and Studio run paths. ``None`` (offline / single-box / ``all_tenants``)
    leaves both packs on their historical static scope — byte-for-byte unchanged.
    """
    import asyncio

    from himmy.api.studio_service import load_studio_spec, resolve_spec_path
    from himmy.config.eval_spec import load_eval_suite
    from himmy.runtime import from_spec
    from himmy.services.evaluation.agent_harness import AgentEvalHarness
    from himmy.services.evaluation.service import EvaluationService
    from himmy.services.tools.ambient import use_tool_authorizer

    with use_tool_authorizer(tool_authorizer):
        suite = load_eval_suite(str(resolve_spec_path(suite_path)))
        spec = load_studio_spec(agent_path, provider=provider, model=model)
        runtime, _registry = await asyncio.to_thread(
            lambda: from_spec.build_runtime_for_spec(
                spec,
                provider=provider,
                model=model,
                durable_defaults=True,
                subject=subject,
            )
        )
        harness = AgentEvalHarness(runtime, EvaluationService())
        return await harness.evaluate_agent(
            suite, spec.to_persona(), llm_config=spec.to_llm_config()
        )


__all__ = ["EvalSuiteInfo", "discover_suites", "run_eval"]
