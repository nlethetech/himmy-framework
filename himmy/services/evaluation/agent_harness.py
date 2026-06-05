"""Agent evaluation harness: run an agent or team over a suite and score it.

The :class:`~himmy.services.evaluation.service.EvaluationService` scores a dict of
``{case_id: output}``. This harness produces that dict by actually *running* an agent
(or a multi-agent team) on each case's input, then delegates to the eval service — so a
prompt/persona/team can be regression-tested with deterministic metrics offline and
LLM-judge / embedding-similarity metrics when a provider/embedder is configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from himmy.agents.base_agent.task import Task
from himmy.services.evaluation.models import EvaluationRun, EvaluationSuite
from himmy.services.evaluation.service import EvaluationService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.agents.personas.persona import Persona
    from himmy.orchestrators import AgentTeam
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.inference.models import LLMConfig
    from himmy.services.tools.registry import ToolRegistry


class AgentEvalHarness:
    """Evaluate an agent/team against an :class:`EvaluationSuite`."""

    def __init__(
        self, runtime: SingleAgentRuntime, eval_service: EvaluationService | None = None
    ) -> None:
        """Wire the runtime that runs cases and the service that scores them."""
        self._runtime = runtime
        self._eval = eval_service or EvaluationService()

    async def evaluate_agent(
        self,
        suite: EvaluationSuite,
        persona: Persona,
        *,
        llm_config: LLMConfig | None = None,
        input_key: str = "prompt",
    ) -> EvaluationRun:
        """Run ``persona`` on each case's ``input[input_key]`` and score the suite."""
        outputs: dict[str, Any] = {}
        for case in suite.cases:
            prompt = str(case.input.get(input_key, ""))
            result = await self._runtime.run_task_detailed(
                persona,
                Task(title=f"eval-{case.case_id}", prompt=prompt),
                llm_config=llm_config,
            )
            outputs[case.case_id] = result.output_text
        return await self._eval.run_suite(suite=suite, actual_outputs=outputs)

    async def evaluate_team(
        self,
        suite: EvaluationSuite,
        team: AgentTeam,
        tool_registry: ToolRegistry,
        *,
        input_key: str = "prompt",
        max_turns: int = 12,
    ) -> EvaluationRun:
        """Route each case's input through a team and score the final answers."""
        from himmy.orchestrators import MultiAgentOrchestrator

        orchestrator = MultiAgentOrchestrator(
            self._runtime, team, tool_registry, max_turns=max_turns
        )
        outputs: dict[str, Any] = {}
        for case in suite.cases:
            prompt = str(case.input.get(input_key, ""))
            result = await orchestrator.run(prompt)
            outputs[case.case_id] = result.output_text
        return await self._eval.run_suite(suite=suite, actual_outputs=outputs)


__all__ = ["AgentEvalHarness"]
