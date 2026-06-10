"""LLM-as-judge grader tier for the benchmark — a thin adapter over the eval metric.

Some benchmark tasks (open-ended summarization, free-text reasoning) have no computable
ground truth, so a deterministic grader can't score them honestly. For those a task may
declare a ``judge`` block: a natural-language rubric plus a pass threshold that an LLM
grades. This module is the deliberate first step of **unifying the benchmark grader
registry with the eval service's metric registry** — rather than duplicate the judge
prompt/schema/parsing here, it reuses
:class:`himmy.services.evaluation.metrics.LLMJudgeMetric`: the rubric is packed into a
synthetic :class:`~himmy.services.evaluation.models.EvaluationCase` and scored through
that one metric, so there is a single judge implementation in the codebase.

Hard properties of the judge tier (enforced here, not by the caller):

* The **judge model is configurable per run** (a :class:`~himmy.benchmark.models.ModelSpec`
  carries the judge provider/model); see :func:`build_judge`.
* The **judge must differ from the candidate** — grading a model's output with the same
  model id is refused (:class:`SameModelJudgeError`), because self-grading is not an
  independent signal.
* Judge tasks are **reported, never gating**: the gate subset excludes them and
  :func:`himmy.benchmark.baseline.compare_to_baseline` ignores judge results entirely.
* A judge that **times out or returns an unparseable verdict** marks the trial *ungraded*
  (a :class:`JudgeVerdict` with ``ungraded=True``) — distinct from a graded fail — so the
  report can show the ungraded count instead of silently counting it against the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.benchmark.models import ModelSpec

#: Default threshold a judge ``score`` must reach (0..1) for the trial to pass when the
#: task's ``judge`` block does not set its own ``threshold``.
DEFAULT_JUDGE_THRESHOLD = 0.6


class SameModelJudgeError(HimmyError):
    """Raised when the configured judge model id equals the candidate model id.

    Self-grading is not an independent signal, so the runner refuses it loudly rather
    than silently produce a meaningless judge score.
    """


@dataclass(frozen=True)
class JudgeVerdict:
    """The outcome of judging one trial's answer against a task's rubric.

    ``ungraded`` is the third state between pass and fail: the judge model failed to
    return a usable verdict (timeout, provider error, unparseable output). An ungraded
    trial is reported separately and never counted as a graded fail.
    """

    passed: bool
    score: float
    rationale: str
    ungraded: bool = False
    judge_model: str = ""


def has_judge(task: Any) -> bool:
    """Whether ``task`` declares a (non-empty) ``judge`` block."""
    return bool(getattr(task, "judge", None))


def build_judge_spec(judge: dict[str, Any]) -> tuple[str, float]:
    """Normalize a task's ``judge`` block into ``(rubric, threshold)`` (fail-loud).

    The block must carry a non-empty ``rubric`` string; ``threshold`` is optional and
    defaults to :data:`DEFAULT_JUDGE_THRESHOLD`. A missing rubric or an out-of-range
    threshold raises :class:`ValueError` (the runner records it as a trial error rather
    than silently judging against an empty rubric).
    """
    rubric = str(judge.get("rubric", "")).strip()
    if not rubric:
        raise ValueError("judge grader requires a non-empty 'rubric' string")
    threshold = float(judge.get("threshold", DEFAULT_JUDGE_THRESHOLD))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"judge threshold must be in [0, 1], got {threshold}")
    return rubric, threshold


#: claude-cli short aliases ⇒ the full model id the CLI resolves them to. A candidate
#: named ``haiku`` and a judge named ``claude-haiku-4-5`` (or vice-versa) are the SAME
#: model once the CLI resolves the alias, so the judge!=candidate check must normalize
#: both sides through this table before comparing. Kept minimal and current; an unknown
#: alias normalizes to itself (no false collision).
_CLAUDE_CLI_ALIASES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}


def _normalize_model_id(provider: str, model: str) -> str:
    """Canonicalize a ``provider:model`` id so alias/tag-equal models compare equal.

    The exact-string ``judge != candidate`` check is bypassed by provider conveniences
    that resolve to the same model only later (in the external CLI / Ollama server):

    * **Ollama** treats ``llama3.2`` and ``llama3.2:latest`` as one tag, so strip a
      trailing ``:latest``.
    * **claude-cli** accepts both the short alias (``haiku``) and the full id
      (``claude-haiku-4-5``); map known aliases to the full id.

    An unrecognized model is returned unchanged, so this never *invents* a collision —
    it only collapses the documented alias/tag equivalences.
    """
    m = model.strip().lower()
    if provider == "ollama":
        if m.endswith(":latest"):
            m = m[: -len(":latest")]
    elif provider == "claude-cli":
        m = _CLAUDE_CLI_ALIASES.get(m, m)
    return f"{provider}:{m}"


def resolve_judge_model(spec: ModelSpec, candidate_provider: str) -> tuple[str, str]:
    """Pick the judge ``(provider, model)`` for a candidate run; enforce judge != candidate.

    The judge model is configured on the candidate's :class:`~himmy.benchmark.models.ModelSpec`
    via ``judge_provider``/``judge_model`` (per-run configurable). When the resolved judge
    id equals the candidate's ``provider:model``, :class:`SameModelJudgeError` is raised —
    a model may not grade itself. Comparison is over *normalized* ids
    (:func:`_normalize_model_id`), so an alias/tag pair that resolves to the same model
    (``ollama:llama3.2`` vs ``ollama:llama3.2:latest``, ``claude-cli:haiku`` vs the full
    ``claude-cli:claude-haiku-4-5``) is still refused — not just literal string equality.
    """
    judge_provider = spec.judge_provider or spec.provider
    judge_model = spec.judge_model
    if not judge_model:
        raise ValueError(
            "judge-tier task requires a judge model; set ModelSpec.judge_model "
            "(and optionally judge_provider) for the run"
        )
    candidate_norm = _normalize_model_id(candidate_provider, spec.model)
    judge_norm = _normalize_model_id(judge_provider, judge_model)
    if judge_norm == candidate_norm:
        raise SameModelJudgeError(
            f"judge model '{judge_provider}:{judge_model}' resolves to the same model as "
            f"the candidate '{candidate_provider}:{spec.model}' "
            f"(both normalize to {judge_norm!r}); a model may not grade its own output — "
            "configure a distinct judge_provider/judge_model"
        )
    return judge_provider, judge_model


def build_judge(spec: ModelSpec, *, inference_factory: Any = None) -> tuple[Any, str]:
    """Build the judge ``(LLMJudgeMetric, judge_id)`` for a candidate ``spec``.

    Reuses :class:`himmy.services.evaluation.metrics.LLMJudgeMetric` over a judge
    :class:`~himmy.services.inference.service.InferenceService` so the judge prompt,
    schema, and verdict parsing have a single implementation. ``inference_factory`` (when
    given) builds the judge inference from a one-off judge spec — tests inject a
    deterministic stub; otherwise the CLI provider builder constructs the real judge
    model. The judge model id is enforced to differ from the candidate.
    """
    from himmy.benchmark.models import ModelSpec as _ModelSpec
    from himmy.services.evaluation.metrics import LLMJudgeMetric

    judge_provider, judge_model = resolve_judge_model(spec, spec.provider)
    judge_id = f"{judge_provider}:{judge_model}"
    judge_spec = _ModelSpec(provider=judge_provider, model=judge_model)
    if inference_factory is not None:
        inference = inference_factory(judge_spec)
    else:
        from himmy.cli.provider import build_inference_for

        inference = build_inference_for(judge_provider, judge_model)
    # threshold is applied at the verdict layer (per task), so the metric's own pass
    # boundary is irrelevant here — we read the raw score from the MetricScore.
    return LLMJudgeMetric(inference), judge_id


async def judge_answer(
    metric: Any,
    *,
    rubric: str,
    threshold: float,
    prompt: str,
    answer: str,
    judge_model: str,
) -> JudgeVerdict:
    """Grade one ``answer`` against ``rubric`` via the shared :class:`LLMJudgeMetric`.

    The rubric + the task prompt are packed into a synthetic
    :class:`~himmy.services.evaluation.models.EvaluationCase` so the metric's existing
    prompt/schema/parsing do the work (no duplicated judge logic). A judge failure
    (timeout, provider error, unparseable verdict) yields an *ungraded* verdict —
    distinct from a graded fail — surfaced via :attr:`JudgeVerdict.ungraded`. A
    successful verdict passes iff ``score >= threshold``.
    """
    import inspect

    from himmy.services.evaluation.models import EvaluationCase

    case = EvaluationCase(
        input={"prompt": prompt},
        expected_output={"rubric": rubric, "question": prompt},
    )
    try:
        result = metric.score(case, answer)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:  # noqa: BLE001 - any judge failure is "ungraded", not fail
        return JudgeVerdict(
            passed=False,
            score=0.0,
            rationale=f"judge error: {exc}",
            ungraded=True,
            judge_model=judge_model,
        )
    # LLMJudgeMetric flags two distinct judge-never-produced-a-verdict cases with a
    # detail prefix: "judge failed:" (provider/transport error or timeout) and
    # "judge unparseable:" (SUCCESS but no usable numeric score — malformed/non-JSON
    # verdict). BOTH mark the trial UNGRADED (the third state) rather than a graded 0.0,
    # so a flaky or malformed judge never silently counts against the candidate. A real
    # numeric verdict below threshold is a genuine graded fail and falls through.
    from himmy.services.evaluation.metrics import JUDGE_UNPARSEABLE_PREFIX

    detail = result.detail or ""
    if detail.startswith("judge failed:") or detail.startswith(
        JUDGE_UNPARSEABLE_PREFIX
    ):
        return JudgeVerdict(
            passed=False,
            score=0.0,
            rationale=detail,
            ungraded=True,
            judge_model=judge_model,
        )
    return JudgeVerdict(
        passed=result.score >= threshold,
        score=result.score,
        rationale=detail,
        ungraded=False,
        judge_model=judge_model,
    )


#: Default location of the hand-labelled judge validation set, anchored to the package
#: (like ``history.HISTORY_PATH``) so it resolves regardless of the current working
#: directory — ``scripts/bench_gate.py judge-agreement`` and the shipped-set test both
#: work from anywhere, not only the repo root.
JUDGE_VALIDATION_PATH = str(
    Path(__file__).resolve().parents[2] / "benchmarks" / "judge_validation.jsonl"
)


@dataclass(frozen=True)
class JudgeAgreement:
    """Agreement of the judge against a hand-labelled validation set.

    ``agreement`` is the fraction of rows where the judge's pass/fail verdict matched the
    human ``good``/``bad`` label (over rows the judge actually graded). ``ungraded`` rows
    (judge timeout / unparseable verdict) are reported but excluded from the rate. Use this
    BEFORE trusting any judge-tier benchmark number: a judge tier is only trustworthy once
    its agreement with obviously-correct labels is high.
    """

    n: int
    graded: int
    matches: int
    ungraded: int
    rows: list[dict[str, Any]]

    @property
    def agreement(self) -> float:
        """Fraction of *graded* rows where the judge matched the human label."""
        return self.matches / self.graded if self.graded else 0.0


def load_judge_validation(path: str | None = None) -> list[dict[str, Any]]:
    """Load the hand-labelled validation rows (JSONL: task, rubric, answer, human_label)."""
    import json

    target = Path(path or JUDGE_VALIDATION_PATH).expanduser()
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


async def run_judge_agreement(
    rows: list[dict[str, Any]],
    *,
    metric: Any,
    judge_model: str = "judge",
    threshold: float = DEFAULT_JUDGE_THRESHOLD,
) -> JudgeAgreement:
    """Run ``metric`` (an :class:`LLMJudgeMetric`) over labelled rows; report agreement.

    Each row is ``{task, rubric, answer, human_label}`` where ``human_label`` is ``"good"``
    (should pass) or ``"bad"`` (should fail). Judges every row, compares the verdict to the
    label, and returns a :class:`JudgeAgreement`. Documented workflow: extend
    ``benchmarks/judge_validation.jsonl`` with real model outputs you hand-label, run this
    against your chosen judge model, and only trust the judge tier once agreement is high.
    """
    out_rows: list[dict[str, Any]] = []
    matches = 0
    graded = 0
    ungraded = 0
    for row in rows:
        label = str(row.get("human_label", "")).lower()
        expected_pass = label == "good"
        verdict = await judge_answer(
            metric,
            rubric=str(row.get("rubric", "")),
            threshold=float(row.get("threshold", threshold)),
            prompt=str(row.get("task", "")),
            answer=str(row.get("answer", "")),
            judge_model=judge_model,
        )
        matched = (not verdict.ungraded) and (verdict.passed == expected_pass)
        if verdict.ungraded:
            ungraded += 1
        else:
            graded += 1
            if matched:
                matches += 1
        out_rows.append(
            {
                "task": row.get("task", ""),
                "human_label": label,
                "judge_passed": None if verdict.ungraded else verdict.passed,
                "judge_score": verdict.score,
                "ungraded": verdict.ungraded,
                "matched": matched,
            }
        )
    return JudgeAgreement(
        n=len(rows),
        graded=graded,
        matches=matches,
        ungraded=ungraded,
        rows=out_rows,
    )


__all__ = [
    "DEFAULT_JUDGE_THRESHOLD",
    "JUDGE_VALIDATION_PATH",
    "SameModelJudgeError",
    "JudgeVerdict",
    "JudgeAgreement",
    "has_judge",
    "build_judge_spec",
    "resolve_judge_model",
    "build_judge",
    "judge_answer",
    "load_judge_validation",
    "run_judge_agreement",
]
