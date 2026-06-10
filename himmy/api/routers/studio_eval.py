"""API kernel: Studio eval router — run suites from the GUI, compare to baseline.

Three surfaces under ``/api/studio/eval`` (same ``studio:use`` guard as the rest
of the Studio family, see :mod:`himmy.api.routers.studio_common`):

  * ``GET  /suites``   — what is runnable: discovered ``*.eval.yaml`` suites plus
    the checked-in bench gate subset, each with a case count.
  * ``POST /run``      — run one suite, streaming per-case progress over SSE
    (``start`` → ``case``… → ``summary``). Defaults to the offline deterministic
    stub so a run completes in seconds; a live provider runs only when the caller
    explicitly picks one.
  * ``GET  /baseline`` — ``benchmarks/baseline.json`` parsed into display rows
    (metric · floor/ceiling · last measured, plus the SHA/date it was cut at).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from himmy.api.routers.studio_common import build_studio_router

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator
    from pathlib import Path

    from himmy.benchmark.models import BenchmarkSuite
    from himmy.services.evaluation.models import EvaluationCase, EvaluationSuite

router = build_studio_router("eval", tag="studio-eval")

# ---- bounds ---------------------------------------------------------------

#: A UI run streams at most this many cases (an eval suite is hand-written;
#: anything larger belongs in CI, not a browser tab).
_MAX_CASES = 200
#: Cap on detail/answer strings sent over the wire (display strings, not data).
_DETAIL_CAP = 240
#: Per-case wall-clock ceiling. The stub answers in milliseconds; live local
#: models (Ollama / the Claude CLI) get a real budget.
_STUB_CASE_TIMEOUT_S = 30.0
_LIVE_CASE_TIMEOUT_S = 300.0

#: The synthetic suite id for "run the bench gate subset".
_BENCH_GATE_ID = "bench:gate"

#: Display model id per provider when the caller doesn't pick one.
_DEFAULT_MODELS = {"stub": "stub", "claude-cli": "haiku", "ollama": "llama3.2"}


def _baseline_file() -> Path:
    from himmy.api.studio_service import project_root

    return project_root() / "benchmarks" / "baseline.json"


def _cap(text: str, n: int = _DETAIL_CAP) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _sse(event: dict[str, Any]) -> str:
    """Encode one event dict as a Server-Sent-Events ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ---- GET /suites ----------------------------------------------------------


class RunnableSuite(BaseModel):
    """One runnable thing: an eval suite file or the checked-in bench gate."""

    id: str  # pass back to POST /run as `suite`
    kind: str  # "eval" | "bench"
    name: str
    cases: int
    source: str  # provenance shown in the GUI meta column


class SuitesResponse(BaseModel):
    suites: list[RunnableSuite]


def _gate_entry() -> RunnableSuite | None:
    """The bench-gate entry, or ``None`` when no parseable baseline is checked in."""
    path = _baseline_file()
    if not path.is_file():
        return None
    try:
        from himmy.benchmark.baseline import load_baseline

        data = load_baseline(path)
    except Exception:  # noqa: BLE001 - a malformed baseline just isn't runnable
        return None
    tasks = list((data.get("gate") or {}).get("tasks") or [])
    if not tasks:
        try:
            from himmy.benchmark import default_suite

            tasks = [t.id for t in default_suite().tasks]
        except Exception:  # noqa: BLE001 - packaged suite unavailable
            return None
    return RunnableSuite(
        id=_BENCH_GATE_ID,
        kind="bench",
        name=str(data.get("suite") or "bench gate"),
        cases=len(tasks),
        source="benchmarks/baseline.json",
    )


@router.get("/suites", response_model=SuitesResponse)
async def list_suites() -> SuitesResponse:
    """Everything runnable from the Evaluation screen, with case counts."""
    from himmy.api import studio_eval as discovery

    out = [
        RunnableSuite(id=s.path, kind="eval", name=s.name, cases=s.cases, source=s.path)
        for s in discovery.discover_suites()
    ]
    gate = _gate_entry()
    if gate is not None:
        out.append(gate)
    return SuitesResponse(suites=out)


# ---- GET /baseline ----------------------------------------------------------


class BaselineRow(BaseModel):
    """One gate metric: its bound (floor or ceiling) and the measured value."""

    metric: str
    bound: str  # "floor" | "ceiling"
    limit: float | None = None
    measured: float | None = None


class BaselineModelView(BaseModel):
    name: str
    rows: list[BaselineRow]


class BaselineView(BaseModel):
    """``benchmarks/baseline.json`` flattened for display (no secrets involved)."""

    exists: bool
    reason: str = ""
    suite: str = ""
    sha: str = ""
    date: str = ""
    trials: int = 0
    min_trials: int = 0
    gate_tasks: list[str] = []
    models: list[BaselineModelView] = []


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _model_rows(entry: dict[str, Any]) -> list[BaselineRow]:
    measured: dict[str, Any] = entry.get("measured") or {}
    floors: dict[str, Any] = entry.get("floors") or {}
    ceilings: dict[str, Any] = entry.get("ceilings") or {}
    rows = [
        BaselineRow(
            metric="accuracy",
            bound="floor",
            limit=_num(floors.get("accuracy")),
            measured=_num(measured.get("accuracy")),
        ),
        BaselineRow(
            metric="tool_call_accuracy",
            bound="floor",
            limit=_num(floors.get("tool_call_accuracy")),
            measured=_num(measured.get("tool_call_accuracy")),
        ),
        BaselineRow(
            metric="error_rate",
            bound="ceiling",
            limit=_num(ceilings.get("error_rate")),
            measured=_num(measured.get("error_rate")),
        ),
    ]
    cat_floors: dict[str, Any] = floors.get("by_category") or {}
    by_category: dict[str, Any] = measured.get("by_category") or {}
    for cat in sorted(set(by_category) | set(cat_floors)):
        rows.append(
            BaselineRow(
                metric=f"cat:{cat}",
                bound="floor",
                limit=_num(cat_floors.get(cat)),
                measured=_num(by_category.get(cat)),
            )
        )
    return rows


@router.get("/baseline", response_model=BaselineView)
async def baseline() -> BaselineView:
    """The checked-in bench baseline, parsed for the gate table."""
    path = _baseline_file()
    if not path.is_file():
        return BaselineView(
            exists=False,
            reason="No benchmarks/baseline.json in this project — cut one with "
            "`make bench-rebaseline` (or scripts/bench_gate.py).",
        )
    try:
        from himmy.benchmark.baseline import load_baseline

        data = load_baseline(path)
    except Exception as exc:  # noqa: BLE001 - malformed file is a display state
        return BaselineView(
            exists=False, reason=f"baseline file could not be parsed: {exc}"
        )
    at: dict[str, Any] = data.get("baselined_at") or {}
    gate: dict[str, Any] = data.get("gate") or {}
    models = [
        BaselineModelView(name=str(name), rows=_model_rows(entry or {}))
        for name, entry in (data.get("models") or {}).items()
    ]
    return BaselineView(
        exists=True,
        suite=str(data.get("suite") or ""),
        sha=str(at.get("sha") or ""),
        date=str(at.get("date") or ""),
        trials=int(at.get("trials") or 0),
        min_trials=int(gate.get("min_trials") or 0),
        gate_tasks=[str(t) for t in (gate.get("tasks") or [])],
        models=models,
    )


# ---- POST /run --------------------------------------------------------------


class RunEvalRequest(BaseModel):
    """Which suite to run, and (optionally) which live provider to run it on."""

    suite: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="an eval suite path (relative to the project root) "
        f"or {_BENCH_GATE_ID!r} for the bench gate subset",
    )
    provider: str | None = Field(None, max_length=64)
    model: str | None = Field(None, max_length=128)


def _case_prompt(case: EvaluationCase) -> str:
    data = case.input or {}
    for key in ("prompt", "question", "input", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(data, ensure_ascii=False) if data else ""


def _case_label(case: EvaluationCase, prompt: str, index: int) -> str:
    meta = case.metadata or {}
    for key in ("name", "label", "title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return _cap(value, 96)
    if prompt.strip():
        return _cap(prompt, 96)
    return f"case {index + 1}"


def _load_eval_suite_checked(rel_path: str) -> EvaluationSuite:
    """Resolve + load an eval suite, mapping failures to clean 4xx responses."""
    from himmy.api.studio_service import resolve_spec_path
    from himmy.config.eval_spec import load_eval_suite

    try:
        path = resolve_spec_path(rel_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"eval suite not found: {rel_path!r}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        suite = load_eval_suite(str(path))
    except Exception as exc:  # noqa: BLE001 - bad YAML is the caller's 400
        raise HTTPException(
            status_code=400, detail=f"could not parse eval suite: {exc}"
        ) from exc
    if not suite.cases:
        raise HTTPException(status_code=400, detail="suite has no cases to run")
    if len(suite.cases) > _MAX_CASES:
        raise HTTPException(
            status_code=400,
            detail=f"suite has {len(suite.cases)} cases; the UI runner caps at "
            f"{_MAX_CASES} — run it with `himmy eval` instead",
        )
    return suite


def _load_gate_suite() -> tuple[BenchmarkSuite, str]:
    """The bench gate sub-suite + display name (4xx when no baseline exists)."""
    path = _baseline_file()
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="no benchmarks/baseline.json in this project — the bench gate "
            "needs a checked-in baseline",
        )
    try:
        from himmy.benchmark.baseline import load_baseline

        data = load_baseline(path)
    except Exception as exc:  # noqa: BLE001 - malformed baseline
        raise HTTPException(
            status_code=400, detail=f"baseline file could not be parsed: {exc}"
        ) from exc
    from himmy.benchmark import default_suite
    from himmy.benchmark.baseline import subset_suite

    suite = default_suite()
    tasks = [str(t) for t in (data.get("gate") or {}).get("tasks") or []]
    if tasks:
        try:
            suite = subset_suite(suite, tasks)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not suite.tasks:
        raise HTTPException(status_code=400, detail="the bench gate has no tasks")
    return suite, str(data.get("suite") or "bench gate")


async def _eval_events(
    suite: EvaluationSuite, provider: str, model: str | None
) -> AsyncIterator[dict[str, Any]]:
    """Per-case events for an eval suite: model output → 5-metric scorecard."""
    from himmy.cli.provider import build_manager_for
    from himmy.services.evaluation.models import EvaluationSuite as Suite
    from himmy.services.evaluation.service import EvaluationService
    from himmy.services.inference.models import InferenceMessage, InferenceRequest
    from himmy.services.inference.service import InferenceService

    inference = InferenceService(build_manager_for(provider, model))
    scorer = EvaluationService()
    timeout = _STUB_CASE_TIMEOUT_S if provider == "stub" else _LIVE_CASE_TIMEOUT_S
    started = time.perf_counter()
    yield {
        "type": "start",
        "kind": "eval",
        "suite": suite.name,
        "cases": len(suite.cases),
        "provider": provider,
    }
    aggregates: list[float] = []
    passed_n = 0
    for i, case in enumerate(suite.cases):
        prompt = _case_prompt(case)
        actual = ""
        note = ""
        try:
            request = InferenceRequest(
                messages=[InferenceMessage(role="user", content=prompt)],
                timeout_seconds=min(timeout, 120.0),
            )
            response = await asyncio.wait_for(inference.run(request), timeout=timeout)
            actual = response.output_text or ""
            if response.error is not None:
                note = response.error.message
        except TimeoutError:
            note = f"inference timed out after {timeout:.0f}s"
        except Exception as exc:  # noqa: BLE001 - a broken case must not kill the run
            note = str(exc)
        one = Suite(suite_id=suite.suite_id, name=suite.name, cases=[case])
        run = await scorer.run_suite(suite=one, actual_outputs={case.case_id: actual})
        result = run.case_results[0]
        aggregates.append(result.aggregate)
        passed_n += 1 if result.passed else 0
        yield {
            "type": "case",
            "index": i,
            "id": case.case_id,
            "case": _case_label(case, prompt, i),
            "score": round(result.aggregate, 4),
            "passed": result.passed,
            "meta": _cap(note or actual),
            "metrics": [
                {
                    "metric": m.metric,
                    "score": round(m.score, 4),
                    "passed": m.passed,
                    "detail": _cap(m.detail),
                }
                for m in result.metric_scores
            ],
        }
    total = len(aggregates)
    yield {
        "type": "summary",
        "kind": "eval",
        "suite": suite.name,
        "aggregate_score": round(sum(aggregates) / total, 4) if total else 0.0,
        "passed": passed_n,
        "failed": total - passed_n,
        "cases": total,
        "duration_s": round(time.perf_counter() - started, 3),
    }


async def _bench_events(
    suite: BenchmarkSuite, suite_name: str, provider: str, model: str | None
) -> AsyncIterator[dict[str, Any]]:
    """Per-task events for the bench gate: one trial per task, graded."""
    from himmy.benchmark.models import BenchmarkSuite as Bench
    from himmy.benchmark.models import ModelSpec
    from himmy.benchmark.runner import BenchmarkRunner
    from himmy.cli.provider import build_manager_for
    from himmy.services.inference.service import InferenceService

    spec = ModelSpec(
        provider=provider, model=model or _DEFAULT_MODELS.get(provider, "default")
    )

    def _factory(_spec: object) -> InferenceService:
        return InferenceService(build_manager_for(provider, model))

    runner = BenchmarkRunner(trials=1, inference_factory=_factory)
    timeout = _STUB_CASE_TIMEOUT_S if provider == "stub" else _LIVE_CASE_TIMEOUT_S
    started = time.perf_counter()
    yield {
        "type": "start",
        "kind": "bench",
        "suite": suite_name,
        "cases": len(suite.tasks),
        "provider": provider,
        "model": spec.name,
    }
    correct = 0
    errors = 0
    tool_hits = 0
    tool_total = 0
    for i, task in enumerate(suite.tasks):
        trial = None
        try:
            cards = await asyncio.wait_for(
                runner.run(Bench(name=suite_name, tasks=[task]), [spec]),
                timeout=timeout,
            )
            trial = cards[0].task_scores[0].trials[0]
        except TimeoutError:
            pass  # surfaced as a failed case below
        if trial is None:
            errors += 1
            yield {
                "type": "case",
                "index": i,
                "id": task.id,
                "case": task.id,
                "score": 0.0,
                "passed": False,
                "meta": f"timed out after {timeout:.0f}s",
                "metrics": [
                    {
                        "metric": "grade",
                        "score": 0.0,
                        "passed": False,
                        "detail": f"timed out after {timeout:.0f}s",
                    }
                ],
            }
            continue
        correct += 1 if trial.correct else 0
        errors += 1 if trial.error else 0
        metrics: list[dict[str, Any]] = [
            {
                "metric": "grade",
                "score": 1.0 if trial.correct else 0.0,
                "passed": trial.correct,
                "detail": _cap(trial.error or trial.answer or "(no answer)"),
            }
        ]
        if trial.tool_ok is not None:
            tool_total += 1
            tool_hits += 1 if trial.tool_ok else 0
            metrics.append(
                {
                    "metric": "tools",
                    "score": 1.0 if trial.tool_ok else 0.0,
                    "passed": bool(trial.tool_ok),
                    "detail": _cap(", ".join(trial.tools_called) or "no tools called"),
                }
            )
        yield {
            "type": "case",
            "index": i,
            "id": task.id,
            "case": task.id,
            "score": 1.0 if trial.correct else 0.0,
            "passed": trial.correct and trial.error is None,
            "meta": f"{trial.latency_s:.2f}s",
            "metrics": metrics,
        }
    total = len(suite.tasks)
    yield {
        "type": "summary",
        "kind": "bench",
        "suite": suite_name,
        "model": spec.name,
        "aggregate_score": round(correct / total, 4) if total else 0.0,
        "passed": correct,
        "failed": total - correct,
        "cases": total,
        "duration_s": round(time.perf_counter() - started, 3),
        "metrics": {
            "accuracy": round(correct / total, 4) if total else 0.0,
            "tool_call_accuracy": round(tool_hits / tool_total, 4)
            if tool_total
            else None,
            "error_rate": round(errors / total, 4) if total else 0.0,
        },
        "note": "1 trial per task — the CI gate runs more",
    }


@router.post("/run")
async def run_suite(body: RunEvalRequest) -> StreamingResponse:
    """Run a suite, streaming ``start`` → ``case``… → ``summary`` frames (SSE).

    With no ``provider`` the run uses the offline deterministic stub (no network,
    no keys — done in seconds). A live provider (``ollama`` / ``claude-cli`` / …)
    runs real model calls and is only used when explicitly requested.
    """
    provider = (body.provider or "stub").strip().lower() or "stub"
    model = body.model.strip() if body.model and body.model.strip() else None
    # Validate the provider up front: a missing extra / unknown name is a clean
    # 4xx with the install hint, never a 500 or a mid-stream surprise.
    try:
        from himmy.cli.provider import build_manager_for

        build_manager_for(provider, model)
    except Exception as exc:  # noqa: BLE001 - ProviderError carries the remedy
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if body.suite.strip() == _BENCH_GATE_ID:
        gate_suite, gate_name = _load_gate_suite()
        events = _bench_events(gate_suite, gate_name, provider, model)
    else:
        events = _eval_events(
            _load_eval_suite_checked(body.suite.strip()), provider, model
        )

    async def _stream() -> AsyncIterator[str]:
        try:
            async for event in events:
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - surface as a terminal error frame
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["router"]
