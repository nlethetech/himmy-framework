"""The shared model-compare + catalog seam (T3d) Studio, /v1, and the CLI share."""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.inference import compare as cmp
from himmy.services.inference.compare import (
    CompareTargetSpec,
    build_model_catalog,
    run_compare,
)
from tests.conftest import run_async


def test_run_compare_preserves_target_order_and_succeeds() -> None:
    cells = run_async(
        run_compare(
            [
                CompareTargetSpec(provider="stub", model="a"),
                CompareTargetSpec(provider="stub", model="b"),
            ],
            prompt="say hello",
        )
    )
    assert [(c.provider, c.model) for c in cells] == [("stub", "a"), ("stub", "b")]
    assert all(c.ok for c in cells)
    assert all(c.output for c in cells)
    # The stub reports a concrete usage split (ints), never None, on success.
    assert all(isinstance(c.input_tokens, int) for c in cells)


def test_run_compare_reports_bad_provider_as_a_cell_not_an_exception() -> None:
    cells = run_async(
        run_compare(
            [
                CompareTargetSpec(provider="stub", model="a"),
                CompareTargetSpec(provider="no-such-provider", model="x"),
            ],
            prompt="hi",
        )
    )
    assert cells[0].ok is True
    bad = cells[1]
    assert bad.ok is False
    assert bad.error
    # A failed cell carries no usage split.
    assert bad.input_tokens is None and bad.cost is None


def test_run_compare_threads_a_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _Mgr:
        async def generate(self, request: Any) -> Any:
            seen["roles"] = [m.role for m in request.messages]
            from himmy.services.inference.models import (
                InferenceResponse,
                InferenceStatus,
            )

            return InferenceResponse(
                request_id="r", status=InferenceStatus.SUCCESS, output_text="ok"
            )

    monkeypatch.setattr(
        "himmy.cli.provider.build_manager_for", lambda p, m: _Mgr()
    )
    run_async(
        run_compare(
            [CompareTargetSpec(provider="stub", model="a")],
            prompt="hi",
            system="be terse",
        )
    )
    assert seen["roles"] == ["system", "user"]


def test_build_model_catalog_shape_and_claude_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _no_ollama(cap: int = 2) -> list[str]:
        return []

    monkeypatch.setattr(cmp, "_ollama_catalog_models", _no_ollama)
    monkeypatch.setattr(cmp.shutil, "which", lambda name: "/usr/bin/claude")

    catalog = run_async(build_model_catalog())
    by = {entry["provider"]: entry for entry in catalog}
    assert set(by) == {"ollama", "claude-cli"}
    assert by["ollama"]["available"] is False and by["ollama"]["models"] == []
    assert by["claude-cli"]["available"] is True
    assert [m["name"] for m in by["claude-cli"]["models"]] == [
        "haiku",
        "sonnet",
        "opus",
    ]


def test_build_model_catalog_decorates_with_bench_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ollama(cap: int = 2) -> list[str]:
        return ["llama3.2:latest"]

    monkeypatch.setattr(cmp, "_ollama_catalog_models", _ollama)
    monkeypatch.setattr(cmp.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cmp,
        "_bench_stats",
        lambda: {"llama3.2:latest": {"accuracy": 0.9, "latency": 1200.0}},
    )
    catalog = run_async(build_model_catalog())
    ollama = next(e for e in catalog if e["provider"] == "ollama")
    model = ollama["models"][0]
    assert model["name"] == "llama3.2:latest"
    assert model["accuracy"] == 0.9 and model["latency"] == 1200.0
