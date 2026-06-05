"""Tier 1.3 — the tool router: narrow to the relevant few tools per query."""

from __future__ import annotations

from typing import Any

from himmy.runtime.tool_router import select_tools
from himmy.services.inference.models import InferenceResponse, InferenceStatus
from tests.conftest import run_async


class _FakeInference:
    """Returns a fixed structured tool selection (and records that it was called)."""

    def __init__(self, tools: list[str]) -> None:
        self._tools = tools
        self.called = False

    async def run(self, request: Any) -> InferenceResponse:
        self.called = True
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_structured={"tools": self._tools},
        )


def _cands(n: int) -> list[tuple[str, str]]:
    return [(f"t{i}", f"description {i}") for i in range(n)]


def test_no_routing_when_few_candidates() -> None:
    fi = _FakeInference(["t0"])
    out = run_async(select_tools(fi, "q", _cands(3), max_tools=4))
    assert out == ["t0", "t1", "t2"]  # all returned
    assert fi.called is False  # no inference call needed


def test_routes_to_valid_subset() -> None:
    fi = _FakeInference(["t2", "t5", "not-a-tool"])
    out = run_async(select_tools(fi, "q", _cands(8), max_tools=4))
    assert out == ["t2", "t5"]  # invalid name dropped
    assert fi.called is True


def test_caps_at_max_tools() -> None:
    fi = _FakeInference([f"t{i}" for i in range(8)])
    out = run_async(select_tools(fi, "q", _cands(10), max_tools=3))
    assert len(out) == 3


def test_failure_falls_back_to_all() -> None:
    class _Boom:
        async def run(self, request: Any) -> InferenceResponse:
            raise RuntimeError("router model down")

    out = run_async(select_tools(_Boom(), "q", _cands(8), max_tools=4))
    assert len(out) == 8  # never narrow on failure


def test_empty_selection_falls_back_to_all() -> None:
    out = run_async(select_tools(_FakeInference([]), "q", _cands(8), max_tools=4))
    assert len(out) == 8  # never strand the agent with zero tools
