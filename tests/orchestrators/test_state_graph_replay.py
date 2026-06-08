"""StateGraph deterministic replay: a graph of inference-using nodes replays exactly.

Because a graph node that needs a model goes through the existing
:class:`InferenceService`, a graph run recorded with
:class:`RecordingClientManager` replays bit-identically with
:class:`ReplayClientManager` — no provider, no tool side effects — giving
"audited, replayable graph workflows" for free on the existing cassette seam.
"""

from __future__ import annotations

from himmy.orchestrators.state_graph import END, StateGraph
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    InferenceMessage,
    InferenceRequest,
)
from himmy.services.inference.replay import (
    RecordingClientManager,
    ReplayClientManager,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async


def _llm_graph(service: InferenceService) -> StateGraph:
    """A graph whose nodes call inference, accumulating their answers."""
    g = StateGraph("llm")
    g.set_reducer("answers", lambda old, new: [*(old or []), *new])

    async def ask_a(state: dict) -> dict:
        resp = await service.run(
            InferenceRequest(
                messages=[InferenceMessage(role="user", content="question A")]
            )
        )
        return {"answers": [resp.output_text]}

    async def ask_b(state: dict) -> dict:
        resp = await service.run(
            InferenceRequest(
                messages=[InferenceMessage(role="user", content="question B")]
            )
        )
        return {"answers": [resp.output_text]}

    g.add_node("ask_a", ask_a)
    g.add_node("ask_b", ask_b)
    g.add_edge("ask_a", "ask_b")
    g.add_edge("ask_b", END)
    g.set_entry_point("ask_a")
    return g


def test_recorded_graph_replays_identically() -> None:
    # 1) Record a live (stub) run.
    recorder = RecordingClientManager(StubClientManager())
    rec_service = InferenceService(recorder)
    res_live = run_async(_llm_graph(rec_service).compile().invoke({}))
    assert res_live.status == "completed"
    assert len(recorder.cassette.entries) == 2

    # 2) Replay from the cassette: no provider, deterministic, identical outputs.
    replay = ReplayClientManager(recorder.cassette)
    replay_service = InferenceService(replay)
    res_replay = run_async(_llm_graph(replay_service).compile().invoke({}))
    assert res_replay.status == "completed"
    assert res_replay.final_state["answers"] == res_live.final_state["answers"]
    assert replay.remaining() == 0


def test_replay_run_uses_no_provider() -> None:
    """A replayed graph never touches the underlying client manager."""

    class _ExplodingManager:
        provider_name = "explode"

        def resolve(self, model_key: str) -> str:
            return model_key

        async def generate(self, request: InferenceRequest) -> object:
            raise AssertionError("replay must not call the provider")

    recorder = RecordingClientManager(StubClientManager())
    run_async(_llm_graph(InferenceService(recorder)).compile().invoke({}))

    # No fallback => any cassette miss would raise; an exploding fallback would
    # raise on use. Neither happens because the cassette fully covers the run.
    replay = ReplayClientManager(recorder.cassette, fallback=_ExplodingManager())
    res = run_async(_llm_graph(InferenceService(replay)).compile().invoke({}))
    assert res.status == "completed"
    assert replay.remaining() == 0


def test_pure_python_graph_is_deterministic_without_replay() -> None:
    """A graph of pure-Python nodes is deterministic by construction (offline)."""
    g = StateGraph("pure")

    async def step(state: dict) -> dict:
        return {"acc": state.get("acc", 0) + 10}

    def route(state: dict) -> str:
        return "step" if state["acc"] < 30 else END

    g.add_node("step", step)
    g.add_conditional_edges("step", route)
    g.set_entry_point("step")

    first = run_async(g.compile().invoke({}))
    second = run_async(g.compile().invoke({}))
    assert first.final_state == second.final_state == {"acc": 30}
    assert first.node_sequence == second.node_sequence
