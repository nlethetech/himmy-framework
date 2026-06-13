"""P0-C: compaction persists its summary as an episodic trace (was discarded).

The compactor already distills "every fact, decision, tool result" into a summary
before replacing the thread middle. Instead of throwing it away, the runtime now
saves it as an episodic memory object — a corpus of "what happened" for learning.
"""

from __future__ import annotations

from himmy.agents.personas.persona import Persona
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.conftest import run_async
from tests.runtime.test_compaction import _FixedSummaryManager, _over_budget_thread


def test_compaction_persists_episodic_summary() -> None:
    storage = StorageService()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_FixedSummaryManager("distilled trace")),
        memory_store=storage,
    )
    thread = _over_budget_thread()
    ctx = {
        "compaction_spec": {"max_tokens": 100, "keep_recent": 2},
        "subject_id": "boss",
    }
    run_async(rt._maybe_compact(Persona(name="a"), thread, ctx, "th:1", None))

    episodes = run_async(storage.list_episodic_memory("boss"))
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.payload["summary"] == "distilled trace"
    assert ep.payload["source"] == "compaction"
    assert ep.payload["tier"] == "archival"
    assert ep.metadata["trace_id"] == "th:1"
    assert ep.subject_id == "boss"


def test_compaction_without_memory_store_is_safe() -> None:
    """A runtime with no episodic store still compacts (the persist is best-effort)."""
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_FixedSummaryManager("s")),
    )
    thread = _over_budget_thread()
    ctx = {"compaction_spec": {"max_tokens": 100, "keep_recent": 2}}
    # No raise, and the in-context summary still applied.
    run_async(rt._maybe_compact(Persona(name="a"), thread, ctx, "th:1", None))
    assert any(m.metadata.get("compacted") for m in thread.messages)
