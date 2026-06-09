"""Live Ollama integration — real local LLM + real local embedding model.

These are the "offline-first means *local models*" tests: they run genuine semantic
recall against a pulled Ollama embedding model (``qwen3-embedding`` by default) and a
real tool-less/▪tool LLM turn against a local ``qwen2.5`` model. Nothing here uses the
deterministic stub — that is the point: it proves the Ollama embedding path and the
``"auto"`` embed-model gate actually work end to end on downloaded local models.

Marked ``integration`` (so the offline unit suite — which the hermetic conftest fixture
forces onto the deterministic backend — is unaffected) and skipped unless Ollama is
reachable AND the required models are pulled. Configure via env:

* ``HIMMY_OLLAMA_EMBED_MODEL`` (default ``qwen3-embedding``)
* ``HIMMY_INTEGRATION_MODEL`` (default ``qwen2.5:3b-instruct``)

Run them with:  ``pytest -m integration tests/integration/test_ollama_live.py -v``
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import httpx
import pytest

from himmy.services.knowledge.local_embedders import (
    default_ollama_embed_model,
    ollama_embed_model_available,
    ollama_reachable,
)
from tests.conftest import run_async

_OLLAMA_URL = os.environ.get("HIMMY_OLLAMA_URL", "http://localhost:11434")
_EMBED_MODEL = default_ollama_embed_model()
_LLM_MODEL = os.environ.get("HIMMY_INTEGRATION_MODEL", "qwen2.5:3b-instruct")


def _llm_available() -> bool:
    """True when Ollama is up and the integration LLM model is pulled."""
    try:
        resp = httpx.get(f"{_OLLAMA_URL}/api/tags", timeout=3.0)
    except Exception:
        return False
    if resp.status_code != 200:
        return False
    pulled = {
        str(m.get("name", "")).split(":", 1)[0] for m in resp.json().get("models", [])
    }
    return _LLM_MODEL.split(":", 1)[0] in pulled


def _embed_available() -> bool:
    return ollama_reachable(_OLLAMA_URL) and ollama_embed_model_available(
        base_url=_OLLAMA_URL
    )


pytestmark = pytest.mark.integration

_needs_embed = pytest.mark.skipif(
    not _embed_available(),
    reason=f"Ollama embed model {_EMBED_MODEL!r} not pulled (ollama pull {_EMBED_MODEL})",
)
_needs_llm = pytest.mark.skipif(
    not _llm_available(), reason=f"Ollama LLM model {_LLM_MODEL!r} not pulled"
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ----------------------------------------------------------- real embedding model
@_needs_embed
def test_qwen3_embedding_ranks_semantically() -> None:
    """The live embed model ranks a semantically-related sentence over an unrelated one.

    A pure lexical embedder (the offline default) could not do this: the related sentence
    shares almost no tokens with the query, yet a real embedding model scores it higher.
    """
    from himmy.services.knowledge.local_embedders import OllamaEmbedder

    emb = OllamaEmbedder(model=_EMBED_MODEL, base_url=_OLLAMA_URL)
    assert emb.dim >= 256  # qwen3-embedding is 4096-d; guard against a wrong-dim model

    query = run_async(emb.embed_query("how do honeybees make honey?"))
    related = run_async(
        emb.embed_query("pollinators in the apiary convert nectar into golden syrup")
    )
    unrelated = run_async(
        emb.embed_query("the central bank raised interest rates on Tuesday")
    )
    assert _cosine(query, related) > _cosine(query, unrelated)


@_needs_embed
def test_auto_resolves_to_ollama_when_embed_model_pulled() -> None:
    """With the embed model genuinely pulled, ``"auto"`` selects the Ollama backend live.

    Runs with NO mocks (the hermetic conftest fixture skips ``integration`` tests), so this
    exercises the real ``ollama_embed_model_available`` gate against the live server.
    """
    from himmy.services.knowledge.local_embedders import (
        fastembed_available,
        resolve_auto_backend,
    )

    expected = "fastembed" if fastembed_available() else "ollama"
    assert resolve_auto_backend(ollama_base_url=_OLLAMA_URL) == expected


@_needs_embed
def test_knowledge_semantic_recall_with_ollama(tmp_path: Path) -> None:
    """End-to-end RAG on real embeddings: a semantic query retrieves the right document."""
    from himmy.services.tools.registry import ToolRegistry
    from himmy.toolkit.config import ToolkitConfig
    from himmy.toolkit.knowledge import register_knowledge_pack

    registry = ToolRegistry()
    register_knowledge_pack(
        registry, ToolkitConfig(embedder="ollama", fs_root=tmp_path)
    )
    ingest = registry.handler_for("kb_ingest")
    search = registry.handler_for("kb_search")

    run_async(
        ingest(
            {
                "text": "Honeybees forage nectar and pollinate orchard blossoms.",
                "title": "bees",
            }
        )
    )
    run_async(
        ingest(
            {
                "text": "The quarterly earnings call covered revenue and dividends.",
                "title": "finance",
            }
        )
    )

    # Semantically about bees, but almost no shared tokens with the bees document.
    found = run_async(
        search({"query": "beekeeping and pollination of fruit trees", "top_k": 2})
    )
    assert found["results"], "expected at least one semantic hit"
    top_text = (found["results"][0]["text"] or "").lower()
    assert "honeybees" in top_text or "pollinate" in top_text, (
        f"semantic recall ranked the wrong doc first: {found['results']!r}"
    )


@_needs_embed
def test_memory_semantic_recall_with_ollama(tmp_path: Path) -> None:
    """A remembered fact is recalled by a *paraphrased* query via real embeddings."""
    from himmy.services.tools.registry import ToolRegistry
    from himmy.toolkit.config import ToolkitConfig
    from himmy.toolkit.memory import register_memory_pack

    cfg = ToolkitConfig(
        embedder="ollama", memory_path=str(tmp_path / "m.db"), memory_subject="s"
    )
    registry = ToolRegistry()
    register_memory_pack(registry, cfg)
    registry.handler_for("remember")(
        {"text": "The customer prefers email over phone calls."}
    )
    found = run_async(
        registry.handler_for("recall")(
            {"query": "what is the best way to contact this client?", "top_k": 3}
        )
    )
    assert found["results"], "real-embedding recall returned nothing"
    assert "email" in found["results"][0]["text"].lower()


# ----------------------------------------------------------------- real local LLM
@_needs_llm
def test_live_llm_agent_answers() -> None:
    """A no-tool agent returns a sensible, non-empty answer from the live local LLM."""
    from himmy import build_runtime
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona
    from himmy.cli.provider import build_inference_for

    inference = build_inference_for("ollama", _LLM_MODEL)
    runtime, _inf, _tools = build_runtime(inference=inference)
    persona = Persona(name="geo", instructions=["Answer concisely."])
    task = Task(title="t", prompt="What is the capital of France? Answer in one word.")

    loop = run_async(runtime.run_agent_loop(persona, task, max_turns=2))
    answer = (loop.final.output_text or "").strip().lower()
    assert answer, "live LLM returned an empty answer"
    assert "paris" in answer, f"expected 'paris', got: {answer!r}"
