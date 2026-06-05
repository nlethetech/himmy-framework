"""Example 12: durable memory — remember a fact, then recall it semantically.

    python examples/12_memory_recall.py                          # offline (deterministic embedder)
    HIMMY_EMBEDDER=ollama HIMMY_EMBEDDER_MODEL=nomic-embed-text \\
        python examples/12_memory_recall.py                      # real semantic embeddings

Shows the memory module directly (no LLM needed): persist facts for a subject and recall
the most relevant one for a query. With a real embedder, recall matches by meaning, not
just shared words.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from himmy.services.memory import InMemoryMemoryStore, MemoryService  # noqa: E402
from himmy.toolkit import ToolkitConfig  # noqa: E402


async def main() -> None:
    """Remember a few facts for a subject, then recall the most relevant one."""
    embedder, _dim = ToolkitConfig.from_env().build_embedder_and_dim()
    memory = MemoryService(InMemoryMemoryStore(), embedder=embedder)

    facts = [
        "The farm in Bardaghat runs poultry, ducks, a fish pond, and an orchard.",
        "Bees pollinate the orchard and produce honey each season.",
        "The closed-loop design feeds duck waste into the fish pond.",
    ]
    for fact in facts:
        memory.remember(fact, subject_id="farm")

    query = os.environ.get("HIMMY_EXAMPLE_QUERY", "how is honey produced on the farm")
    hits = await memory.recall(query, subject_id="farm", top_k=2)

    print("=== Example 12: memory recall ===")
    print(f"embedder : {ToolkitConfig.from_env().embedder}")
    print(f"query    : {query}")
    for hit in hits:
        print(f"  [{hit.similarity:.3f}] {hit.record.text}")


if __name__ == "__main__":
    asyncio.run(main())
