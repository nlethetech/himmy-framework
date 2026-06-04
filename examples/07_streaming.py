"""Example 07: streaming — incremental token deltas via ``InferenceService.run_stream``.

Runs offline against the deterministic stub by default. The stub client manager
exposes ``generate_stream``, the offline analogue of provider token streaming, so
``run_stream`` yields incremental :class:`StreamDelta` chunks and a final ``done``
frame that carries the fully-materialized :class:`InferenceResponse`. With a real
provider configured (``pydantic_ai`` + a key + ``OPENSIMS_EXAMPLES_MODEL``) the
exact same loop streams genuine provider deltas — no code change.

    python examples/07_streaming.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Make the sibling ``_runtime`` module importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _runtime import build_inference  # noqa: E402

from opensims.services.inference.models import (  # noqa: E402
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
)


async def main() -> None:
    """Stream one inference response and print deltas as they arrive."""
    inference = build_inference()

    request = InferenceRequest(
        messages=[
            InferenceMessage(
                role="system",
                content="You are a concise market-research analyst.",
            ),
            InferenceMessage(
                role="user",
                content="Summarize why semiconductor demand is cyclical.",
            ),
        ]
    )

    print("=== Example 07: streaming ===")
    print(f"request_id : {request.request_id}")
    print("-" * 60)
    print("streamed deltas (incremental):")

    deltas: list = []
    assembled: list[str] = []
    final = None
    # The same loop works for the offline stub and for real providers.
    async for delta in inference.run_stream(request, chunk_size=24):
        deltas.append(delta)
        if delta.done:
            final = delta.response
            break
        assembled.append(delta.delta)
        # Print each chunk inline, no newline, to show progressive arrival.
        print(f"  [chunk {delta.index:>2}] {delta.delta!r}")

    print("-" * 60)
    reassembled = "".join(assembled)
    print("reassembled text:")
    print(reassembled)
    print("-" * 60)

    assert final is not None, "stream must end with a done frame carrying a response"
    assert final.status == InferenceStatus.SUCCESS
    # The reassembled deltas must exactly equal the final buffered output.
    assert reassembled == (final.output_text or ""), "deltas must reassemble cleanly"
    # Every delta correlates to the originating request.
    assert all(d.request_id == request.request_id for d in deltas)

    print(f"deltas       : {len(deltas)} ({len(deltas) - 1} content + 1 done frame)")
    print(f"final status : {final.status.value}")
    print(f"provider     : {final.provider_name or 'stub'}")
    print(f"model_path   : {final.model_path}")
    print(f"output_tokens: {final.output_tokens}")
    print("stream reassembled cleanly and matched the final response.")


if __name__ == "__main__":
    asyncio.run(main())
