"""Query transformation — multi-query expansion and HyDE, opt-in and offline-safe.

A single user query phrasing can miss relevant chunks that say the same thing
differently. Two documented levers help:

* **Multi-query expansion** — ask an LLM for a handful of alternative phrasings,
  retrieve each, and fold every candidate list into the SAME RRF pool so a chunk
  any phrasing surfaces is fused in.
* **HyDE (Hypothetical Document Embeddings)** — ask the LLM to *write* a plausible
  answer, then retrieve with that hypothetical document instead of the bare query;
  it can sharpen dense recall but is a mixed lever (sometimes underperforms vanilla
  dense), so it is strictly opt-in and must be proven by the eval harness.

Both rewriters are backed by the existing :class:`ClientManager` seam (Claude-CLI /
Ollama / stub all qualify) and are lazy/guarded: the zero-config default rewriter is
:class:`IdentityRewriter`, which performs no rewrite and needs no model — so turning
``query_rewrite`` off (the default) keeps retrieval fully offline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from himmy.services.inference.client_manager import ClientManager
from himmy.services.inference.models import (
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    ResponseFormat,
)


@runtime_checkable
class QueryRewriterProtocol(Protocol):
    """Transform one query string into one or more retrieval queries.

    The returned list is what the hybrid pipeline retrieves over; each query's
    candidates are fused into a single RRF pool. An implementation that does not
    want to change anything returns ``[query]`` unchanged.
    """

    async def rewrite(self, query: str) -> list[str]:
        """Return the retrieval queries derived from ``query`` (>= 1 element)."""
        ...


class IdentityRewriter:
    """The offline default: no rewrite, returns the original query verbatim.

    Keeps the zero-config path model-free — selecting it (or leaving
    ``query_rewrite=False``) means nothing is sent to any provider.
    """

    async def rewrite(self, query: str) -> list[str]:
        """Return ``[query]`` unchanged."""
        return [query]


def _split_lines(text: str, *, limit: int) -> list[str]:
    """Split LLM output into clean, de-duplicated lines, bounded to ``limit``.

    Tolerates the common list shapes a model emits (numbered, bulleted, or plain
    lines) by stripping leading enumeration/bullet markers.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        # Strip a leading "1." / "1)" / "-" / "*" / "•" list marker.
        while line[:1] in {"-", "*", "•"}:
            line = line[1:].strip()
        if line[:1].isdigit():
            for sep in (".", ")"):
                head, _, rest = line.partition(sep)
                if head.isdigit():
                    line = rest.strip()
                    break
        if line and line.lower() not in seen:
            seen.add(line.lower())
            out.append(line)
        if len(out) >= limit:
            break
    return out


class MultiQueryRewriter:
    """Expand a query into N alternative phrasings via a :class:`ClientManager`.

    The original query is always kept (so expansion can only add recall, never lose
    the user's own wording), with up to ``num_expansions`` model-suggested
    alternatives appended. If the model call fails or returns nothing usable, the
    rewriter degrades gracefully to ``[query]`` — never a hard error mid-retrieval.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        *,
        num_expansions: int = 3,
        model_key: str = "default",
    ) -> None:
        """Wire the rewriter to a client manager and expansion budget."""
        if num_expansions < 0:
            raise ValueError("num_expansions must be >= 0")
        self._client = client_manager
        self._num_expansions = num_expansions
        self._model_key = model_key

    async def rewrite(self, query: str) -> list[str]:
        """Return ``[query, *alternatives]`` (original always first)."""
        if self._num_expansions == 0:
            return [query]
        prompt = (
            f"Rewrite the search query below into {self._num_expansions} alternative "
            "phrasings that a retrieval system could use to find the same "
            "information. Output one phrasing per line, no numbering, no extra text.\n"
            f"Query: {query}"
        )
        request = InferenceRequest(
            model_key=self._model_key,
            response_format=ResponseFormat.TEXT,
            messages=[InferenceMessage(role="user", content=prompt)],
        )
        response = await self._client.generate(request)
        queries = [query]
        if response.status == InferenceStatus.SUCCESS and response.output_text:
            for alt in _split_lines(response.output_text, limit=self._num_expansions):
                if alt.lower() != query.lower():
                    queries.append(alt)
        return queries


class HyDERewriter:
    """HyDE: generate a hypothetical answer and retrieve with it (opt-in).

    Returns the synthesized hypothetical document as the retrieval query (optionally
    keeping the original too, so dense+lexical still see the literal terms). Falls
    back to ``[query]`` if generation fails — HyDE never breaks a search.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        *,
        keep_original: bool = True,
        model_key: str = "default",
    ) -> None:
        """Wire the rewriter; ``keep_original`` also retrieves with the raw query."""
        self._client = client_manager
        self._keep_original = keep_original
        self._model_key = model_key

    async def rewrite(self, query: str) -> list[str]:
        """Return the hypothetical document (and optionally the original query)."""
        prompt = (
            "Write a short, factual passage that would directly answer the question "
            "below, as if it were an excerpt from a relevant document. Output only "
            "the passage.\n"
            f"Question: {query}"
        )
        request = InferenceRequest(
            model_key=self._model_key,
            response_format=ResponseFormat.TEXT,
            messages=[InferenceMessage(role="user", content=prompt)],
        )
        response = await self._client.generate(request)
        queries: list[str] = [query] if self._keep_original else []
        if (
            response.status == InferenceStatus.SUCCESS
            and response.output_text
            and response.output_text.strip()
        ):
            queries.append(response.output_text.strip())
        return queries or [query]


__all__ = [
    "QueryRewriterProtocol",
    "IdentityRewriter",
    "MultiQueryRewriter",
    "HyDERewriter",
]
