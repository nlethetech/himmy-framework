"""Lexical (BM25) retrieval — the sparse leg of hybrid search.

Dense vectors are strong on paraphrase but systematically miss exact-match queries
(rare tokens, codes, identifiers, proper names) because hashing/embedding dilutes
a single decisive term. BM25 is the opposite: a pure-Python, dependency-free
sparse ranker that rewards exact term overlap weighted by inverse document
frequency and length-normalized term saturation. Fusing the two (see
:mod:`himmy.services.knowledge.retrieval.fusion`) gets the best of both.

This index keys chunks by ``chunk_id`` — the SAME identity the in-memory
KnowledgeBase uses — so fusion over the two legs is well-defined and the index can
be pruned in lock-step with the KB on delete/replace. Tokenization is shared with
:class:`~himmy.services.knowledge.embedder.DeterministicEmbedder` so the offline
lexical and dense legs agree on what a "token" is.

The index is incremental (add/remove single chunks) and recomputes corpus IDF
lazily on the first search after any mutation, so ingest/delete stay O(chunk) and
the O(corpus) IDF pass is paid once per search-after-change, never per write.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from himmy.services.knowledge.embedder import DeterministicEmbedder

#: BM25 term-frequency saturation parameter (Robertson/Okapi default).
BM25_K1 = 1.5
#: BM25 document-length normalization parameter (Okapi default).
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Tokenize text the same way the offline dense embedder does.

    Sharing the tokenizer keeps the lexical and dense legs consistent about word
    boundaries so a hybrid run reasons over one notion of "token".
    """
    return DeterministicEmbedder._tokenize(text)


@runtime_checkable
class LexicalIndex(Protocol):
    """The lexical-retrieval contract the hybrid pipeline depends on.

    An implementation maintains a per-corpus inverted index keyed by ``chunk_id``
    and answers ``search`` with a ranked ``(chunk_id, score)`` list.
    """

    def add(self, chunk_id: str, text: str) -> None:
        """Index one chunk's text under ``chunk_id`` (idempotent on re-add)."""
        ...

    def remove(self, chunk_id: str) -> None:
        """Drop a chunk from the index (no-op if absent)."""
        ...

    def search(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(chunk_id, score)`` pairs, best first."""
        ...


class BM25Index:
    """A pure-Python in-memory BM25 (Okapi) inverted index keyed by ``chunk_id``.

    No third-party dependency — the sparse leg works in the zero-config offline
    path. Documents are added/removed one chunk at a time; corpus statistics (doc
    frequency, average length) are kept current incrementally and the IDF table is
    rebuilt lazily on the first search after a mutation.
    """

    def __init__(self, *, k1: float = BM25_K1, b: float = BM25_B) -> None:
        """Configure BM25 saturation (``k1``) and length-normalization (``b``)."""
        self.k1 = k1
        self.b = b
        #: chunk_id -> {token: term_frequency}
        self._postings: dict[str, dict[str, int]] = {}
        #: chunk_id -> document length in tokens
        self._lengths: dict[str, int] = {}
        #: token -> number of documents containing it
        self._doc_freq: dict[str, int] = {}
        self._total_len = 0
        #: Cached token -> idf, invalidated on any mutation.
        self._idf: dict[str, float] | None = None

    @property
    def num_documents(self) -> int:
        """How many chunks are currently indexed."""
        return len(self._postings)

    def add(self, chunk_id: str, text: str) -> None:
        """Index ``text`` under ``chunk_id``; re-adding the same id replaces it."""
        if chunk_id in self._postings:
            # Replace cleanly so corpus stats never double-count a re-ingested chunk.
            self.remove(chunk_id)
        tokens = tokenize(text)
        if not tokens:
            return
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        self._postings[chunk_id] = tf
        self._lengths[chunk_id] = len(tokens)
        self._total_len += len(tokens)
        for token in tf:
            self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._idf = None

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk and decrement the corpus statistics it contributed."""
        tf = self._postings.pop(chunk_id, None)
        if tf is None:
            return
        self._total_len -= self._lengths.pop(chunk_id, 0)
        for token in tf:
            remaining = self._doc_freq.get(token, 0) - 1
            if remaining <= 0:
                self._doc_freq.pop(token, None)
            else:
                self._doc_freq[token] = remaining
        self._idf = None

    def _ensure_idf(self) -> dict[str, float]:
        """Return the (cached) IDF table, rebuilding it after any mutation.

        Uses the BM25 "plus-one" IDF, ``ln(1 + (N - df + 0.5) / (df + 0.5))``,
        which is non-negative for every term so a frequent term never contributes a
        negative score that could rank a matching chunk below a non-matching one.
        """
        if self._idf is not None:
            return self._idf
        n = len(self._postings)
        idf: dict[str, float] = {}
        for token, df in self._doc_freq.items():
            idf[token] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._idf = idf
        return idf

    def search(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        """Score every chunk that shares a query token and return the top ``top_k``.

        Chunks with zero query-token overlap score nothing and are never returned,
        so the lexical leg contributes only genuine lexical matches to the fusion
        pool. Ties are broken deterministically by ``chunk_id``.
        """
        if top_k <= 0 or not self._postings:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        idf = self._ensure_idf()
        avg_len = self._total_len / len(self._postings) if self._postings else 0.0
        scored: list[tuple[str, float]] = []
        for chunk_id, tf in self._postings.items():
            length = self._lengths[chunk_id]
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token)
                if not freq:
                    continue
                token_idf = idf.get(token, 0.0)
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * (length / avg_len if avg_len else 0.0)
                )
                score += token_idf * (freq * (self.k1 + 1.0)) / denom if denom else 0.0
            if score > 0.0:
                scored.append((chunk_id, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]


__all__ = [
    "LexicalIndex",
    "BM25Index",
    "tokenize",
    "BM25_K1",
    "BM25_B",
]
