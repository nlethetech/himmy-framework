"""Hybrid retrieval: BM25 + dense fused by RRF, with optional rerank / query rewrite.

This subpackage is additive on top of the dense KnowledgeBase. The pieces:

* :mod:`config` — :class:`RetrievalConfig`, the single knob the KB is built with
  (default reproduces the dense path byte-for-byte);
* :mod:`fusion` — :func:`reciprocal_rank_fusion`, the pure-function RRF;
* :mod:`lexical` — :class:`BM25Index`, the pure-Python sparse leg (offline);
* :mod:`hybrid` — :class:`HybridRetriever`, the dense+lexical->RRF->rerank engine;
* :mod:`reranker` — the opt-in cross-encoder seam (behind the [embeddings] extra);
* :mod:`query_rewrite` — opt-in multi-query / HyDE seams (ClientManager-backed).
"""

from __future__ import annotations

from himmy.services.knowledge.retrieval.config import (
    DEFAULT_RETRIEVAL_CONFIG,
    RetrievalConfig,
    RetrievalMode,
)
from himmy.services.knowledge.retrieval.fusion import (
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
)
from himmy.services.knowledge.retrieval.hybrid import HybridRetriever
from himmy.services.knowledge.retrieval.lexical import (
    BM25Index,
    LexicalIndex,
    tokenize,
)
from himmy.services.knowledge.retrieval.query_rewrite import (
    HyDERewriter,
    IdentityRewriter,
    MultiQueryRewriter,
    QueryRewriterProtocol,
)
from himmy.services.knowledge.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    FastEmbedReranker,
    RerankerProtocol,
    build_reranker,
    fastembed_rerank_available,
)

__all__ = [
    "RetrievalConfig",
    "RetrievalMode",
    "DEFAULT_RETRIEVAL_CONFIG",
    "reciprocal_rank_fusion",
    "DEFAULT_RRF_K",
    "BM25Index",
    "LexicalIndex",
    "tokenize",
    "HybridRetriever",
    "RerankerProtocol",
    "FastEmbedReranker",
    "build_reranker",
    "fastembed_rerank_available",
    "DEFAULT_RERANKER_MODEL",
    "QueryRewriterProtocol",
    "IdentityRewriter",
    "MultiQueryRewriter",
    "HyDERewriter",
]
