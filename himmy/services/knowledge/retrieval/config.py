"""RetrievalConfig — the single knob that selects the retrieval pipeline.

The KnowledgeBase is constructed with one of these. The DEFAULT instance
(``mode='dense'``, no fusion / rerank / rewrite) reproduces the pre-hybrid dense
cosine path byte-for-byte, so an existing caller that passes no ``retrieval`` kwarg
sees no behavioural change. Switching ``mode='hybrid'`` turns on BM25+dense RRF
fusion (still fully offline), and ``rerank`` / ``query_rewrite`` layer on the
optional, opt-in cross-encoder and query-transformation stages.

Every knob is intended to be proven before it ships: see
:func:`~himmy.services.knowledge.retrieval_eval.compare_retrieval`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from himmy.services.knowledge.retrieval.fusion import DEFAULT_RRF_K
from himmy.services.knowledge.retrieval.query_rewrite import QueryRewriterProtocol
from himmy.services.knowledge.retrieval.reranker import RerankerProtocol

RetrievalMode = Literal["dense", "hybrid"]


@dataclass(slots=True)
class RetrievalConfig:
    """Declarative selection of the retrieval pipeline and its optional stages.

    Attributes:
        mode: ``'dense'`` (default) is today's pure cosine path; ``'hybrid'`` fuses
            BM25 (lexical) and dense via Reciprocal Rank Fusion.
        rrf_k: The RRF damping constant used when fusing the two legs.
        candidate_pool: How deep each leg (and each rewritten query) retrieves
            before fusion — a wide pool then fused/reranked down to ``top_k``.
        rerank: When True, re-score the fused top-``candidate_pool`` with
            ``reranker`` before the final ``top_k`` cut.
        reranker: The cross-encoder to use when ``rerank`` is True. Required when
            ``rerank`` is set.
        query_rewrite: When True, transform the query via ``rewriter`` (multi-query
            / HyDE) and fold every variant's candidates into one RRF pool.
        rewriter: The query rewriter to use when ``query_rewrite`` is True. Required
            when ``query_rewrite`` is set.
    """

    mode: RetrievalMode = "dense"
    rrf_k: int = DEFAULT_RRF_K
    candidate_pool: int = 50
    rerank: bool = False
    reranker: RerankerProtocol | None = None
    query_rewrite: bool = False
    rewriter: QueryRewriterProtocol | None = None

    def __post_init__(self) -> None:
        """Validate the config so a misconfigured stage fails at construction.

        A turned-on stage must have its collaborator (a ``rerank`` with no reranker,
        or a ``query_rewrite`` with no rewriter, would silently no-op — an error is
        clearer). Numeric knobs are range-checked here rather than per-search.
        """
        if self.mode not in ("dense", "hybrid"):
            raise ValueError(f"mode must be 'dense' or 'hybrid', got {self.mode!r}")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        if self.candidate_pool <= 0:
            raise ValueError("candidate_pool must be a positive integer")
        if self.rerank and self.reranker is None:
            raise ValueError("rerank=True requires a reranker")
        if self.query_rewrite and self.rewriter is None:
            raise ValueError("query_rewrite=True requires a rewriter")

    @property
    def is_default_dense(self) -> bool:
        """True when this config is the untouched dense path (no new stages).

        The KnowledgeBase uses this to take the original, zero-risk code path
        verbatim, guaranteeing the additive promise for callers that opt into
        nothing.
        """
        return self.mode == "dense" and not self.rerank and not self.query_rewrite


#: The framework default — exactly the pre-hybrid dense cosine behaviour.
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()


__all__ = ["RetrievalConfig", "RetrievalMode", "DEFAULT_RETRIEVAL_CONFIG"]
