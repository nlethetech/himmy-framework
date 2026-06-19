"""Knowledge kernel: per-client embedded knowledge bases with evidenced retrieval."""

from __future__ import annotations

from himmy.services.knowledge.backend import (
    HALFVEC_MAX_DIM,
    VECTOR_MAX_DIM,
    KnowledgeBackendProtocol,
    LexicalSearchProtocol,
    PgVectorKnowledgeBackend,
    build_knowledge_schema_ddl,
    build_lexical_index_ddl,
    resolve_column_and_index,
)
from himmy.services.knowledge.chunker import MarkdownAwareChunker, SemanticChunker
from himmy.services.knowledge.embedder import (
    EMBEDDER_FINGERPRINT_KEY,
    DeterministicEmbedder,
    EmbedderProtocol,
    OpenAIMultimodalEmbeddingModel,
    build_openai_compatible_embedder,
    embedder_fingerprint,
    embedder_is_multimodal,
)
from himmy.services.knowledge.models import (
    DocumentInput,
    KnowledgeBaseRecord,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievedChunk,
)
from himmy.services.knowledge.readers import (
    CsvReader,
    DocumentReader,
    DocumentReaderFactory,
    ExcelReader,
    PDFReader,
    TextReader,
)
from himmy.services.knowledge.retrieval import (
    DEFAULT_RETRIEVAL_CONFIG,
    DEFAULT_RRF_K,
    BM25Index,
    FastEmbedReranker,
    HybridRetriever,
    HyDERewriter,
    IdentityRewriter,
    LexicalIndex,
    MultiQueryRewriter,
    QueryRewriterProtocol,
    RerankerProtocol,
    RetrievalConfig,
    RetrievalMode,
    build_reranker,
    fastembed_rerank_available,
    reciprocal_rank_fusion,
)
from himmy.services.knowledge.retrieval_eval import (
    RetrievalEvalCase,
    RetrievalEvalReport,
    RetrievalScore,
    aggregate_scores,
    compare_retrieval,
    evaluate_retrieval,
    score_retrieval,
)
from himmy.services.knowledge.service import (
    KnowledgeBase,
    KnowledgeBaseAdapter,
    build_kb_context_field,
)
from himmy.services.knowledge.sqlite_backend import SqliteKnowledgeBackend
from himmy.services.knowledge.tools import (
    KB_SEARCH_ARGS_SCHEMA,
    register_kb_search_tool,
)

__all__ = [
    "KnowledgeBase",
    "KnowledgeBaseAdapter",
    "build_kb_context_field",
    "KnowledgeBaseRecord",
    "KnowledgeDocument",
    "KnowledgeChunk",
    "RetrievedChunk",
    "DocumentInput",
    "DeterministicEmbedder",
    "EmbedderProtocol",
    "EMBEDDER_FINGERPRINT_KEY",
    "embedder_fingerprint",
    "embedder_is_multimodal",
    "OpenAIMultimodalEmbeddingModel",
    "build_openai_compatible_embedder",
    "SemanticChunker",
    "MarkdownAwareChunker",
    "DocumentReader",
    "TextReader",
    "PDFReader",
    "CsvReader",
    "ExcelReader",
    "DocumentReaderFactory",
    # durable disk-backed backend (stdlib sqlite3 — desktop/offline persistence)
    "SqliteKnowledgeBackend",
    # pgvector backend (requires [postgres,knowledge] + a live DB)
    "PgVectorKnowledgeBackend",
    "KnowledgeBackendProtocol",
    "LexicalSearchProtocol",
    "build_knowledge_schema_ddl",
    "build_lexical_index_ddl",
    "resolve_column_and_index",
    "VECTOR_MAX_DIM",
    "HALFVEC_MAX_DIM",
    # in-run kb_search tool
    "register_kb_search_tool",
    "KB_SEARCH_ARGS_SCHEMA",
    # hybrid retrieval (BM25 + dense RRF, optional rerank / query rewrite)
    "RetrievalConfig",
    "RetrievalMode",
    "DEFAULT_RETRIEVAL_CONFIG",
    "reciprocal_rank_fusion",
    "DEFAULT_RRF_K",
    "BM25Index",
    "LexicalIndex",
    "HybridRetriever",
    "RerankerProtocol",
    "FastEmbedReranker",
    "build_reranker",
    "fastembed_rerank_available",
    "QueryRewriterProtocol",
    "IdentityRewriter",
    "MultiQueryRewriter",
    "HyDERewriter",
    # retrieval-quality evaluation
    "RetrievalEvalCase",
    "RetrievalScore",
    "RetrievalEvalReport",
    "score_retrieval",
    "aggregate_scores",
    "evaluate_retrieval",
    "compare_retrieval",
]
