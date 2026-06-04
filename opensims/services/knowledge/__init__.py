"""Knowledge kernel: per-client embedded knowledge bases with evidenced retrieval."""

from __future__ import annotations

from opensims.services.knowledge.backend import (
    HALFVEC_MAX_DIM,
    VECTOR_MAX_DIM,
    KnowledgeBackendProtocol,
    PgVectorKnowledgeBackend,
    build_knowledge_schema_ddl,
    resolve_column_and_index,
)
from opensims.services.knowledge.chunker import SemanticChunker
from opensims.services.knowledge.embedder import (
    DeterministicEmbedder,
    EmbedderProtocol,
    OpenAIMultimodalEmbeddingModel,
    build_openai_compatible_embedder,
    embedder_is_multimodal,
)
from opensims.services.knowledge.models import (
    DocumentInput,
    KnowledgeBaseRecord,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievedChunk,
)
from opensims.services.knowledge.readers import (
    DocumentReader,
    DocumentReaderFactory,
    PDFReader,
    TextReader,
)
from opensims.services.knowledge.service import (
    KnowledgeBase,
    KnowledgeBaseAdapter,
    build_kb_context_field,
)
from opensims.services.knowledge.tools import (
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
    "embedder_is_multimodal",
    "OpenAIMultimodalEmbeddingModel",
    "build_openai_compatible_embedder",
    "SemanticChunker",
    "DocumentReader",
    "TextReader",
    "PDFReader",
    "DocumentReaderFactory",
    # pgvector backend (requires [postgres,knowledge] + a live DB)
    "PgVectorKnowledgeBackend",
    "KnowledgeBackendProtocol",
    "build_knowledge_schema_ddl",
    "resolve_column_and_index",
    "VECTOR_MAX_DIM",
    "HALFVEC_MAX_DIM",
    # in-run kb_search tool
    "register_kb_search_tool",
    "KB_SEARCH_ARGS_SCHEMA",
]
