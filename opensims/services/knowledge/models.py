"""Knowledge kernel: KB / document / chunk / retrieval record types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from opensims.core.ids import new_uuid


class KnowledgeBaseRecord(BaseModel):
    """One named, per-client knowledge index, scoped by (workspace, client, name)."""

    kb_id: str = Field(default_factory=new_uuid)
    workspace_id: str
    client_id: str
    name: str
    vector_dim: int = 64
    metadata: dict[str, Any] = {}


class KnowledgeDocument(BaseModel):
    """One ingested source. Holds the full original text for windowed retrieval.

    ``text`` is None for image documents (the binary lives elsewhere).
    """

    document_id: str = Field(default_factory=new_uuid)
    kb_id: str
    title: str | None = None
    source_uri: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = {}


class KnowledgeChunk(BaseModel):
    """One embedded segment of a document (text-kind or image-kind)."""

    chunk_id: str = Field(default_factory=new_uuid)
    document_id: str
    kb_id: str
    text: str | None = None
    start_pos: int = 0
    end_pos: int = 0
    embedding: list[float] = []
    chunk_kind: str = "text"
    image_uri: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = {}


class RetrievedChunk(BaseModel):
    """A search result — a chunk with its similarity and parent-doc context window."""

    text: str | None = None
    similarity: float = 0.0
    context_window: str | None = None
    document_id: str
    source_uri: str | None = None
    chunk_kind: str = "text"
    metadata: dict[str, Any] = {}


class DocumentInput(BaseModel):
    """A programmatic ingestion item — exactly one of ``text`` or ``file``."""

    text: str | None = None
    file: str | None = None
    title: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def _exactly_one_source(self) -> DocumentInput:
        """Require exactly one *non-empty* ``text`` / ``file`` source.

        An empty/whitespace-only ``text`` or an empty ``file`` path would otherwise
        pass the bare ``is not None`` check and ingest as a silent no-op document
        (chunks to nothing, never retrievable), so both are rejected here.
        """
        has_text = self.text is not None and self.text.strip() != ""
        has_file = self.file is not None and self.file.strip() != ""
        if has_text == has_file:
            raise ValueError(
                "DocumentInput requires exactly one non-empty source: 'text' or 'file'"
            )
        return self


__all__ = [
    "KnowledgeBaseRecord",
    "KnowledgeDocument",
    "KnowledgeChunk",
    "RetrievedChunk",
    "DocumentInput",
]
