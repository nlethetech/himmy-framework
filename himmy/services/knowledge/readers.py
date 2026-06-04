"""Knowledge kernel: document readers + the extension-routing reader factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from himmy.core.errors import HimmyError


class DocumentReader(ABC):
    """Reads a document at a path into plain text. Routed by file extension."""

    #: File extensions this reader handles (lowercase, with leading dot).
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def read(self, path: str) -> str:
        """Read the document at ``path`` and return its text."""
        raise NotImplementedError


class TextReader(DocumentReader):
    """Reads plain-text and Markdown documents (``.txt`` / ``.md``)."""

    extensions = (".txt", ".md")

    def read(self, path: str) -> str:
        """Read a UTF-8 text/markdown file."""
        return Path(path).read_text(encoding="utf-8")


class PDFReader(DocumentReader):
    """Reads PDF documents via ``pypdf`` (lazy import; clear error if missing)."""

    extensions = (".pdf",)

    def read(self, path: str) -> str:
        """Extract text from a PDF, page by page (requires the [knowledge] extra)."""
        pypdf = self._require_pypdf()
        reader = pypdf.PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    @staticmethod
    def _require_pypdf() -> Any:
        """Import pypdf lazily, raising a clear error when the extra is missing."""
        try:
            import pypdf  # type: ignore
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise HimmyError(
                "PDFReader requires the [knowledge] extra "
                "(pip install 'himmy[knowledge]') for pypdf."
            ) from exc
        return pypdf


class DocumentReaderFactory:
    """Routes a path to the right :class:`DocumentReader` by extension."""

    def __init__(self) -> None:
        """Register the built-in text and PDF readers."""
        self._by_ext: dict[str, DocumentReader] = {}
        self.register(TextReader())
        self.register(PDFReader())

    def register(self, reader: DocumentReader) -> None:
        """Register a reader for each of its declared extensions."""
        for ext in reader.extensions:
            self._by_ext[ext.lower()] = reader

    def reader_for(self, path: str) -> DocumentReader | None:
        """Return the reader registered for the path's extension, or None."""
        return self._by_ext.get(Path(path).suffix.lower())

    def read(self, path: str) -> str:
        """Read a document, choosing the reader by its extension."""
        reader = self.reader_for(path)
        if reader is None:
            raise HimmyError(
                f"No registered reader for extension {Path(path).suffix!r}"
            )
        return reader.read(path)


__all__ = [
    "DocumentReader",
    "TextReader",
    "PDFReader",
    "DocumentReaderFactory",
]
