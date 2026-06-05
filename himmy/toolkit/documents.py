"""Documents pack: ``read_document`` — extract text from a PDF / text / Markdown file.

Wraps the framework's existing
:class:`~himmy.services.knowledge.readers.DocumentReaderFactory` (``.pdf``/``.txt``/
``.md``), jailed to the toolkit's sandbox root so an agent can only read files the
operator has exposed. Pairs naturally with ``kb_ingest`` (ingest the extracted text).
"""

from __future__ import annotations

from typing import Any

from himmy.services.knowledge import DocumentReaderFactory
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.files import _safe_path

_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "A .pdf/.txt/.md/.csv/.xlsx file under the root.",
        },
        "max_chars": {"type": "integer", "minimum": 100, "default": 20000},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def register_documents_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register the ``read_document`` tool jailed to ``config.fs_root``."""
    factory = DocumentReaderFactory()

    def read_document(args: dict[str, Any]) -> dict[str, Any]:
        path = str(args["path"])
        max_chars = int(args.get("max_chars", 20000))
        target = _safe_path(config.fs_root, path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        text = factory.read(str(target))
        return {
            "path": path,
            "text": text[:max_chars],
            "chars": len(text),
            "truncated": len(text) > max_chars,
        }

    register_local_tool(
        registry,
        name="read_document",
        handler=read_document,
        description="Extract text from a PDF/text/Markdown/CSV/Excel file under the root.",
        args_json_schema=_READ_SCHEMA,
        metadata={"pack": "documents"},
    )


__all__ = ["register_documents_pack"]
