"""API kernel: Studio knowledge file upload + KB document listing.

Mounted under ``/api/studio/kb`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`):

- ``POST /api/studio/kb/{kb_id}/upload`` — multipart file upload (pdf/txt/md/csv,
  ≤15 MB). The file is written to a generated temp path (the client filename is
  never used to build a filesystem path, so it cannot traverse), text is extracted
  with the existing :class:`~himmy.services.knowledge.readers.DocumentReaderFactory`,
  and the result is ingested into the same Studio KB singleton the paste/search
  endpoints use — so uploads are immediately searchable via
  ``POST /api/studio/knowledge/{kb_id}/search``.
- ``GET /api/studio/kb/{kb_id}/documents`` — the documents currently in a KB
  (in-memory store only; a Postgres-backed KB degrades to an empty list).

Offline-first: when ``python-multipart`` is absent the upload route stays
importable and answers with a clear 400 instead of failing at import time
(which is why the form is parsed via ``request.form()`` rather than FastAPI's
``File(...)`` dependency — the latter raises during route definition).
"""

from __future__ import annotations

import asyncio
import importlib.util
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException

from himmy.api.routers.studio_common import build_studio_router, studio_permission
from himmy.core.errors import HimmyError

router = build_studio_router("kb", tag="studio-knowledge-upload")

#: Uploading a document into a knowledge base mutates retrieval state — a privileged
#: mutation gated by ``studio.kb:write`` (admin-only by default), additively on top of
#: the router's ``studio.kb:read`` baseline so a read-only role can browse but not
#: ingest documents.
_kb_write = Depends(studio_permission("studio.kb", "write"))

#: Hard cap on the uploaded file body (15 MB).
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
#: Allowance for multipart framing overhead when pre-checking Content-Length.
_BODY_OVERHEAD = 64 * 1024
_READ_CHUNK = 1024 * 1024
#: Same bound as ``POST /api/studio/knowledge/{kb_id}/ingest`` (KbIngestRequest).
_MAX_TEXT_CHARS = 1_000_000
_MAX_FILENAME = 255
_MAX_TITLE = 300
_MAX_KB_ID = 200

#: extension → content types accepted for it. A missing/blank content type is
#: allowed (some clients omit it); extension is always validated.
_ALLOWED_CONTENT_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset(
        {"application/pdf", "application/x-pdf", "application/octet-stream"}
    ),
    ".txt": frozenset({"text/plain", "application/octet-stream"}),
    ".md": frozenset(
        {
            "text/markdown",
            "text/x-markdown",
            "text/plain",
            "application/octet-stream",
        }
    ),
    ".csv": frozenset(
        {
            "text/csv",
            "application/csv",
            "application/vnd.ms-excel",
            "text/plain",
            "application/octet-stream",
        }
    ),
}


class UploadResult(BaseModel):
    document_id: str
    chunks: int
    characters: int
    title: str | None = None
    source_uri: str | None = None


class DocumentInfo(BaseModel):
    document_id: str
    title: str | None = None
    source_uri: str | None = None
    characters: int = 0
    chunks: int = 0


def _multipart_available() -> bool:
    """True when the ``python-multipart`` form parser is installed.

    Mirrors starlette's own lookup (``python_multipart`` with a legacy
    ``multipart`` fallback) without importing either.
    """
    return (
        importlib.util.find_spec("python_multipart") is not None
        or importlib.util.find_spec("multipart") is not None
    )


def _kb_service() -> Any:
    """The process-wide Studio KB singleton (shared with the paste/search routes)."""
    from himmy.api import studio_knowledge

    return studio_knowledge._service()


def _check_kb_id(kb_id: str) -> None:
    if not kb_id or len(kb_id) > _MAX_KB_ID:
        raise HTTPException(
            status_code=422, detail=f"kb_id must be 1–{_MAX_KB_ID} characters"
        )


async def _require_kb(
    svc: Any, kb_id: str, *, workspace_id: str, client_id: str
) -> None:
    """404 when ``kb_id`` is unknown OR outside the caller's tenant scope.

    The ``_authorize_kb`` guard verifies the resolved KB belongs to ``(workspace_id,
    client_id)`` so a tenant cannot reach another tenant's KB by raw id; both unknown and
    cross-tenant fold to the same 404 (existence never leaks).
    """
    from himmy.core.errors import HimmyError

    if await svc.get_kb(kb_id) is None:
        raise HTTPException(
            status_code=404, detail=f"knowledge base {kb_id!r} not found"
        )
    authorize = getattr(svc, "_authorize_kb", None)
    if authorize is not None:
        try:
            await authorize(kb_id, workspace_id, client_id, missing_ok=True)
        except HimmyError as exc:
            raise HTTPException(
                status_code=404, detail=f"knowledge base {kb_id!r} not found"
            ) from exc


async def _read_bounded(upload: UploadFile) -> bytes:
    """Read the upload, rejecting empty bodies and bodies over the size cap."""
    blobs: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
                "upload limit",
            )
        blobs.append(chunk)
    if total == 0:
        raise HTTPException(status_code=422, detail="uploaded file is empty")
    return b"".join(blobs)


def _extract_text(data: bytes, ext: str) -> str:
    """Write the bytes to a generated temp path and run the reader factory.

    The temp file name comes from ``tempfile`` plus the already-validated
    extension — the client-supplied filename never reaches the filesystem.
    """
    from himmy.services.knowledge.readers import DocumentReaderFactory

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return DocumentReaderFactory().read(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _chunk_count(svc: Any, text: str) -> int:
    """Chunk the text with the service's own chunker (fallback: the default)."""
    chunker = getattr(svc, "_chunker", None)
    if chunker is None:  # pragma: no cover - defensive against service changes
        from himmy.services.knowledge.chunker import SemanticChunker

        chunker = SemanticChunker()
    return len(chunker.chunk(text))


def _sync_doc_count(kb_id: str) -> None:
    """Keep the Studio KB list's per-KB document count in step after an upload.

    The in-memory store is authoritative when present (dedup/replace means an
    upload is not always +1); otherwise (Postgres backend) fall back to +1.
    """
    from himmy.api import studio_knowledge

    svc = _kb_service()
    docs_map = getattr(svc, "_documents", None)
    if isinstance(docs_map, dict) and kb_id in docs_map:
        studio_knowledge._DOCS[kb_id] = len(docs_map[kb_id])
    else:  # pragma: no cover - only with a configured KB backend
        studio_knowledge._DOCS[kb_id] = studio_knowledge._DOCS.get(kb_id, 0) + 1


def _validated_file_part(form: Any) -> tuple[UploadFile, str, str]:
    """Pull the ``file`` part out of the form → (upload, safe filename, ext)."""
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(
            status_code=422, detail="multipart field 'file' (a file part) is required"
        )
    # Basename only — a client filename like ../../x.txt carries no path here.
    filename = Path(upload.filename or "").name.strip()
    if not filename or len(filename) > _MAX_FILENAME:
        raise HTTPException(
            status_code=422,
            detail=f"uploaded file needs a filename of 1–{_MAX_FILENAME} characters",
        )
    ext = Path(filename).suffix.lower()
    allowed = _ALLOWED_CONTENT_TYPES.get(ext)
    if allowed is None:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {ext or '(no extension)'}; "
            "allowed: .pdf, .txt, .md, .csv",
        )
    ctype = (upload.content_type or "").split(";")[0].strip().lower()
    if ctype and ctype not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"content type {ctype!r} does not match a {ext} upload",
        )
    return upload, filename, ext


@router.post("/{kb_id}/upload", dependencies=[_kb_write])
async def kb_upload(kb_id: str, request: Request) -> UploadResult:
    """Upload one file into a KB: validate → extract text → chunk + embed."""
    _check_kb_id(kb_id)
    if not _multipart_available():
        raise HTTPException(
            status_code=400,
            detail="File upload requires the 'python-multipart' package "
            "(pip install 'himmy[studio]'). Paste text via "
            "POST /api/studio/knowledge/{kb_id}/ingest instead.",
        )
    content_length = request.headers.get("content-length", "")
    if content_length.isdigit() and int(content_length) > (
        MAX_UPLOAD_BYTES + _BODY_OVERHEAD
    ):
        raise HTTPException(
            status_code=413,
            detail=f"request body exceeds the "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit",
        )

    from himmy.api import studio_knowledge

    workspace_id, client_id = studio_knowledge.scope_keys(request)
    svc = _kb_service()
    await _require_kb(svc, kb_id, workspace_id=workspace_id, client_id=client_id)

    try:
        form = await request.form(max_files=2, max_fields=4)
    except MultiPartException as exc:
        raise HTTPException(
            status_code=400, detail=f"malformed multipart body: {exc}"
        ) from exc
    try:
        upload, filename, ext = _validated_file_part(form)
        raw_title = form.get("title")
        title = (
            raw_title.strip()[:_MAX_TITLE]
            if isinstance(raw_title, str) and raw_title.strip()
            else None
        )
        data = await _read_bounded(upload)
    finally:
        await form.close()

    try:
        text = await asyncio.to_thread(_extract_text, data, ext)
    except HimmyError as exc:
        # e.g. the [knowledge] extra (pypdf) is missing — actionable, not a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"{filename} is not valid UTF-8 text"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"could not extract text from {filename}: {exc}"
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=f"no extractable text in {filename} "
            "(a scanned PDF without a text layer extracts as empty)",
        )
    if len(text) > _MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"extracted text is {len(text):,} characters; the limit is "
            f"{_MAX_TEXT_CHARS:,}. Split the file and upload the parts.",
        )

    try:
        doc = await svc.ingest_text(
            kb_id,
            text,
            title=title or Path(filename).stem,
            source_uri=filename,
            metadata={"filename": filename},
        )
    except HimmyError as exc:
        status = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    _sync_doc_count(kb_id)
    return UploadResult(
        document_id=doc.document_id,
        chunks=_chunk_count(svc, text),
        characters=len(text),
        title=doc.title,
        source_uri=doc.source_uri,
    )


@router.get("/{kb_id}/documents")
async def kb_documents(kb_id: str, request: Request) -> list[DocumentInfo]:
    """List the documents in a KB (in-memory store; backend KBs return [])."""
    from himmy.api import studio_knowledge

    _check_kb_id(kb_id)
    workspace_id, client_id = studio_knowledge.scope_keys(request)
    svc = _kb_service()
    await _require_kb(svc, kb_id, workspace_id=workspace_id, client_id=client_id)
    docs_map = getattr(svc, "_documents", {}).get(kb_id, {})
    chunks_map = getattr(svc, "_chunks", {}).get(kb_id, {})
    counts: dict[str, int] = {}
    for chunk in chunks_map.values():
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
    return [
        DocumentInfo(
            document_id=d.document_id,
            title=d.title,
            source_uri=d.source_uri,
            characters=len(d.text or ""),
            chunks=counts.get(d.document_id, 0),
        )
        for d in docs_map.values()
    ]


__all__ = ["router", "UploadResult", "DocumentInfo", "MAX_UPLOAD_BYTES"]
