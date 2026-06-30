"""API kernel: Studio files router — the agent file-sandbox ledger.

Mounted under ``/api/studio/files`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). Two read-only surfaces over the
toolkit's filesystem sandbox (the ``fs_root`` that ``write_file`` lands in):

* ``GET  /api/studio/files``           — list sandbox files (name, path, size, mtime)
* ``GET  /api/studio/files/download``  — stream one file as an attachment

Both share the toolkit's jail discipline (mirroring
:func:`himmy.toolkit.files._safe_path` and the ``resolve_spec_path`` canon):
absolute paths and ``..`` traversal are rejected, every component below the
root is refused if it is a symlink, hidden (dot-prefixed) entries are never
listed nor served — the sandbox root is the launch cwd by default, so dotfiles
like ``.env`` / ``.himmy/`` must stay invisible — and the download re-opens the
target one path segment at a time with ``O_NOFOLLOW`` on *every* component
(``openat``-style, anchored to the resolved root ``fd``) so a directory swapped
for a symlink between check and use still fails closed.
"""

from __future__ import annotations

import errno
import mimetypes
import os
import stat
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from himmy.api.routers.studio_common import build_studio_router

router = build_studio_router("files", tag="studio-files")


def _require_unscoped_sandbox_reader(request: Request) -> None:
    """Refuse the shared agent file-sandbox to a tenant-/subject-scoped principal (BOLA).

    The agent file sandbox (``ToolkitConfig.fs_root``) is a SINGLE process-global directory
    with NO tenant dimension — unlike the memory/KB/tasks/notes stores, it was never
    per-tenant namespaced. On a multi-tenant shared-process deployment that makes
    ``GET /api/studio/files`` and ``/download`` a cross-tenant BOLA: tenant A's agent
    ``write_file`` outputs land in the same root that tenant B (any ``files:read`` holder)
    could enumerate and download.

    Since a shared sandbox is the intended single-box design, this is closed by restricting
    the two file routes to an UNSCOPED operator/offline principal — exactly the
    ``reveal_host_posture`` cross-workspace pattern. ``all_tenants`` (offline default /
    ANONYMOUS / trusted shared key) is a NO-OP, so the zero-config single-box path is
    byte-unchanged; a tenant-bound or ``subject_scoped`` principal gets a uniform 404 (the
    sandbox is operator-cross-workspace, not tenant-keyed data it may browse).
    """
    from himmy.api.auth.context import get_principal

    principal = get_principal(request)
    # ``all_tenants`` is the unrestricted offline/ANONYMOUS/trusted-shared-key boundary
    # (mirrors ``studio_tenant_filter`` returning None) — a NO-OP, byte-unchanged. Any other
    # principal is tenant-/subject-bound and may not browse the shared, un-namespaced sandbox.
    if principal.all_tenants:
        return
    raise HTTPException(status_code=404, detail="not found")

#: Directories never listed (mirrors the agent-spec scanner's skip set).
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".himmy"}
#: Hard ceiling on directory entries examined per listing (bounds pathological trees).
_MAX_SCAN = 20_000
#: Largest file the download endpoint will serve (it is a GUI affordance, not rsync).
_MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
_CHUNK = 64 * 1024
#: ``O_NOFOLLOW`` where the platform has it (0 elsewhere — the explicit symlink
#: rejection in :func:`_validate_rel_path` still guards those platforms).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
#: ``O_DIRECTORY`` where the platform has it (0 elsewhere); applied to every
#: intermediate segment of the openat-style download traversal so an intermediate
#: component swapped for a symlink-to-a-file is also refused.
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class SandboxFile(BaseModel):
    """One file under the sandbox root, as shown in the Files ledger."""

    name: str
    path: str  # relative to the sandbox root (the GUI's stable id)
    size: int
    mtime: str  # ISO-8601 UTC


class FileListResponse(BaseModel):
    root: str
    items: list[SandboxFile]
    total: int
    truncated: bool


def _sandbox_root() -> Path:
    """The toolkit's ``fs_root`` exactly as a Studio run resolves it.

    Same precedence as :func:`himmy.runtime.from_spec.build_runtime_for_spec`
    (env > ``himmy.toml`` ``[toolkit]`` > cwd default), read fresh per request so
    the ledger always describes the sandbox the *next* run would use.
    """
    from himmy.config.project import load_project_config
    from himmy.toolkit.config import ToolkitConfig

    cfg = ToolkitConfig.from_sources(load_project_config().get("toolkit"))
    return cfg.fs_root


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _validate_rel_path(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root`` with the full sandbox jail discipline.

    Raises :class:`HTTPException` (400) on absolute paths, ``..`` traversal,
    hidden components, or any symlinked component below the root — the same
    two-layer check as ``toolkit.files._safe_path`` / ``resolve_spec_path``
    (resolved-path containment + per-component unresolved symlink rejection).
    """

    def _bad(reason: str) -> HTTPException:
        # The client-supplied path is echoed nowhere else; keep the reason terse.
        return HTTPException(status_code=400, detail=f"invalid path: {reason}")

    if not rel or rel.strip() != rel:
        raise _bad("blank or padded")
    if "\x00" in rel:
        raise _bad("contains NUL")
    if Path(rel).is_absolute() or PurePosixPath(rel).is_absolute() or rel[:1] == "~":
        raise _bad("absolute paths are not allowed")
    parts = PurePosixPath(rel).parts
    if any(p == ".." for p in parts) or ".." in Path(rel).parts:
        raise _bad("'..' traversal is not allowed")
    if any(_is_hidden(p) for p in parts if p != "."):
        raise _bad("hidden files are not served")

    root_resolved = root.resolve()
    candidate = (root_resolved / rel).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise _bad("escapes the sandbox root")
    probe = root_resolved
    for part in PurePosixPath(rel).parts:
        if part == ".":
            continue
        probe = probe / part
        if probe.is_symlink():
            raise _bad("symlinks are not allowed inside the sandbox")
    return candidate


def _open_nofollow_segments(root_resolved: Path, rel: str) -> int:
    """Open ``rel`` under ``root_resolved`` one segment at a time, ``O_NOFOLLOW``-jailed.

    ``O_NOFOLLOW`` on a single ``os.open`` only guards the *final* component, so a
    validated path whose intermediate directory is swapped for a symlink (TOCTOU)
    would still open through it. This walks each segment anchored to the previous
    segment's directory ``fd`` (``openat`` semantics) with ``O_NOFOLLOW`` on all of
    them — intermediates additionally require ``O_DIRECTORY`` — so a symlinked
    component anywhere along the path is rejected atomically at open time.

    Returns an open read-only ``fd`` for the final file (caller owns closing it).
    Raises :class:`OSError` (``ELOOP`` on a symlinked component, ``ENOENT`` if a
    component vanished, ``ENOTDIR`` if an intermediate is not a directory).
    """
    parts = [p for p in PurePosixPath(rel).parts if p != "."]
    dir_fd = os.open(root_resolved, os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY,
                dir_fd=dir_fd,
            )
            os.close(dir_fd)
            dir_fd = next_fd
        return os.open(parts[-1], os.O_RDONLY | _O_NOFOLLOW, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


@router.get(
    "",
    response_model=FileListResponse,
    dependencies=[Depends(_require_unscoped_sandbox_reader)],
)
async def list_files(
    limit: int = Query(200, ge=1, le=1000),
) -> FileListResponse:
    """List regular files under the agent file-sandbox root, newest first.

    Symlinks and hidden (dot-prefixed) entries are skipped outright; the walk is
    capped at ``_MAX_SCAN`` directory entries so a pathological tree cannot stall
    the GUI. A missing root is an empty ledger, not an error.
    """
    root = _sandbox_root()
    try:
        root_resolved = root.resolve()
    except OSError as exc:
        raise HTTPException(
            status_code=409, detail=f"sandbox root is not resolvable: {exc}"
        ) from exc
    if not root_resolved.is_dir():
        return FileListResponse(
            root=str(root_resolved), items=[], total=0, truncated=False
        )

    found: list[tuple[float, SandboxFile]] = []
    scanned = 0
    scan_capped = False
    for dirpath, dirnames, filenames in os.walk(root_resolved, followlinks=False):
        # Prune in place: hidden dirs, the skip set, and symlinked dirs never recurse.
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if not _is_hidden(d)
            and d not in _SKIP_DIRS
            and not Path(dirpath, d).is_symlink()
        ]
        for fname in filenames:
            scanned += 1
            if scanned > _MAX_SCAN:
                scan_capped = True
                break
            if _is_hidden(fname):
                continue
            fpath = Path(dirpath) / fname
            try:
                st = os.lstat(fpath)
            except OSError:
                continue  # raced away mid-walk
            if not stat.S_ISREG(st.st_mode):
                continue  # symlinks, fifos, sockets, devices
            rel = fpath.relative_to(root_resolved).as_posix()
            found.append(
                (
                    st.st_mtime,
                    SandboxFile(
                        name=fname,
                        path=rel,
                        size=st.st_size,
                        mtime=datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(
                            timespec="seconds"
                        ),
                    ),
                )
            )
        if scan_capped:
            break

    found.sort(key=lambda t: (-t[0], t[1].path))
    items = [f for _, f in found[:limit]]
    return FileListResponse(
        root=str(root_resolved),
        items=items,
        total=len(found),
        truncated=scan_capped or len(found) > limit,
    )


@router.get("/download", dependencies=[Depends(_require_unscoped_sandbox_reader)])
async def download_file(
    path: str = Query(..., min_length=1, max_length=1024),
) -> StreamingResponse:
    """Stream one sandbox file as an attachment, with the full jail discipline.

    Rejects absolute paths, ``..`` traversal, hidden components, and symlinks
    (400); missing files and directories are 404; oversized files are 413. The
    open walks the path one segment at a time with ``O_NOFOLLOW`` on *every*
    component (anchored to the resolved root ``fd``), so a directory swapped for a
    symlink between check and use — including an intermediate one — fails closed.
    """
    root = _sandbox_root()
    target = _validate_rel_path(root, path)
    try:
        root_resolved = root.resolve()
    except OSError as exc:
        raise HTTPException(
            status_code=409, detail="sandbox root is not resolvable"
        ) from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail="file not found in the sandbox")
    if target.is_dir():
        raise HTTPException(status_code=404, detail="path is a directory, not a file")

    try:
        fd = _open_nofollow_segments(root_resolved, path)
    except OSError as exc:
        # O_NOFOLLOW on a symlinked component raises ELOOP (link→file) or, on
        # some platforms (macOS), ENOTDIR when an intermediate link→dir is
        # refused; an intermediate that is a plain non-dir file is ENOTDIR too.
        # All are invalid-path rejections, not "file missing".
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise HTTPException(
                status_code=400, detail="invalid path: symlinks are not allowed"
            ) from exc
        if exc.errno == errno.ENOENT:
            raise HTTPException(
                status_code=404, detail="file not found in the sandbox"
            ) from exc
        raise HTTPException(status_code=409, detail="file could not be opened") from exc

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise HTTPException(status_code=404, detail="not a regular file")
        if st.st_size > _MAX_DOWNLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="file exceeds the 200 MB download limit",
            )
    except HTTPException:
        os.close(fd)
        raise

    handle = os.fdopen(fd, "rb")

    def _iter() -> Iterator[bytes]:
        with handle:
            while chunk := handle.read(_CHUNK):
                yield chunk

    # ASCII-safe fallback + RFC 5987 UTF-8 name; quotes/control chars stripped.
    ascii_name = "".join(
        c for c in target.name if c.isascii() and c.isprintable() and c not in '"\\'
    )
    disposition = (
        f'attachment; filename="{ascii_name or "download"}"; '
        f"filename*=UTF-8''{quote(target.name)}"
    )
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return StreamingResponse(
        _iter(),
        media_type=media_type,
        headers={
            "Content-Disposition": disposition,
            "Content-Length": str(st.st_size),
        },
    )


__all__ = ["router"]
