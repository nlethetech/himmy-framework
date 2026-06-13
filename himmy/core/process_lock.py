"""Cross-process advisory file locks (``flock``) for single-flight guards.

A small POSIX-``flock`` wrapper used where an in-process registry is not enough
because two SEPARATE processes can both try to do the same side-effecting thing.
The motivating case (T3c) is a routine: :class:`himmy.api.routines.RoutineScheduler`
tracks in-flight runs in a per-process ``_running`` dict, so the Studio server
process never double-executes one routine on its own. But a ``himmy routines
run-now`` invocation runs in a DIFFERENT process and cannot see that dict — so
without a cross-process guard the CLI and the Studio scheduler could fire the SAME
routine at the same instant, double-running its gated tools / deliveries.

:func:`process_lock` is that guard: a non-blocking exclusive ``flock`` on a lock
file under ``.himmy/locks/`` keyed by a stable name (e.g. ``routine-<id>``). The
first holder runs; a second concurrent attempt (in this process or any other on the
same machine) gets :class:`ProcessLockBusy` immediately rather than blocking or
silently proceeding. The OS releases the lock automatically if the holder crashes,
so there is no stale-lock cleanup to get wrong.

Scope + portability:

* The lock is **advisory** and **same-host** (``flock`` is local to one kernel) —
  exactly the boundary the routine guard needs (CLI + Studio on one machine).
* ``fcntl`` is POSIX-only. On a platform without it (Windows) the context manager
  degrades to a no-op that always "acquires": the in-process registry still
  prevents same-process overlap, and himmy's supported deployment targets are
  Linux/macOS, so the cross-process guard is a POSIX feature documented as such
  rather than faked on Windows.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

try:  # POSIX advisory locks; absent on Windows.
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False


class ProcessLockBusy(RuntimeError):
    """The lock is already held (by this or another process); the caller backs off.

    Carries the lock ``name`` so a caller can surface a precise message (e.g. map it
    to an HTTP 409) instead of a generic failure.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"resource {name!r} is locked by another runner")


def _locks_dir(base: str | Path | None) -> Path:
    """Resolve (and create) the directory the lock files live in.

    Defaults to ``<project-root>/.himmy/locks`` so a lock is co-located with the
    other durable per-project state and every interface launched from inside the
    same tree agrees on the path. An explicit ``base`` (a tmp dir in tests) wins.
    """
    if base is not None:
        root = Path(base)
    else:
        from himmy.config.project import himmy_dir

        root = himmy_dir()
    locks = root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    return locks


def _safe_name(name: str) -> str:
    """Make ``name`` safe to use as a single path segment (no separators/escapes)."""
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return cleaned or "lock"


class ProcessLockHandle:
    """A held cross-process lock you release explicitly (not scope-bound).

    :func:`process_lock` is the right tool when the guarded work is a single call you
    can wrap in a ``with`` block. A long-lived listener (T3g: the Studio Telegram
    long-poll loop) instead acquires the lock when it STARTS and releases it when it
    STOPS — a span that does not nest cleanly in one stack frame — so it holds a handle
    across its whole lifetime and calls :meth:`release` on shutdown. The OS still drops
    the lock automatically if the process dies, so a crashed listener never wedges the
    token. Idempotent: a second :meth:`release` is a no-op.

    On a non-POSIX host (no ``fcntl``) the handle is inert (``release`` is a no-op) — the
    in-process manager registry remains the guard there.
    """

    def __init__(self, name: str, fd: int | None) -> None:
        self.name = name
        self._fd = fd

    def release(self) -> None:
        """Release the lock and close the descriptor (idempotent)."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        if _HAVE_FCNTL:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)


def acquire_process_lock(
    name: str, *, base: str | Path | None = None
) -> ProcessLockHandle:
    """Acquire a non-blocking cross-process exclusive lock and return its handle.

    Like :func:`process_lock` but the lock is HELD until the caller calls
    :meth:`ProcessLockHandle.release` (or the process dies) rather than the end of a
    ``with`` block — the shape a long-running listener needs. Raises
    :class:`ProcessLockBusy` immediately if the lock is held anywhere on this host, so a
    second poller of the same bot token is refused (mapped to a 409) instead of starting
    a competing long-poll.

    On a non-POSIX host (no ``fcntl``) this always "acquires" an inert handle.
    """
    if not _HAVE_FCNTL:  # pragma: no cover - non-POSIX fallback
        return ProcessLockHandle(name, None)
    path = _locks_dir(base) / f"{_safe_name(name)}.lock"
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.close(fd)
        # EAGAIN/EACCES: the lock is held elsewhere. Refuse rather than block.
        raise ProcessLockBusy(name) from exc
    return ProcessLockHandle(name, fd)


@contextlib.contextmanager
def process_lock(
    name: str, *, base: str | Path | None = None
) -> Iterator[None]:
    """Hold a non-blocking cross-process exclusive lock named ``name``.

    Acquires an exclusive ``flock`` on ``<locks>/<name>.lock`` without blocking. On
    success the body runs and the lock is released (and the descriptor closed) on
    exit — including on exception, and automatically by the OS if the process dies.
    A lock already held anywhere on this host raises :class:`ProcessLockBusy`
    *before* the body runs, so the caller never does the guarded work twice.

    On a non-POSIX host (no ``fcntl``) this is a no-op acquire — the same-process
    registry remains the guard there (see the module docstring).
    """
    handle = acquire_process_lock(name, base=base)
    try:
        yield
    finally:
        handle.release()


__all__ = [
    "ProcessLockBusy",
    "ProcessLockHandle",
    "acquire_process_lock",
    "process_lock",
]
