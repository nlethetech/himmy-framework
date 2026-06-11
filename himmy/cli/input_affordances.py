"""REPL input affordances: ``@file`` expansion and ``!cmd`` / ``!!cmd`` shell escapes.

These are local conveniences the USER types at the prompt — never agent-initiated —
so they carry no approval gate. They are factored here as small, pure-ish functions so
they can be unit-tested in isolation from the REPL's event loop:

* :func:`expand_at_files` — replaces ``@path`` tokens that point at a real file with the
  file's contents in a fenced block (size-capped), warning on a missing path.
* :func:`classify_bang` / :func:`run_shell` — parse and execute a ``!cmd`` (run-and-show)
  or ``!!cmd`` (run-and-send-to-agent) line through the user's shell.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: Per-file cap when inlining an ``@file`` (~200 KB).
AT_FILE_CAP = 200 * 1024
#: Cap on captured ``!cmd`` output (~50 KB).
SHELL_OUTPUT_CAP = 50 * 1024
#: Wall-clock timeout for a ``!cmd`` shell escape.
SHELL_TIMEOUT_SECONDS = 30


def _looks_like_path(token: str) -> bool:
    """Whether an ``@``-stripped token is plausibly a path (not an email/handle).

    Only tokens containing a ``/`` or a ``.`` — or that name an existing file — are
    treated as file references, so ``@someone`` (a handle) and ``user@host`` fragments
    are left untouched. The caller has already split on whitespace.
    """
    return "/" in token or "." in token or Path(token).exists()


def expand_at_files(
    text: str,
    *,
    cap: int = AT_FILE_CAP,
    on_warn: Callable[[str], None] | None = None,
    cwd: Path | None = None,
) -> str:
    """Replace ``@path`` tokens that point at a real file with a fenced content block.

    Each whitespace-delimited ``@token`` that *looks like a path* and resolves to an
    existing file is replaced inline by ``\\n```<path>\\n<contents>\\n```\\n`` (contents
    capped at ``cap`` bytes with a visible ``… (truncated)`` marker). A token that looks
    like a path but does not exist triggers ``on_warn`` and is left verbatim. Tokens that
    don't look like paths (emails, ``@handle``) are never touched. ``cwd`` overrides the
    base directory for relative paths (defaults to the process cwd).
    """
    base = cwd or Path.cwd()
    out_words: list[str] = []
    for word in text.split(" "):
        if not word.startswith("@") or len(word) < 2:
            out_words.append(word)
            continue
        token = word[1:]
        if not _looks_like_path(token):
            out_words.append(word)
            continue
        path = Path(token)
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            if on_warn is not None:
                on_warn(f"@{token}: no such file (left as-is)")
            out_words.append(word)
            continue
        out_words.append(_render_file_block(token, path, cap))
    return " ".join(out_words)


def _render_file_block(label: str, path: Path, cap: int) -> str:
    """Read ``path`` (size-capped) and render it as a fenced, labeled block."""
    try:
        raw = path.read_bytes()
    except OSError as exc:  # unreadable despite is_file — degrade to a note
        return f"\n```{label}\n(could not read: {exc})\n```\n"
    truncated = len(raw) > cap
    body = raw[:cap].decode("utf-8", errors="replace")
    marker = "\n… (truncated)" if truncated else ""
    return f"\n```{label}\n{body}{marker}\n```\n"


@dataclass(frozen=True)
class BangCommand:
    """A parsed ``!``/``!!`` shell-escape line."""

    command: str
    #: True for ``!!`` (run, then send the framed output to the agent); False for ``!``.
    send_to_agent: bool


def classify_bang(line: str) -> BangCommand | None:
    """Parse a ``!cmd`` / ``!!cmd`` line, or return ``None`` if it isn't one.

    ``!!cmd`` runs and then feeds the framed command+output to the agent;
    ``!cmd`` runs and only shows the output locally. A bare ``!`` / ``!!`` with no
    command is not a shell escape (returns ``None``).
    """
    if not line.startswith("!"):
        return None
    if line.startswith("!!"):
        command = line[2:].strip()
        return BangCommand(command=command, send_to_agent=True) if command else None
    command = line[1:].strip()
    return BangCommand(command=command, send_to_agent=False) if command else None


@dataclass(frozen=True)
class ShellResult:
    """The captured outcome of a ``!cmd`` shell escape."""

    command: str
    output: str
    exit_code: int
    timed_out: bool


def run_shell(
    command: str,
    *,
    timeout: int = SHELL_TIMEOUT_SECONDS,
    cap: int = SHELL_OUTPUT_CAP,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> ShellResult:
    """Run ``command`` through ``/bin/sh -c`` and capture its combined output, capped.

    ``runner`` is injectable for tests (defaults to :func:`subprocess.run`). stdout and
    stderr are merged in order, capped at ``cap`` bytes with a visible truncation marker.
    A timeout yields ``timed_out=True`` and exit code ``124`` (the conventional code);
    any captured partial output is preserved.
    """
    run = runner or subprocess.run
    try:
        completed = run(
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        partial = _merge_streams(_as_text(exc.stdout), _as_text(exc.stderr), cap=cap)
        note = f"(timed out after {timeout}s)"
        output = f"{partial}\n{note}" if partial else note
        return ShellResult(
            command=command, output=output, exit_code=124, timed_out=True
        )
    output = _merge_streams(completed.stdout, completed.stderr, cap=cap)
    return ShellResult(
        command=command,
        output=output,
        exit_code=completed.returncode,
        timed_out=False,
    )


def _as_text(value: str | bytes | None) -> str:
    """Coerce a possibly-bytes captured stream to text (TimeoutExpired may hand bytes)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _merge_streams(stdout: str, stderr: str, *, cap: int) -> str:
    """Merge stdout + stderr into one capped block with a truncation marker."""
    merged = (stdout or "") + (stderr or "")
    if len(merged) > cap:
        return merged[:cap] + "\n… (truncated)"
    return merged


def frame_for_agent(result: ShellResult) -> str:
    """Frame a ``!!cmd`` result as the message sent to the agent (clear provenance)."""
    status = "" if result.exit_code == 0 else f" (exit {result.exit_code})"
    return f"I ran: `{result.command}`{status}\noutput:\n```\n{result.output}\n```"


__all__ = [
    "AT_FILE_CAP",
    "SHELL_OUTPUT_CAP",
    "SHELL_TIMEOUT_SECONDS",
    "BangCommand",
    "ShellResult",
    "classify_bang",
    "expand_at_files",
    "frame_for_agent",
    "run_shell",
]
