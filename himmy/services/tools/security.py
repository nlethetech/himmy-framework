"""Tools kernel: HTTP security helpers (SSRF / path-traversal) and secret redaction.

The HTTP connector executes paths assembled from model-synthesized arguments, so
it is the most security-sensitive surface in the framework. These helpers:

* URL-encode path arguments and reject separators / control characters so a
  prompt-injected model cannot escape the configured base path (path traversal),
  inject a query string, or reach another host (SSRF);
* verify the resolved final URL still points at the configured base host;
* redact known-sensitive argument and header keys before they are emitted on
  events, so secrets never leak into the audit trail.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import quote, urlsplit

# Method allow-list for HTTP tools — anything else is rejected up-front.
ALLOWED_HTTP_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
)

# Characters in a path arg that would let it escape the configured path/host.
_FORBIDDEN_PATH_CHARS: frozenset[str] = frozenset({"/", "\\", "?", "#"})

# Substrings in an arg/header KEY that mark its value a secret to redact.
_SENSITIVE_KEY_HINTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "authorization",
    "auth",
    "credential",
    "private_key",
    "client_secret",
    "session",
    "cookie",
    "bearer",
)

#: The placeholder substituted for redacted secret values.
REDACTED = "***REDACTED***"


class ToolSecurityError(Exception):
    """Raised when a tool argument or URL fails a security check."""


def _has_control_chars(text: str) -> bool:
    """True when ``text`` contains ASCII control characters (incl. CR/LF)."""
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text)


def safe_path_arg(value: Any) -> str:
    """URL-encode a single path argument, rejecting traversal / injection.

    Rejects values containing path separators (``/`` ``\\``), query/fragment
    markers (``?`` ``#``), control characters, or a literal ``..`` segment.
    Returns the percent-encoded value (``safe=''`` so nothing is left raw).
    """
    text = str(value)
    if _has_control_chars(text):
        raise ToolSecurityError("path argument contains control characters")
    if any(ch in _FORBIDDEN_PATH_CHARS for ch in text):
        raise ToolSecurityError(
            "path argument contains a path/query separator (/, \\, ?, or #)"
        )
    if text == ".." or text == ".":
        raise ToolSecurityError("path argument is a relative-path segment")
    return quote(text, safe="")


def build_safe_path(path_template: str, path_args: dict[str, Any]) -> str:
    """Format ``path_template`` with URL-encoded, validated ``path_args``.

    Only keys actually referenced by ``{name}`` placeholders in the template are
    consumed; a missing referenced key raises ``KeyError`` (handled by the
    caller as ``INVALID_REQUEST``). Every substituted value is run through
    :func:`safe_path_arg`.
    """
    encoded = {k: safe_path_arg(v) for k, v in path_args.items()}
    return path_template.format(**encoded)


def assemble_url(base_url: str, path: str) -> str:
    """Join ``base_url`` and an already-encoded ``path`` and verify the host.

    The resolved URL's scheme+host (netloc) must equal the configured base URL's,
    so an encoded-but-malicious path cannot pivot to another origin.
    """
    base = base_url.rstrip("/")
    final = base + "/" + path.lstrip("/")
    base_parts = urlsplit(base_url)
    final_parts = urlsplit(final)
    if (
        final_parts.scheme != base_parts.scheme
        or final_parts.netloc != base_parts.netloc
    ):
        raise ToolSecurityError(
            "resolved URL host/scheme does not match the configured base URL"
        )
    if final_parts.scheme not in ("http", "https"):
        raise ToolSecurityError(
            f"unsupported URL scheme {final_parts.scheme!r} (expected http/https)"
        )
    return final


def is_sensitive_key(key: str, extra: Iterable[str] = ()) -> bool:
    """True when ``key`` names a secret (by hint substring or explicit ``extra``)."""
    lowered = key.lower()
    if any(e.lower() == lowered for e in extra):
        return True
    return any(hint in lowered for hint in _SENSITIVE_KEY_HINTS)


def _redact_value(value: Any, extra: tuple[str, ...], placeholder: str) -> Any:
    """Recurse into a non-sensitive value, redacting any nested sensitive keys.

    A dict is walked key-by-key (a sensitive key's value is fully replaced); a list
    or tuple has each element recursed into; any other value is returned unchanged.
    This is what makes :func:`redact_mapping` reach secrets nested at any depth — a
    credential inside a ``json_body`` arg never reaches the audit spine in plaintext.
    """
    if isinstance(value, dict):
        return {
            k: (
                placeholder
                if is_sensitive_key(str(k), extra)
                else _redact_value(v, extra, placeholder)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, extra, placeholder) for item in value]
    return value


def redact_mapping(
    data: dict[str, Any],
    *,
    extra_keys: Iterable[str] = (),
    placeholder: str = REDACTED,
) -> dict[str, Any]:
    """Return a deep copy of ``data`` with sensitive values replaced, recursively.

    Used before emitting argument/header dicts onto events so secrets never reach
    the audit trail. ``extra_keys`` names tool-specific sensitive arguments.
    Redaction RECURSES into nested dicts and lists at any depth, so a secret buried
    inside a ``json_body`` (or any nested structure) is masked too. ``placeholder``
    overrides the substituted marker (e.g. the approvals UI's ``••••``).
    """
    extra = tuple(extra_keys)
    return {
        k: (
            placeholder
            if is_sensitive_key(str(k), extra)
            else _redact_value(v, extra, placeholder)
        )
        for k, v in data.items()
    }
