"""Tool-call repair: turn a model's near-miss into a recovered call or a clear fix.

Small local models frequently get tool calls *almost* right — a misspelled tool name,
a missing required argument. Rather than fail opaquely, this module:

* resolves a near-miss tool name to the real tool when there's a single confident
  match (so an ``egg_total`` → ``egg_totals`` typo recovers in zero extra turns), and
* composes a concrete, model-facing correction (did-you-mean + the available tools, or
  the exact arg schema) so the *next* turn self-corrects instead of guessing again.

Pure functions over plain data — reused by the inference layer (text-protocol
providers) and safe to call anywhere.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

# Similarity at/above which a single close name is auto-applied (a confident typo).
_AUTO_RESOLVE_CUTOFF = 0.82
# Lower bar for *suggesting* names in a correction message.
_SUGGEST_CUTOFF = 0.5
_MAX_SUGGESTIONS = 3


@dataclass
class NameResolution:
    """The outcome of resolving a model-supplied tool name against the real tools."""

    name: str | None  # the resolved real tool name, or None if unresolved
    auto_corrected: bool = False  # True when a near-miss was confidently remapped
    original: str = ""  # the name the model emitted
    suggestions: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.suggestions is None:
            self.suggestions = []


def resolve_tool_name(name: str, available: list[str]) -> NameResolution:
    """Resolve ``name`` to a real tool: exact, a confident typo-fix, or unresolved.

    * Exact match → resolved as-is.
    * Exactly one close match at/above the auto cutoff → resolved + ``auto_corrected``.
    * Otherwise → unresolved, with ranked ``suggestions`` for a correction message.
    """
    if name in available:
        return NameResolution(name=name, original=name)

    # Case-insensitive exact (models sometimes change case).
    lower = {a.lower(): a for a in available}
    if name.lower() in lower:
        real = lower[name.lower()]
        return NameResolution(name=real, auto_corrected=True, original=name)

    close = difflib.get_close_matches(
        name, available, n=_MAX_SUGGESTIONS, cutoff=_SUGGEST_CUTOFF
    )
    confident = [
        c
        for c in close
        if difflib.SequenceMatcher(None, name, c).ratio() >= _AUTO_RESOLVE_CUTOFF
    ]
    if len(confident) == 1:
        return NameResolution(
            name=confident[0], auto_corrected=True, original=name, suggestions=close
        )
    return NameResolution(name=None, original=name, suggestions=close)


def unknown_tool_message(resolution: NameResolution, available: list[str]) -> str:
    """A concrete correction the model can act on next turn for an unknown tool."""
    parts = [f"Unknown tool {resolution.original!r}."]
    if resolution.suggestions:
        if len(resolution.suggestions) == 1:
            parts.append(f"Did you mean {resolution.suggestions[0]!r}?")
        else:
            quoted = ", ".join(repr(s) for s in resolution.suggestions)
            parts.append(f"Did you mean one of: {quoted}?")
    listed = ", ".join(available) if available else "(none)"
    parts.append(f"Available tools: {listed}. Call one of these exactly by name.")
    return " ".join(parts)


__all__ = ["NameResolution", "resolve_tool_name", "unknown_tool_message"]
