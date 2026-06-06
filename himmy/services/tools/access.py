"""Tool access intent: does a tool *read* data or *change* it?

Small models reliably emit a well-formed call but often pick the wrong tool — a
*writer* when they meant a *reader* ("how many eggs?" → ``log_eggs`` instead of
``egg_totals``). The fix is to tell the model, in each tool's description, whether the
tool returns data or mutates state, so a look-up never lands on an action.

Intent is taken from an explicit ``read_only`` flag a tool author may set; otherwise it
is inferred from the tool's verb (``list_…``/``…_totals`` read, ``log_…``/``add_…``
write). The inference only ever *adds* a hint — an ambiguous name yields no tag rather
than a wrong one.
"""

from __future__ import annotations

# Verb tokens that strongly imply a read (returns data, no side effects).
_READ_VERBS = frozenset(
    {
        "get",
        "list",
        "read",
        "search",
        "fetch",
        "find",
        "show",
        "view",
        "status",
        "summary",
        "summarize",
        "count",
        "recall",
        "query",
        "totals",
        "total",
        "recent",
        "history",
        "overview",
        "describe",
        "inspect",
        "latest",
        "info",
        "lookup",
        "check",
        "report",
        "stats",
    }
)
# Verb tokens that strongly imply a write / action (changes state or acts on the world).
_WRITE_VERBS = frozenset(
    {
        "write",
        "log",
        "add",
        "record",
        "send",
        "create",
        "update",
        "delete",
        "remove",
        "set",
        "ingest",
        "remember",
        "complete",
        "save",
        "post",
        "put",
        "submit",
        "schedule",
        "cancel",
        "assign",
        "pay",
        "transfer",
        "execute",
        "run",
        "make",
        "register",
        "book",
        "order",
        "approve",
        "reject",
        "upsert",
    }
)


def classify_read_only(name: str) -> bool | None:
    """Infer whether a tool is read-only from its name; None when ambiguous.

    The first token (verb-first naming is the norm) decides; if it's unknown, a
    later token may still settle it, but only when the name doesn't mix read+write
    signals.
    """
    tokens = [t for t in name.lower().replace("-", "_").split("_") if t]
    if not tokens:
        return None
    first = tokens[0]
    if first in _WRITE_VERBS:
        return False
    if first in _READ_VERBS:
        return True
    has_read = any(t in _READ_VERBS for t in tokens)
    has_write = any(t in _WRITE_VERBS for t in tokens)
    if has_write and not has_read:
        return False
    if has_read and not has_write:
        return True
    return None


_READ_TAG = " — [read-only: returns data; safe to call for look-ups]"
_WRITE_TAG = (
    " — [WRITE: changes data; only call to record/modify something, "
    "never just to look something up]"
)


def describe_for_model(
    name: str, description: str, read_only: bool | None = None
) -> str:
    """Append a read/write intent hint to a tool's description for the model.

    ``read_only`` (an explicit author flag) wins; otherwise the intent is inferred
    from the name. Already-tagged descriptions are left unchanged.
    """
    description = description or ""
    if "[read-only:" in description or "[WRITE:" in description:
        return description
    intent = read_only if read_only is not None else classify_read_only(name)
    if intent is True:
        return description + _READ_TAG
    if intent is False:
        return description + _WRITE_TAG
    return description


__all__ = ["classify_read_only", "describe_for_model"]
