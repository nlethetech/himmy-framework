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
        # Action verbs that pair with an unambiguous read verb in composite names
        # (``get_and_charge``, ``list_and_publish``, ``describe_and_deploy``, …). Without
        # these the deny-list was fail-OPEN: such a side-effecting tool was wrongly judged
        # parallel-safe AND retried on timeout (double-fire). They must barrier.
        "sync",
        "notify",
        "email",
        "publish",
        "dispatch",
        "trigger",
        "apply",
        "charge",
        "refund",
        "deploy",
        "sign",
        "commit",
        "push",
        "enqueue",
        "emit",
        "reset",
        "purge",
        "clear",
        "flush",
        "drop",
        "mint",
        "revoke",
        "invite",
        "share",
        "fire",
        "increment",
        "decrement",
        "ping",
        "track",
        "activate",
        "deactivate",
        "enable",
        "disable",
        "start",
        "stop",
        "restart",
        "kill",
        "terminate",
        "provision",
        "rollback",
        "migrate",
        "release",
        "grant",
        "unshare",
        "archive",
        "restore",
        # Physical/actuation + control-flow + generation verbs that mutate the world or
        # emit side effects. Without these a composite carrying an unambiguous read verb
        # (``file_get_move``, ``get_and_actuate``, ``read_and_toggle``) was fail-OPEN:
        # judged parallel-safe and hoisted into a concurrent read-batch ahead of a
        # dependent read. They must barrier like any other write verb.
        "actuate",
        "toggle",
        "move",
        "copy",
        "open",
        "close",
        "call",
        "invoke",
        "render",
        "generate",
        "build",
        "forward",
        "place",
        "wire",
        "process",
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


# DUAL-USE read verbs: they read in some names but head an ACTION in others
# (``report_incident``/``report_bug`` file something; ``check_out`` mutates; ``search_replace``
# writes; ``count_and_reset`` resets). They must NOT, on their own, qualify a multi-token
# name as safe to run concurrently — only a name that is ENTIRELY read verbs does.
_AMBIGUOUS_READ_VERBS = frozenset(
    {
        "report",
        "check",
        "status",
        "count",
        "view",
        "history",
        "search",
        "find",
        "fetch",
    }
)
#: Read verbs whose presence as the FIRST token reliably implies no side effect.
_UNAMBIGUOUS_READ_VERBS = _READ_VERBS - _AMBIGUOUS_READ_VERBS


def classify_parallel_safe(name: str) -> bool:
    """Whether a NAME is safe to run CONCURRENTLY with siblings (strict, fail-closed).

    This is deliberately STRICTER than :func:`classify_read_only`. The latter is
    first-token-wins (a model-facing *hint* only) and so infers a name like
    ``fetch_and_delete``/``get_or_create``/``report_incident`` as read-only even though it
    may mutate state — harmless for a description tag, but UNSAFE to gate parallelism on,
    because such a call must never be hoisted into a concurrent read-batch ahead of a
    dependent read.

    A name is parallel-safe ONLY when BOTH hold:

    * NO write verb appears anywhere in its tokens (``fetch_and_delete``, ``get_or_create``,
      ``status_update`` carry ``delete``/``create``/``update`` → barrier); and
    * it is unambiguously a reader — either the FIRST token is an *unambiguous* read verb
      (``get``/``list``/``read``/``describe``/…), or EVERY token is a read verb. A DUAL-USE
      read verb (``report``/``check``/``search``/``find``/``fetch``/``count``/``view``/
      ``status``/``history``) heading a name with a non-read tail (``report_incident``,
      ``check_out``, ``count_and_reset``, ``view_and_ack``, ``history_purge``,
      ``search_replace``) is NOT parallel-safe — it may be an action.

    Everything else (mixed read+write, ambiguous, dual-use+action) stays sequential.
    Authors flag genuine look-ups explicitly with ``read_only=True`` to opt such tools back
    into concurrency.
    """
    tokens = [t for t in name.lower().replace("-", "_").split("_") if t]
    if not tokens:
        return False
    if any(t in _WRITE_VERBS for t in tokens):
        return False
    # A DUAL-USE read verb heading a name whose tail isn't itself all-read may be an
    # ACTION (``report_incident``, ``check_out``, ``search_replace``) → not parallel-safe.
    if tokens[0] in _AMBIGUOUS_READ_VERBS and not all(t in _READ_VERBS for t in tokens):
        return False
    # DEFENCE IN DEPTH against a mutating verb the deny-list has not learned yet. A NAME
    # classifier is not an authorization boundary, so ``read_only=None`` tools must not be
    # auto-parallelised on a fail-OPEN heuristic. When the name is not entirely read verbs
    # (i.e. it carries at least one NON-read "tail" token — a noun like ``egg_totals`` or a
    # verb we may not recognise), require an AUTHORITATIVE reader shape: the FIRST token
    # must be an unambiguous read verb (``get_valve_state``, ``list_open_orders``) OR the
    # LAST token — the one that carries the action in composite names — must itself be a
    # read verb (``egg_totals``, ``ticket_status``). This bars ``valve_get_open``,
    # ``file_get_move``, ``get_and_actuate`` (unknown mutating tail) while keeping genuine
    # readers safe. A truly all-read name skips this and falls through to the check below.
    all_read = all(t in _READ_VERBS for t in tokens)
    if (
        not all_read
        and tokens[0] not in _UNAMBIGUOUS_READ_VERBS
        and tokens[-1] not in _READ_VERBS
    ):
        return False
    # Otherwise: an unambiguous read verb ANYWHERE (noun-first ``egg_totals`` too), or a
    # name that is entirely read verbs, is safe.
    if any(t in _UNAMBIGUOUS_READ_VERBS for t in tokens):
        return True
    return all_read


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


__all__ = ["classify_parallel_safe", "classify_read_only", "describe_for_model"]
