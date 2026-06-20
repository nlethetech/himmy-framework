"""Tolerant parsing of tool calls from a text-only model reply.

Text-protocol providers (Claude CLI, and Ollama models that answer in prose instead
of using the native tool API) don't reliably emit one canonical format. This parser
accepts the common shapes small/instruction-tuned models actually produce:

* ``TOOL_CALL <name> {json}``                         (the documented protocol)
* a fenced ```json block`` holding ``{"tool": ..., "args": {...}}``
* a bare JSON-only reply: ``{"name": "x", "arguments": {...}}`` or a list of them

Tool name and argument keys are normalized across vendor conventions. For the
*ambiguous* shapes (fenced / bare JSON, which could just be a model's final answer
that happens to contain JSON), a call is only accepted when its name matches — or is a
close match to — a known bound tool, so a normal JSON answer is never mistaken for a
call. Explicit ``TOOL_CALL`` markers are always honored (the repair layer fixes a
typo'd name later). Identical calls are de-duplicated.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from typing import Any

from himmy.core.ids import new_uuid
from himmy.services.inference.models import ToolCallRecord

# Marker form: TOOL_CALL <name> ... (args decoded as balanced JSON after the name).
_MARKER_RE = re.compile(r"TOOL_CALL\s+([^\s{]+)")
# Fenced code blocks: ```lang\n ... \n``` (lang optional).
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)

# Vendor-spelling-tolerant key names.
_NAME_KEYS = ("tool", "name", "tool_name", "function", "action", "tool_call")
_ARG_KEYS = ("args", "arguments", "parameters", "params", "input", "action_input")
_CLOSE_CUTOFF = 0.82


def _normalize_call(obj: Any) -> tuple[str, dict[str, Any]] | None:
    """Pull a ``(name, args)`` pair out of a dict using tolerant key spellings."""
    if not isinstance(obj, dict):
        return None
    # Some vendors nest under "function": {"name": ..., "arguments": ...}.
    if "function" in obj and isinstance(obj["function"], dict):
        inner = _normalize_call(obj["function"])
        if inner is not None:
            return inner
    name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str)), None)
    if not name:
        return None
    args: Any = next((obj[k] for k in _ARG_KEYS if k in obj), {})
    if isinstance(args, str):  # arguments delivered as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return str(name), args


def _name_is_plausible(name: str, known: set[str]) -> bool:
    """True when an ambiguous-source name matches (or is close to) a known tool."""
    if not known:
        return True  # nothing to check against — accept (no false-positive risk basis)
    if name in known:
        return True
    return bool(difflib.get_close_matches(name, list(known), n=1, cutoff=_CLOSE_CUTOFF))


def _iter_json_objects(text: str) -> list[Any]:
    """Best-effort: every top-level JSON value embedded in ``text`` (objects/arrays)."""
    decoder = json.JSONDecoder()
    out: list[Any] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "{[":
            try:
                obj, end = decoder.raw_decode(text[i:])
                out.append(obj)
                i += end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return out


def parse_text_tool_calls(
    text: str, known_names: set[str] | None = None
) -> list[ToolCallRecord]:
    """Parse tool calls from a text reply across the supported formats.

    ``known_names`` (the bound tools) guards the ambiguous JSON formats against
    false positives; ``TOOL_CALL`` markers are always accepted. Returns de-duplicated
    :class:`ToolCallRecord`s in first-seen order.
    """
    known = known_names or set()
    calls: list[ToolCallRecord] = []
    seen: set[str] = set()

    def _add(name: str, args: dict[str, Any]) -> None:
        key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if key in seen:
            return
        seen.add(key)
        calls.append(ToolCallRecord(tool_call_id=new_uuid(), tool_name=name, args=args))

    # 1) Explicit TOOL_CALL markers (balanced JSON after the name). Always accepted.
    decoder = json.JSONDecoder()
    for m in _MARKER_RE.finditer(text):
        rest = text[m.end() :]
        brace = rest.find("{")
        if brace == -1:
            continue
        try:
            args, _ = decoder.raw_decode(rest[brace:])
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            _add(m.group(1), args)

    # 2) Fenced code blocks → JSON tool-call object(s) (name must be plausible).
    for fence in _FENCE_RE.findall(text):
        for obj in _iter_json_objects(fence):
            candidates = obj if isinstance(obj, list) else [obj]
            for c in candidates:
                norm = _normalize_call(c)
                if norm and _name_is_plausible(norm[0], known):
                    _add(*norm)

    # 3) Bare JSON-only reply (no markers/fences found yet) → tool-call object(s).
    if not calls:
        stripped = text.strip()
        if stripped[:1] in "{[":
            for obj in _iter_json_objects(stripped):
                candidates = obj if isinstance(obj, list) else [obj]
                for c in candidates:
                    norm = _normalize_call(c)
                    if norm and _name_is_plausible(norm[0], known):
                        _add(*norm)

    return calls


# --------------------------------------------------------------------------- #
# Python-call fallback — recover a tool call from PYTHON-call syntax a small model
# emits instead of <tool_call>. Small open-weight models (HimalayaGPT-0.5b) often
# answer with ``get_weather(city="Paris")`` — a bare call line, a ```python``` fenced
# block, or a ``<|python_start|>...<|python_end|>`` span (nanochat tool-token style).
# The GENERIC tolerant parser MISSES these (counts as 0 emit) even when the tool choice
# was right. This pass recovers them, but is GATED behind a format flag and only fires
# for a name that matches a bound tool (preserving _name_is_plausible), so it can only
# ADD hits — it never raises, never false-fires on prose, and never touches GENERIC.
# --------------------------------------------------------------------------- #

#: Strip the nanochat python tool-token span ``<|python_start|>...<|python_end|>`` so
#: the call expression inside it is parsed like any other bare/fenced call.
_PYTHON_TOKEN_RE = re.compile(
    r"<\|python_start\|>(.*?)<\|python_end\|>", re.DOTALL
)
#: A bare ``name(...)`` call line: a python identifier immediately followed by a
#: balanced-looking ``(...)``. Greedy on the args; ``ast`` validates the real shape.
_BARE_CALL_RE = re.compile(
    r"(?:^|\n)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?=\n|$)", re.DOTALL
)


def _call_from_expr(expr: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one python call expression ``name(arg=val, ...)`` into ``(name, args)``.

    Uses ``ast`` (never ``eval``): the expression must be a single ``ast.Call`` on a
    plain name; keyword arguments are read into a dict via :func:`ast.literal_eval`
    (so only literals are accepted — no code execution). Positional args are tolerated
    but ignored (they carry no key). Anything that is not a literal-keyword call
    returns ``None``.
    """
    expr = expr.strip()
    if not expr:
        return None
    try:
        node = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    call = node.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    name = call.func.id
    args: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs splat — no static key, skip
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError, TypeError):
            continue
    return name, args


def parse_python_tool_calls(
    text: str, known_names: set[str] | None = None
) -> list[ToolCallRecord]:
    """Recover tool calls from PYTHON-call syntax (fenced / bare / python-token).

    Three sources, in order: a ``<|python_start|>...<|python_end|>`` span, a
    ```` ```python ```` / ```` ```tool_call ```` fenced block, and a bare ``name(...)``
    line. Each candidate expression is parsed with :func:`_call_from_expr` and accepted
    ONLY when its name matches a known bound tool (:func:`_name_is_plausible`), so a
    prose sentence that merely mentions a tool name never false-fires. De-duplicated in
    first-seen order; never raises.
    """
    known = known_names or set()
    calls: list[ToolCallRecord] = []
    seen: set[str] = set()

    def _add(name: str, args: dict[str, Any]) -> None:
        if not _name_is_plausible(name, known):
            return
        key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if key in seen:
            return
        seen.add(key)
        calls.append(ToolCallRecord(tool_call_id=new_uuid(), tool_name=name, args=args))

    candidates: list[str] = []
    # 1) nanochat python-token spans — the whole span is one call expression.
    candidates.extend(m.strip() for m in _PYTHON_TOKEN_RE.findall(text))
    # 2) ```python``` / ```tool_call``` fenced blocks (any fence body may hold a call).
    candidates.extend(f.strip() for f in _FENCE_RE.findall(text))
    # 3) bare ``name(args)`` call lines anywhere in the reply.
    for m in _BARE_CALL_RE.finditer(text):
        candidates.append(f"{m.group(1)}({m.group(2)})")

    for cand in candidates:
        norm = _call_from_expr(cand)
        if norm is not None:
            _add(*norm)
    return calls


__all__ = ["parse_text_tool_calls", "parse_python_tool_calls"]
