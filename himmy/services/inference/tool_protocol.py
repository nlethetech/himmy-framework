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


__all__ = ["parse_text_tool_calls"]
