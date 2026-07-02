"""Gemma-4 tool-call grammar — render + parse, ported faithfully from the model's own
``chat_template.jinja`` (himalaya-ai/himalaya-gemma-4-e2b-it).

This lets himmy's format registry advertise tools, parse calls, and thread tool results
in the EXACT grammar the model was trained on, so the himmy "format-registry (prompting)"
arm is a fair test of himmy's thesis on a model whose native tool format is Gemma's —
not Hermes. The three public callables match himmy's ``ToolCallFormat`` contract:

    render_system_manifest(tools, provider) -> str   # <|tool>declaration:...<tool|>
    parse(text, known) -> list[ToolCallRecord]        # <|tool_call>call:NAME{...}<tool_call|>
    render_tool_results(results) -> str               # <|tool_response>response:NAME{...}<tool_response|>

Plus ``render_assistant_calls`` so multi-turn history is re-rendered in-grammar.

The manifest renderer is a 1:1 port of the template's ``format_function_declaration`` /
``format_parameters`` / ``format_argument`` macros and is byte-checked against the real
tokenizer template in tests/test_gemma4_grammar.py.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from typing import Any

from himmy.services.inference.models import (
    BoundTool,
    ToolCallRecord,
    ToolReturnRecord,
)

STR = '<|"|>'  # the 5-char string delimiter the grammar wraps string literals in
_STANDARD_KEYS = ("description", "type", "properties", "required", "nullable")


# --------------------------------------------------------------------------- #
# RENDER: tool declarations (mirrors the jinja macros exactly)
# --------------------------------------------------------------------------- #
def _fmt_arg(value: Any, escape_keys: bool = True) -> str:
    """Port of the template ``format_argument`` macro."""
    if isinstance(value, str):
        return f"{STR}{value}{STR}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        parts = []
        for k in sorted(value):  # jinja ``| dictsort``
            key = f"{STR}{k}{STR}" if escape_keys else k
            parts.append(f"{key}:{_fmt_arg(value[k], escape_keys)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt_arg(v, escape_keys) for v in value) + "]"
    return str(value)


def _fmt_parameters(properties: dict[str, Any], required: list[str], filter_keys: bool = False) -> str:
    """Port of ``format_parameters``: render a properties mapping in-grammar."""
    out: list[str] = []
    for key in sorted(properties):  # dictsort
        if filter_keys and key in _STANDARD_KEYS:
            continue
        value = properties[key]
        if not isinstance(value, dict):
            # Defensive: a non-object property still needs a type tag.
            out.append(f"{key}:{{type:{STR}{str(value).upper()}{STR}}}")
            continue
        body: list[str] = []
        vtype = str(value.get("type", "")).upper()
        if value.get("description"):
            body.append(f"description:{STR}{value['description']}{STR}")
        if vtype == "STRING" and value.get("enum"):
            body.append(f"enum:{_fmt_arg(value['enum'])}")
        elif vtype == "ARRAY" and isinstance(value.get("items"), dict) and value["items"]:
            items_parts: list[str] = []
            for ik in sorted(value["items"]):
                iv = value["items"][ik]
                if iv is None:
                    continue
                if ik == "properties":
                    inner = _fmt_parameters(iv, value["items"].get("required", [])) if isinstance(iv, dict) else ""
                    items_parts.append(f"properties:{{{inner}}}")
                elif ik == "required":
                    reqs = ",".join(f"{STR}{r}{STR}" for r in iv)
                    items_parts.append(f"required:[{reqs}]")
                elif ik == "type":
                    if isinstance(iv, str):
                        items_parts.append(f"type:{_fmt_arg(iv.upper())}")
                    else:
                        items_parts.append(f"type:{_fmt_arg([x.upper() for x in iv])}")
                else:
                    items_parts.append(f"{ik}:{_fmt_arg(iv)}")
            body.append("items:{" + ",".join(items_parts) + "}")
        if value.get("nullable"):
            body.append("nullable:true")
        if vtype == "OBJECT":
            if isinstance(value.get("properties"), dict):
                body.append(f"properties:{{{_fmt_parameters(value['properties'], value.get('required', []))}}}")
            if value.get("required"):
                reqs = ",".join(f"{STR}{r}{STR}" for r in value["required"])
                body.append(f"required:[{reqs}]")
        body.append(f"type:{STR}{vtype}{STR}")
        out.append(f"{key}:{{" + ",".join(body) + "}")
    return ",".join(out)


def _declaration(tool: BoundTool) -> str:
    """Port of ``format_function_declaration`` for one tool."""
    params = tool.args_json_schema or {}
    parts = [f"declaration:{tool.name}{{description:{STR}{tool.description or ''}{STR}"]
    if params:
        seg = ",parameters:{"
        inner: list[str] = []
        if params.get("properties"):
            inner.append("properties:{" + _fmt_parameters(params["properties"], params.get("required", [])) + "}")
        if params.get("required"):
            reqs = ",".join(f"{STR}{r}{STR}" for r in params["required"])
            inner.append(f"required:[{reqs}]")
        ptype = str(params.get("type", "")).upper()
        if ptype:
            inner.append(f"type:{STR}{ptype}{STR}")
        # Close the parameters object exactly ONCE, decoupled from the (optional) type
        # field. A schema with `properties` but no top-level `type` (valid JSON Schema —
        # `type` is optional) previously left the object unbalanced because the closing
        # brace was only emitted as part of the type element.
        seg += ",".join(inner) + "}"
        parts.append(seg)
    parts.append("}")
    return "".join(parts)


_MANIFEST_INSTRUCTION = (
    "\n\nWhen a function is needed, emit it as "
    "<|tool_call>call:NAME{arg:value,...}<tool_call|> using the grammar above "
    "(string values wrapped in the string delimiter). Emit several back-to-back "
    "<|tool_call>...<tool_call|> blocks for multiple calls. If no function fits, "
    "answer the user directly."
)


def render_system_manifest(tools: Sequence[BoundTool], provider: str) -> str:
    """Advertise tools in the model's native ``<|tool>declaration:...<tool|>`` grammar."""
    decls = "".join(f"<|tool>{_declaration(t)}<tool|>" for t in tools)
    return decls + _MANIFEST_INSTRUCTION


# --------------------------------------------------------------------------- #
# PARSE: <|tool_call>call:NAME{...}<tool_call|>  (recursive-descent over the args)
# --------------------------------------------------------------------------- #
_CALL_RE = re.compile(r"<\|tool_call>call:(?P<name>[^{]+)\{(?P<args>.*?)\}<tool_call\|>", re.DOTALL)

#: Tolerant fallback for a DEGRADED call that dropped the ``<|tool_call>call:`` opener.
#: Observed on the real himalaya-gemma-4 model: it gets the name + args right but writes
#: ``NAME{...}<tool_call|>`` (or ``call:NAME{...}<tool_call|>``) with no opener, so the
#: strict grammar above never matches and the tool silently never fires. We recover it
#: ONLY when NAME is a bound tool (name-guarded, so ordinary prose like ``note{...}`` is
#: never mistaken for a call) and only when it is not already captured by the strict pass
#: — keeping the tool firing whether or not the model remembers the opener.
_LENIENT_CALL_RE = re.compile(
    r"(?:call:)?(?P<name>[A-Za-z_][\w.-]*)\s*\{(?P<args>.*?)\}<tool_call\|>", re.DOTALL
)


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i


def _parse_value(s: str, i: int) -> tuple[Any, int]:
    i = _skip_ws(s, i)
    if i >= len(s):
        return "", i
    if s.startswith(STR, i):
        start = i + len(STR)
        end = s.find(STR, start)
        if end == -1:
            return s[start:], len(s)
        return s[start:end], end + len(STR)
    if s[i] == "{":
        obj: dict[str, Any] = {}
        i += 1
        while True:
            i = _skip_ws(s, i)
            if i >= len(s) or s[i] == "}":
                return obj, (i + 1 if i < len(s) else i)
            colon = s.find(":", i)
            if colon == -1:
                return obj, len(s)
            key = s[i:colon].strip()
            val, i = _parse_value(s, colon + 1)
            obj[key] = val
            i = _skip_ws(s, i)
            if i < len(s) and s[i] == ",":
                i += 1
    if s[i] == "[":
        arr: list[Any] = []
        i += 1
        while True:
            i = _skip_ws(s, i)
            if i >= len(s) or s[i] == "]":
                return arr, (i + 1 if i < len(s) else i)
            val, i = _parse_value(s, i)
            arr.append(val)
            i = _skip_ws(s, i)
            if i < len(s) and s[i] == ",":
                i += 1
    j = i
    depth = 0
    while j < len(s):
        c = s[j]
        if c in "{[":
            depth += 1
        elif c in "}]":
            if depth == 0:
                break
            depth -= 1
        elif c == "," and depth == 0:
            break
        j += 1
    raw = s[i:j].strip()
    return _coerce_scalar(raw), j


def _coerce_scalar(raw: str) -> Any:
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw in ("null", "None"):
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d*\.\d+([eE][-+]?\d+)?", raw):
        return float(raw)
    return raw


def _parse_args(body: str) -> dict[str, Any]:
    body = body.strip()
    if not body:
        return {}
    val, _ = _parse_value("{" + body + "}", 0)
    return val if isinstance(val, dict) else {}


def parse(text: str, known: set[str]) -> list[ToolCallRecord]:
    """Fail-open parse: return only calls whose name is a known bound tool.

    The strict native grammar ``<|tool_call>call:NAME{...}<tool_call|>`` is matched first.
    Then, WHEN tool names are known, a tolerant pass recovers an opener-less
    ``NAME{...}<tool_call|>`` (a real degradation the gemma-4 model exhibits) — name-guarded
    so only a BOUND tool can fire, and de-duplicated against the strict pass by end
    position. Calls are emitted in the order they appear in the text.
    """
    text = text or ""
    seen_ends: set[int] = set()
    hits: list[tuple[int, str, str]] = []  # (start_pos, name, args_body)
    for m in _CALL_RE.finditer(text):
        seen_ends.add(m.end())
        hits.append((m.start(), m.group("name").strip(), m.group("args")))
    # Tolerant recovery is name-guarded, so it only runs when the bound-tool set is known.
    if known:
        for m in _LENIENT_CALL_RE.finditer(text):
            if m.end() in seen_ends:
                continue  # shares a closer with a strict hit — already captured
            name = m.group("name").strip()
            if name in known:
                hits.append((m.start(), name, m.group("args")))
    hits.sort(key=lambda h: h[0])
    out: list[ToolCallRecord] = []
    for _start, name, args_body in hits:
        if known and name not in known:
            continue
        out.append(
            ToolCallRecord(
                tool_call_id=f"gemma4-{uuid.uuid4().hex}",
                tool_name=name,
                args=_parse_args(args_body),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# RENDER tool results + assistant calls (for multi-turn history fidelity)
# --------------------------------------------------------------------------- #
def render_tool_results(results: Sequence[ToolReturnRecord]) -> str:
    """Thread tool results back in the native ``<|tool_response>...<tool_response|>`` grammar."""
    parts: list[str] = []
    for r in results:
        content = r.content
        if isinstance(content, dict):
            body = ",".join(f"{k}:{_fmt_arg(content[k], escape_keys=False)}" for k in sorted(content))
            inner = f"response:{r.tool_name}{{{body}}}"
        else:
            inner = f"response:{r.tool_name}{{value:{_fmt_arg(str(content), escape_keys=False)}}}"
        parts.append(f"<|tool_response>{inner}<tool_response|>")
    return "".join(parts)


def render_assistant_calls(calls: Sequence[Any]) -> str:
    """Re-render assistant tool calls (objects with .name/.arguments) in native grammar."""
    parts: list[str] = []
    for tc in calls:
        name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "")
        args = getattr(tc, "arguments", None)
        if args is None and isinstance(tc, dict):
            args = tc.get("arguments")
        if isinstance(args, str):
            import json

            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        body = ",".join(f"{k}:{_fmt_arg(args[k], escape_keys=False)}" for k in sorted(args))
        parts.append(f"<|tool_call>call:{name}{{{body}}}<tool_call|>")
    return "".join(parts)
