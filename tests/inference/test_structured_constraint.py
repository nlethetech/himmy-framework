"""Tests for guaranteed structured output: schema -> constraint (all offline).

Covers the pure schema->constraint compiler
(:mod:`himmy.services.inference.structured`) plus its opt-in, fail-open wiring into
the Ollama / HimalayaGPT(llama.cpp) / OpenRouter managers:

* a GBNF GOLDEN pinning the compiler's deterministic byte output;
* a PROPERTY test — for a battery of schemas + valid instances, the compiled GBNF
  matches every valid instance and rejects representative invalid ones (a minimal
  but faithful GBNF acceptor stands in for llama.cpp's grammar engine offline);
* opt-in gating: an un-opted-in request is byte-identical to today on every path;
* manager wiring: HimalayaGPT passes a compiled grammar to a grammar-capable
  generate-fn (and NOT to a plain ``(prompt)->str`` fn), Ollama normalizes the
  ``format`` schema only when opted in, and OpenRouter upgrades an open model to
  ``strict`` only when opted in.
"""

from __future__ import annotations

import json
import re
from typing import Any

from himmy.services.inference import (
    HimalayaGptClientManager,
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    OllamaClientManager,
)
from himmy.services.inference.models import ResponseFormat
from himmy.services.inference.openai_manager import OpenAIClientManager
from himmy.services.inference.structured import (
    CONSTRAINED_DECODING_KEY,
    constrained_decoding_requested,
    ollama_format,
    schema_to_gbnf,
)
from himmy.services.tools.validation import validate_against_schema
from tests.conftest import run_async


# --------------------------------------------------------------------------- #
# A tiny GBNF acceptor: enough of llama.cpp's GBNF to decide whether a STRING is
# in the language of a grammar this compiler emits. It is NOT a general GBNF
# engine — it supports exactly the constructs ``schema_to_gbnf`` produces
# (named rules, ``|`` alternation, sequence, ``( )`` groups, ``*``/``?`` repeat,
# ``"lit"`` literals, ``[..]`` char classes) — so the property test can assert
# accept/reject offline without spawning llama-server. The real-server e2e is run
# separately.
# --------------------------------------------------------------------------- #
#: The fixed JSON primitives (verbatim from llama.cpp's ``json.gbnf``) are matched
#: by dedicated, known-correct Python matchers rather than re-interpreting their
#: gnarly GBNF bodies (negated classes, ``\\u`` escapes). The acceptor only
#: INTERPRETS the structural rules the compiler GENERATES (objects/arrays/enums/
#: members/chains), which is where correctness actually matters.
def _match_json_string(text: str, pos: int) -> set[int]:
    if pos >= len(text) or text[pos] != '"':
        return set()
    i = pos + 1
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return {i + 1}
        i += 1
    return set()


def _match_number(text: str, pos: int, *, integer: bool) -> set[int]:
    pattern = r"-?(0|[1-9][0-9]*)" + ("" if integer else r"(\.[0-9]+)?([eE][-+]?[0-9]+)?")
    m = re.match(pattern, text[pos:])
    return {pos + m.end()} if m else set()


_PRIMITIVE_MATCHERS = {
    "string": lambda t, p: _match_json_string(t, p),
    "number": lambda t, p: _match_number(t, p, integer=False),
    "boolean": lambda t, p: ({p + 4} if t.startswith("true", p) else {p + 5} if t.startswith("false", p) else set()),
    "null": lambda t, p: ({p + 4} if t.startswith("null", p) else set()),
    "ws": lambda t, p: {p + len(re.match(r"[ \t\n]*", t[p:]).group(0))},  # type: ignore[union-attr]
}


class _GbnfAcceptor:
    def __init__(self, grammar: str) -> None:
        self.rules: dict[str, str] = {}
        for line in grammar.splitlines():
            if "::=" not in line:
                continue
            name, body = line.split("::=", 1)
            self.rules[name.strip()] = body.strip()

    def accepts(self, text: str) -> bool:
        ends = self._match_seq(self._tokenize(self.rules["root"]), text, 0)
        return any(end == len(text) for end in ends)

    # -- tokenization of a rule body into terms ------------------------------
    def _tokenize(self, body: str) -> list[str]:
        toks: list[str] = []
        i = 0
        while i < len(body):
            c = body[i]
            if c.isspace():
                i += 1
            elif c == '"':
                j = i + 1
                while j < len(body) and body[j] != '"':
                    if body[j] == "\\":
                        j += 1
                    j += 1
                toks.append(body[i : j + 1])
                i = j + 1
            elif c == "[":
                j = body.index("]", i)
                # allow an escaped ] inside the class
                while body[j - 1] == "\\":
                    j = body.index("]", j + 1)
                toks.append(body[i : j + 1])
                i = j + 1
            elif c in "()|*?":
                toks.append(c)
                i += 1
            else:
                j = i
                while j < len(body) and (body[j].isalnum() or body[j] in "-_"):
                    j += 1
                toks.append(body[i:j])
                i = j
        return toks

    # -- recursive-descent matcher returning the set of possible end indices --
    def _match_seq(self, toks: list[str], text: str, pos: int) -> set[int]:
        # Split on top-level ``|`` into alternatives, then match each as a sequence.
        alts = self._split_top(toks, "|")
        ends: set[int] = set()
        for alt in alts:
            ends |= self._match_terms(alt, text, {pos})
        return ends

    def _split_top(self, toks: list[str], sep: str) -> list[list[str]]:
        out: list[list[str]] = []
        cur: list[str] = []
        depth = 0
        for t in toks:
            if t == "(":
                depth += 1
                cur.append(t)
            elif t == ")":
                depth -= 1
                cur.append(t)
            elif t == sep and depth == 0:
                out.append(cur)
                cur = []
            else:
                cur.append(t)
        out.append(cur)
        return out

    def _match_terms(self, toks: list[str], text: str, starts: set[int]) -> set[int]:
        positions = set(starts)
        i = 0
        while i < len(toks):
            term = toks[i]
            quant = ""
            # group
            if term == "(":
                depth = 1
                j = i + 1
                while depth:
                    if toks[j] == "(":
                        depth += 1
                    elif toks[j] == ")":
                        depth -= 1
                    j += 1
                inner = toks[i + 1 : j - 1]
                if j < len(toks) and toks[j] in "*?":
                    quant = toks[j]
                    j += 1
                positions = self._apply(
                    lambda p, inner=inner: self._match_seq(inner, text, p),
                    positions,
                    quant,
                )
                i = j
                continue
            if i + 1 < len(toks) and toks[i + 1] in "*?":
                quant = toks[i + 1]
            positions = self._apply(
                lambda p, term=term: self._match_atom(term, text, p), positions, quant
            )
            i += 2 if quant else 1
        return positions

    def _apply(self, fn: Any, positions: set[int], quant: str) -> set[int]:
        if quant == "":
            out: set[int] = set()
            for p in positions:
                out |= fn(p)
            return out
        if quant == "?":
            out = set(positions)
            for p in positions:
                out |= fn(p)
            return out
        # ``*``: zero or more (fixed point)
        out = set(positions)
        frontier = set(positions)
        while frontier:
            nxt: set[int] = set()
            for p in frontier:
                for e in fn(p):
                    if e not in out:
                        nxt.add(e)
            out |= nxt
            frontier = nxt
        return out

    def _match_atom(self, term: str, text: str, pos: int) -> set[int]:
        if term.startswith('"') and term.endswith('"'):
            lit = term[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            if text.startswith(lit, pos):
                return {pos + len(lit)}
            return set()
        if term.startswith("[") and term.endswith("]"):
            if pos < len(text) and self._class_match(term[1:-1], text[pos]):
                return {pos + 1}
            return set()
        # Known-correct primitive matchers (avoid re-interpreting their GBNF bodies).
        if term in _PRIMITIVE_MATCHERS:
            return _PRIMITIVE_MATCHERS[term](text, pos)
        if term == "value":
            return self._match_value(text, pos)
        if term == "object":
            return self._match_generic_object(text, pos)
        if term == "array":
            return self._match_generic_array(text, pos)
        # Generated ``integer-N`` rules are a tighter number; match as integer.
        if term.split("-")[0] == "integer":
            return _match_number(text, pos, integer=True)
        # rule reference (generated structural rule)
        if term in self.rules:
            return self._match_seq(self._tokenize(self.rules[term]), text, pos)
        return set()

    # -- generic JSON value/object/array (for the permissive ``value`` primitive) --
    def _match_value(self, text: str, pos: int) -> set[int]:
        out: set[int] = set()
        out |= self._match_generic_object(text, pos)
        out |= self._match_generic_array(text, pos)
        out |= _match_json_string(text, pos)
        out |= _match_number(text, pos, integer=False)
        out |= _PRIMITIVE_MATCHERS["boolean"](text, pos)
        out |= _PRIMITIVE_MATCHERS["null"](text, pos)
        return out

    def _match_generic_object(self, text: str, pos: int) -> set[int]:
        try:
            obj, end = json.JSONDecoder().raw_decode(text, pos)
        except ValueError:
            return set()
        return {end} if isinstance(obj, dict) else set()

    def _match_generic_array(self, text: str, pos: int) -> set[int]:
        try:
            arr, end = json.JSONDecoder().raw_decode(text, pos)
        except ValueError:
            return set()
        return {end} if isinstance(arr, list) else set()

    def _class_match(self, body: str, ch: str) -> bool:
        # Expand a GBNF char class like ``a-z0-9\t`` (with \\-escapes) into a regex.
        pattern = "[" + body + "]"
        try:
            return re.match(pattern, ch) is not None
        except re.error:
            # Fall back to a literal-member check.
            return ch in body


# ----------------------------------------------------------------- GBNF golden
def test_gbnf_golden_is_byte_stable() -> None:
    """The compiler's GBNF is deterministic and pinned byte-for-byte (golden).

    A change to the compiler MUST update this golden consciously — it is the
    contract a llama-server backend relies on.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "status": {"type": "string", "enum": ["active", "inactive"]},
        },
        "required": ["name", "status"],
        "additionalProperties": False,
    }
    expected = "\n".join(
        [
            "root ::= obj-9",
            'member-0 ::= "\\"name\\"" ws ":" ws string',
            'integer-1 ::= "-"? ("0" | [1-9] [0-9]*)',
            'member-2 ::= "\\"age\\"" ws ":" ws integer-1',
            'enum-3 ::= "\\"active\\"" | "\\"inactive\\""',
            'member-4 ::= "\\"status\\"" ws ":" ws enum-3',
            'end-5 ::= ""',
            'chain-after-6 ::= "," ws member-4 end-5',
            "chain-after-7 ::= ( \",\" ws member-2 chain-after-6 ) | ( chain-after-6 )",
            "chain-fresh-8 ::= member-0 chain-after-7",
            'obj-9 ::= "{" ws chain-fresh-8 ws "}"',
            'string ::= "\\"" ( [^"\\\\] | "\\\\" ["\\\\/bfnrt] | "\\\\u" '
            "[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* \"\\\"\"",
            'ws ::= [ \\t\\n]*',
        ]
    )
    assert schema_to_gbnf(schema) == expected


def test_gbnf_is_deterministic() -> None:
    """Compiling the same schema twice yields byte-identical grammar."""
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    assert schema_to_gbnf(schema) == schema_to_gbnf(schema)


def test_gbnf_failopen_on_unknown_widens_to_value() -> None:
    """A non-dict / empty schema widens to the permissive JSON value grammar."""
    g = schema_to_gbnf(None)
    assert g.splitlines()[0] == "root ::= value"
    assert "value ::=" in g


# ----------------------------------------------------- constrained-output property
_SCHEMAS: list[tuple[dict[str, Any], list[Any], list[Any]]] = [
    # (schema, valid instances, invalid instances)
    (
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        [{"name": "Sita"}, {"name": "Ram", "age": 30}],
        [{"age": 30}, {"name": 5}, [1, 2]],
    ),
    (
        {"type": "string", "enum": ["red", "green", "blue"]},
        ["red", "green", "blue"],
        ["yellow", "Red", 1],
    ),
    (
        {"type": "array", "items": {"type": "integer"}},
        [[], [1], [1, 2, 3]],
        [["x"], [1.5], {"a": 1}],
    ),
    (
        {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "active": {"type": "boolean"},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        [
            {"id": 1},
            {"id": 2, "tags": ["a", "b"], "active": True},
            {"id": 3, "active": False},
        ],
        [{"tags": ["a"]}, {"id": "x"}, {"id": 1, "active": "yes"}],
    ),
    (
        {"type": "object", "properties": {"score": {"type": "number"}}},
        [{"score": 1.5}, {"score": -3}, {}],
        ["not an object", 5],
    ),
]


def test_compiled_grammar_accepts_every_valid_instance() -> None:
    """For each schema, the compiled GBNF matches every valid (compact) JSON instance.

    This is the "constrained output is always valid" property in the contrapositive:
    a constrained decoder can ONLY emit strings the grammar accepts, so the grammar
    MUST accept every well-formed instance of the schema (else valid output would be
    impossible). We assert acceptance of each valid instance, and that each instance
    is itself schema-valid (sanity).
    """
    for schema, valids, _ in _SCHEMAS:
        grammar = schema_to_gbnf(schema)
        acc = _GbnfAcceptor(grammar)
        for value in valids:
            assert validate_against_schema(value, schema) is None, value
            rendered = json.dumps(value, separators=(",", ":"))
            assert acc.accepts(rendered), (schema, rendered, grammar)


def test_compiled_grammar_rejects_invalid_instances() -> None:
    """The compiled GBNF rejects representative schema-invalid strings.

    The GBNF subset cannot encode EVERY JSON-Schema constraint (it widens past
    unsupported keywords), so this only asserts the cases the subset DOES encode:
    wrong scalar type, a missing required key, a non-enum value, a wrong array
    element type. Each is also confirmed schema-invalid.
    """
    for schema, _, invalids in _SCHEMAS:
        grammar = schema_to_gbnf(schema)
        acc = _GbnfAcceptor(grammar)
        for value in invalids:
            assert validate_against_schema(value, schema) is not None, value
            rendered = json.dumps(value, separators=(",", ":"))
            assert not acc.accepts(rendered), (schema, rendered, grammar)


def test_acceptor_handles_whitespace_variants() -> None:
    """A constrained decoder may emit JSON whitespace; the grammar permits it."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    acc = _GbnfAcceptor(schema_to_gbnf(schema))
    assert acc.accepts('{"a": 1}')
    assert acc.accepts('{ "a" : 1 }')
    assert acc.accepts('{"a":1}')


# ----------------------------------------------------------------- opt-in gate
def test_constrained_decoding_requested_flag() -> None:
    base = InferenceRequest(messages=[InferenceMessage(role="user", content="hi")])
    assert constrained_decoding_requested(base) is False
    opted = InferenceRequest(
        messages=[InferenceMessage(role="user", content="hi")],
        metadata={CONSTRAINED_DECODING_KEY: True},
    )
    assert constrained_decoding_requested(opted) is True


# ----------------------------------------------------------------- Ollama wiring
def _structured_req(*, opted_in: bool) -> InferenceRequest:
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }
    meta = {CONSTRAINED_DECODING_KEY: True} if opted_in else {}
    return InferenceRequest(
        messages=[InferenceMessage(role="user", content="classify")],
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema=schema,
        metadata=meta,
    )


def test_ollama_default_format_is_raw_schema_byte_identical() -> None:
    """Un-opted-in: Ollama's ``format`` is the RAW schema, exactly as before."""
    seen: dict[str, Any] = {}

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = payload
        return {"message": {"content": '{"label":"x"}'}}

    mgr = OllamaClientManager(model="qwen2.5:0.5b", transport=transport)
    req = _structured_req(opted_in=False)
    run_async(mgr.generate(req))
    assert seen["payload"]["format"] == req.output_json_schema


def test_ollama_optin_format_is_normalized() -> None:
    """Opted-in: Ollama's ``format`` is the NORMALIZED schema (inlines $ref etc.)."""
    seen: dict[str, Any] = {}

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = payload
        return {"message": {"content": '{"label":"x"}'}}

    schema = {
        "type": "object",
        "properties": {"label": {"$ref": "#/$defs/Label"}},
        "required": ["label"],
        "$defs": {"Label": {"type": "string"}},
    }
    mgr = OllamaClientManager(model="qwen2.5:0.5b", transport=transport)
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="x")],
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema=schema,
        metadata={CONSTRAINED_DECODING_KEY: True},
    )
    run_async(mgr.generate(req))
    fmt = seen["payload"]["format"]
    # $defs inlined away; the ref resolved to a string schema.
    assert "$defs" not in fmt
    assert fmt["properties"]["label"] == {"type": "string"}
    assert ollama_format(schema) == fmt


# ------------------------------------------------------- HimalayaGPT (llama.cpp) wiring
def test_himalaya_passes_grammar_to_grammar_capable_fn() -> None:
    """Opted-in structured output: a compiled GBNF grammar reaches the backend fn."""
    captured: dict[str, Any] = {}

    def grammar_fn(prompt: str, *, grammar: str | None = None) -> str:
        captured["grammar"] = grammar
        return '{"label":"ok"}'

    mgr = HimalayaGptClientManager(generate_fn=grammar_fn)
    resp = run_async(mgr.generate(_structured_req(opted_in=True)))
    assert resp.status == InferenceStatus.SUCCESS
    assert captured["grammar"] is not None
    assert captured["grammar"].splitlines()[0].startswith("root ::=")
    # The constrained reply is parsed into output_structured.
    assert resp.output_structured == {"label": "ok"}


def test_himalaya_no_grammar_when_not_opted_in() -> None:
    """Un-opted-in: the backend fn is called WITHOUT a grammar (byte-identical path)."""
    captured: dict[str, Any] = {}

    def grammar_fn(prompt: str, *, grammar: str | None = None) -> str:
        captured["grammar"] = grammar
        return "free text"

    mgr = HimalayaGptClientManager(generate_fn=grammar_fn)
    run_async(mgr.generate(_structured_req(opted_in=False)))
    assert captured["grammar"] is None


def test_himalaya_plain_fn_never_receives_grammar() -> None:
    """A plain ``(prompt)->str`` fn (no ``grammar`` kw) is called with a single arg."""
    calls: dict[str, Any] = {"args": None}

    def plain_fn(prompt: str) -> str:
        calls["args"] = prompt
        return '{"label":"x"}'

    mgr = HimalayaGptClientManager(generate_fn=plain_fn)
    # Even opted-in, a fn that cannot take a grammar must not be passed one (no crash).
    resp = run_async(mgr.generate(_structured_req(opted_in=True)))
    assert resp.status == InferenceStatus.SUCCESS
    assert calls["args"] is not None


def test_himalaya_grammar_suppressed_for_tool_turn() -> None:
    """A tool-bound turn is left unconstrained even when opted in (so tools can fire)."""
    captured: dict[str, Any] = {}

    def grammar_fn(prompt: str, *, grammar: str | None = None) -> str:
        captured["grammar"] = grammar
        return "no tool call here"

    from himmy.services.inference.models import BoundTool

    mgr = HimalayaGptClientManager(generate_fn=grammar_fn)
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="x")],
        bound_tools=[BoundTool(name="t", description="d", args_json_schema={})],
        metadata={CONSTRAINED_DECODING_KEY: True},
    )
    run_async(mgr.generate(req))
    assert captured["grammar"] is None


# ----------------------------------------------------------------- OpenRouter strict
def test_openrouter_optin_upgrades_open_model_to_strict() -> None:
    """Opted-in OpenRouter open model gains ``json_schema.strict`` (hard constraint)."""
    seen: dict[str, Any] = {}

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                async def create(**payload: Any) -> Any:
                    seen["payload"] = payload

                    class _M:
                        content = '{"label":"x"}'
                        tool_calls = None

                    class _C:
                        message = _M()

                    class _Comp:
                        choices = [_C()]
                        usage = None

                    return _Comp()

    mgr = OpenAIClientManager(
        model="meta-llama/llama-3.1-8b-instruct",
        provider_name="openrouter",
        client=_FakeClient(),
    )
    run_async(mgr.generate(_structured_req(opted_in=True)))
    rf = seen["payload"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"].get("strict") is True


def test_openrouter_default_open_model_is_lenient() -> None:
    """Un-opted-in OpenRouter open model stays lenient (no ``strict``) — byte-identical."""
    seen: dict[str, Any] = {}

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                async def create(**payload: Any) -> Any:
                    seen["payload"] = payload

                    class _M:
                        content = '{"label":"x"}'
                        tool_calls = None

                    class _C:
                        message = _M()

                    class _Comp:
                        choices = [_C()]
                        usage = None

                    return _Comp()

    mgr = OpenAIClientManager(
        model="meta-llama/llama-3.1-8b-instruct",
        provider_name="openrouter",
        client=_FakeClient(),
    )
    run_async(mgr.generate(_structured_req(opted_in=False)))
    rf = seen["payload"]["response_format"]
    assert rf["type"] == "json_schema"
    assert "strict" not in rf["json_schema"]
