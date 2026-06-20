"""Inference kernel: schema -> constraint compiler (guaranteed structured output).

A single, PURE, offline-testable module that turns a JSON-Schema (a request's
``output_json_schema``, or a forced tool's argument schema) into the *constraint*
each runtime needs so the model PHYSICALLY CANNOT emit invalid output:

* **Ollama** — its ``/api/chat`` ``format`` field already takes a JSON schema
  verbatim; :func:`ollama_format` is the (normalized) pass-through.
* **llama.cpp ``llama-server``** — its ``/completion`` accepts a GBNF ``grammar``
  (or a ``json_schema``); :func:`schema_to_gbnf` compiles the common JSON-Schema
  subset into a GBNF grammar that constrains decoding token-by-token.
* **OpenAI / OpenRouter** — ``response_format={"type":"json_schema", strict}`` and
  guided tool-args are handled in :mod:`himmy.services.inference.openai_manager`;
  this module exposes :func:`constrained_decoding_requested` so those paths share
  one opt-in gate.
* **anything else** — no constraint (today's fail-open behaviour).

Design principles (mirrors :mod:`himmy.services.tools.schema_normalize`):

* **Opt-in.** Constraint is applied ONLY when the request opts in
  (:func:`constrained_decoding_requested` — a ``metadata`` flag). Every
  un-opted path is byte/behaviour-identical to before.
* **Fail-open.** A schema the compiler does not understand widens to the
  permissive ``root ::= value`` JSON grammar (accept any JSON) rather than
  raising — a feature flag must never take a working call down.
* **Widen, never narrow on the parts we cannot express.** The GBNF subset covers
  object/properties/required/enum/string/number/integer/boolean/null/array-items.
  Unsupported keywords (``pattern``/``minimum``/``format``/combinators/…) are
  SKIPPED (the field stays the widest grammar for its type), never half-encoded
  into a grammar that would reject valid output.
* **Pure + offline.** No network, no provider SDK import — schema-in / string-out.

The compiled GBNF is deterministic (stable rule order + names) so a golden test
can pin it byte-for-byte.
"""

from __future__ import annotations

import json
from typing import Any

from himmy.services.inference.models import InferenceRequest
from himmy.services.tools.schema_normalize import normalize_tool_schema

#: Request ``metadata`` key that opts a call into provider-native constrained
#: decoding (GBNF grammar on llama-server, json_schema/guided-args elsewhere).
#: Absent / falsy = today's unconstrained behaviour (fail-open). A string is
#: truthy too, so ``metadata={"constrained_decoding": "1"}`` also opts in.
CONSTRAINED_DECODING_KEY = "constrained_decoding"

#: Recursion ceiling: a schema deeper than this widens to the permissive JSON
#: value grammar rather than risking the stack on an adversarial input.
_MAX_DEPTH = 32


def constrained_decoding_requested(request: InferenceRequest) -> bool:
    """True when this request opted into provider-native constrained decoding.

    Reads the :data:`CONSTRAINED_DECODING_KEY` request-metadata flag. Returns
    ``False`` for any request that did not set it, so a manager guarding a
    constraint path on this stays byte-identical to today by default.
    """
    return bool(request.metadata.get(CONSTRAINED_DECODING_KEY))


def ollama_format(schema: Any, *, provider: str = "ollama") -> dict[str, Any]:
    """Project a JSON-Schema into Ollama's ``format`` field (normalized pass-through).

    Ollama constrains its reply to whatever JSON schema is in the ``format`` field
    of ``/api/chat``; it needs no GBNF. We run the schema through the shared
    cross-provider normalizer (``$ref`` inlining, nullable collapse, …) so a schema
    another provider tolerates still parses for Ollama, then hand it over verbatim.
    """
    return normalize_tool_schema(schema, provider)


# --------------------------------------------------------------------------- #
# GBNF compiler (llama.cpp grammar). The output is a set of named rules; the
# entry rule is always ``root``. llama.cpp's GBNF is a PEG-ish EBNF: ``::=`` binds
# a rule, ``|`` alternates, ``*``/``?`` repeat, ``"..."`` is a literal, ``[...]``
# is a char class, and whitespace between terms is implicit-but-ignored.
# --------------------------------------------------------------------------- #

#: Shared primitive rules emitted once and referenced by generated rules. These
#: are the llama.cpp ``json.gbnf`` primitives, verbatim, so the grammar matches
#: well-formed JSON values/strings/numbers and JSON whitespace.
_PRIMITIVES: dict[str, str] = {
    "ws": r"""[ \t\n]*""",
    "string": r'''"\"" ( [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* "\""''',
    "number": r'''"-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?''',
    "boolean": r'''"true" | "false"''',
    "null": r'''"null"''',
    "value": r"object | array | string | number | boolean | null",
    "object": r'''"{" ws ( string ws ":" ws value ( ws "," ws string ws ":" ws value )* )? ws "}"''',
    "array": r'''"[" ws ( value ( ws "," ws value )* )? ws "]"''',
}


class _GbnfBuilder:
    """Accumulates named GBNF rules with deterministic, collision-free names.

    Rules are emitted in INSERTION ORDER after ``root``, and an identical
    sub-schema reuses the same rule (content-addressed by its compiled body) so
    the grammar is compact and the byte output is stable for a given schema.
    """

    def __init__(self) -> None:
        self._rules: dict[str, str] = {}
        self._order: list[str] = []
        self._used_primitives: set[str] = set()
        self._by_body: dict[str, str] = {}
        self._counter = 0

    def add(self, name: str, body: str) -> str:
        """Register ``name ::= body`` (idempotent on name); return the name."""
        if name not in self._rules:
            self._rules[name] = body
            self._order.append(name)
        return name

    def intern(self, prefix: str, body: str) -> str:
        """Register an anonymous rule for ``body``, reusing one with the same body."""
        existing = self._by_body.get(body)
        if existing is not None:
            return existing
        name = f"{prefix}-{self._counter}"
        self._counter += 1
        self._by_body[body] = name
        self.add(name, body)
        return name

    def use_primitive(self, name: str) -> str:
        """Mark a shared primitive (and its transitive deps) as used; return its name."""
        if name in self._used_primitives:
            return name
        self._used_primitives.add(name)
        # ``value``/``object``/``array`` pull in the scalar primitives transitively.
        deps = {
            "value": ("object", "array", "string", "number", "boolean", "null"),
            "object": ("ws", "string", "value"),
            "array": ("ws", "value"),
        }.get(name, ())
        for dep in deps:
            self.use_primitive(dep)
        return name

    def render(self, root_body: str) -> str:
        """Emit the full grammar text: ``root`` first, then rules, then primitives."""
        lines = [f"root ::= {root_body}"]
        for name in self._order:
            lines.append(f"{name} ::= {self._rules[name]}")
        # Primitives last, in a stable canonical order, only if referenced.
        for name in (
            "value",
            "object",
            "array",
            "string",
            "number",
            "boolean",
            "null",
            "ws",
        ):
            if name in self._used_primitives:
                lines.append(f"{name} ::= {_PRIMITIVES[name]}")
        return "\n".join(lines)


def _gbnf_string_literal(text: str) -> str:
    """A GBNF literal matching the JSON-encoded ``text`` (an enum/const value).

    The value is JSON-serialized (so a string gets its quotes + escapes) and each
    character emitted as a GBNF double-quoted literal. This matches the exact JSON
    token, so an ``enum`` becomes an alternation of exact literals.
    """
    encoded = json.dumps(text)
    # GBNF string literals are double-quoted; escape backslash and double-quote.
    safe = encoded.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{safe}"'


def _gbnf_json_literal(value: Any) -> str:
    """A GBNF literal matching a JSON scalar (string/number/bool/null) ``value``."""
    encoded = json.dumps(value)
    safe = encoded.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{safe}"'


def _compile(node: Any, builder: _GbnfBuilder, *, depth: int) -> str:
    """Compile a JSON-Schema ``node`` to a GBNF rule body (or primitive reference).

    Returns a rule NAME or an inline body the caller can reference. Fail-open: a
    shape the subset does not cover widens to the permissive ``value`` primitive.
    """
    if depth > _MAX_DEPTH or not isinstance(node, dict) or not node:
        return builder.use_primitive("value")

    # ``enum`` / ``const`` pin an exact set of JSON literals (most constraining).
    if "const" in node:
        return builder.intern("const", _gbnf_json_literal(node["const"]))
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        alts = " | ".join(_gbnf_json_literal(v) for v in enum)
        return builder.intern("enum", alts)

    types = _type_set(node)
    # A multi-type union (other than the nullable case folded below) is widened to
    # the permissive value grammar rather than a half-correct alternation.
    nullable = "null" in types
    concrete = [t for t in types if t != "null"]
    if len(concrete) != 1:
        body = builder.use_primitive("value")
        return body

    type_name = concrete[0]
    if type_name == "object":
        body = _compile_object(node, builder, depth=depth)
    elif type_name == "array":
        body = _compile_array(node, builder, depth=depth)
    elif type_name == "string":
        body = builder.use_primitive("string")
    elif type_name == "integer":
        body = builder.intern("integer", _PRIMITIVES_INTEGER)
    elif type_name == "number":
        body = builder.use_primitive("number")
    elif type_name == "boolean":
        body = builder.use_primitive("boolean")
    elif type_name == "null":
        body = builder.use_primitive("null")
    else:  # unknown type token -> widen
        body = builder.use_primitive("value")

    if nullable:
        # A nullable typed field accepts its type OR a literal ``null``.
        builder.use_primitive("null")
        return builder.intern("nullable", f"{body} | null")
    return body


#: GBNF for a JSON integer (no fraction/exponent) — a tighter ``number`` variant.
_PRIMITIVES_INTEGER = r'''"-"? ("0" | [1-9] [0-9]*)'''


def _compile_object(node: dict[str, Any], builder: _GbnfBuilder, *, depth: int) -> str:
    """Compile a ``type:object`` schema to a GBNF object rule.

    Encodes ``properties`` as named members in a STABLE (schema-declared) order,
    each member optional unless listed in ``required``. ``additionalProperties``
    that is not ``False`` keeps the object open (extra free-form members allowed).
    An object with no enumerable ``properties`` widens to the generic ``object``
    primitive (accept any JSON object) rather than forbidding all members.
    """
    props = node.get("properties")
    if not isinstance(props, dict) or not props:
        return builder.use_primitive("object")

    required = node.get("required")
    required_set = (
        {r for r in required if isinstance(r, str)}
        if isinstance(required, list)
        else set()
    )
    builder.use_primitive("ws")
    builder.use_primitive("string")
    open_object = node.get("additionalProperties") is not False

    # Build a member rule per property: ``"key" ws ":" ws <value-rule>``.
    member_terms: list[tuple[str, bool]] = []  # (member-rule-name, is-required)
    for key, sub in props.items():
        value_rule = _compile(sub, builder, depth=depth + 1)
        key_literal = _gbnf_string_literal(key)
        member_body = f'{key_literal} ws ":" ws {value_rule}'
        member_name = builder.intern("member", member_body)
        member_terms.append((member_name, key in required_set))

    # Compose the brace body via a recursive member chain (handles every subset of
    # the optional members, in declared order, with correct comma placement). When
    # the object is OPEN (``additionalProperties`` != False) a free-form tail accepts
    # extra keys (widening), so the constraint never rejects a valid object.
    inner = _compose_object_members(member_terms, open_object, builder)
    body = f'"{{" ws {inner} ws "}}"'
    return builder.intern("obj", body)


def _compose_object_members(
    member_terms: list[tuple[str, bool]],
    open_object: bool,
    builder: _GbnfBuilder,
) -> str:
    """Compose the inside-the-braces GBNF for an object's members (comma-correct).

    A naive ``m0 ("," m1)? ("," m2)?`` is WRONG: if an earlier optional is omitted
    a later member's hard-coded leading comma produces invalid JSON (``{,"x":1}``)
    or a missing one. We instead emit a recursive chain of helper rules, two per
    member position:

    * ``chain-fresh-i`` — members from index ``i`` on, with NOTHING emitted yet, so
      the next emitted member carries NO leading comma; and
    * ``chain-after-i`` — members from index ``i`` on, with something ALREADY
      emitted, so the next emitted member carries a leading ``"," ws``.

    A required member forces the chain through it; an optional member alternates
    between "emit it (then continue in the *after* state)" and "skip it (continue in
    the SAME state)". The terminal rule (past the last member) is empty, plus — for
    an OPEN object — a free-form extra-member tail so unanticipated keys are still
    accepted (widening). The result is a context-free grammar that accepts exactly
    the in-order objects the schema allows, for every combination of present
    optionals.
    """
    n = len(member_terms)
    if n == 0:
        return ""
    builder.use_primitive("ws")
    # The open-object tail accepts zero-or-more extra free-form members.
    if open_object:
        builder.use_primitive("string")
        builder.use_primitive("value")
        tail = '( ws "," ws string ws ":" ws value )*'
    else:
        tail = ""

    def chain(idx: int, *, after: bool) -> str:
        """Return the rule name for the member chain at ``idx`` (after/fresh state)."""
        if idx >= n:
            # Past the last member: an OPEN object may still emit extra members IF
            # something already preceded them (so the comma is valid); a FRESH-state
            # terminal is empty (an open object with no declared members emitted).
            body = tail if (open_object and after) else ""
            return builder.intern("end", body or '""')
        name, is_required = member_terms[idx]
        member = name
        sep = '"," ws ' if after else ""
        emit = f"{sep}{member} {chain(idx + 1, after=True)}"
        if is_required:
            body = emit
        else:
            # Optional: either emit this member (then continue in the after-state),
            # or skip it (continue in the SAME state so the comma decision is intact).
            skip = chain(idx + 1, after=after)
            body = f"( {emit} ) | ( {skip} )"
        prefix = "chain-after" if after else "chain-fresh"
        return builder.intern(prefix, body)

    return chain(0, after=False)


def _compile_array(node: dict[str, Any], builder: _GbnfBuilder, *, depth: int) -> str:
    """Compile a ``type:array`` schema to a GBNF array rule.

    A single ``items`` schema constrains every element; a tuple-form ``items``
    (list) or an absent ``items`` widens the element to the permissive ``value``
    primitive (accept any JSON value) so we never reject a valid array.
    """
    items = node.get("items")
    if isinstance(items, dict):
        item_rule = _compile(items, builder, depth=depth + 1)
    else:
        item_rule = builder.use_primitive("value")
    builder.use_primitive("ws")
    body = f'"[" ws ( {item_rule} ( ws "," ws {item_rule} )* )? ws "]"'
    return builder.intern("arr", body)


def _type_set(node: dict[str, Any]) -> set[str]:
    """Declared JSON-Schema types as a set (handles the ``["X","null"]`` form)."""
    t = node.get("type")
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    if isinstance(t, str):
        return {t}
    # No declared type but enumerable object/array keywords present -> infer.
    if isinstance(node.get("properties"), dict):
        return {"object"}
    if "items" in node:
        return {"array"}
    return set()


def schema_to_gbnf(schema: Any, *, provider: str = "ollama") -> str:
    """Compile a JSON-Schema into a llama.cpp GBNF grammar; pure + fail-open.

    The schema is first run through the shared cross-provider normalizer (``$ref``
    inlining, nullable collapse, …) so a referenced/exotic schema is flattened into
    the subset this compiler understands. The result is a GBNF grammar string whose
    entry rule is ``root``; feed it to ``llama-server``'s ``/completion`` ``grammar``
    field to constrain decoding token-by-token.

    Fail-open contract: a schema (or sub-schema) outside the supported subset
    widens to the permissive JSON grammar (``root ::= value``) rather than raising
    — a feature flag must never break a call. The output is DETERMINISTIC for a
    given schema (stable rule order + names) so it can be pinned by a golden test.
    """
    builder = _GbnfBuilder()
    normalized = normalize_tool_schema(schema, provider)
    if not isinstance(normalized, dict) or not normalized:
        # Empty / non-dict schema -> accept any JSON value.
        builder.use_primitive("value")
        return builder.render("value")
    try:
        root_body = _compile(normalized, builder, depth=0)
    except Exception:  # noqa: BLE001 - fail-open: never let a grammar bug take a call down
        builder = _GbnfBuilder()
        builder.use_primitive("value")
        return builder.render("value")
    return builder.render(root_body)


__all__ = [
    "CONSTRAINED_DECODING_KEY",
    "constrained_decoding_requested",
    "ollama_format",
    "schema_to_gbnf",
]
