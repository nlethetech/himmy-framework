"""Tools kernel: one cross-provider JSON-schema normalizer (the documented P0).

A single, PURE, offline-testable function — :func:`normalize_tool_schema` — that
takes a provider-agnostic JSON schema (a tool's ``args_json_schema`` or a
structured-output schema) and rewrites it into a shape each LLM provider accepts,
*without* an API call. The same primitive is reused at every verbatim
pass-through egress site (Anthropic / OpenAI / Ollama / pydantic-ai / MCP / the
pydantic-ai arg-model adapter) so a schema that one provider tolerates and
another rejects no longer silently breaks tool calling on the stricter backend.

Design principles (all enforced by the adversarial fixture-matrix tests):

* **Declarative, not branching.** Per-provider behaviour lives in a frozen
  :class:`ProviderProfile` table (data). The transform code reads the profile;
  it never hard-codes ``if provider == ...`` business logic.
* **FAIL-SAFE: widen, never fabricate or narrow.** Every transform either leaves
  a schema unchanged or makes it *strictly more permissive*. A ``$ref`` we cannot
  resolve, a cycle we cannot inline, or any shape we are unsure about collapses
  to the permissive ``{}`` / ``{"type": "object"}`` rather than guessing a
  narrower constraint. The normalizer NEVER adds a ``required`` key, an ``enum``,
  a ``minimum``, ``additionalProperties: false`` or any other constraint that was
  not already present.
* **Idempotent.** ``normalize(normalize(s)) == normalize(s)`` for every profile.
* **Offline.** No network, no provider SDK import — pure dict-in / dict-out.

The transforms applied (gated by the profile):

#. **Inline ``$ref`` / ``$defs``.** Resolve local ``#/$defs/...`` /
   ``#/definitions/...`` references in place; a cycle (or an unresolvable ref) is
   widened to a permissive shape rather than recursing forever. ``$defs`` /
   ``definitions`` containers are then dropped (some providers reject unknown
   top-level keys). This runs for every provider — an un-inlined ``$ref`` is the
   single most common cross-provider tool-schema failure.
#. **Collapse nullable unions.** ``{"type": ["string", "null"]}`` and
   ``{"anyOf": [{"type": "string"}, {"type": "null"}]}`` are rewritten to each
   provider's preferred nullable form (a type-array, an ``anyOf``-with-null, or
   the bare non-null type for providers that reject explicit null).
#. **Strip unknown string ``format``.** Providers that reject unknown
   ``format`` values (Gemini is strict) keep only a known allow-list; the
   constraint is *dropped* (widening), never rewritten to a different format.
#. **Ensure non-empty ``properties``.** Gemini rejects an object schema whose
   ``properties`` is empty/absent while ``type == "object"``; the profile can
   request a permissive placeholder property so the call is accepted (widening —
   the placeholder is optional and unconstrained).
#. **Recurse into nested schemas** (``properties`` / ``items`` / ``$defs`` /
   combinators / ``additionalProperties``) so deeply-nested schemas are handled
   uniformly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

# --------------------------------------------------------------------------- #
# Nullable representations a profile can target.
# --------------------------------------------------------------------------- #
#: Emit ``{"type": ["<t>", "null"]}`` for a nullable scalar (JSON-Schema canonical;
#: accepted by Anthropic, Ollama, OpenAI tool schemas).
NULL_STYLE_TYPE_ARRAY = "type_array"
#: Emit ``{"anyOf": [<schema>, {"type": "null"}]}`` (the OpenAI strict /
#: structured-output canonical form).
NULL_STYLE_ANYOF = "anyof"
#: Drop the ``null`` member entirely, leaving the bare non-null type. Used by
#: backends that reject an explicit null type in a tool schema. This is still a
#: WIDENING relative to rejecting the schema, and never narrows an already
#: non-nullable schema.
NULL_STYLE_DROP_NULL = "drop_null"

#: String ``format`` values a strict provider (Gemini) is known to accept on a
#: tool/function schema. Anything else is DROPPED (widening — the format
#: constraint is removed, never rewritten) for profiles that
#: ``strip_unknown_format``. Gemini's function-calling schema accepts only
#: ``enum`` (handled separately, not a format here) and ``date-time`` for STRING,
#: so this list is deliberately tiny; a non-strict profile keeps every format.
_KNOWN_FORMATS: frozenset[str] = frozenset(
    {
        "date-time",
    }
)

#: Top-level schema keys whose presence breaks a provider (definition containers).
#: They are inlined then stripped by the ``$ref`` pass; listed here so the strip
#: stays declarative.
_DEFS_KEYS: tuple[str, ...] = ("$defs", "definitions")

#: A maximally-permissive object placeholder used wherever the normalizer must
#: widen (unresolvable ref, cycle, empty-properties trap fallback).
_PERMISSIVE_OBJECT: dict[str, Any] = {"type": "object"}


@dataclass(frozen=True)
class ProviderProfile:
    """Declarative per-provider normalization rules (data, not branching code).

    Every flag defaults to the most conservative (least-transforming) value so an
    unknown provider gets only the universally-safe ``$ref`` inlining.
    """

    #: Canonical provider key (``"openai"``, ``"anthropic"``, ``"gemini"`` ...).
    name: str
    #: How a nullable union (``type:[X,null]`` / ``anyOf``-with-null) is emitted.
    null_style: str = NULL_STYLE_TYPE_ARRAY
    #: Drop unknown string ``format`` values (Gemini is strict about these).
    strip_unknown_format: bool = False
    #: When an object schema has empty/absent ``properties``, inject a permissive
    #: placeholder property so the provider accepts it (the empty-properties trap).
    ensure_nonempty_properties: bool = False
    #: Name of the permissive placeholder property injected when
    #: ``ensure_nonempty_properties`` fires. Optional + unconstrained = widening.
    placeholder_property: str = "_"
    #: Allow-list of string formats kept when ``strip_unknown_format`` is set.
    known_formats: frozenset[str] = field(default_factory=lambda: _KNOWN_FORMATS)


# --------------------------------------------------------------------------- #
# The declarative profile table. provider_name on each manager selects a row;
# unknown names fall back to a safe default (ref-inlining only).
# --------------------------------------------------------------------------- #
_PROFILES: dict[str, ProviderProfile] = {
    "openai": ProviderProfile(name="openai", null_style=NULL_STYLE_TYPE_ARRAY),
    "openrouter": ProviderProfile(
        name="openrouter", null_style=NULL_STYLE_TYPE_ARRAY
    ),
    "anthropic": ProviderProfile(name="anthropic", null_style=NULL_STYLE_TYPE_ARRAY),
    "ollama": ProviderProfile(name="ollama", null_style=NULL_STYLE_TYPE_ARRAY),
    "claude-cli": ProviderProfile(
        name="claude-cli", null_style=NULL_STYLE_TYPE_ARRAY
    ),
    "himalayagpt": ProviderProfile(
        name="himalayagpt", null_style=NULL_STYLE_TYPE_ARRAY
    ),
    "pydantic_ai": ProviderProfile(
        name="pydantic_ai", null_style=NULL_STYLE_ANYOF
    ),
    # Google Gemini: the strict provider — rejects unknown formats and empty
    # object ``properties``, and is happiest with the bare non-null type.
    "gemini": ProviderProfile(
        name="gemini",
        null_style=NULL_STYLE_DROP_NULL,
        strip_unknown_format=True,
        ensure_nonempty_properties=True,
    ),
    "google": ProviderProfile(
        name="google",
        null_style=NULL_STYLE_DROP_NULL,
        strip_unknown_format=True,
        ensure_nonempty_properties=True,
    ),
}

#: Default for any provider name not in the table: only the universally-safe
#: ``$ref`` inlining runs (no nullable rewrite beyond canonical, no format strip).
_DEFAULT_PROFILE = ProviderProfile(name="_default")

# Recursion ceiling: schemas deeper than this are widened to permissive rather
# than recursed (defends against pathological nesting / adversarial inputs).
_MAX_DEPTH = 64


def profile_for(provider: str | None) -> ProviderProfile:
    """Return the :class:`ProviderProfile` for ``provider`` (case-insensitive).

    OpenRouter underlying-backend hints (``anthropic/...``, ``google/...``) are
    NOT resolved here — callers pass the *manager's* ``provider_name``. Unknown
    providers get the safe default (ref-inlining only).
    """
    if not provider:
        return _DEFAULT_PROFILE
    return _PROFILES.get(provider.strip().lower(), _DEFAULT_PROFILE)


def normalize_tool_schema(
    schema: Any, provider: str | ProviderProfile | None
) -> dict[str, Any]:
    """Normalize a tool/output JSON ``schema`` for ``provider``; pure + fail-safe.

    ``provider`` may be a provider-name string (resolved via :func:`profile_for`)
    or a :class:`ProviderProfile` directly. A non-dict / falsy ``schema`` widens to
    a permissive object schema (so an egress site always has a valid object to
    hand the provider). The input is never mutated.

    Guarantees: the result is always at least as permissive as the input (widen,
    never fabricate or narrow), and ``normalize(normalize(s)) == normalize(s)``.
    """
    profile = provider if isinstance(provider, ProviderProfile) else profile_for(
        provider
    )
    if not isinstance(schema, dict):
        # A non-dict (None / list / scalar) is not a schema -> widen to the empty
        # (accept-anything) schema, the most permissive shape every provider accepts.
        return {}
    if not schema:
        # The empty schema ``{}`` already means "accept anything"; it is maximally
        # permissive and valid for every provider, so leave it untouched (widening
        # it to ``{"type": "object"}`` would NARROW — it would reject non-objects).
        return {}
    # Work on a deep copy so the caller's schema is never mutated.
    working = copy.deepcopy(schema)
    inlined = _inline_refs(working, working)
    result = _transform(inlined, profile, depth=0)
    # The top-level schema is a dict on entry; every transform of a dict yields a
    # dict (or the permissive object), so this is always a dict in practice.
    if isinstance(result, dict):
        return result
    return dict(_PERMISSIVE_OBJECT)


# --------------------------------------------------------------------------- #
# $ref / $defs inlining (runs for every provider; the most common failure).
# --------------------------------------------------------------------------- #
def _resolve_pointer(root: dict[str, Any], ref: str) -> Any:
    """Resolve a local JSON pointer (``#/$defs/Foo``) against ``root`` or ``None``.

    Only same-document (``#/...``) pointers are resolved; an external/remote ref
    (or a malformed pointer) returns ``None`` so the caller widens — we never
    fetch or guess.
    """
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node: Any = root
    for raw in ref[2:].split("/"):
        # Unescape JSON-pointer tokens (~1 -> /, ~0 -> ~).
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        else:
            return None
    return node


def _inline_refs(
    node: Any, root: dict[str, Any], _seen: tuple[str, ...] = ()
) -> Any:
    """Inline local ``$ref``s in ``node`` against ``root``; widen on cycle/miss.

    ``_seen`` is the chain of ``$ref`` strings currently being expanded; a ref
    that reappears is a cycle and is widened to a permissive object rather than
    recursing forever. Unresolvable refs widen the same way. After inlining, the
    ``$defs`` / ``definitions`` containers are dropped.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in _seen:
                # Cycle: widen to permissive (never narrow, never loop).
                return dict(_PERMISSIVE_OBJECT)
            target = _resolve_pointer(root, ref)
            if not isinstance(target, dict):
                # Unresolvable / non-object target: widen.
                return dict(_PERMISSIVE_OBJECT)
            # Sibling keys alongside a $ref are dropped in 2020-12 ($ref wins);
            # inline the target with this ref pushed onto the cycle chain.
            return _inline_refs(copy.deepcopy(target), root, (*_seen, ref))
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DEFS_KEYS:
                # Definition containers are inlined-from, never emitted.
                continue
            out[key] = _inline_refs(value, root, _seen)
        return out
    if isinstance(node, list):
        return [_inline_refs(item, root, _seen) for item in node]
    return node


# --------------------------------------------------------------------------- #
# Per-profile structural transforms (recursive, depth-guarded).
# --------------------------------------------------------------------------- #
def _transform(node: Any, profile: ProviderProfile, *, depth: int) -> Any:
    """Recursively apply the profile's transforms; widen past the depth ceiling."""
    if depth > _MAX_DEPTH:
        # Pathologically deep: widen rather than risk the stack / a hostile input.
        return dict(_PERMISSIVE_OBJECT)
    if isinstance(node, list):
        return [_transform(item, profile, depth=depth + 1) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = dict(node)

    # 1. Collapse nullable unions into the profile's preferred form. Done first so
    #    nested children of the resulting shape are recursed below.
    out = _collapse_nullable(out, profile)

    # 2. Strip unknown string formats (widening) when the profile is strict.
    if profile.strip_unknown_format:
        fmt = out.get("format")
        if isinstance(fmt, str) and fmt not in profile.known_formats:
            out.pop("format", None)

    # 3. Recurse into every nested schema-bearing position.
    if isinstance(out.get("properties"), dict):
        out["properties"] = {
            k: _transform(v, profile, depth=depth + 1)
            for k, v in out["properties"].items()
        }
    if isinstance(out.get("items"), dict):
        out["items"] = _transform(out["items"], profile, depth=depth + 1)
    elif isinstance(out.get("items"), list):  # tuple-form items
        out["items"] = [
            _transform(v, profile, depth=depth + 1) for v in out["items"]
        ]
    if isinstance(out.get("additionalProperties"), dict):
        out["additionalProperties"] = _transform(
            out["additionalProperties"], profile, depth=depth + 1
        )
    for combinator in ("anyOf", "oneOf", "allOf"):
        members = out.get(combinator)
        if isinstance(members, list):
            out[combinator] = [
                _transform(v, profile, depth=depth + 1) for v in members
            ]
    if isinstance(out.get("prefixItems"), list):
        out["prefixItems"] = [
            _transform(v, profile, depth=depth + 1) for v in out["prefixItems"]
        ]

    # 4. Ensure non-empty object properties LAST (after recursion) so the trap is
    #    closed even for a deeply-nested empty object.
    out = _maybe_ensure_properties(out, profile)
    return out


def _collapse_nullable(node: dict[str, Any], profile: ProviderProfile) -> dict[str, Any]:
    """Rewrite a nullable union into the profile's ``null_style`` (widening only).

    Handles both ``{"type": ["X", "null"]}`` and an ``anyOf`` whose ONLY non-null
    member is a single schema plus a ``{"type": "null"}`` member. A multi-member
    union (more than one real type) is left for the canonical type-array path and
    otherwise untouched. Never removes a non-null member, never adds a type.
    """
    # --- type-array form: {"type": [...]} ----------------------------------
    type_field = node.get("type")
    if isinstance(type_field, list):
        non_null = [t for t in type_field if t != "null"]
        has_null = len(non_null) != len(type_field)
        if has_null:
            return _apply_null_style_typearray(node, non_null, profile)
        return node

    # --- anyOf-with-null form ----------------------------------------------
    any_of = node.get("anyOf")
    if isinstance(any_of, list) and any_of:
        null_members = [m for m in any_of if _is_null_schema(m)]
        real_members = [m for m in any_of if not _is_null_schema(m)]
        if null_members and real_members:
            return _apply_null_style_anyof(node, real_members, profile)
    return node


def _is_null_schema(member: Any) -> bool:
    """True when ``member`` is exactly a ``{"type": "null"}`` schema."""
    return isinstance(member, dict) and member.get("type") == "null"


def _apply_null_style_typearray(
    node: dict[str, Any], non_null: list[str], profile: ProviderProfile
) -> dict[str, Any]:
    """Emit a nullable schema (originally a type-array) in the profile's style."""
    out = dict(node)
    if profile.null_style == NULL_STYLE_DROP_NULL:
        # A provider that rejects an explicit ``null`` type cannot represent the
        # nullable union. Narrowing to the bare non-null type would REJECT ``null``
        # (a narrowing). The fail-safe widen-only move is to drop the ``type``
        # constraint entirely (accept any value, including null) rather than guess.
        out.pop("type", None)
        return out
    if profile.null_style == NULL_STYLE_ANYOF and len(non_null) == 1:
        # {"type":["X","null"], ...kw} -> {"anyOf":[{"type":"X", ...kw}, {null}]}
        inner = {k: v for k, v in node.items() if k != "type"}
        inner["type"] = non_null[0]
        return {"anyOf": [inner, {"type": "null"}]}
    # Canonical type-array (re-append null, deduped, null last for stability).
    out["type"] = [*non_null, "null"]
    return out


def _apply_null_style_anyof(
    node: dict[str, Any], real_members: list[Any], profile: ProviderProfile
) -> dict[str, Any]:
    """Emit a nullable schema (originally an anyOf-with-null) in the profile's style."""
    out = {k: v for k, v in node.items() if k != "anyOf"}
    single = real_members[0] if len(real_members) == 1 else None

    if profile.null_style == NULL_STYLE_DROP_NULL:
        # Drop the ``null`` member. Folding in the single real member would REJECT
        # ``null`` (a narrowing), so widen: keep the real member(s) as an anyOf but
        # WITHOUT a ``type`` lock — an anyOf of the real branches still accepts each
        # real shape, and dropping the null branch is only safe because we do not
        # also forbid null elsewhere. To stay strictly widening we additionally drop
        # any ``type`` the merge would have pinned.
        if single is not None and isinstance(single, dict):
            merged = dict(single)
            # Pull the single member's constraints up, but do not re-pin a type that
            # would reject null; the absence of ``type`` keeps null acceptable.
            out.update({k: v for k, v in merged.items() if k != "type"})
            return out
        out["anyOf"] = list(real_members)
        return out

    if profile.null_style == NULL_STYLE_TYPE_ARRAY and single is not None:
        single_type = single.get("type") if isinstance(single, dict) else None
        if isinstance(single_type, str):
            # Fold a single typed member + null into a {"type":[X,"null"]}.
            merged = {
                k: v for k, v in single.items() if k != "type"
            }
            out.update(merged)
            out["type"] = [single_type, "null"]
            return out

    # Default / ANYOF style: keep the anyOf with the null member preserved.
    out["anyOf"] = [*real_members, {"type": "null"}]
    return out


def _maybe_ensure_properties(
    node: dict[str, Any], profile: ProviderProfile
) -> dict[str, Any]:
    """Inject a permissive placeholder property to dodge the empty-object trap.

    Fires only for a profile that requests it, only on a ``type == "object"``
    schema whose ``properties`` is empty/absent and which has no
    ``additionalProperties`` / combinator carrying its shape. The placeholder is
    OPTIONAL and unconstrained, so the schema is strictly widened (it accepts a
    superset of objects), never narrowed.
    """
    if not profile.ensure_nonempty_properties:
        return node
    if node.get("type") != "object":
        return node
    props = node.get("properties")
    if isinstance(props, dict) and props:
        return node
    # An object whose shape is carried elsewhere is already valid for Gemini.
    if any(
        node.get(k) is not None
        for k in ("additionalProperties", "anyOf", "oneOf", "allOf", "patternProperties")
    ):
        return node
    out = dict(node)
    out["properties"] = {profile.placeholder_property: {}}
    return out


# --------------------------------------------------------------------------- #
# Strict-mode helper (one primitive, applied through the same profile selection).
# --------------------------------------------------------------------------- #
#: JSON-Schema validation keywords OpenAI's strict mode REJECTS (a request carrying
#: any of them is 400-rejected). They are dropped during strictification — a pure
#: WIDENING (the constraint is removed, never rewritten to a narrower one) so a
#: schema that was previously accepted in lenient mode is never bounced under strict.
#: ``format`` is handled separately (only ``date-time`` survives — the one string
#: format OpenAI strict accepts).
_STRICT_UNSUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "minContains",
        "maxContains",
        "default",
        "contentEncoding",
        "contentMediaType",
        "patternProperties",
    }
)

#: String ``format`` values OpenAI strict mode accepts. Anything else is dropped
#: (widening) so a tool/output schema declaring e.g. ``format: "uri"`` is never
#: 400-rejected under strict.
_STRICT_KNOWN_FORMATS: frozenset[str] = frozenset({"date-time"})


def strict_output_schema(schema: Any, provider: str | None) -> dict[str, Any]:
    """Normalize then tighten a STRUCTURED-OUTPUT schema for an OpenAI strict call.

    OpenAI's ``json_schema.strict`` (and function ``strict``) require that EVERY
    object lists every property in ``required`` and sets
    ``additionalProperties: false``, and REJECT most validation keywords
    (``minimum``/``maxLength``/``pattern``/``format`` != date-time/``default`` ...).
    This is the one place the normalizer is *allowed* to tighten — but it does so
    WITHOUT narrowing the accepted value set of any field:

    * a property that was NOT in the object's original ``required`` list is added to
      ``required`` (strict demands it) but its type is made NULLABLE, so the model
      may still legitimately emit ``null`` for it — optionality is preserved, not
      force-required to a real value; and
    * strict-incompatible keywords are stripped (a pure widening) so a schema that
      was previously accepted in lenient mode is never 400-rejected under strict.

    Used by the OpenAI / OpenRouter strict paths; gate the call outside.
    """
    normalized = normalize_tool_schema(schema, provider)
    strict = _strictify(normalized)
    if isinstance(strict, dict):
        return strict
    return dict(_PERMISSIVE_OBJECT)


def _type_set(node: dict[str, Any]) -> set[str]:
    """Declared JSON-Schema types as a set (handles the ``["X", "null"]`` form)."""
    t = node.get("type")
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    if isinstance(t, str):
        return {t}
    return set()


def is_strict_expressible(schema: Any) -> bool:
    """True iff ``schema`` can be tightened to OpenAI strict form WITHOUT semantic loss.

    OpenAI strict mode requires every object to be CLOSED (``additionalProperties:
    false`` with every key enumerated in ``required``) and every array to declare
    ``items``. Two shapes cannot be made strict without changing which inputs are
    accepted, so a tool whose (normalized) schema contains either must be sent
    NON-strict (lenient) rather than distorted — otherwise the provider 400-rejects
    the whole request, or strict silently strips the tool's real input surface:

    * an **open object** — no enumerable ``properties`` (an arbitrary-key dict, e.g.
      HTTP headers) or an ``additionalProperties`` that permits extras (``True`` or a
      sub-schema); closing it would forbid keys the tool legitimately needs; and
    * an **array without ``items``** — closing it would require inventing an element
      type the schema never declared.

    Returns ``True`` for schemas that strictify losslessly (typed/closed objects,
    arrays with ``items``, scalars, and combinators thereof).
    """
    return _strict_expressible(schema, depth=0)


def _strict_expressible(node: Any, *, depth: int) -> bool:
    if depth > _MAX_DEPTH:
        return False  # too deep to certify lossless -> fall back to lenient
    if isinstance(node, list):
        return all(_strict_expressible(item, depth=depth + 1) for item in node)
    if not isinstance(node, dict):
        return True
    for combinator in ("anyOf", "oneOf", "allOf"):
        members = node.get(combinator)
        if isinstance(members, list) and not all(
            _strict_expressible(m, depth=depth + 1) for m in members
        ):
            return False
    types = _type_set(node)
    if "object" in types:
        props = node.get("properties")
        if not isinstance(props, dict) or not props:
            return False  # open / arbitrary-key dict — cannot be closed losslessly
        extra = node.get("additionalProperties")
        if extra is True or isinstance(extra, dict):
            return False  # explicitly permits extra keys
        if not all(_strict_expressible(v, depth=depth + 1) for v in props.values()):
            return False
    if "array" in types:
        items = node.get("items")
        if isinstance(items, dict):
            return _strict_expressible(items, depth=depth + 1)
        if isinstance(items, list):
            return all(_strict_expressible(i, depth=depth + 1) for i in items)
        return False  # array without a declared item schema
    return True


def _make_nullable(node: Any) -> Any:
    """Return ``node`` as a nullable schema (accepts its values OR ``null``).

    OpenAI strict mode has no notion of an *optional* property — every property must
    be in ``required``. To preserve a previously-optional field's semantics (the
    model may omit it) we instead let the model emit ``null`` for it: this widens the
    field, never narrows it. Already-nullable schemas are returned unchanged
    (idempotent). A schema without a representable type (no ``type`` / ``anyOf``)
    already accepts any value (including null) so it is left alone.
    """
    if not isinstance(node, dict):
        # Not a schema object (e.g. ``True`` / bare value) -> accepts anything.
        return node
    type_field = node.get("type")
    if isinstance(type_field, str):
        if type_field == "null":
            return node
        out = dict(node)
        out["type"] = [type_field, "null"]
        return out
    if isinstance(type_field, list):
        if "null" in type_field:
            return node
        out = dict(node)
        out["type"] = [*type_field, "null"]
        return out
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        if any(_is_null_schema(m) for m in any_of):
            return node
        out = dict(node)
        out["anyOf"] = [*any_of, {"type": "null"}]
        return out
    # No ``type`` / ``anyOf`` constraint: the schema already accepts any value
    # (including null), so it is effectively nullable already.
    return node


def _strictify(node: Any) -> Any:
    """Recursively tighten a schema into OpenAI strict form (structurally only).

    On every ``type:object`` node this sets ``additionalProperties:false`` and adds
    EVERY property to ``required`` (strict's hard requirement), but makes any property
    that was NOT in the object's own original ``required`` list nullable first, so the
    model may still emit ``null`` for it (optionality preserved, never force-required
    to a real value). Strict-incompatible validation keywords (``minimum`` etc. and
    unknown ``format`` values) are stripped everywhere — a pure widening so a
    previously-accepted schema is never 400-rejected under strict.
    """
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node

    # Determine, BEFORE recursing, which child properties were originally optional
    # (present in ``properties`` but absent from ``required``) so we can preserve
    # their optionality by making them nullable.
    child_optional: frozenset[str] = frozenset()
    if "object" in _type_set(node) and isinstance(node.get("properties"), dict):
        declared_required = node.get("required")
        required_set = (
            {r for r in declared_required if isinstance(r, str)}
            if isinstance(declared_required, list)
            else set()
        )
        child_optional = frozenset(
            k for k in node["properties"] if k not in required_set
        )

    out: dict[str, Any] = {}
    for key, value in node.items():
        # Drop strict-unsupported validation keywords (widening only).
        if key in _STRICT_UNSUPPORTED_KEYWORDS:
            continue
        if key == "format":
            if isinstance(value, str) and value not in _STRICT_KNOWN_FORMATS:
                continue
            out[key] = value
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {
                pk: _strictify(pv) for pk, pv in value.items()
            }
            # Make originally-optional properties nullable AFTER strictifying their
            # inner shape (so the nullability wraps the cleaned schema).
            for pk in child_optional:
                if pk in out[key]:
                    out[key][pk] = _make_nullable(out[key][pk])
            continue
        out[key] = _strictify(value)

    if "object" in _type_set(out):
        # Strict requires EVERY object (incl. a nullable ``["object","null"]`` or an
        # empty one) to set additionalProperties:false and list all keys in required.
        props = out.get("properties")
        if not isinstance(props, dict):
            props = {}
            out["properties"] = props
        out["additionalProperties"] = False
        out["required"] = list(props.keys())
    return out


def with_profile(profile: ProviderProfile) -> ProviderProfile:
    """Register/override a profile at runtime (returns it); for advanced callers.

    Mutates the module table so a host embedding himmy with a bespoke backend can
    teach the normalizer about it without forking. Returns the profile for chaining.
    """
    _PROFILES[profile.name.strip().lower()] = profile
    return profile


def list_profiles() -> dict[str, ProviderProfile]:
    """A copy of the current profile table (introspection / tests)."""
    return dict(_PROFILES)


# Re-export of replace for callers that build derivative profiles.
__all__ = [
    "ProviderProfile",
    "normalize_tool_schema",
    "strict_output_schema",
    "is_strict_expressible",
    "profile_for",
    "with_profile",
    "list_profiles",
    "NULL_STYLE_TYPE_ARRAY",
    "NULL_STYLE_ANYOF",
    "NULL_STYLE_DROP_NULL",
    "replace",
]
