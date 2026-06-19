"""Adversarial fixture-matrix tests for the cross-provider schema normalizer.

These tests are fully OFFLINE (pure dict-in / dict-out) and exercise every core
transform against every provider profile, asserting:

* each provider profile's *structural invariants* (nullable form, no ``$ref``
  survives, no empty-properties object survives for Gemini, etc.);
* the **fail-safe / widen-only** contract — normalization never makes a schema
  reject a value the original schema accepted (checked by validating a battery of
  sample values against both the original and the normalized schema);
* **idempotence** — ``normalize(normalize(s)) == normalize(s)``.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from himmy.services.tools.schema_normalize import (
    NULL_STYLE_ANYOF,
    NULL_STYLE_DROP_NULL,
    NULL_STYLE_TYPE_ARRAY,
    ProviderProfile,
    list_profiles,
    normalize_tool_schema,
    profile_for,
    strict_output_schema,
    with_profile,
)
from himmy.services.tools.validation import validate_against_schema

# Every distinct provider key the managers pass as ``provider_name``.
PROVIDERS: tuple[str, ...] = (
    "openai",
    "openrouter",
    "anthropic",
    "ollama",
    "claude-cli",
    "himalayagpt",
    "pydantic_ai",
    "gemini",
    "google",
    None,  # the safe default profile
)

# A battery of sample values used to prove widening: if the ORIGINAL schema
# accepts a value, the NORMALIZED schema must accept it too.
_SAMPLE_VALUES: tuple[Any, ...] = (
    None,
    True,
    False,
    0,
    1,
    -3,
    3.14,
    "",
    "hello",
    "2020-01-01",
    [],
    [1, 2, 3],
    ["a", "b"],
    {},
    {"x": 1},
    {"a": "s", "b": 2},
    {"name": "n", "age": 30},
)


def _accepts(schema: dict[str, Any], value: Any) -> bool:
    """True when ``schema`` accepts ``value`` (offline subset, no jsonschema)."""
    return validate_against_schema(value, schema, use_jsonschema=False) is None


def _assert_widens(original: dict[str, Any], provider: str | None) -> dict[str, Any]:
    """Normalize and assert the result accepts every value the original accepts."""
    normalized = normalize_tool_schema(original, provider)
    for value in _SAMPLE_VALUES:
        if _accepts(original, value):
            assert _accepts(normalized, value), (
                f"provider={provider!r}: normalization NARROWED — original accepted "
                f"{value!r} but normalized rejected it.\n"
                f"original={original}\nnormalized={normalized}"
            )
    return normalized


def _no_ref_anywhere(node: Any) -> bool:
    """True when no ``$ref`` / ``$defs`` / ``definitions`` survives anywhere."""
    if isinstance(node, dict):
        if "$ref" in node or "$defs" in node or "definitions" in node:
            return False
        return all(_no_ref_anywhere(v) for v in node.values())
    if isinstance(node, list):
        return all(_no_ref_anywhere(v) for v in node)
    return True


# --------------------------------------------------------------------------- #
# 1. $ref / $defs inlining
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("provider", PROVIDERS)
def test_ref_inlining(provider: str | None) -> None:
    """Local ``$ref``/``$defs`` are inlined and the container is dropped, everywhere."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"$ref": "#/$defs/Str"},
            "b": {"type": "array", "items": {"$ref": "#/$defs/Num"}},
        },
        "$defs": {"Str": {"type": "string"}, "Num": {"type": "integer"}},
    }
    normalized = _assert_widens(schema, provider)
    assert _no_ref_anywhere(normalized), normalized
    assert normalized["properties"]["a"]["type"] == "string"
    assert normalized["properties"]["b"]["items"]["type"] == "integer"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_ref_definitions_legacy_key(provider: str | None) -> None:
    """The legacy ``definitions`` container is inlined + stripped like ``$defs``."""
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/Foo"}},
        "definitions": {"Foo": {"type": "string"}},
    }
    normalized = _assert_widens(schema, provider)
    assert _no_ref_anywhere(normalized)
    assert normalized["properties"]["a"]["type"] == "string"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_ref_cycle_widens_not_loops(provider: str | None) -> None:
    """A recursive ``$ref`` cycle widens to a permissive object (no infinite loop)."""
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
    }
    normalized = _assert_widens(schema, provider)
    assert _no_ref_anywhere(normalized)
    # The recursive position collapsed to a permissive object (no constraints
    # beyond what the profile must add to make an empty object acceptable).
    child = normalized["properties"]["node"]["properties"]["child"]
    assert child.get("type") == "object"
    # No fabricated constraints (required / enum / minimum / additionalProperties).
    assert "required" not in child
    assert "enum" not in child


@pytest.mark.parametrize("provider", PROVIDERS)
def test_unresolvable_ref_widens(provider: str | None) -> None:
    """An unresolvable / external ``$ref`` widens to permissive rather than failing."""
    for ref in ("#/$defs/Missing", "https://example.com/schema.json", "not-a-pointer"):
        schema = {"type": "object", "properties": {"x": {"$ref": ref}}}
        normalized = normalize_tool_schema(schema, provider)
        assert _no_ref_anywhere(normalized)
        widened = normalized["properties"]["x"]
        assert widened.get("type") == "object"
        assert "required" not in widened and "enum" not in widened


# --------------------------------------------------------------------------- #
# 2. nullable-union collapse (per-profile form)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("provider", PROVIDERS)
def test_null_union_type_array(provider: str | None) -> None:
    """``type:[X,"null"]`` collapses into the profile's nullable form, always widening."""
    schema = {"type": ["string", "null"]}
    normalized = _assert_widens(schema, provider)
    style = profile_for(provider).null_style
    if style == NULL_STYLE_DROP_NULL:
        # Widen-only: the type lock is dropped entirely (still accepts null + string).
        assert "type" not in normalized
    elif style == NULL_STYLE_ANYOF:
        assert {"type": "null"} in normalized["anyOf"]
        assert any(m.get("type") == "string" for m in normalized["anyOf"])
    else:  # type-array canonical
        assert set(normalized["type"]) == {"string", "null"}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_anyof_with_null_collapse(provider: str | None) -> None:
    """``anyOf`` with a ``{"type":"null"}`` member collapses to the profile's form."""
    schema = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    normalized = _assert_widens(schema, provider)
    style = profile_for(provider).null_style
    if style == NULL_STYLE_DROP_NULL:
        # null member dropped, but the type lock is NOT re-pinned (widen-only).
        assert "type" not in normalized
        assert "anyOf" not in normalized
    elif style == NULL_STYLE_TYPE_ARRAY:
        assert set(normalized["type"]) == {"integer", "null"}
    else:  # anyOf preserved
        assert {"type": "null"} in normalized["anyOf"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_nested_null_union_in_properties(provider: str | None) -> None:
    """A nullable union nested inside ``properties`` is collapsed too."""
    schema = {
        "type": "object",
        "properties": {"opt": {"type": ["number", "null"], "minimum": 0}},
    }
    normalized = _assert_widens(schema, provider)
    opt = normalized["properties"]["opt"]
    # The numeric keyword survives somewhere in the collapsed shape (either at the
    # top of the property for type-array/drop styles, or inside the anyOf branch).
    has_minimum = opt.get("minimum") == 0 or any(
        isinstance(m, dict) and m.get("minimum") == 0 for m in opt.get("anyOf", [])
    )
    assert has_minimum, opt
    assert _no_ref_anywhere(normalized)


# --------------------------------------------------------------------------- #
# 3. empty-properties Gemini trap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("provider", PROVIDERS)
def test_empty_properties_object(provider: str | None) -> None:
    """A ``type:object`` with no properties gets a placeholder ONLY for strict profiles."""
    schema = {"type": "object"}
    normalized = _assert_widens(schema, provider)
    if profile_for(provider).ensure_nonempty_properties:
        assert normalized.get("properties"), normalized
        # The placeholder is optional + unconstrained (pure widening).
        assert "required" not in normalized
        placeholder = profile_for(provider).placeholder_property
        assert normalized["properties"] == {placeholder: {}}
    else:
        assert "properties" not in normalized or not normalized["properties"]


@pytest.mark.parametrize("provider", ("gemini", "google"))
def test_nested_empty_properties_object(provider: str) -> None:
    """A deeply-nested empty object also gets the placeholder for strict profiles."""
    schema = {
        "type": "object",
        "properties": {"inner": {"type": "object"}},
    }
    normalized = _assert_widens(schema, provider)
    assert normalized["properties"]["inner"]["properties"], normalized


@pytest.mark.parametrize("provider", ("gemini", "google"))
def test_object_with_additionalproperties_not_padded(provider: str) -> None:
    """An object carrying its shape via ``additionalProperties`` keeps no placeholder."""
    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    normalized = _assert_widens(schema, provider)
    assert "properties" not in normalized or "_" not in normalized.get("properties", {})


# --------------------------------------------------------------------------- #
# 4. unknown string format stripping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("provider", PROVIDERS)
def test_unknown_format_stripped_for_strict(provider: str | None) -> None:
    """An unknown ``format`` is dropped for strict profiles, kept otherwise."""
    schema = {"type": "string", "format": "date"}  # 'date' unsupported by Gemini tools
    normalized = _assert_widens(schema, provider)
    if profile_for(provider).strip_unknown_format:
        assert "format" not in normalized, normalized
    else:
        assert normalized.get("format") == "date"


@pytest.mark.parametrize("provider", ("gemini", "google"))
def test_known_format_kept_for_strict(provider: str) -> None:
    """A known format (``date-time``) is preserved even for strict profiles."""
    schema = {"type": "string", "format": "date-time"}
    normalized = normalize_tool_schema(schema, provider)
    assert normalized.get("format") == "date-time"


# --------------------------------------------------------------------------- #
# 5. deep nesting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("provider", PROVIDERS)
def test_deep_nesting_handled(provider: str | None) -> None:
    """A deeply-nested schema is transformed throughout (refs gone, nulls collapsed)."""
    # Build a 12-deep object chain ending in a nullable string behind a $ref.
    schema: dict[str, Any] = {"$ref": "#/$defs/Leaf"}
    defs: dict[str, Any] = {"Leaf": {"type": ["string", "null"]}}
    node = schema
    for _ in range(12):
        wrapper = {"type": "object", "properties": {"child": node}}
        node = wrapper
    root = {**node, "$defs": defs}
    normalized = _assert_widens(root, provider)
    assert _no_ref_anywhere(normalized)

    # Walk to the leaf and confirm the nullable collapse reached it.
    cur = normalized
    for _ in range(12):
        cur = cur["properties"]["child"]
    style = profile_for(provider).null_style
    if style == NULL_STYLE_DROP_NULL:
        assert "type" not in cur
    elif style == NULL_STYLE_ANYOF:
        assert "anyOf" in cur
    else:
        assert set(cur["type"]) == {"string", "null"}


def test_pathological_depth_widens() -> None:
    """A schema deeper than the recursion ceiling widens instead of recursing forever."""
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    cur = schema
    for _ in range(200):
        child: dict[str, Any] = {"type": "object", "properties": {}}
        cur["properties"] = {"c": child}
        cur = child
    # Must not raise (RecursionError) and must return a dict.
    out = normalize_tool_schema(schema, "openai")
    assert isinstance(out, dict)


# --------------------------------------------------------------------------- #
# Cross-cutting invariants
# --------------------------------------------------------------------------- #
_FIXTURES: tuple[dict[str, Any], ...] = (
    {"type": "object", "properties": {"a": {"$ref": "#/$defs/S"}}, "$defs": {"S": {"type": "string"}}},
    {"type": ["string", "null"]},
    {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    {"type": "object"},
    {"type": "string", "format": "date"},
    {"type": "string", "format": "date-time"},
    {
        "type": "object",
        "required": ["a"],
        "properties": {
            "a": {"type": "string", "minLength": 1},
            "b": {"type": ["integer", "null"]},
            "c": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
        },
        "$defs": {"Item": {"type": "object", "properties": {"k": {"type": "string"}}}},
    },
    {},
    {"type": "boolean"},
)


@pytest.mark.parametrize(
    "fixture,provider", list(itertools.product(_FIXTURES, PROVIDERS))
)
def test_idempotence(fixture: dict[str, Any], provider: str | None) -> None:
    """``normalize(normalize(s)) == normalize(s)`` for every fixture x provider."""
    once = normalize_tool_schema(fixture, provider)
    twice = normalize_tool_schema(once, provider)
    assert once == twice, f"not idempotent for provider={provider!r}: {fixture}"


@pytest.mark.parametrize(
    "fixture,provider", list(itertools.product(_FIXTURES, PROVIDERS))
)
def test_input_never_mutated(fixture: dict[str, Any], provider: str | None) -> None:
    """The caller's schema is never mutated in place."""
    import copy

    snapshot = copy.deepcopy(fixture)
    normalize_tool_schema(fixture, provider)
    assert fixture == snapshot


@pytest.mark.parametrize(
    "fixture,provider", list(itertools.product(_FIXTURES, PROVIDERS))
)
def test_global_widening_invariant(
    fixture: dict[str, Any], provider: str | None
) -> None:
    """Across the whole fixture matrix, normalization only ever WIDENS."""
    _assert_widens(fixture, provider)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_non_dict_schema_widens_to_permissive(provider: str | None) -> None:
    """A falsy / non-dict schema widens to the maximally-permissive empty schema.

    ``{}`` (accept-anything) is the safe, non-narrowing result — pinning a
    ``type: object`` would REJECT non-object values the absent schema accepted.
    """
    for bad in (None, {}, "not a schema", 123, []):
        out = normalize_tool_schema(bad, provider)
        assert isinstance(out, dict)
        assert out == {}
        # The empty schema accepts every sample value (it is maximally permissive).
        for value in _SAMPLE_VALUES:
            assert _accepts(out, value)


# --------------------------------------------------------------------------- #
# Strict-output mode (OpenAI / OpenRouter strict structured output + tool args)
# --------------------------------------------------------------------------- #
def test_strict_output_forces_required_and_no_additional() -> None:
    """Strict mode lists every property in ``required`` + sets additionalProperties:false."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "object", "properties": {"c": {"type": "integer"}}},
        },
    }
    strict = strict_output_schema(schema, "openai")
    assert strict["additionalProperties"] is False
    assert sorted(strict["required"]) == ["a", "b"]
    inner = strict["properties"]["b"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["c"]


def test_strict_output_still_inlines_refs() -> None:
    """Strict mode runs the full normalizer first (so ``$ref`` is gone)."""
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/S"}},
        "required": ["a"],  # required -> inlined type stays the plain scalar
        "$defs": {"S": {"type": "string"}},
    }
    strict = strict_output_schema(schema, "openai")
    assert _no_ref_anywhere(strict)
    assert strict["properties"]["a"]["type"] == "string"


def test_strict_output_empty_object_no_required() -> None:
    """An object with no properties stays valid under strict (no spurious required)."""
    strict = strict_output_schema({"type": "object"}, "openai")
    # No properties -> no additionalProperties:false / required injected.
    assert "required" not in strict or strict["required"] == []


def test_strict_output_preserves_optional_field_as_nullable() -> None:
    """A field absent from ``required`` is kept optional by being made NULLABLE.

    OpenAI strict has no optional properties — every one must be in ``required``.
    To avoid FORCING a previously-optional field to a real value, strictification
    adds it to ``required`` but makes its type nullable so the model may emit
    ``null``. This pins the websearch/news/nrb-style ``required:[query]`` + optional
    integer contract: ``query`` stays a plain string, the optional field stays
    null-acceptable.
    """
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    strict = strict_output_schema(schema, "openai")

    # Strict demands every property in required + additionalProperties:false.
    assert sorted(strict["required"]) == ["count", "query"]
    assert strict["additionalProperties"] is False

    # The originally-required field is untouched (still a plain non-null string).
    assert strict["properties"]["query"]["type"] == "string"

    # The originally-OPTIONAL field is now nullable (model may legitimately emit
    # null) rather than force-required to a real integer.
    count = strict["properties"]["count"]
    assert "null" in count["type"]
    assert "integer" in count["type"]


def test_strict_output_strips_unsupported_keywords() -> None:
    """Strict mode drops keywords OpenAI's strict engine 400-rejects (widening only).

    A schema declaring ``minimum``/``maximum``/``minLength``/``pattern``/``default``
    and a non-``date-time`` ``format`` would be rejected by OpenAI under strict; the
    strictifier strips them so a previously-accepted schema is never bounced.
    """
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            "s": {
                "type": "string",
                "minLength": 1,
                "maxLength": 50,
                "pattern": "^x",
                "format": "uri",
            },
            "when": {"type": "string", "format": "date-time"},
        },
        "required": ["n", "s", "when"],
    }
    strict = strict_output_schema(schema, "openai")

    n = strict["properties"]["n"]
    assert "minimum" not in n and "maximum" not in n and "default" not in n
    s = strict["properties"]["s"]
    for k in ("minLength", "maxLength", "pattern", "format"):
        assert k not in s, (k, s)
    # The one strict-supported format survives.
    assert strict["properties"]["when"]["format"] == "date-time"


def test_strict_output_optional_nullable_is_idempotent() -> None:
    """An already-nullable optional field is not double-wrapped (idempotent)."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": ["string", "null"]}},
        "required": [],
    }
    strict = strict_output_schema(schema, "openai")
    x = strict["properties"]["x"]
    assert x["type"].count("null") == 1
    assert set(x["type"]) == {"string", "null"}


# --------------------------------------------------------------------------- #
# Profile table introspection / extensibility
# --------------------------------------------------------------------------- #
def test_profile_for_unknown_is_safe_default() -> None:
    """An unknown provider falls back to the safe (ref-inlining-only) default."""
    prof = profile_for("some-future-backend")
    assert prof.null_style == NULL_STYLE_TYPE_ARRAY
    assert prof.strip_unknown_format is False
    assert prof.ensure_nonempty_properties is False


def test_profile_for_is_case_insensitive() -> None:
    """Provider lookup is case-insensitive and whitespace-tolerant."""
    assert profile_for("  GEMINI ").name == "gemini"
    assert profile_for("OpenAI").name == "openai"


def test_known_profiles_present() -> None:
    """Every manager ``provider_name`` resolves to a concrete profile row."""
    table = list_profiles()
    for name in (
        "openai",
        "openrouter",
        "anthropic",
        "ollama",
        "claude-cli",
        "himalayagpt",
        "pydantic_ai",
        "gemini",
        "google",
    ):
        assert name in table


def test_with_profile_registers_runtime_override() -> None:
    """A host can register a bespoke backend profile at runtime."""
    import himmy.services.tools.schema_normalize as mod

    custom = ProviderProfile(
        name="my-edge-llm",
        null_style=NULL_STYLE_ANYOF,
        strip_unknown_format=True,
    )
    try:
        with_profile(custom)
        assert profile_for("my-edge-llm").null_style == NULL_STYLE_ANYOF
        out = normalize_tool_schema({"type": "string", "format": "weird"}, "my-edge-llm")
        assert "format" not in out
    finally:
        mod._PROFILES.pop("my-edge-llm", None)
