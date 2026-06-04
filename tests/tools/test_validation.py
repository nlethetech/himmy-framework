"""Tests for the tools-kernel JSON-schema validator (offline subset + delegation)."""

from __future__ import annotations

from opensims.services.tools import build_arg_model
from opensims.services.tools.validation import _validate_subset, validate_against_schema


def test_subset_type_and_required() -> None:
    """The offline subset enforces type and required keywords."""
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer"}},
    }
    assert _validate_subset({"a": 1}, schema) is None
    assert _validate_subset({}, schema) is not None
    assert _validate_subset({"a": "x"}, schema) is not None


def test_subset_additional_properties_false() -> None:
    """additionalProperties:false rejects unexpected keys (per the doc example)."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
    }
    assert _validate_subset({"a": "ok"}, schema) is None
    assert _validate_subset({"a": "ok", "b": 1}, schema) is not None


def test_subset_const_and_enum() -> None:
    """const and enum are honored by the offline subset."""
    assert _validate_subset("X", {"const": "X"}) is None
    assert _validate_subset("Y", {"const": "X"}) is not None
    assert _validate_subset("b", {"enum": ["a", "b"]}) is None
    assert _validate_subset("c", {"enum": ["a", "b"]}) is not None


def test_subset_numeric_and_string_constraints() -> None:
    """minimum/maximum and minLength/maxLength/pattern are checked."""
    assert _validate_subset(5, {"type": "integer", "minimum": 10}) is not None
    assert _validate_subset(15, {"type": "integer", "minimum": 10}) is None
    assert _validate_subset("ab", {"type": "string", "minLength": 3}) is not None
    assert _validate_subset("abcd", {"type": "string", "maxLength": 3}) is not None
    assert _validate_subset("abc", {"type": "string", "pattern": "^a.c$"}) is None


def test_subset_oneof_anyof_allof() -> None:
    """Combinators behave per JSON-schema semantics in the offline subset."""
    one_of = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert _validate_subset("x", one_of) is None
    assert _validate_subset(1, one_of) is None
    assert _validate_subset([], one_of) is not None  # matches neither
    any_of = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert _validate_subset(True is False and 0 or "ok", any_of) is None


def test_subset_array_constraints() -> None:
    """items, minItems, and maxItems are enforced for arrays."""
    schema = {"type": "array", "items": {"type": "integer"}, "minItems": 2}
    assert _validate_subset([1, 2], schema) is None
    assert _validate_subset([1], schema) is not None
    assert _validate_subset([1, "x"], schema) is not None


def test_validate_against_schema_public_entry() -> None:
    """The public entry validates a valid value as None and an invalid one not-None."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    assert validate_against_schema({"x": "ok"}, schema) is None
    assert validate_against_schema({"x": 1}, schema) is not None


def test_build_arg_model_from_object_schema() -> None:
    """build_arg_model derives a pydantic model with required/optional fields."""
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
    }
    model = build_arg_model("my_tool", schema)
    instance = model(a=1)
    assert instance.a == 1
    assert instance.b is None
    # Missing required field raises.
    try:
        model(b="x")
        raised = False
    except Exception:
        raised = True
    assert raised


def test_build_arg_model_empty_schema() -> None:
    """A tool with no object schema builds an empty (no-field) model."""
    model = build_arg_model("noargs", {})
    assert model().model_dump() == {}


def test_falls_back_to_subset_when_jsonschema_absent(monkeypatch) -> None:
    """With jsonschema unavailable, validation still works via the offline subset."""
    import opensims.services.tools.validation as v

    monkeypatch.setattr(v, "_JSONSCHEMA_MODULE", False)
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": False,
    }
    assert v.validate_against_schema({"a": 1}, schema) is None
    assert v.validate_against_schema({"a": 1, "x": 2}, schema) is not None
    assert v.validate_against_schema({}, schema) is not None
