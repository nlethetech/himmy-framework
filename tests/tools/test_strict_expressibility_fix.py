"""Regression tests for the per-tool strict-expressibility fix.

The cross-provider normalizer's OpenAI strict path used to 400-reject two built-in
tool packs under an OpenAI-family model: an OPEN dict param (e.g. http_request's
``headers``) strictified to an object with no ``additionalProperties: false``, and an
items-less array (sql_query's ``params``). The fix makes strict a PER-TOOL decision
gated on lossless expressibility, and hardens ``_strictify`` to close nullable
(``["object","null"]``) objects too.
"""

from __future__ import annotations

from himmy.services.inference.models import BoundTool
from himmy.services.inference.openai_manager import _openai_tools
from himmy.services.tools.schema_normalize import (
    is_strict_expressible,
    strict_output_schema,
)

# --------------------------------------------------------- is_strict_expressible


def test_closed_typed_object_is_expressible() -> None:
    assert is_strict_expressible(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    )


def test_open_dict_is_not_expressible() -> None:
    # An object with no enumerable properties = arbitrary-key dict (e.g. HTTP headers).
    assert not is_strict_expressible({"type": "object"})
    assert not is_strict_expressible({"type": "object", "properties": {}})


def test_additional_properties_open_is_not_expressible() -> None:
    assert not is_strict_expressible(
        {"type": "object", "properties": {"a": {"type": "string"}},
         "additionalProperties": True}
    )
    assert not is_strict_expressible(
        {"type": "object", "properties": {"a": {"type": "string"}},
         "additionalProperties": {"type": "string"}}
    )


def test_array_without_items_is_not_expressible() -> None:
    assert not is_strict_expressible({"type": "array"})


def test_array_with_items_is_expressible() -> None:
    assert is_strict_expressible({"type": "array", "items": {"type": "string"}})


def test_nested_open_dict_makes_whole_schema_inexpressible() -> None:
    # The http_request shape: a closed top-level object whose `headers` is an open dict.
    schema = {
        "type": "object",
        "properties": {"headers": {"type": "object"}},
        "required": ["headers"],
    }
    assert not is_strict_expressible(schema)


def test_nullable_typed_object_is_expressible() -> None:
    assert is_strict_expressible(
        {"type": ["object", "null"], "properties": {"a": {"type": "string"}},
         "required": ["a"]}
    )


# --------------------------------------------------------- _strictify completeness


def test_strictify_closes_object_with_nullable_type_list() -> None:
    """A nested object whose type is the LIST ``["object","null"]`` must still close."""
    schema = {
        "type": "object",
        "properties": {
            "obj": {
                "type": ["object", "null"],
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            }
        },
        "required": ["obj"],
    }
    out = strict_output_schema(schema, "openrouter")
    assert out["additionalProperties"] is False
    obj = out["properties"]["obj"]
    # The pre-fix bug: a ["object","null"] node never got additionalProperties:false.
    assert obj.get("additionalProperties") is False


# --------------------------------------------------------- per-tool strict gate


def test_open_dict_tool_falls_back_to_lenient_clean_tool_stays_strict() -> None:
    open_tool = BoundTool(
        name="http_request",
        description="make an http request",
        args_json_schema={
            "type": "object",
            "properties": {"headers": {"type": "object"}},  # open dict
            "required": ["headers"],
        },
    )
    array_tool = BoundTool(
        name="sql_query",
        description="run a query",
        args_json_schema={
            "type": "object",
            "properties": {"params": {"type": "array"}},  # items-less array
            "required": ["params"],
        },
    )
    clean_tool = BoundTool(
        name="calc",
        description="calculate",
        args_json_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    )
    fns = {
        t["function"]["name"]: t["function"]
        for t in _openai_tools(
            [open_tool, array_tool, clean_tool], provider="openrouter", strict=True
        )
    }
    # Inexpressible tools fall back to lenient (NO strict flag) -> no provider 400.
    assert "strict" not in fns["http_request"]
    assert "strict" not in fns["sql_query"]
    # An expressible tool still goes strict + closed.
    assert fns["calc"].get("strict") is True
    assert fns["calc"]["parameters"]["additionalProperties"] is False


def test_strict_false_never_marks_any_tool_strict() -> None:
    clean_tool = BoundTool(
        name="calc",
        description="calculate",
        args_json_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    )
    fns = _openai_tools([clean_tool], provider="openrouter", strict=False)
    assert "strict" not in fns[0]["function"]
