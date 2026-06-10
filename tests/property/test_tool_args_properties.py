"""Property tests: tool-arg lenient coercion + JSON-schema validation (tools kernel).

Hypothesis fuzzes the seam small models hit hardest: arbitrary JSON-ish argument
dicts must never crash the validator or the lenient coercion, already-valid args
must stay valid through coercion, and coercion must never INVENT keys or values
(it only ever drops). Skips gracefully when ``hypothesis`` is not installed
(it is a dev-only dependency).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from himmy.services.tools.repair import (  # noqa: E402
    resolve_tool_name,
    unknown_tool_message,
)
from himmy.services.tools.service import _coerce_lenient_args  # noqa: E402
from himmy.services.tools.validation import validate_against_schema  # noqa: E402

# Moderate example counts, no deadline, derandomized => fast and seed-stable in CI.
_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True, database=None)

_JSON_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**9), max_value=10**9)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=20)
)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=8), children, max_size=3)
    ),
    max_leaves=10,
)

_PROP_NAMES = ("alpha", "beta", "gamma", "delta")
#: ``None`` = an untyped property spec (every JSON value is valid for it).
_PROP_TYPES = (
    "string",
    "integer",
    "number",
    "boolean",
    "array",
    "object",
    "null",
    None,
)


@st.composite
def _object_schemas(draw: st.DrawFn) -> dict[str, Any]:
    """A realistic tool-args object schema: typed properties, required, extras."""
    names = draw(
        st.lists(st.sampled_from(_PROP_NAMES), unique=True, min_size=1, max_size=4)
    )
    properties: dict[str, Any] = {}
    for name in names:
        ptype = draw(st.sampled_from(_PROP_TYPES))
        spec: dict[str, Any] = {} if ptype is None else {"type": ptype}
        if ptype == "string" and draw(st.booleans()):
            spec["minLength"] = draw(st.integers(min_value=0, max_value=3))
        if ptype in ("integer", "number") and draw(st.booleans()):
            spec["minimum"] = draw(st.integers(min_value=-5, max_value=5))
        properties[name] = spec
    required = draw(st.lists(st.sampled_from(names), unique=True, max_size=len(names)))
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    additional = draw(st.sampled_from((True, False, None)))
    if additional is not None:
        schema["additionalProperties"] = additional
    return schema


def _valid_value_for(draw: st.DrawFn, spec: dict[str, Any]) -> Any:
    """Draw a value that satisfies one property spec from :func:`_object_schemas`."""
    ptype = spec.get("type")
    if ptype == "string":
        return draw(st.text(min_size=spec.get("minLength", 0), max_size=12))
    if ptype == "integer":
        return draw(st.integers(min_value=spec.get("minimum", -100), max_value=100))
    if ptype == "number":
        minimum = spec.get("minimum", -100)
        return draw(
            st.floats(min_value=minimum, max_value=minimum + 200, allow_nan=False)
        )
    if ptype == "boolean":
        return draw(st.booleans())
    if ptype == "array":
        return draw(st.lists(_JSON_SCALARS, max_size=3))
    if ptype == "object":
        return draw(st.dictionaries(st.text(max_size=5), _JSON_SCALARS, max_size=3))
    if ptype == "null":
        return None
    return draw(_JSON_VALUES)  # untyped property: anything goes


@st.composite
def _schemas_with_valid_args(
    draw: st.DrawFn,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """An object schema plus an args dict that validates against it."""
    schema = draw(_object_schemas())
    required = set(schema["required"])
    args: dict[str, Any] = {}
    for name, spec in schema["properties"].items():
        if name in required or draw(st.booleans()):
            args[name] = _valid_value_for(draw, spec)
    return schema, args


@_SETTINGS
@given(value=_JSON_VALUES, schema=_object_schemas())
def test_validator_is_total_over_arbitrary_json(
    value: Any, schema: dict[str, Any]
) -> None:
    """Arbitrary JSON-ish instances never crash either validator path."""
    for use_jsonschema in (True, False):
        outcome = validate_against_schema(value, schema, use_jsonschema=use_jsonschema)
        assert outcome is None or isinstance(outcome, str)


@_SETTINGS
@given(case=_schemas_with_valid_args())
def test_valid_args_round_trip_through_lenient_coercion(
    case: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Schema-valid args stay valid after coercion; coercion only ever drops."""
    schema, args = case
    # Sanity: the generated args really are valid under both validator paths.
    assert validate_against_schema(args, schema, use_jsonschema=False) is None
    assert validate_against_schema(args, schema, use_jsonschema=True) is None
    coerced = _coerce_lenient_args(args, schema)
    assert validate_against_schema(coerced, schema, use_jsonschema=False) is None
    # Never invents: every coerced entry existed identically in the input.
    assert set(coerced) <= set(args)
    assert all(coerced[key] == args[key] for key in coerced)
    # Required keys are never stripped.
    assert set(schema["required"]) <= set(coerced)


@_SETTINGS
@given(
    args=st.dictionaries(
        st.sampled_from(_PROP_NAMES) | st.text(max_size=12),
        _JSON_VALUES,
        max_size=5,
    ),
    schema=_object_schemas(),
)
def test_coercion_is_total_subsetting_and_idempotent(
    args: dict[str, Any], schema: dict[str, Any]
) -> None:
    """Arbitrary args never crash coercion; output ⊆ input; fixed point reached."""
    coerced = _coerce_lenient_args(args, schema)
    assert isinstance(coerced, dict)
    assert set(coerced) <= set(args)
    assert all(coerced[key] == args[key] for key in coerced)
    assert _coerce_lenient_args(coerced, schema) == coerced


@_SETTINGS
@given(
    name=st.text(max_size=24),
    available=st.lists(st.text(min_size=1, max_size=16), unique=True, max_size=6),
)
def test_tool_name_resolution_is_total_and_sound(
    name: str, available: list[str]
) -> None:
    """Resolution never crashes, never resolves outside ``available``."""
    resolution = resolve_tool_name(name, available)
    assert resolution.original == name
    assert resolution.name is None or resolution.name in available
    assert all(suggestion in available for suggestion in resolution.suggestions)
    message = unknown_tool_message(resolution, available)
    assert repr(name) in message


@_SETTINGS
@given(
    available=st.lists(
        st.text(min_size=1, max_size=16), unique=True, min_size=1, max_size=6
    ),
    data=st.data(),
)
def test_exact_tool_name_resolves_to_itself(
    available: list[str], data: st.DataObject
) -> None:
    """An exact name always resolves as-is, never flagged as a correction."""
    name = data.draw(st.sampled_from(available))
    resolution = resolve_tool_name(name, available)
    assert resolution.name == name
    assert not resolution.auto_corrected
