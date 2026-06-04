"""Tools kernel: JSON-schema validation for tool inputs and outputs.

The framework validates both incoming tool arguments and tool outputs against
their declared JSON schemas. By default a self-contained offline validator runs
(no dependency beyond the stdlib); when the optional ``jsonschema`` library is
importable it is delegated to for full Draft-2020-12 fidelity. Both paths are
import-safe — ``jsonschema`` is imported lazily inside the delegating helper.
"""

from __future__ import annotations

import re
from typing import Any

# Cache the lazy ``jsonschema`` import result: ``None`` = not yet probed,
# ``False`` = probed and absent, otherwise the imported module.
_JSONSCHEMA_MODULE: Any = None


def _load_jsonschema() -> Any:
    """Lazily import ``jsonschema`` once; return the module or ``None``."""
    global _JSONSCHEMA_MODULE
    if _JSONSCHEMA_MODULE is None:
        try:
            import jsonschema  # type: ignore

            _JSONSCHEMA_MODULE = jsonschema
        except Exception:  # noqa: BLE001 - any import failure means "absent"
            _JSONSCHEMA_MODULE = False
    return _JSONSCHEMA_MODULE or None


def validate_against_schema(
    value: Any, schema: dict[str, Any], *, use_jsonschema: bool = True
) -> str | None:
    """Validate ``value`` against ``schema``; return an error string or ``None``.

    When ``jsonschema`` is installed (and ``use_jsonschema`` is true) it is used
    for full fidelity. Otherwise the built-in offline subset runs. The offline
    subset honors: ``type`` (object/array/string/number/integer/boolean/null),
    ``required``, nested ``properties``, ``additionalProperties`` (incl. ``false``
    and sub-schema), array ``items``/``minItems``/``maxItems``, ``enum``,
    ``const``, numeric ``minimum``/``maximum``/``exclusiveMinimum``/
    ``exclusiveMaximum``, string ``minLength``/``maxLength``/``pattern``, and
    ``oneOf``/``anyOf``/``allOf``. Unknown keywords are ignored permissively.
    """
    if not isinstance(schema, dict):
        return None
    if use_jsonschema:
        module = _load_jsonschema()
        if module is not None:
            try:
                module.validate(instance=value, schema=schema)
                return None
            except module.ValidationError as exc:  # type: ignore[attr-defined]
                # Normalize to a short, stable error string.
                path = ".".join(str(p) for p in exc.path) if exc.path else ""
                prefix = f"{path}: " if path else ""
                return f"{prefix}{exc.message}"
            except module.SchemaError:  # type: ignore[attr-defined]
                # A malformed schema falls back to the permissive subset.
                pass
    return _validate_subset(value, schema)


def _type_matches(value: Any, expected: str) -> bool:
    """True when ``value`` matches a single JSON-schema primitive ``expected``."""
    type_map: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "array": (list, tuple),
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    if expected == "integer" and isinstance(value, bool):
        return False
    if expected == "number" and isinstance(value, bool):
        return False
    pytype = type_map.get(expected)
    if pytype is None:
        return True  # unknown type keyword -> permissive
    return isinstance(value, pytype)


def _validate_subset(value: Any, schema: dict[str, Any]) -> str | None:
    """Self-contained offline validator (see :func:`validate_against_schema`)."""
    # --- const / enum -------------------------------------------------------
    if "const" in schema and value != schema["const"]:
        return f"value {value!r} != const {schema['const']!r}"
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return f"value {value!r} not in enum {enum!r}"

    # --- combinators --------------------------------------------------------
    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = sum(1 for sub in one_of if _validate_subset(value, sub) is None)
        if matches != 1:
            return f"value matched {matches} of oneOf branches (expected 1)"
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if not any(_validate_subset(value, sub) is None for sub in any_of):
            return "value did not match any anyOf branch"
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for sub in all_of:
            err = _validate_subset(value, sub)
            if err is not None:
                return f"allOf: {err}"

    # --- type ---------------------------------------------------------------
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_type_matches(value, t) for t in expected):
            return f"expected one of {expected}, got {type(value).__name__}"
        # When a union of types, only run keyword checks for the matched type.
        expected = next((t for t in expected if _type_matches(value, t)), None)
    elif isinstance(expected, str):
        if not _type_matches(value, expected):
            return f"expected {expected}, got {type(value).__name__}"

    # --- string constraints -------------------------------------------------
    if expected == "string" and isinstance(value, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(value) < min_len:
            return f"string shorter than minLength {min_len}"
        max_len = schema.get("maxLength")
        if isinstance(max_len, int) and len(value) > max_len:
            return f"string longer than maxLength {max_len}"
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    return f"string does not match pattern {pattern!r}"
            except re.error:
                pass

    # --- numeric constraints ------------------------------------------------
    if (
        expected in ("number", "integer")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            return f"value {value} < minimum {minimum}"
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            return f"value {value} > maximum {maximum}"
        excl_min = schema.get("exclusiveMinimum")
        if isinstance(excl_min, (int, float)) and value <= excl_min:
            return f"value {value} <= exclusiveMinimum {excl_min}"
        excl_max = schema.get("exclusiveMaximum")
        if isinstance(excl_max, (int, float)) and value >= excl_max:
            return f"value {value} >= exclusiveMaximum {excl_max}"

    # --- object constraints -------------------------------------------------
    if expected == "object" and isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f"missing required property {key!r}"
        properties = schema.get("properties", {})
        for key, sub in properties.items():
            if key in value and isinstance(sub, dict):
                err = _validate_subset(value[key], sub)
                if err is not None:
                    return f"{key}: {err}"
        additional = schema.get("additionalProperties")
        if additional is False:
            extra = [k for k in value if k not in properties]
            if extra:
                return (
                    f"unexpected propert{'y' if len(extra) == 1 else 'ies'} {extra!r}"
                )
        elif isinstance(additional, dict):
            for key, sub_value in value.items():
                if key not in properties:
                    err = _validate_subset(sub_value, additional)
                    if err is not None:
                        return f"{key}: {err}"
        min_props = schema.get("minProperties")
        if isinstance(min_props, int) and len(value) < min_props:
            return f"object has fewer than minProperties {min_props}"

    # --- array constraints --------------------------------------------------
    if expected == "array" and isinstance(value, (list, tuple)):
        items = schema.get("items")
        if isinstance(items, dict):
            for idx, item in enumerate(value):
                err = _validate_subset(item, items)
                if err is not None:
                    return f"[{idx}]: {err}"
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return f"array has fewer than minItems {min_items}"
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            return f"array has more than maxItems {max_items}"

    return None
