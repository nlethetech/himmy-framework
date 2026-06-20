"""FIX 1 — python-call parser fallback (the EMIT lever for the 0.5b).

Small open-weight models (HimalayaGPT-0.5b) often emit a tool call as PYTHON syntax
instead of a ``<tool_call>`` block: a ``get_weather(city="Paris")`` line, a ```python```
fenced block, or a ``<|python_start|>...<|python_end|>`` span. The generic/Hermes JSON
parsers MISS these (count as 0 emit). :func:`parse_python_tool_calls` recovers them, and
the few-shot Hermes format OR-s it in as a final add-only, name-guarded pass.

These tests pin: each python-call shape is recovered; a name match is required (prose that
merely mentions a tool name does NOT false-fire); the fallback is OFF for GENERIC and the
canonical Hermes format (byte/recall parity); and the few-shot format never drops or
changes a real ``<tool_call>`` parse.
"""

from __future__ import annotations

from himmy.services.inference.tool_formats import (
    GENERIC,
    HERMES_CHATML_XML,
    HERMES_CHATML_XML_FEWSHOT,
)
from himmy.services.inference.tool_protocol import (
    parse_python_tool_calls,
    parse_text_tool_calls,
)

_KNOWN = {"get_weather", "add_numbers", "set_timer"}


def _names_args(calls: list) -> list[tuple[str, dict]]:
    return [(c.tool_name, c.args) for c in calls]


# ---- the three python-call shapes are recovered --------------------------


def test_bare_call_line_recovered() -> None:
    text = 'get_weather(city="Paris")'
    assert _names_args(parse_python_tool_calls(text, _KNOWN)) == [
        ("get_weather", {"city": "Paris"})
    ]


def test_bare_call_with_prose_around_it() -> None:
    text = "Sure, let me do that.\nget_weather(city=\"Tokyo\")\nDone."
    assert _names_args(parse_python_tool_calls(text, _KNOWN)) == [
        ("get_weather", {"city": "Tokyo"})
    ]


def test_python_fenced_block_recovered() -> None:
    text = '```python\nget_weather(city="Paris")\n```'
    assert _names_args(parse_python_tool_calls(text, _KNOWN)) == [
        ("get_weather", {"city": "Paris"})
    ]


def test_tool_call_fenced_block_recovered() -> None:
    text = '```tool_call\nadd_numbers(a=7, b=5)\n```'
    assert _names_args(parse_python_tool_calls(text, _KNOWN)) == [
        ("add_numbers", {"a": 7, "b": 5})
    ]


def test_python_token_span_recovered() -> None:
    text = '<|python_start|>set_timer(minutes=10)<|python_end|>'
    assert _names_args(parse_python_tool_calls(text, _KNOWN)) == [
        ("set_timer", {"minutes": 10})
    ]


def test_int_and_keyword_args_literal_evaled() -> None:
    text = "add_numbers(a=42, b=8)"
    assert _names_args(parse_python_tool_calls(text, _KNOWN)) == [
        ("add_numbers", {"a": 42, "b": 8})
    ]


def test_positional_args_tolerated_and_ignored() -> None:
    """A positional-only call has no arg keys; the name still fires with empty args."""
    text = 'get_weather("Paris")'
    assert _names_args(parse_python_tool_calls(text, _KNOWN)) == [
        ("get_weather", {})
    ]


# ---- name guard: prose must NOT false-fire -------------------------------


def test_prose_mentioning_tool_name_does_not_fire() -> None:
    """A sentence that merely names a tool (no call syntax) recovers nothing."""
    text = "You can use the get_weather tool to look up the weather."
    assert parse_python_tool_calls(text, _KNOWN) == []


def test_unknown_name_not_accepted() -> None:
    """A python call to an unbound name is rejected (name guard)."""
    text = "frobnicate(x=1)"
    assert parse_python_tool_calls(text, _KNOWN) == []


def test_no_false_fire_on_plain_prose() -> None:
    assert parse_python_tool_calls("It is sunny in Paris today.", _KNOWN) == []


def test_empty_known_set_still_safe_on_prose() -> None:
    """With no bound tools the name guard accepts, but prose has no call to recover."""
    assert parse_python_tool_calls("Just prose, no call.", set()) == []


def test_never_raises_on_garbage() -> None:
    for text in ["get_weather(", "(((", "def f(): pass", ""]:
        assert isinstance(parse_python_tool_calls(text, _KNOWN), list)


def test_dedup_same_call() -> None:
    text = 'get_weather(city="Paris")\nget_weather(city="Paris")'
    assert len(parse_python_tool_calls(text, _KNOWN)) == 1


# ---- GENERIC byte/recall parity: fallback is OFF -------------------------


def test_generic_does_not_use_python_fallback() -> None:
    """GENERIC.parse stays EXACTLY parse_text_tool_calls — it misses the python call."""
    text = 'get_weather(city="Paris")'
    assert parse_text_tool_calls(text, _KNOWN) == []  # generic tolerant parser misses it
    assert GENERIC.parse(text, _KNOWN) == []  # so GENERIC must miss it too (parity)


def test_canonical_hermes_does_not_use_python_fallback() -> None:
    """Canonical Hermes/Qwen recall is unchanged: no python-call fallback."""
    text = 'get_weather(city="Paris")'
    assert HERMES_CHATML_XML.parse(text, _KNOWN) == []
    assert HERMES_CHATML_XML.flags.python_call_fallback is False


# ---- few-shot format: fallback ON, add-only ------------------------------


def test_fewshot_recovers_python_call() -> None:
    text = 'get_weather(city="Paris")'
    assert HERMES_CHATML_XML_FEWSHOT.flags.python_call_fallback is True
    assert _names_args(HERMES_CHATML_XML_FEWSHOT.parse(text, _KNOWN)) == [
        ("get_weather", {"city": "Paris"})
    ]


def test_fewshot_real_tool_call_still_wins_unchanged() -> None:
    """A successful <tool_call> parse is not changed/dropped by the fallback."""
    text = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "KTM"}}\n'
        "</tool_call>"
    )
    assert _names_args(HERMES_CHATML_XML_FEWSHOT.parse(text, _KNOWN)) == [
        ("get_weather", {"city": "KTM"})
    ]


def test_fewshot_tool_call_plus_trailing_python_dedupes() -> None:
    """A <tool_call> block AND a redundant python line for the same call → one record."""
    text = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        "</tool_call>\n"
        'get_weather(city="Paris")'
    )
    assert _names_args(HERMES_CHATML_XML_FEWSHOT.parse(text, _KNOWN)) == [
        ("get_weather", {"city": "Paris"})
    ]


def test_fewshot_python_fallback_is_add_only_not_replace() -> None:
    """A <tool_call> for one tool + a python line for ANOTHER → both, in order."""
    text = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        "</tool_call>\n"
        "set_timer(minutes=5)"
    )
    assert _names_args(HERMES_CHATML_XML_FEWSHOT.parse(text, _KNOWN)) == [
        ("get_weather", {"city": "Paris"}),
        ("set_timer", {"minutes": 5}),
    ]
