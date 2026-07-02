"""Tests for the Gemma-4 tool-call grammar (himalaya-ai/himalaya-gemma-4-e2b-it).

The manifest renderer is a 1:1 port of the model's own ``chat_template.jinja``. The
golden string below was captured by rendering the SAME tool schema through the real HF
tokenizer template (``apply_chat_template(tools=...)``) and asserting byte-equality at
build time; freezing it here keeps the port faithful without a transformers dependency
in the test env.
"""

from __future__ import annotations

from himmy.services.inference import _gemma4_grammar as g
from himmy.services.inference.models import BoundTool, ToolReturnRecord
from himmy.services.inference.tool_formats import GEMMA4, format_for

_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["location"],
}

# Byte-identical to the model's own chat_template.jinja for _SCHEMA (validated live).
_GOLDEN_DECL = (
    "<|tool>declaration:get_weather{description:<|\"|>Get weather.<|\"|>,"
    "parameters:{properties:{"
    "days:{type:<|\"|>INTEGER<|\"|>},"
    "location:{description:<|\"|>City<|\"|>,type:<|\"|>STRING<|\"|>},"
    "tags:{items:{type:<|\"|>STRING<|\"|>},type:<|\"|>ARRAY<|\"|>},"
    "unit:{enum:[<|\"|>celsius<|\"|>,<|\"|>fahrenheit<|\"|>],type:<|\"|>STRING<|\"|>}},"
    "required:[<|\"|>location<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"
)


def _bound() -> list[BoundTool]:
    return [BoundTool(name="get_weather", description="Get weather.", args_json_schema=_SCHEMA)]


def test_manifest_matches_template_grammar():
    manifest = g.render_system_manifest(_bound(), "openrouter")
    assert manifest.startswith(_GOLDEN_DECL), manifest


def test_registry_resolution():
    # Auto-selects by model tag, and by explicit override.
    assert format_for("openai/himalaya-ai/himalaya-gemma-4-e2b-it").name == "gemma4"
    assert format_for(None, "gemma4").name == "gemma4"
    assert GEMMA4.flags.use_text_tool_path is True


def test_parse_roundtrip_scalars_and_nesting():
    text = (
        '<|tool_call>call:get_weather{location:<|"|>Kathmandu, Nepal<|"|>,'
        'days:3,active:true,opts:{deep:false},tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>'
    )
    calls = g.parse(text, {"get_weather"})
    assert len(calls) == 1
    c = calls[0]
    assert c.tool_name == "get_weather"
    assert c.args == {
        "location": "Kathmandu, Nepal",
        "days": 3,
        "active": True,
        "opts": {"deep": False},
        "tags": ["a", "b"],
    }


def test_parse_is_fail_open_on_unknown_tools():
    # A call to a tool that is not bound is dropped (fail-open: only known hits).
    text = '<|tool_call>call:not_a_tool{x:1}<tool_call|>'
    assert g.parse(text, {"get_weather"}) == []


def test_parse_multiple_back_to_back_calls():
    text = (
        '<|tool_call>call:get_weather{location:<|"|>A<|"|>}<tool_call|>'
        '<|tool_call>call:get_weather{location:<|"|>B<|"|>}<tool_call|>'
    )
    calls = g.parse(text, {"get_weather"})
    assert [c.args["location"] for c in calls] == ["A", "B"]


def test_render_tool_results_grammar():
    out = g.render_tool_results(
        [ToolReturnRecord(tool_call_id="x", tool_name="get_weather", content='{"t":22}')]
    )
    assert out == '<|tool_response>response:get_weather{value:<|"|>{"t":22}<|"|>}<tool_response|>'


def test_render_assistant_calls_grammar():
    class _TC:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    out = g.render_assistant_calls([_TC("get_order", {"order_id": "W123", "full": True})])
    assert out == '<|tool_call>call:get_order{full:true,order_id:<|"|>W123<|"|>}<tool_call|>'


# --------------------------------------------------------------------------- #
# Tolerant recovery of a DEGRADED opener-less call (the real himalaya-gemma-4
# symptom: name+args correct, but written as ``NAME{...}<tool_call|>`` with no
# ``<|tool_call>call:`` opener, so the strict grammar never fires the tool).
# --------------------------------------------------------------------------- #
_DEGRADED = 'book_appointment{name:<|"|>Sam<|"|>,phone:<|"|>555<|"|>}<tool_call|>'
_CORRECT = '<|tool_call>call:book_appointment{name:<|"|>Sam<|"|>,phone:<|"|>555<|"|>}<tool_call|>'


def test_parse_recovers_openerless_degraded_call():
    # The EXACT shape the user reported. Before the fix this returned [].
    calls = g.parse(_DEGRADED, {"book_appointment"})
    assert len(calls) == 1
    assert calls[0].tool_name == "book_appointment"
    assert calls[0].args == {"name": "Sam", "phone": "555"}


def test_parse_recovers_call_prefix_without_opener():
    # ``call:NAME{...}<tool_call|>`` — has the ``call:`` but no ``<|tool_call>`` opener.
    text = 'call:book_appointment{name:<|"|>Sam<|"|>}<tool_call|>'
    calls = g.parse(text, {"book_appointment"})
    assert [c.tool_name for c in calls] == ["book_appointment"]


def test_parse_degraded_is_name_guarded():
    # An opener-less blob whose name is NOT a bound tool must NOT fire (prose safety).
    assert g.parse(_DEGRADED, {"some_other_tool"}) == []
    assert g.parse('note{body:<|"|>hi<|"|>}<tool_call|>', {"book_appointment"}) == []


def test_parse_degraded_not_recovered_without_known_tools():
    # With no bound-tool set we cannot name-guard, so tolerant recovery stays OFF
    # (strict-only) — byte-for-byte the pre-fix behavior for the unknown-tools case.
    assert g.parse(_DEGRADED, set()) == []


def test_parse_correct_form_not_double_counted():
    # The strict + tolerant passes must not both count the same well-formed call.
    calls = g.parse(_CORRECT, {"book_appointment"})
    assert len(calls) == 1
    assert calls[0].args == {"name": "Sam", "phone": "555"}


def test_parse_mixed_correct_and_degraded_in_order():
    text = (
        '<|tool_call>call:get_weather{location:<|"|>A<|"|>}<tool_call|>'
        'book_appointment{name:<|"|>Sam<|"|>}<tool_call|>'
    )
    calls = g.parse(text, {"get_weather", "book_appointment"})
    assert [c.tool_name for c in calls] == ["get_weather", "book_appointment"]


def test_registry_gemma4_parse_recovers_degraded():
    # Same recovery through the format the manager actually uses for gemma-4.
    calls = GEMMA4.parse(_DEGRADED, {"book_appointment"})
    assert [c.tool_name for c in calls] == ["book_appointment"]
    # And gemma-4 auto-selects this format by model tag.
    assert format_for("himalaya-gemma-4-e2b-it").name == "gemma4"
