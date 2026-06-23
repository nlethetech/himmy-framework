"""Llama-3.1 (``llama3_json``) + Mistral v3 (``mistral_v3``) tool-call FORMAT tests.

These two native open-weight formats are ADDITIVE registry rows that mirror
HERMES_CHATML_XML: a per-family manifest, a FAIL-OPEN parser (the native pass OR-ed with
the generic tolerant pass and a ``<tool_call>`` fallback), and a per-family result
renderer. The tests pin:

* the GOLDEN manifests (byte-for-byte) + a SHA pin so a "no-op" change can't drift them;
* the native parse fixtures incl. PARALLEL calls (Llama JSON array, Mistral
  ``[TOOL_CALLS]`` array) and the off-grammar fail-open fall-through;
* the Mistral 9-char ``[a-zA-Z0-9]`` ``call_id`` being RENDER-TIME-ONLY — minted at
  serialization from the himmy UUID, which itself NEVER leaves the record;
* the selection rows (llama3*/llama-3*, mistral*/mixtral*) + the coder/guard/vision/embed
  excludes;
* the CRITICAL guarantee that GENERIC + canonical HERMES_CHATML_XML stay byte-identical
  (SHA-pinned) — these new families touch only their own opt-in rows.
"""

from __future__ import annotations

import hashlib
import json as _json
from typing import Any

from himmy.services.inference.models import (
    BoundTool,
    InferenceMessage,
    InferenceRequest,
    ToolReturnRecord,
)
from himmy.services.inference.tool_formats import (
    GENERIC,
    HERMES_CHATML_XML,
    LLAMA3_JSON,
    MISTRAL_V3,
    ToolCallFormat,
    format_for,
    get_format,
    mint_mistral_tool_call_id,
)
from himmy.services.inference.tool_protocol import parse_text_tool_calls
from tests.conftest import run_async

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
_WEATHER_TOOL = BoundTool(
    name="get_weather",
    description="Get weather",
    args_json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    read_only=True,
)
_TIME_TOOL = BoundTool(
    name="get_time",
    description="Get the current time in a timezone.",
    args_json_schema={
        "type": "object",
        "properties": {"tz": {"type": "string"}},
    },
    read_only=True,
)
_KNOWN = {"get_weather", "get_time"}


def _names_args(fmt: ToolCallFormat, text: str, known: set[str]) -> list[tuple[str, dict]]:
    return [(c.tool_name, c.args) for c in fmt.parse(text, known)]


# ========================================================================== #
# LLAMA3_JSON — golden manifest
# ========================================================================== #

_GOLDEN_LLAMA3_MANIFEST = (
    "You have access to the following functions. To call a function, respond with a "
    "JSON for a function call with its proper arguments that best answers the given "
    "prompt.\n\n"
    'Respond in the format {"name": function name, "parameters": dictionary of '
    'argument name and its value}. Do not use variables.\n\n'
    "[\n"
    "    {\n"
    '        "type": "function",\n'
    '        "function": {\n'
    '            "name": "get_weather",\n'
    '            "description": "Get weather — [read-only: returns data; safe to '
    'call for look-ups]",\n'
    '            "parameters": {\n'
    '                "type": "object",\n'
    '                "properties": {\n'
    '                    "city": {\n'
    '                        "type": "string"\n'
    "                    }\n"
    "                },\n"
    '                "required": [\n'
    '                    "city"\n'
    "                ]\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "]"
    "\n\n"
    "If multiple independent calls are needed, respond with a JSON list of such "
    "objects. If none of the functions are relevant, respond in plain text instead "
    "of a function call."
)


def test_llama3_render_manifest_golden() -> None:
    """The Llama-3.1 manifest = Meta's literal preamble + the JSON list of specs."""
    got = LLAMA3_JSON.render_system_manifest([_WEATHER_TOOL], "ollama")
    assert got == _GOLDEN_LLAMA3_MANIFEST


def test_llama3_manifest_uses_parameters_key_and_first_user_framing() -> None:
    """The manifest is the first-user-message style ('respond with a JSON ...') and the
    advertised arg key in the framing is ``parameters`` (Llama-3.1's convention)."""
    m = LLAMA3_JSON.render_system_manifest([_WEATHER_TOOL], "ollama")
    assert m.startswith("You have access to the following functions.")
    assert '"parameters": dictionary of argument name' in m
    # The JSON list (not a per-line <tools> block) carries the specs; general
    # format-native guidance prose now trails the list after a blank line.
    body = m.split("\n\n", 2)[2].rsplit("\n\n", 1)[0]
    parsed = _json.loads(body)
    assert isinstance(parsed, list) and parsed[0]["function"]["name"] == "get_weather"


def test_llama3_manifest_normalizes_per_tool() -> None:
    """normalize_tool_schema runs per tool: a $ref is inlined inside `parameters`."""
    tool = BoundTool(
        name="lookup",
        description="Look up",
        args_json_schema={
            "type": "object",
            "properties": {"q": {"$ref": "#/$defs/Q"}},
            "$defs": {"Q": {"type": "string"}},
        },
    )
    m = LLAMA3_JSON.render_system_manifest([tool], "ollama")
    body = m.split("\n\n", 2)[2].rsplit("\n\n", 1)[0]
    spec = _json.loads(body)[0]
    assert spec["function"]["parameters"]["properties"]["q"] == {"type": "string"}


# ========================================================================== #
# LLAMA3_JSON — native parse (bare {"name","parameters"}) + fail-open
# ========================================================================== #


def test_llama3_parse_single_bare_call() -> None:
    text = '{"name": "get_weather", "parameters": {"city": "KTM"}}'
    assert _names_args(LLAMA3_JSON, text, _KNOWN) == [("get_weather", {"city": "KTM"})]


def test_llama3_parse_parallel_json_array() -> None:
    """Parallel = a JSON ARRAY of bare call objects (Llama-3.1 multi-call shape)."""
    text = (
        '[{"name": "get_weather", "parameters": {"city": "KTM"}}, '
        '{"name": "get_time", "parameters": {}}]'
    )
    assert _names_args(LLAMA3_JSON, text, _KNOWN) == [
        ("get_weather", {"city": "KTM"}),
        ("get_time", {}),
    ]


def test_llama3_parse_call_with_surrounding_prose() -> None:
    """A bare call object embedded in prose is still recovered."""
    text = (
        "Sure, let me check.\n"
        '{"name": "get_weather", "parameters": {"city": "PKR"}}\n'
        "I will call the tool."
    )
    assert _names_args(LLAMA3_JSON, text, _KNOWN) == [("get_weather", {"city": "PKR"})]


def test_llama3_parse_parameters_as_json_string_decoded() -> None:
    """parameters delivered as a JSON string are decoded to a dict."""
    text = '{"name": "get_weather", "parameters": "{\\"city\\": \\"KTM\\"}"}'
    assert _names_args(LLAMA3_JSON, text, _KNOWN) == [("get_weather", {"city": "KTM"})]


def test_llama3_parse_unknown_name_rejected() -> None:
    """A JSON object whose name is NOT a bound tool is not mis-read as a call."""
    text = '{"name": "totally_unrelated", "parameters": {"x": 1}}'
    assert LLAMA3_JSON.parse(text, _KNOWN) == []


def test_llama3_parse_plain_structured_json_not_a_call() -> None:
    """A generic JSON object answer (no 'name' key) is never mis-parsed as a call."""
    text = '{"city": "KTM", "population": 975453}'
    assert LLAMA3_JSON.parse(text, _KNOWN) == []


def test_llama3_parse_fail_open_to_tool_call_envelope() -> None:
    """A Hermes-style <tool_call> envelope (some Llama fine-tunes drift onto it) is
    recovered by the OR-ed fallback pass — read with the parameters key."""
    text = (
        "<tool_call>\n"
        '{"name": "get_weather", "parameters": {"city": "BKT"}}\n'
        "</tool_call>"
    )
    assert _names_args(LLAMA3_JSON, text, _KNOWN) == [("get_weather", {"city": "BKT"})]


def test_llama3_parse_fail_open_to_tool_call_marker() -> None:
    """An off-grammar TOOL_CALL marker is recovered by the generic tolerant pass."""
    text = 'TOOL_CALL get_time {"tz": "Asia/Kathmandu"}'
    assert _names_args(LLAMA3_JSON, text, _KNOWN) == [
        ("get_time", {"tz": "Asia/Kathmandu"})
    ]


def test_llama3_parse_dedupes_across_passes() -> None:
    """The same call surfaced by the native + fallback passes is emitted once."""
    text = '{"name": "get_weather", "parameters": {"city": "KTM"}}'
    assert len(LLAMA3_JSON.parse(text, _KNOWN)) == 1


def test_llama3_parse_empty_and_prose_are_empty() -> None:
    assert LLAMA3_JSON.parse("", _KNOWN) == []
    assert LLAMA3_JSON.parse("Just a prose answer, no call.", _KNOWN) == []


def test_llama3_parse_idempotent() -> None:
    text = (
        '[{"name": "get_weather", "parameters": {"city": "KTM"}}, '
        '{"name": "get_time", "parameters": {}}]'
    )
    assert _names_args(LLAMA3_JSON, text, _KNOWN) == _names_args(
        LLAMA3_JSON, text, _KNOWN
    )


# ========================================================================== #
# LLAMA3_JSON — render_tool_results (ipython role)
# ========================================================================== #


def test_llama3_render_tool_results_ipython_block() -> None:
    results = [
        ToolReturnRecord(tool_call_id="a", tool_name="get_weather", content="sunny, 24C")
    ]
    assert LLAMA3_JSON.render_tool_results(results) == (
        "<|start_header_id|>ipython<|end_header_id|>\n\nsunny, 24C<|eot_id|>"
    )


def test_llama3_render_tool_results_batches_consecutive() -> None:
    """Consecutive results share ONE ipython turn (joined by blank lines)."""
    results = [
        ToolReturnRecord(tool_call_id="a", tool_name="get_weather", content="sunny"),
        ToolReturnRecord(tool_call_id="b", tool_name="get_time", content="10:00"),
    ]
    assert LLAMA3_JSON.render_tool_results(results) == (
        "<|start_header_id|>ipython<|end_header_id|>\n\nsunny\n\n10:00<|eot_id|>"
    )


def test_llama3_render_tool_results_empty() -> None:
    assert LLAMA3_JSON.render_tool_results([]) == ""


# ========================================================================== #
# LLAMA3_JSON — selection rows
# ========================================================================== #


def test_llama3_selected_for_llama3_tags() -> None:
    for tag in (
        "llama3.2",
        "llama3:8b",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.2-3b-instruct",
    ):
        assert format_for(tag) is LLAMA3_JSON, tag


def test_llama3_excludes_code_guard_vision_variants() -> None:
    """Code Llama / Llama Guard / Llama-3.2-Vision do NOT use this grammar → GENERIC."""
    assert format_for("codellama:13b") is GENERIC
    assert format_for("meta-llama/CodeLlama-3-7b") is GENERIC
    assert format_for("meta-llama/llama-guard-3-8b") is GENERIC
    assert format_for("llama-3.2-90b-vision-instruct") is GENERIC


def test_non_llama3_base_stays_generic() -> None:
    """A non-3 base llama (no JSON-function tool calling) stays on GENERIC."""
    assert format_for("llama2:13b") is GENERIC
    assert format_for("llama") is GENERIC


def test_llama3_override_beats_exclude() -> None:
    """An explicit override wins even over a code/guard exclude (operator is trusted)."""
    assert format_for("codellama:13b", override="llama3_json") is LLAMA3_JSON


def test_llama3_flags_pinned() -> None:
    f = LLAMA3_JSON.flags
    assert f.arg_key == "parameters"  # the Llama-3.1 arg key
    assert f.name_key == "name"
    assert f.result_role == "ipython"
    assert f.parallel_supported is True
    assert f.use_text_tool_path is True


# ========================================================================== #
# MISTRAL_V3 — golden manifest
# ========================================================================== #

_GOLDEN_MISTRAL_MANIFEST = (
    "[AVAILABLE_TOOLS] "
    '[{"type": "function", "function": {"name": "get_weather", "description": '
    '"Get weather — [read-only: returns data; safe to call for look-ups]", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, '
    '"required": ["city"]}}}]'
    " [/AVAILABLE_TOOLS]"
    " When multiple independent calls are needed, include them as multiple entries "
    "in a single [TOOL_CALLS] array. If no available tool fits, reply in plain text "
    "without a [TOOL_CALLS] block."
)


def test_mistral_render_manifest_golden() -> None:
    """The Mistral manifest = the [AVAILABLE_TOOLS] envelope around the JSON spec list."""
    got = MISTRAL_V3.render_system_manifest([_WEATHER_TOOL], "ollama")
    assert got == _GOLDEN_MISTRAL_MANIFEST


def test_mistral_manifest_is_available_tools_json_array() -> None:
    m = MISTRAL_V3.render_system_manifest([_WEATHER_TOOL, _TIME_TOOL], "ollama")
    assert m.startswith("[AVAILABLE_TOOLS] ")
    # General format-native guidance prose now trails the envelope; slice the JSON
    # array out from between the canonical [AVAILABLE_TOOLS] ... [/AVAILABLE_TOOLS]
    # markers rather than off the end of the string.
    prefix = "[AVAILABLE_TOOLS] "
    suffix = " [/AVAILABLE_TOOLS]"
    assert suffix in m
    inner = m[len(prefix) : m.index(suffix)]
    arr = _json.loads(inner)
    assert [s["function"]["name"] for s in arr] == ["get_weather", "get_time"]


# ========================================================================== #
# MISTRAL_V3 — native parse ([TOOL_CALLS] array, native parallel) + fail-open
# ========================================================================== #


def test_mistral_parse_single_tool_calls() -> None:
    text = '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "KTM"}}]'
    assert _names_args(MISTRAL_V3, text, _KNOWN) == [("get_weather", {"city": "KTM"})]


def test_mistral_parse_parallel_array() -> None:
    """Native parallel = ONE [TOOL_CALLS] marker + a JSON array of N entries."""
    text = (
        "[TOOL_CALLS]["
        '{"name": "get_weather", "arguments": {"city": "KTM"}}, '
        '{"name": "get_time", "arguments": {}}'
        "]"
    )
    assert _names_args(MISTRAL_V3, text, _KNOWN) == [
        ("get_weather", {"city": "KTM"}),
        ("get_time", {}),
    ]


def test_mistral_parse_arguments_as_json_string_decoded() -> None:
    text = '[TOOL_CALLS][{"name": "get_weather", "arguments": "{\\"city\\": \\"KTM\\"}"}]'
    assert _names_args(MISTRAL_V3, text, _KNOWN) == [("get_weather", {"city": "KTM"})]


def test_mistral_parse_marker_then_single_object() -> None:
    """A [TOOL_CALLS] marker followed by a single object (not an array) still parses."""
    text = '[TOOL_CALLS]{"name": "get_time", "arguments": {}}'
    assert _names_args(MISTRAL_V3, text, _KNOWN) == [("get_time", {})]


def test_mistral_parse_fail_open_to_tool_call_envelope() -> None:
    """A <tool_call> envelope a Mistral fine-tune emits is recovered by the fallback."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "BKT"}}\n</tool_call>'
    assert _names_args(MISTRAL_V3, text, _KNOWN) == [("get_weather", {"city": "BKT"})]


def test_mistral_parse_fail_open_to_marker() -> None:
    text = 'TOOL_CALL get_time {"tz": "Asia/Kathmandu"}'
    assert _names_args(MISTRAL_V3, text, _KNOWN) == [
        ("get_time", {"tz": "Asia/Kathmandu"})
    ]


def test_mistral_parse_malformed_array_skipped_never_raises() -> None:
    text = "[TOOL_CALLS][not valid json"
    assert MISTRAL_V3.parse(text, _KNOWN) == []


def test_mistral_parse_empty_and_prose_are_empty() -> None:
    assert MISTRAL_V3.parse("", _KNOWN) == []
    assert MISTRAL_V3.parse("Just prose.", _KNOWN) == []


def test_mistral_parse_dedupes_across_passes() -> None:
    text = '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "KTM"}}]'
    assert len(MISTRAL_V3.parse(text, _KNOWN)) == 1


def test_mistral_parse_idempotent() -> None:
    text = (
        "[TOOL_CALLS]["
        '{"name": "get_weather", "arguments": {"city": "KTM"}}, '
        '{"name": "get_time", "arguments": {}}]'
    )
    assert _names_args(MISTRAL_V3, text, _KNOWN) == _names_args(
        MISTRAL_V3, text, _KNOWN
    )


# ========================================================================== #
# MISTRAL_V3 — the 9-char tool_call_id is RENDER-TIME-ONLY
# ========================================================================== #


def test_mint_mistral_id_is_9_char_alnum() -> None:
    mid = mint_mistral_tool_call_id("11111111-2222-3333-4444-555555555555")
    assert len(mid) == 9
    assert mid.isalnum()
    assert all(c.isascii() for c in mid)


def test_mint_mistral_id_is_deterministic_and_pinned() -> None:
    """The same himmy id always maps to the same 9-char wire id (golden-pinned)."""
    assert (
        mint_mistral_tool_call_id("11111111-2222-3333-4444-555555555555") == "HO7Xq5qLC"
    )
    assert (
        mint_mistral_tool_call_id("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee") == "8TTWB2U0V"
    )


def test_mint_mistral_id_distinct_for_distinct_ids() -> None:
    ids = {
        mint_mistral_tool_call_id(f"id-{i:06d}-uuid-value-here") for i in range(200)
    }
    # No collisions over a realistic per-turn id population.
    assert len(ids) == 200


def test_mistral_render_tool_results_uses_render_time_9char_id() -> None:
    """[TOOL_RESULTS] carries the 9-char call_id minted from the himmy id — and the
    himmy UUID itself NEVER appears on the wire."""
    himmy_uuid = "11111111-2222-3333-4444-555555555555"
    r = ToolReturnRecord(tool_call_id=himmy_uuid, tool_name="get_weather", content="sunny")
    wire = MISTRAL_V3.render_tool_results([r])
    assert wire == '[TOOL_RESULTS] {"call_id": "HO7Xq5qLC", "content": "sunny"} [/TOOL_RESULTS]'
    # The himmy UUID must NOT leak onto the wire.
    assert himmy_uuid not in wire
    # The record itself is unchanged (the himmy id stays on it).
    assert r.tool_call_id == himmy_uuid


def test_mistral_call_result_id_pairing_is_stable_within_a_render() -> None:
    """A call and its result share the SAME 9-char wire id within one serialization
    (the pairing Mistral needs), derived from the one himmy id on both records."""
    himmy_uuid = "abc12345-0000-0000-0000-000000000000"
    result = ToolReturnRecord(
        tool_call_id=himmy_uuid, tool_name="get_weather", content="sunny"
    )
    wire = MISTRAL_V3.render_tool_results([result])
    rendered_id = _json.loads(
        wire[len("[TOOL_RESULTS] ") : -len(" [/TOOL_RESULTS]")]
    )["call_id"]
    assert rendered_id == mint_mistral_tool_call_id(himmy_uuid)


def test_mistral_render_tool_results_one_block_per_result() -> None:
    results = [
        ToolReturnRecord(tool_call_id="id-a", tool_name="get_weather", content="sunny"),
        ToolReturnRecord(tool_call_id="id-b", tool_name="get_time", content="10:00"),
    ]
    wire = MISTRAL_V3.render_tool_results(results)
    assert wire.count("[TOOL_RESULTS]") == 2
    assert wire.count("[/TOOL_RESULTS]") == 2
    # Distinct himmy ids → distinct 9-char wire ids.
    ids = [
        _json.loads(blk.split("[TOOL_RESULTS] ", 1)[1])["call_id"]
        for blk in wire.split(" [/TOOL_RESULTS]")
        if "[TOOL_RESULTS]" in blk
    ]
    assert ids[0] != ids[1]


def test_mistral_render_tool_results_empty() -> None:
    assert MISTRAL_V3.render_tool_results([]) == ""


# ========================================================================== #
# MISTRAL_V3 — selection rows
# ========================================================================== #


def test_mistral_selected_for_mistral_tags() -> None:
    for tag in (
        "mistral:7b",
        "mistralai/mistral-nemo",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mixtral:8x7b",
        "ministral-8b",
        "magistral-small",
    ):
        assert format_for(tag) is MISTRAL_V3, tag


def test_mistral_excludes_codestral_and_embed() -> None:
    """Codestral (a code model) + Mistral embedding models do NOT use this grammar."""
    assert format_for("mistralai/codestral-22b") is GENERIC
    assert format_for("mistral-embed") is GENERIC


def test_mistral_override_beats_exclude() -> None:
    assert format_for("mistralai/codestral-22b", override="mistral_v3") is MISTRAL_V3


def test_mistral_flags_pinned() -> None:
    f = MISTRAL_V3.flags
    assert f.arg_key == "arguments"  # the Mistral arg key
    assert f.name_key == "name"
    assert f.parallel_supported is True
    assert f.use_text_tool_path is True


# ========================================================================== #
# Registry registration
# ========================================================================== #


def test_both_formats_registered_by_name() -> None:
    assert get_format("llama3_json") is LLAMA3_JSON
    assert get_format("mistral_v3") is MISTRAL_V3
    # case-insensitive
    assert get_format("LLAMA3_JSON") is LLAMA3_JSON
    assert get_format("MISTRAL_V3") is MISTRAL_V3


# ========================================================================== #
# GENERIC + canonical HERMES stay BYTE-IDENTICAL (these families are add-only)
# ========================================================================== #

_SHA_PIN_TOOLS = [
    BoundTool(
        name="get_weather",
        description="Get weather",
        args_json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        read_only=True,
    ),
    BoundTool(
        name="add_numbers",
        description="Add two integers together and return the sum.",
        args_json_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        read_only=True,
    ),
]
# These two SHAs are the SAME values pinned in test_tool_formats.py — re-pinning them
# here proves adding LLAMA3_JSON / MISTRAL_V3 did not perturb GENERIC or canonical Hermes.
_GENERIC_MANIFEST_SHA = (
    "a3843801b22db0510d330b12d6e7156e706ef7ddf54eeba0ab98fd2dc64b3e74"
)
_HERMES_MANIFEST_SHA = (
    "6b8be9a85924495a245c43bd6aa02184a7f6664717f12b73717873038e92cd8c"
)
# The new families' own manifest SHAs (golden-pinned: any drift is a wire-grammar change).
_LLAMA3_MANIFEST_SHA = (
    "b12b9f6ea051ed6a1fc3fdf74fc488523acebc239943fe631ba2f13cccce5f1e"
)
_MISTRAL_MANIFEST_SHA = (
    "34ba5cdb6ec3becc63ba13803c7603a8adeae9b1bc100555c20326475fe68833"
)


def _manifest_sha(fmt: ToolCallFormat) -> str:
    blob = ""
    for provider in ("claude-cli", "ollama", "himalayagpt"):
        blob += fmt.render_system_manifest(_SHA_PIN_TOOLS, provider) + "\x1e"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_generic_manifest_sha_unchanged_by_new_families() -> None:
    assert _manifest_sha(GENERIC) == _GENERIC_MANIFEST_SHA


def test_canonical_hermes_manifest_sha_unchanged_by_new_families() -> None:
    assert _manifest_sha(HERMES_CHATML_XML) == _HERMES_MANIFEST_SHA


def test_llama3_manifest_sha_pinned() -> None:
    assert _manifest_sha(LLAMA3_JSON) == _LLAMA3_MANIFEST_SHA


def test_mistral_manifest_sha_pinned() -> None:
    assert _manifest_sha(MISTRAL_V3) == _MISTRAL_MANIFEST_SHA


def test_generic_parse_record_identical_to_text_parser() -> None:
    """GENERIC parse is still byte/record-identical to parse_text_tool_calls."""
    samples = [
        'TOOL_CALL get_weather {"city": "KTM"}',
        '{"name": "add_numbers", "arguments": {"a": 1, "b": 2}}',
        "Just prose.",
    ]
    known = {"get_weather", "add_numbers"}
    for text in samples:
        assert [(c.tool_name, c.args) for c in GENERIC.parse(text, known)] == [
            (c.tool_name, c.args) for c in parse_text_tool_calls(text, known)
        ]


# ========================================================================== #
# Manager wiring (text path) — the formats slot into the existing generic seam
# ========================================================================== #


def _ollama_payload_for(model: str) -> dict[str, Any]:
    """Capture the Ollama /api/chat payload a tools request produces for ``model``."""
    captured: dict[str, Any] = {}

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return {"message": {"content": ""}}

    from himmy.services.inference.local import OllamaClientManager

    mgr = OllamaClientManager(
        model=model, transport=transport, model_registry={"default": model}
    )
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="weather in KTM?")],
        bound_tools=[_WEATHER_TOOL],
    )
    run_async(mgr.generate(req))
    return captured["payload"]


def test_ollama_llama3_drives_text_path_with_json_manifest() -> None:
    """A Llama-3.x model suppresses Ollama's native tools= and injects the JSON manifest."""
    payload = _ollama_payload_for("llama3.1:8b-instruct")
    assert "tools" not in payload  # native function-tool API suppressed (text path)
    system = payload["messages"][0]
    assert system["role"] == "system"
    assert system["content"] == LLAMA3_JSON.render_system_manifest(
        [_WEATHER_TOOL], "ollama"
    )


def test_ollama_mistral_drives_text_path_with_available_tools_manifest() -> None:
    """A Mistral model suppresses native tools= and injects [AVAILABLE_TOOLS]."""
    payload = _ollama_payload_for("mistral-nemo")
    assert "tools" not in payload
    system = payload["messages"][0]
    assert system["role"] == "system"
    assert system["content"] == MISTRAL_V3.render_system_manifest(
        [_WEATHER_TOOL], "ollama"
    )


def test_ollama_llama3_text_path_parses_bare_json_call_end_to_end() -> None:
    """On the text path, a bare {"name","parameters"} reply becomes an executed call."""

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "message": {
                "content": '{"name": "get_weather", "parameters": {"city": "KTM"}}'
            }
        }

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        return ToolReturnRecord(
            tool_call_id="x", tool_name=name, content=f"sunny in {args['city']}"
        )

    from himmy.services.inference.local import OllamaClientManager

    mgr = OllamaClientManager(
        model="llama3.1:8b-instruct",
        transport=transport,
        model_registry={"default": "llama3.1:8b-instruct"},
    )
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="weather?")],
        bound_tools=[_WEATHER_TOOL],
        tool_executor=executor,
    )
    resp = run_async(mgr.generate(req))
    assert [(c.tool_name, c.args) for c in resp.tool_calls] == [
        ("get_weather", {"city": "KTM"})
    ]
    assert resp.tool_returns[0].content == "sunny in KTM"
    assert resp.output_text == ""  # the reply was a tool call, not a final answer


def test_ollama_mistral_text_path_parses_tool_calls_array_end_to_end() -> None:
    """On the text path, a [TOOL_CALLS] array reply becomes executed (parallel) calls."""

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "message": {
                "content": (
                    "[TOOL_CALLS]["
                    '{"name": "get_weather", "arguments": {"city": "KTM"}}'
                    "]"
                )
            }
        }

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        return ToolReturnRecord(
            tool_call_id="x", tool_name=name, content=f"sunny in {args['city']}"
        )

    from himmy.services.inference.local import OllamaClientManager

    mgr = OllamaClientManager(
        model="mistral-nemo",
        transport=transport,
        model_registry={"default": "mistral-nemo"},
    )
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="weather?")],
        bound_tools=[_WEATHER_TOOL],
        tool_executor=executor,
    )
    resp = run_async(mgr.generate(req))
    assert [(c.tool_name, c.args) for c in resp.tool_calls] == [
        ("get_weather", {"city": "KTM"})
    ]
    assert resp.tool_returns[0].content == "sunny in KTM"
    assert resp.output_text == ""
