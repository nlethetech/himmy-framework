"""Tool-call FORMAT registry — Phase 0 skeleton + byte-parity guarantees.

Phase 0 ships only the GENERIC format, which must reproduce today's behavior
EXACTLY: its renderer is ``_react_tool_manifest`` byte-for-byte, its parser is
``parse_text_tool_calls`` record-for-record, and threading a format through the
managers is a strict no-op. These tests pin that contract.
"""

from __future__ import annotations

from typing import Any

from himmy.services.inference.local import (
    ClaudeCliClientManager,
    OllamaClientManager,
    _react_tool_manifest,
)
from himmy.services.inference.models import (
    BoundTool,
    InferenceMessage,
    InferenceRequest,
    ToolReturnRecord,
)
from himmy.services.inference.tool_formats import (
    GENERIC,
    HERMES_CHATML_XML,
    ToolCallFormat,
    ToolCallFormatRegistry,
    format_for,
    get_format,
    list_formats,
    register_format,
)
from himmy.services.inference.tool_protocol import parse_text_tool_calls
from tests.conftest import run_async

_TOOLS = [
    BoundTool(
        name="egg_totals",
        description="Sum eggs collected.",
        args_json_schema={
            "type": "object",
            "properties": {"days": {"type": "integer"}},
        },
        read_only=True,
    ),
    BoundTool(
        name="add_task",
        description="Add a task.",
        args_json_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}},
        },
        read_only=False,
    ),
]
_KNOWN = {"egg_totals", "add_task"}


# ---- GENERIC render byte-parity -----------------------------------------


def test_generic_render_matches_react_manifest_byte_for_byte() -> None:
    """GENERIC.render_system_manifest == _react_tool_manifest, every byte."""
    for provider in ("claude-cli", "ollama", "himalayagpt"):
        expected = _react_tool_manifest(_TOOLS, provider=provider)
        got = GENERIC.render_system_manifest(_TOOLS, provider)
        assert got == expected


def test_generic_render_empty_tools_matches() -> None:
    """The no-bound-tools manifest (fallback example name) is reproduced too."""
    assert GENERIC.render_system_manifest([], "claude-cli") == _react_tool_manifest(
        [], provider="claude-cli"
    )


# ---- GENERIC parse parity -----------------------------------------------


def test_generic_parse_matches_parse_text_tool_calls() -> None:
    """GENERIC.parse == parse_text_tool_calls, record for record."""
    samples = [
        'TOOL_CALL egg_totals {"days": 7}',
        '```json\n{"tool": "add_task", "args": {"title": "net litchi"}}\n```',
        '{"name": "egg_totals", "arguments": {"days": 3}}',
        "Just prose, no calls here.",
        '<tool_call>\n{"name": "egg_totals", "arguments": {"days": 7}}\n</tool_call>',
    ]
    for text in samples:
        expected = parse_text_tool_calls(text, _KNOWN)
        got = GENERIC.parse(text, _KNOWN)
        # tool_call_id is a fresh uuid each call — compare the meaningful fields.
        assert [(c.tool_name, c.args) for c in got] == [
            (c.tool_name, c.args) for c in expected
        ]


def test_generic_parse_empty_known_set() -> None:
    text = 'TOOL_CALL egg_totals {"days": 1}'
    assert [(c.tool_name, c.args) for c in GENERIC.parse(text, set())] == [
        (c.tool_name, c.args) for c in parse_text_tool_calls(text, set())
    ]


# ---- GENERIC render_tool_results reproduces the [Tool result] label ------


def test_generic_render_tool_results_label() -> None:
    results = [
        ToolReturnRecord(tool_call_id="a", tool_name="egg_totals", content="42 eggs"),
        ToolReturnRecord(tool_call_id="b", tool_name="add_task", content="added"),
    ]
    assert (
        GENERIC.render_tool_results(results)
        == "[Tool result]\n42 eggs\n\n[Tool result]\nadded"
    )


# ---- Registry resolution -------------------------------------------------


def test_format_for_defaults_to_generic() -> None:
    """No override, unknown tag → GENERIC (the no-op default)."""
    assert format_for("llama3.2") is GENERIC
    assert format_for("mistral:7b") is GENERIC
    assert format_for(None) is GENERIC
    assert format_for("") is GENERIC


def test_format_for_unknown_override_falls_back_to_generic() -> None:
    """A bogus override never raises — it resolves to GENERIC."""
    assert format_for("anything", override="does-not-exist") is GENERIC


def test_registry_register_get_list() -> None:
    reg = ToolCallFormatRegistry()
    assert get_format("generic") is GENERIC  # process registry seeded with GENERIC
    fmt = ToolCallFormat(
        name="probe",
        render_system_manifest=GENERIC.render_system_manifest,
        parse=GENERIC.parse,
        render_tool_results=GENERIC.render_tool_results,
        model_tags=frozenset({"probe-model"}),
    )
    reg.register_format(fmt)
    assert reg.get_format("PROBE") is fmt  # case-insensitive
    assert "probe" in reg.list_formats()
    # override by name wins
    assert reg.format_for("anything", override="probe") is fmt
    # auto-select by model-tag substring
    assert reg.format_for("vendor/probe-model:latest") is fmt
    # non-matching tag → default GENERIC
    assert reg.format_for("vendor/other-model") is GENERIC


def test_registry_select_never_raises_on_bad_tags() -> None:
    reg = ToolCallFormatRegistry()
    # Defensive: a non-string-ish tag still resolves (to GENERIC), never raises.
    assert reg.format_for(None, None) is GENERIC


def test_register_format_on_process_registry_is_visible() -> None:
    fmt = ToolCallFormat(
        name="ephemeral_probe",
        render_system_manifest=GENERIC.render_system_manifest,
        parse=GENERIC.parse,
        render_tool_results=GENERIC.render_tool_results,
    )
    register_format(fmt)
    assert "ephemeral_probe" in list_formats()
    assert get_format("ephemeral_probe") is fmt


# ---- Manager seam is a strict no-op -------------------------------------


def _cli_prompt_for(tool_call_format: str | None) -> str:
    """Capture the stdin prompt the CLI runner receives for a tools request."""
    captured: dict[str, str] = {}

    def runner(argv: list[str], stdin: str) -> str:
        captured["stdin"] = stdin
        return "ok"

    mgr = ClaudeCliClientManager(
        model="haiku", runner=runner, tool_call_format=tool_call_format
    )
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="hello")],
        bound_tools=_TOOLS,
    )
    run_async(mgr.generate(req))
    return captured["stdin"]


def test_cli_manifest_threading_is_a_no_op() -> None:
    """Default (None) and explicit 'generic' both embed the verbatim manifest."""
    expected_manifest = _react_tool_manifest(_TOOLS, provider="claude-cli")
    default_prompt = _cli_prompt_for(None)
    explicit_prompt = _cli_prompt_for("generic")
    assert expected_manifest in default_prompt
    assert default_prompt == explicit_prompt


# ========================================================================== #
# PHASE 1 — HERMES_CHATML_XML (Hermes 2 Pro / Hermes 3 / Qwen2.5-Instruct)
# ========================================================================== #

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
_WEATHER_KNOWN = {"get_weather", "get_time"}


# ---- golden manifest render (pinned, byte-for-byte) ----------------------

#: The exact manifest body the local qwen2.5:*-instruct chat template expects for the
#: single get_weather tool above. Pinned verbatim — any drift is a wire-grammar change.
_GOLDEN_HERMES_MANIFEST = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "get_weather", "description": '
    '"Get weather — [read-only: returns data; safe to call for look-ups]", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, '
    '"required": ["city"]}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


def test_hermes_render_manifest_golden() -> None:
    """render_system_manifest reproduces the pinned ChatML-XML <tools> body exactly."""
    got = HERMES_CHATML_XML.render_system_manifest([_WEATHER_TOOL], "ollama")
    assert got == _GOLDEN_HERMES_MANIFEST


def test_hermes_render_manifest_one_line_per_tool() -> None:
    """Each tool is exactly one line inside <tools>...</tools>, no trailing comma."""
    manifest = HERMES_CHATML_XML.render_system_manifest(_TOOLS, "ollama")
    body = manifest.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    tool_lines = body.splitlines()
    assert len(tool_lines) == len(_TOOLS)
    for line in tool_lines:
        assert line.startswith('{"type": "function", "function": ')
        assert not line.rstrip().endswith(",")


def test_hermes_render_manifest_normalizes_per_tool() -> None:
    """normalize_tool_schema runs per tool: $ref is inlined inside `parameters`."""
    import json as _json

    tool = BoundTool(
        name="lookup",
        description="Look up",
        args_json_schema={
            "type": "object",
            "properties": {"q": {"$ref": "#/$defs/Q"}},
            "$defs": {"Q": {"type": "string"}},
        },
    )
    manifest = HERMES_CHATML_XML.render_system_manifest([tool], "ollama")
    line = manifest.split("<tools>\n", 1)[1].splitlines()[0]
    obj = _json.loads(line)
    params = obj["function"]["parameters"]
    # The $ref is inlined to its target (no dangling $ref in the advertised schema).
    assert params["properties"]["q"] == {"type": "string"}


# ---- native parse fixtures ----------------------------------------------


def _names_args(text: str, known: set[str]) -> list[tuple[str, dict]]:
    return [(c.tool_name, c.args) for c in HERMES_CHATML_XML.parse(text, known)]


def test_hermes_parse_single_tool_call() -> None:
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "KTM"}}\n</tool_call>'
    assert _names_args(text, _WEATHER_KNOWN) == [("get_weather", {"city": "KTM"})]


def test_hermes_parse_is_fail_open_where_generic_misses() -> None:
    """The native block the GENERIC parser returns [] for becomes a hit (add-only)."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "KTM"}}\n</tool_call>'
    assert parse_text_tool_calls(text, _WEATHER_KNOWN) == []  # generic misses it
    assert _names_args(text, _WEATHER_KNOWN) == [("get_weather", {"city": "KTM"})]


def test_hermes_parse_parallel_blocks() -> None:
    """Parallel = N back-to-back <tool_call> blocks (NOT a JSON array)."""
    text = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "KTM"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
    )
    assert _names_args(text, _WEATHER_KNOWN) == [
        ("get_weather", {"city": "KTM"}),
        ("get_time", {}),
    ]


def test_hermes_parse_order_insensitive_keys() -> None:
    """arguments-before-name still parses (key order does not matter)."""
    text = '<tool_call>\n{"arguments": {"city": "PKR"}, "name": "get_weather"}\n</tool_call>'
    assert _names_args(text, _WEATHER_KNOWN) == [("get_weather", {"city": "PKR"})]


def test_hermes_parse_skips_scratch_pad() -> None:
    """Hermes-3 <scratch_pad> reasoning is ignored; only <tool_call> is read."""
    text = (
        "<scratch_pad>I should check the weather for the user.</scratch_pad>\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "BKT"}}\n</tool_call>'
    )
    assert _names_args(text, _WEATHER_KNOWN) == [("get_weather", {"city": "BKT"})]


def test_hermes_parse_off_grammar_falls_through_to_tolerant_pass() -> None:
    """A TOOL_CALL marker (not <tool_call> XML) is recovered by the OR-in generic pass."""
    text = 'TOOL_CALL get_time {"tz": "Asia/Kathmandu"}'
    assert _names_args(text, _WEATHER_KNOWN) == [
        ("get_time", {"tz": "Asia/Kathmandu"})
    ]


def test_hermes_parse_unions_native_and_generic_without_dupes() -> None:
    """Native block + a generic-recoverable marker for distinct calls → both, once."""
    text = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "KTM"}}\n</tool_call>\n'
        "TOOL_CALL get_time {}"
    )
    assert _names_args(text, _WEATHER_KNOWN) == [
        ("get_weather", {"city": "KTM"}),
        ("get_time", {}),
    ]


def test_hermes_parse_dedupes_same_call() -> None:
    """The same call is never emitted twice across the native + generic passes."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "KTM"}}\n</tool_call>'
    assert len(HERMES_CHATML_XML.parse(text, _WEATHER_KNOWN)) == 1


def test_hermes_parse_string_arguments_decoded() -> None:
    """arguments delivered as a JSON string are decoded to a dict."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": "{\\"city\\": \\"KTM\\"}"}\n</tool_call>'
    assert _names_args(text, _WEATHER_KNOWN) == [("get_weather", {"city": "KTM"})]


def test_hermes_parse_malformed_block_skipped_never_raises() -> None:
    """A malformed JSON block is skipped; a later valid block still parses."""
    text = (
        "<tool_call>\n{not valid json}\n</tool_call>\n"
        '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
    )
    assert _names_args(text, _WEATHER_KNOWN) == [("get_time", {})]


def test_hermes_parse_missing_name_skipped() -> None:
    text = '<tool_call>\n{"arguments": {"city": "KTM"}}\n</tool_call>'
    assert HERMES_CHATML_XML.parse(text, _WEATHER_KNOWN) == []


def test_hermes_parse_empty_and_prose_are_empty() -> None:
    assert HERMES_CHATML_XML.parse("", _WEATHER_KNOWN) == []
    assert HERMES_CHATML_XML.parse("Just a prose answer.", _WEATHER_KNOWN) == []


def test_hermes_parse_idempotent() -> None:
    """Parsing the same text twice yields the same (name, args) records."""
    text = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "KTM"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
    )
    assert _names_args(text, _WEATHER_KNOWN) == _names_args(text, _WEATHER_KNOWN)


# ---- render_tool_results (<tool_response>) -------------------------------


def test_hermes_render_tool_results_single() -> None:
    results = [
        ToolReturnRecord(
            tool_call_id="a", tool_name="get_weather", content="sunny, 24C"
        )
    ]
    assert HERMES_CHATML_XML.render_tool_results(results) == (
        "<|im_start|>user\n<tool_response>\nsunny, 24C\n</tool_response><|im_end|>\n"
    )


def test_hermes_render_tool_results_batches_consecutive() -> None:
    """Consecutive results share ONE user turn (HF-canonical Qwen2.5 batching)."""
    results = [
        ToolReturnRecord(
            tool_call_id="a", tool_name="get_weather", content="sunny"
        ),
        ToolReturnRecord(tool_call_id="b", tool_name="get_time", content="10:00"),
    ]
    assert HERMES_CHATML_XML.render_tool_results(results) == (
        "<|im_start|>user"
        "\n<tool_response>\nsunny\n</tool_response>"
        "\n<tool_response>\n10:00\n</tool_response>"
        "<|im_end|>\n"
    )


def test_hermes_render_tool_results_empty() -> None:
    assert HERMES_CHATML_XML.render_tool_results([]) == ""


# ---- selection rows ------------------------------------------------------


def test_hermes_selected_for_qwen_instruct_tags() -> None:
    for tag in (
        "qwen2.5:0.5b-instruct",
        "qwen2.5:3b-instruct",
        "qwen2.5:7b-instruct",
    ):
        assert format_for(tag) is HERMES_CHATML_XML


def test_hermes_selected_for_hermes_tags() -> None:
    for tag in ("hermes3:8b", "NousResearch/Hermes-2-Pro-Llama-3-8B", "Hermes-3"):
        assert format_for(tag) is HERMES_CHATML_XML


def test_qwen_coder_excluded_to_generic() -> None:
    """qwen2.5-coder is NOT this grammar → it stays on GENERIC."""
    assert format_for("qwen2.5-coder:7b") is GENERIC
    assert format_for("qwen2.5-coder:32b-instruct") is GENERIC


def test_override_beats_exclude() -> None:
    """An explicit override wins even over an exclude veto (operator is trusted)."""
    assert (
        format_for("qwen2.5-coder:7b", override="hermes_chatml_xml")
        is HERMES_CHATML_XML
    )


def test_hermes_case_insensitive_and_suffix_tolerant() -> None:
    assert format_for("QWEN2.5:7B-INSTRUCT") is HERMES_CHATML_XML
    assert format_for("vendor/hermes-2-pro:latest") is HERMES_CHATML_XML


# ---- flags are data-only -------------------------------------------------


def test_hermes_flags_pinned() -> None:
    f = HERMES_CHATML_XML.flags
    assert f.arg_key == "arguments"
    assert f.name_key == "name"
    assert f.result_role == "user"
    assert f.batch_consecutive_results is True
    assert f.parallel_supported is True
    assert f.use_text_tool_path is True


# ---- Ollama text-path wiring (FORMAT is the single A/B variable) ----------


def _ollama_payload_for(model: str) -> dict[str, Any]:
    """Capture the Ollama /api/chat payload a tools request produces for ``model``."""
    captured: dict[str, Any] = {}

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        # No native tool_calls → exercises the text/prose parse path.
        return {"message": {"content": ""}}

    mgr = OllamaClientManager(
        model=model, transport=transport, model_registry={"default": model}
    )
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="weather in KTM?")],
        bound_tools=[_WEATHER_TOOL],
    )
    run_async(mgr.generate(req))
    return captured["payload"]


def test_ollama_generic_keeps_native_tools_field() -> None:
    """A GENERIC model (llama) keeps Ollama's native `tools=` field (today's path)."""
    payload = _ollama_payload_for("llama3.2")
    assert "tools" in payload
    assert payload["tools"][0]["function"]["name"] == "get_weather"
    # No manifest system message injected on the native path.
    assert all(m["role"] != "system" for m in payload["messages"])


def test_ollama_hermes_suppresses_native_tools_and_injects_manifest() -> None:
    """A Hermes/Qwen model drives the text path: no native tools, manifest injected."""
    payload = _ollama_payload_for("qwen2.5:7b-instruct")
    assert "tools" not in payload  # native function-tool API suppressed
    system = payload["messages"][0]
    assert system["role"] == "system"
    assert system["content"] == HERMES_CHATML_XML.render_system_manifest(
        [_WEATHER_TOOL], "ollama"
    )


def test_ollama_hermes_text_path_parses_xml_tool_call() -> None:
    """On the text path, an emitted <tool_call> block is parsed into a tool call."""

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "message": {
                "content": (
                    "<tool_call>\n"
                    '{"name": "get_weather", "arguments": {"city": "KTM"}}\n'
                    "</tool_call>"
                )
            }
        }

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        return ToolReturnRecord(
            tool_call_id="x", tool_name=name, content=f"sunny in {args['city']}"
        )

    mgr = OllamaClientManager(
        model="qwen2.5:7b-instruct",
        transport=transport,
        model_registry={"default": "qwen2.5:7b-instruct"},
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
