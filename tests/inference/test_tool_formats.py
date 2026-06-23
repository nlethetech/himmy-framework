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
    HimalayaGptClientManager,
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
    HERMES_CHATML_XML_FEWSHOT,
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
    """No override, unknown tag → GENERIC (the no-op default).

    NOTE: ``llama3``/``mistral`` tags now auto-select their native formats (LLAMA3_JSON /
    MISTRAL_V3) — see the LLAMA3_JSON / MISTRAL_V3 selection tests. The tags here are
    genuinely unrecognized families that still fall through to GENERIC.
    """
    assert format_for("phi3:mini") is GENERIC
    assert format_for("gemma2:9b") is GENERIC
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
    "\n\n"
    "When several independent calls are needed, emit multiple "
    "<tool_call></tool_call> blocks back-to-back in the same reply (one JSON object "
    "per block). If no available function fits the request, do not emit a "
    "<tool_call>; answer the user directly."
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
    """A GENERIC model keeps Ollama's native `tools=` field (today's path).

    Uses a family with no native format row (gemma2) — ``llama3.2`` now auto-selects
    LLAMA3_JSON and drives the text path, which is asserted separately.
    """
    payload = _ollama_payload_for("gemma2:9b")
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


# ========================================================================== #
# PHASE 2 — HERMES_CHATML_XML_FEWSHOT (the recipe that makes the 0.5b call tools)
# ========================================================================== #

# The few-shot variant shares the canonical Hermes grammar; only the MANIFEST differs
# (synthesized exemplars + the "output only <tool_call>, no code" nudge). The canonical
# HERMES_CHATML_XML and GENERIC manifests must stay byte-identical (no regression).


def test_fewshot_registered_and_opt_in_only() -> None:
    """The few-shot format is registered but NEVER auto-selects by model tag."""
    assert get_format("hermes_chatml_xml_fewshot") is HERMES_CHATML_XML_FEWSHOT
    assert HERMES_CHATML_XML_FEWSHOT.model_tags == frozenset()
    # A bare tag (no override) keeps the lean zero-shot Hermes manifest, not few-shot.
    assert format_for("qwen2.5:7b-instruct") is HERMES_CHATML_XML
    # Opt-in via override resolves it.
    assert (
        format_for("anything", override="hermes_chatml_xml_fewshot")
        is HERMES_CHATML_XML_FEWSHOT
    )


def test_fewshot_flags_pinned() -> None:
    f = HERMES_CHATML_XML_FEWSHOT.flags
    assert f.fewshot == 2
    assert f.no_code_instruction is True
    assert f.use_text_tool_path is True
    # Shares the wire-grammar knobs with canonical Hermes (only the manifest differs).
    assert f.arg_key == "arguments"
    assert f.name_key == "name"
    assert f.result_role == "user"


def test_canonical_hermes_manifest_unaffected_by_fewshot() -> None:
    """Adding the few-shot format must not regress the canonical Hermes manifest."""
    assert (
        HERMES_CHATML_XML.render_system_manifest([_WEATHER_TOOL], "ollama")
        == _GOLDEN_HERMES_MANIFEST
    )
    # The canonical manifest carries NO examples / no-code nudge.
    assert "# Examples" not in _GOLDEN_HERMES_MANIFEST
    assert "output ONLY" not in HERMES_CHATML_XML.render_system_manifest(
        [_WEATHER_TOOL], "ollama"
    )


#: The exact few-shot manifest for the single get_weather tool — the canonical body, the
#: no-code nudge, the anti-parrot framing line, then a single synthesized exemplar per
#: fewshot count (2). The placeholder value is an obviously-fake "<example-text>" sentinel
#: (FIX 3) and the framing tells the model to fill args from the user, not the example.
_GOLDEN_FEWSHOT_MANIFEST = (
    _GOLDEN_HERMES_MANIFEST
    + "\n\n"
    + (
        "When a tool is needed, output ONLY the <tool_call>...</tool_call> line and "
        "NOTHING else — do not write Python, do not use code fences, do not explain. "
        "If no tool is needed, answer the user directly in prose."
    )
    + "\n\n# Examples\n\n"
    + (
        "Below are examples of the FORMAT only. Fill the arguments from the USER's "
        "request, NOT from the examples."
    )
    + "\n\n"
    + (
        "Example — user: Use the get_weather tool.\n"
        "Example — assistant:\n<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "<example-text>"}}\n'
        "</tool_call>"
    )
)


def test_fewshot_render_manifest_golden_single_tool() -> None:
    """The few-shot manifest = canonical body + no-code nudge + framing + 1 exemplar.

    With one bound tool, fewshot=2 still yields a single exemplar (n is capped at the
    tool count) — the golden pins exemplar shape, placeholder fill, framing, and ordering.
    """
    got = HERMES_CHATML_XML_FEWSHOT.render_system_manifest([_WEATHER_TOOL], "ollama")
    assert got == _GOLDEN_FEWSHOT_MANIFEST


def test_fewshot_exemplar_count_capped_at_two() -> None:
    """fewshot=2 over many tools emits exactly two exemplars over distinct tools."""
    many = [
        BoundTool(name=f"tool_{i}", description="d", args_json_schema={"type": "object"})
        for i in range(4)
    ]
    manifest = HERMES_CHATML_XML_FEWSHOT.render_system_manifest(many, "ollama")
    assert manifest.count("Example — user:") == 2
    # The two exemplars cover the first two DISTINCT tools (anti-parrot diversity).
    assert "Use the tool_0 tool." in manifest
    assert "Use the tool_1 tool." in manifest


def test_fewshot_exemplar_fills_typed_placeholder() -> None:
    """The synthesized exemplar fills the first property with a fake type-correct value."""
    int_tool = BoundTool(
        name="egg_totals",
        description="Sum eggs.",
        args_json_schema={
            "type": "object",
            "properties": {"days": {"type": "integer"}},
        },
    )
    manifest = HERMES_CHATML_XML_FEWSHOT.render_system_manifest([int_tool], "ollama")
    # FIX 3: int placeholder is a distinctive 4242 (not a small round int the model
    # parrots like 99/2/5, and not a value any plausible task would carry).
    assert '{"name": "egg_totals", "arguments": {"days": 4242}}' in manifest


def test_fewshot_empty_tools_no_examples_block() -> None:
    """No bound tools → no exemplars (the synthesizer needs a tool to copy)."""
    manifest = HERMES_CHATML_XML_FEWSHOT.render_system_manifest([], "ollama")
    assert "# Examples" not in manifest


def test_fewshot_parser_is_canonical_hermes_parser() -> None:
    """The few-shot variant parses identically to canonical Hermes (same grammar)."""
    text = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "KTM"}}\n'
        "</tool_call>"
    )
    assert [
        (c.tool_name, c.args) for c in HERMES_CHATML_XML_FEWSHOT.parse(text, _WEATHER_KNOWN)
    ] == [("get_weather", {"city": "KTM"})]


def test_fewshot_render_tool_results_is_canonical() -> None:
    """Result feedback uses the same <tool_response> grammar as canonical Hermes."""
    results = [
        ToolReturnRecord(tool_call_id="a", tool_name="get_weather", content="sunny")
    ]
    assert HERMES_CHATML_XML_FEWSHOT.render_tool_results(
        results
    ) == HERMES_CHATML_XML.render_tool_results(results)


# ---- HimalayaGptClientManager render/parse path --------------------------


def _hgpt_prompt_and_resp(
    *,
    reply: str,
    tools: list[BoundTool],
    tool_call_format: str | None = None,
    extra_messages: list[InferenceMessage] | None = None,
    executor: Any = None,
) -> tuple[str, Any]:
    """Drive the manager with a fake generate_fn; return (prompt seen, response)."""
    captured: dict[str, str] = {}

    def gen(prompt: str) -> str:
        captured["prompt"] = prompt
        return reply

    mgr = HimalayaGptClientManager(
        generate_fn=gen, tool_call_format=tool_call_format
    )
    messages = [InferenceMessage(role="user", content="weather in KTM?")]
    if extra_messages:
        messages.extend(extra_messages)
    req = InferenceRequest(
        messages=messages, bound_tools=tools, tool_executor=executor
    )
    resp = run_async(mgr.generate(req))
    return captured["prompt"], resp


def test_hgpt_defaults_to_fewshot_hermes_format() -> None:
    """No explicit format → the 0.5b manager renders the FEW-SHOT Hermes manifest.

    With multi-turn few-shot (FIX 2) the flattened in-manifest ``# Examples`` block is
    replaced by REAL prior turns, so the manifest no longer carries ``# Examples``; the
    exemplar surfaces as a ``[User]``/``[Assistant]`` turn pair instead.
    """
    prompt, _ = _hgpt_prompt_and_resp(reply="hi", tools=[_WEATHER_TOOL])
    assert "<tools>" in prompt  # tool manifest injected
    assert "# Examples" not in prompt  # flattened block gone (exemplars are turns now)
    assert "Use the get_weather tool." in prompt  # exemplar present as a real turn
    assert "output ONLY" in prompt  # the no-code nudge present
    assert "[System]" in prompt  # injected as a system block


def test_hgpt_parses_and_executes_tool_call_through_manager() -> None:
    """End-to-end: the manager renders, the model emits <tool_call>, it is executed."""

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        return ToolReturnRecord(
            tool_call_id="x", tool_name=name, content=f"sunny in {args['city']}"
        )

    reply = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "KTM"}}\n'
        "</tool_call>"
    )
    _, resp = _hgpt_prompt_and_resp(
        reply=reply, tools=[_WEATHER_TOOL], executor=executor
    )
    assert [(c.tool_name, c.args) for c in resp.tool_calls] == [
        ("get_weather", {"city": "KTM"})
    ]
    assert resp.tool_returns[0].content == "sunny in KTM"
    assert resp.output_text == ""  # a tool call, not a final answer


def test_hgpt_prose_reply_passes_through_when_no_tool_call() -> None:
    """A prose answer (no <tool_call>) is returned as the final answer untouched."""
    _, resp = _hgpt_prompt_and_resp(
        reply="It is sunny today.", tools=[_WEATHER_TOOL]
    )
    assert resp.tool_calls == []
    assert resp.output_text == "It is sunny today."


def test_hgpt_no_tools_is_plain_chat_no_manifest() -> None:
    """With no bound tools the prompt is the plain composed chat (strict no-op)."""
    captured: dict[str, str] = {}

    def gen(prompt: str) -> str:
        captured["prompt"] = prompt
        return "काठमाडौं"

    mgr = HimalayaGptClientManager(generate_fn=gen)
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="राजधानी?")]
    )
    resp = run_async(mgr.generate(req))
    assert "<tools>" not in captured["prompt"]
    assert "# Examples" not in captured["prompt"]
    assert resp.output_text == "काठमाडौं"


def test_hgpt_feeds_tool_results_in_format_grammar() -> None:
    """Prior tool returns on the thread are re-rendered via the format's grammar."""
    prompt, _ = _hgpt_prompt_and_resp(
        reply="done",
        tools=[_WEATHER_TOOL],
        extra_messages=[InferenceMessage(role="tool", content="sunny, 24C")],
    )
    # The tool result is fed back inside the Hermes <tool_response> envelope.
    assert "<tool_response>\nsunny, 24C\n</tool_response>" in prompt


def test_hgpt_explicit_generic_format_overrides_default() -> None:
    """Pinning 'generic' opts OUT of the few-shot Hermes manifest (operator trusted)."""
    prompt, _ = _hgpt_prompt_and_resp(
        reply="hi", tools=[_WEATHER_TOOL], tool_call_format="generic"
    )
    # GENERIC is not a text-path format → the manager uses the plain composed prompt,
    # so neither the Hermes <tools> block nor the few-shot examples appear.
    assert "<tools>" not in prompt
    assert "# Examples" not in prompt


# ========================================================================== #
# FIX 2 — TRUE MULTI-TURN FEW-SHOT (exemplars as REAL prior turns)
# ========================================================================== #

from himmy.services.inference.tool_formats import render_fewshot_turns  # noqa: E402


def test_render_fewshot_turns_are_user_assistant_pairs() -> None:
    """render_fewshot_turns emits a (user, assistant) pair per exemplar tool."""
    turns = HERMES_CHATML_XML_FEWSHOT.fewshot_turns([_WEATHER_TOOL])
    assert turns == [
        ("user", "Use the get_weather tool."),
        (
            "assistant",
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "<example-text>"}}\n</tool_call>',
        ),
    ]


def test_render_fewshot_turns_two_distinct_tools() -> None:
    """fewshot=2 over many tools → two (user, assistant) pairs over the first two tools."""
    many = [
        BoundTool(name=f"tool_{i}", description="d", args_json_schema={"type": "object"})
        for i in range(4)
    ]
    turns = render_fewshot_turns(many, HERMES_CHATML_XML_FEWSHOT.flags)
    assert len(turns) == 4  # 2 exemplars * (user + assistant)
    assert turns[0] == ("user", "Use the tool_0 tool.")
    assert turns[2] == ("user", "Use the tool_1 tool.")


def test_render_fewshot_turns_empty_without_tools() -> None:
    assert HERMES_CHATML_XML_FEWSHOT.fewshot_turns([]) == []


def test_canonical_hermes_has_no_fewshot_turns() -> None:
    """Single-turn formats expose no multi-turn renderer (opt-in only)."""
    assert HERMES_CHATML_XML.fewshot_turns is None
    assert GENERIC.fewshot_turns is None


def test_base_manifest_omits_flattened_examples() -> None:
    """render_system_manifest_base drops the flattened # Examples (used with turns)."""
    base = HERMES_CHATML_XML_FEWSHOT.render_system_manifest_base(
        [_WEATHER_TOOL], "ollama"
    )
    assert "# Examples" not in base
    # ...but keeps the no-code nudge + canonical body.
    assert "output ONLY" in base
    assert "<tools>" in base


def test_hgpt_renders_fewshot_as_real_prior_turns() -> None:
    """The 0.5b manager composes few-shot as REAL [User]/[Assistant] turns, no flatten."""
    prompt, _ = _hgpt_prompt_and_resp(reply="hi", tools=[_WEATHER_TOOL])
    # The flattened in-manifest block is GONE (exemplars are turns now).
    assert "# Examples" not in prompt
    assert "Example — user:" not in prompt
    # The exemplar is a real prior turn pair, in order, between manifest and real user.
    sys_idx = prompt.index("[System]")
    ex_user_idx = prompt.index("[User]\nUse the get_weather tool.")
    ex_asst_idx = prompt.index("[Assistant]\n<tool_call>")
    real_user_idx = prompt.index("[User]\nweather in KTM?")
    assert sys_idx < ex_user_idx < ex_asst_idx < real_user_idx
    # The system block still advertises the tools + the no-code nudge.
    assert "<tools>" in prompt
    assert "output ONLY" in prompt


def test_hgpt_generic_format_has_no_multi_turn_fewshot() -> None:
    """A format without fewshot_turns (generic) keeps the plain single-turn prompt."""
    prompt, _ = _hgpt_prompt_and_resp(
        reply="hi", tools=[_WEATHER_TOOL], tool_call_format="generic"
    )
    assert "Use the get_weather tool." not in prompt


def test_hgpt_canonical_hermes_format_stays_single_turn() -> None:
    """Canonical Hermes on the 0.5b manager: no exemplar turns, no flattened examples."""
    prompt, _ = _hgpt_prompt_and_resp(
        reply="hi", tools=[_WEATHER_TOOL], tool_call_format="hermes_chatml_xml"
    )
    assert "# Examples" not in prompt
    assert "Use the get_weather tool." not in prompt
    assert "<tools>" in prompt  # but the manifest is still rendered


# ========================================================================== #
# FIX 3 — ANTI-PARROT EXEMPLARS (values can't collide with eval inputs)
# ========================================================================== #


def test_fewshot_framing_instruction_present() -> None:
    """The 'fill from the USER's request' anti-parrot framing is in the manifest."""
    manifest = HERMES_CHATML_XML_FEWSHOT.render_system_manifest([_WEATHER_TOOL], "ollama")
    assert "Fill the arguments from the USER's request, NOT from the examples" in manifest


def test_fewshot_exemplar_values_differ_from_eval_inputs() -> None:
    """The synthesized exemplar values do NOT collide with the A/B eval-suite inputs.

    If they did, the 0.5b could parrot the exemplar literal and still 'pass'. Assert the
    placeholder values (the only literals the exemplar carries) are disjoint from every
    arg value the eval tasks expect.
    """
    import json as _json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "himalayagpt"))
    from tool_suite import TASKS, TOOLS  # type: ignore[import-not-found]

    # Every literal value the exemplars carry (across both fewshot tools).
    turns = render_fewshot_turns(TOOLS, HERMES_CHATML_XML_FEWSHOT.flags)
    exemplar_values: set[str] = set()
    for role, content in turns:
        if role != "assistant":
            continue
        body = content.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
        args = _json.loads(body)["arguments"]
        for v in args.values():
            exemplar_values.add(str(v))

    # Every expected arg value across the eval suite.
    eval_values: set[str] = set()
    for task in TASKS:
        for v in task["expect_args"].values():
            eval_values.add(str(v))

    assert exemplar_values  # exemplars actually carry values
    assert exemplar_values.isdisjoint(eval_values), (
        f"exemplar values {exemplar_values} collide with eval inputs {eval_values} "
        "— the model could parrot them"
    )


# ========================================================================== #
# FIX 4 — EXEMPLAR SYNTHESIZER FILLS *EVERY* REQUIRED PROP (the EN 50% ceiling)
# ========================================================================== #

from himmy.services.inference.tool_formats import (  # noqa: E402
    _example_args_for,
    _fewshot_tools,
)

# Two- and three-arg tools the single-property synthesizer could never satisfy.
_ADD_TOOL = BoundTool(
    name="add_numbers",
    description="Add two integers together and return the sum.",
    args_json_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
    read_only=True,
)
_TRANSLATE_TOOL = BoundTool(
    name="translate_text",
    description="Translate text into a target language.",
    args_json_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "target_language": {"type": "string"},
        },
        "required": ["text", "target_language"],
    },
    read_only=True,
)


def test_example_args_fills_all_required_props_two_arg() -> None:
    """A 2-arg tool's exemplar fills BOTH required args (was only the first → 50% cap)."""
    args = _example_args_for(_ADD_TOOL)
    assert set(args) == {"a", "b"}
    assert args["a"] == 4242 and args["b"] == 4242  # both int sentinels


def test_example_args_two_string_args_get_distinct_values() -> None:
    """A 2-string-arg tool's exemplar gives each slot a DISTINCT fake value (not one
    repeated literal) so the exemplar demonstrates two real, different arguments."""
    args = _example_args_for(_TRANSLATE_TOOL)
    assert set(args) == {"text", "target_language"}
    assert args["text"] != args["target_language"]
    # both are obviously-fake bracketed sentinels
    assert args["text"].startswith("<") and args["target_language"].startswith("<")


def test_example_args_uses_required_not_just_first_prop() -> None:
    """Only the REQUIRED props are filled; an optional prop is left out of the exemplar."""
    tool = BoundTool(
        name="search",
        description="Search.",
        args_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    args = _example_args_for(tool)
    assert set(args) == {"query"}  # only the required prop


def test_example_args_no_required_key_fills_all_declared() -> None:
    """When a schema omits `required`, the exemplar falls back to all declared props."""
    tool = BoundTool(
        name="noreq",
        description="No required key.",
        args_json_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "string"}},
        },
    )
    args = _example_args_for(tool)
    assert set(args) == {"x", "y"}


def _suite_tools() -> list[BoundTool]:
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "experiments" / "himalayagpt")
    )
    from tool_suite import TOOLS  # type: ignore[import-not-found]

    return list(TOOLS)


def test_every_synthesized_exemplar_fills_all_required_props_over_suite() -> None:
    """Over the real A/B suite: EVERY synthesized exemplar fills ALL of its tool's
    required props — the contract that lets 2-arg tools (add_numbers, translate_text)
    pass the args check and lifts the EN ceiling past 50%."""
    import json as _json

    tools = _suite_tools()
    chosen = _fewshot_tools(tools, HERMES_CHATML_XML_FEWSHOT.flags)
    assert chosen  # the synthesizer actually selected exemplar tools

    by_name = {t.name: t for t in tools}
    turns = render_fewshot_turns(tools, HERMES_CHATML_XML_FEWSHOT.flags)
    asst = [c for r, c in turns if r == "assistant"]
    assert len(asst) == len(chosen)
    for content in asst:
        body = content.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
        call = _json.loads(body)
        tool = by_name[call["name"]]
        required = set((tool.args_json_schema or {}).get("required") or [])
        assert required.issubset(set(call["arguments"])), (
            f"exemplar for {call['name']} missing required props "
            f"{required - set(call['arguments'])}"
        )


def test_fewshot_exemplar_set_demonstrates_a_two_arg_tool() -> None:
    """The exemplar set covers >=2 distinct tools AND includes at least one 2-arg tool
    (when the toolset has one) — a single-arg-only exemplar set is what capped EN at 50%."""
    tools = _suite_tools()
    chosen = _fewshot_tools(tools, HERMES_CHATML_XML_FEWSHOT.flags)
    assert len(chosen) >= 2  # >=2 diverse exemplars
    assert len({t.name for t in chosen}) == len(chosen)  # over DIFFERENT tools

    def _req_count(t: BoundTool) -> int:
        schema = t.args_json_schema or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or list(props)
        return len([k for k in props if k in required])

    assert any(_req_count(t) >= 2 for t in chosen), (
        "exemplar set has no multi-arg tool — 2-arg tasks can never be demonstrated"
    )


# ========================================================================== #
# PRE-BOOST BASELINE FORMAT — the A/B baseline arm. It reproduces main's ORIGINAL
# single-turn few-shot so the boost is the SOLE difference in the powered A/B.
# ========================================================================== #

from himmy.services.inference.tool_formats import (  # noqa: E402
    HERMES_CHATML_XML_FEWSHOT_BASELINE,
)


def test_baseline_format_registered_and_opt_in_only() -> None:
    """The baseline format is registered by name and NEVER auto-selects (opt-in)."""
    assert (
        get_format("hermes_chatml_xml_fewshot_baseline")
        is HERMES_CHATML_XML_FEWSHOT_BASELINE
    )
    assert HERMES_CHATML_XML_FEWSHOT_BASELINE.model_tags == frozenset()
    # An explicit override resolves to it; nothing auto-selects it by tag.
    assert (
        format_for("qwen2.5", override="hermes_chatml_xml_fewshot_baseline")
        is HERMES_CHATML_XML_FEWSHOT_BASELINE
    )


def test_baseline_is_single_turn_no_python_fallback() -> None:
    """The baseline has NO multi-turn few-shot and NO python-call fallback (pre-boost)."""
    assert HERMES_CHATML_XML_FEWSHOT_BASELINE.fewshot_turns is None
    assert HERMES_CHATML_XML_FEWSHOT_BASELINE.render_system_manifest_base is None
    assert HERMES_CHATML_XML_FEWSHOT_BASELINE.flags.python_call_fallback is False
    # It still carries the pre-boost few-shot knobs (those existed before the boost).
    assert HERMES_CHATML_XML_FEWSHOT_BASELINE.flags.fewshot == 2
    assert HERMES_CHATML_XML_FEWSHOT_BASELINE.flags.no_code_instruction is True


def test_baseline_exemplar_fills_only_first_prop() -> None:
    """The baseline 2-arg exemplar shows ONLY the first arg (the documented 50% cap)."""
    import json as _json
    import re

    manifest = HERMES_CHATML_XML_FEWSHOT_BASELINE.render_system_manifest(
        _suite_tools(), "himalayagpt"
    )
    # Find the add_numbers exemplar in the flattened # Examples block.
    assert "# Examples" in manifest
    block = manifest.split("# Examples", 1)[1]
    # Each exemplar is a full ``{"name": ..., "arguments": ...}`` JSON object inside a
    # <tool_call> tag; pull the add_numbers one and confirm it shows only the first prop.
    add_call = next(
        c
        for c in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", block, re.DOTALL)
        if '"add_numbers"' in c
    )
    args = _json.loads(add_call)["arguments"]
    assert set(args) == {"a"}, f"baseline exemplar should show only first prop, got {args}"
    assert args["a"] == 1  # the parrotable pre-boost int placeholder


def test_baseline_manifest_byte_identical_to_main_prefix() -> None:
    """The baseline manifest carries no anti-parrot framing line (pre-boost)."""
    manifest = HERMES_CHATML_XML_FEWSHOT_BASELINE.render_system_manifest(
        _suite_tools(), "himalayagpt"
    )
    assert "Fill the arguments from the USER's" not in manifest  # no anti-parrot framing


def test_fewshot_swaps_in_multi_arg_when_leading_slice_all_single_arg() -> None:
    """If the first-N slice is all single-arg, a multi-arg tool is swapped into the last
    slot so the exemplar set always demonstrates a 2-arg call when one is available."""
    single_a = BoundTool(
        name="s_a", description="d",
        args_json_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    )
    single_b = BoundTool(
        name="s_b", description="d",
        args_json_schema={"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]},
    )
    multi = BoundTool(
        name="m", description="d",
        args_json_schema={
            "type": "object",
            "properties": {"p": {"type": "string"}, "q": {"type": "string"}},
            "required": ["p", "q"],
        },
    )
    # multi is LAST, outside the leading fewshot=2 slice [s_a, s_b].
    chosen = _fewshot_tools([single_a, single_b, multi], HERMES_CHATML_XML_FEWSHOT.flags)
    names = {t.name for t in chosen}
    assert "m" in names  # the multi-arg tool was swapped in
    assert len(chosen) == 2


# ========================================================================== #
# FIX 4 — GENERIC + canonical-Hermes manifests stay BYTE-IDENTICAL (re-pinned SHA)
# ========================================================================== #

import hashlib  # noqa: E402

#: Re-pinned SHA-256 of the GENERIC + canonical-Hermes manifests over a representative
#: tool set (incl. a 2-arg tool) across all three providers, RU-separated. The exemplar-
#: synthesizer rework touches ONLY the opt-in few-shot path; if either of these hashes
#: drifts, a "no-op" change leaked into GENERIC or canonical Hermes — a regression.
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
_GENERIC_MANIFEST_SHA = (
    "a3843801b22db0510d330b12d6e7156e706ef7ddf54eeba0ab98fd2dc64b3e74"
)
_HERMES_MANIFEST_SHA = (
    "6b8be9a85924495a245c43bd6aa02184a7f6664717f12b73717873038e92cd8c"
)


def _manifest_sha(fmt: ToolCallFormat) -> str:
    blob = ""
    for provider in ("claude-cli", "ollama", "himalayagpt"):
        blob += fmt.render_system_manifest(_SHA_PIN_TOOLS, provider) + "\x1e"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_generic_manifest_sha_unchanged() -> None:
    """GENERIC manifest is byte-identical to the pinned SHA (the rework is a no-op here)."""
    assert _manifest_sha(GENERIC) == _GENERIC_MANIFEST_SHA


def test_canonical_hermes_manifest_sha_unchanged() -> None:
    """Canonical Hermes manifest is byte-identical to the pinned SHA (no fewshot leakage)."""
    assert _manifest_sha(HERMES_CHATML_XML) == _HERMES_MANIFEST_SHA


def test_generic_and_canonical_parse_byte_unchanged() -> None:
    """GENERIC + canonical-Hermes PARSE are unchanged by the few-shot rework.

    GENERIC stays record-identical to parse_text_tool_calls; canonical Hermes parses a
    native <tool_call> the same way (the synthesizer touches rendering only, not parse)."""
    samples = [
        'TOOL_CALL get_weather {"city": "KTM"}',
        '<tool_call>\n{"name": "add_numbers", "arguments": {"a": 1, "b": 2}}\n</tool_call>',
        "Just prose.",
    ]
    known = {"get_weather", "add_numbers"}
    for text in samples:
        assert [(c.tool_name, c.args) for c in GENERIC.parse(text, known)] == [
            (c.tool_name, c.args) for c in parse_text_tool_calls(text, known)
        ]
    native = '<tool_call>\n{"name": "add_numbers", "arguments": {"a": 1, "b": 2}}\n</tool_call>'
    assert [
        (c.tool_name, c.args) for c in HERMES_CHATML_XML.parse(native, known)
    ] == [("add_numbers", {"a": 1, "b": 2})]
