"""The shared tool-use A/B suite: bound tools + tasks (EN + NE), and the graders.

Tools are English-named (tool names stay English even in the Nepali arm — only the
USER prompt is translated). Each task declares:
  * the prompt (EN) and its Nepali translation (NE),
  * the EXPECTED tool name (selection),
  * the EXPECTED args (argument-correctness), graded via the shipped benchmark
    ``grade_trajectory`` ``tool_called_with`` predicate (each listed arg present and
    scalar-equal; extra args ignored), so this is the real arg-level grader, not a
    hand-rolled compare.

The suite is deliberately small + unambiguous (one obvious tool per task) so a 0.5b
has a fair shot and so the SELECTION signal is clean. N is bounded by CPU speed; the
runner reports the N actually completed per cell.
"""

from __future__ import annotations

from himmy.services.inference.models import BoundTool

# --------------------------------------------------------------------------- #
# Bound tools (pure data; English-named). A small, distinct toolset so tool
# SELECTION is a real choice (the model must pick the right one of several).
# --------------------------------------------------------------------------- #
TOOLS: list[BoundTool] = [
    BoundTool(
        name="get_weather",
        description="Get the current weather for a city.",
        args_json_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
            },
            "required": ["city"],
        },
        read_only=True,
    ),
    BoundTool(
        name="add_numbers",
        description="Add two integers together and return the sum.",
        args_json_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First integer."},
                "b": {"type": "integer", "description": "Second integer."},
            },
            "required": ["a", "b"],
        },
        read_only=True,
    ),
    BoundTool(
        name="get_stock_price",
        description="Get the latest stock price for a ticker symbol.",
        args_json_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL."},
            },
            "required": ["ticker"],
        },
        read_only=True,
    ),
    BoundTool(
        name="translate_text",
        description="Translate text into a target language.",
        args_json_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate."},
                "target_language": {
                    "type": "string",
                    "description": "Target language, e.g. French.",
                },
            },
            "required": ["text", "target_language"],
        },
        read_only=True,
    ),
    BoundTool(
        name="set_timer",
        description="Set a countdown timer for a number of minutes.",
        args_json_schema={
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Duration in minutes."},
            },
            "required": ["minutes"],
        },
        read_only=False,
    ),
]


# --------------------------------------------------------------------------- #
# Tasks. ``expect_tool`` = the right tool (selection). ``expect_args`` = the args the
# call must carry (argument-correctness). ``en`` / ``ne`` are the user prompts; only the
# prompt is translated — tool names + arg keys stay English.
# --------------------------------------------------------------------------- #
TASKS: list[dict] = [
    {
        "id": "weather_paris",
        "en": "What is the weather in Paris right now?",
        "ne": "अहिले प्यारिसको मौसम कस्तो छ?",
        "expect_tool": "get_weather",
        "expect_args": {"city": "Paris"},
    },
    {
        "id": "weather_tokyo",
        "en": "Tell me the current weather in Tokyo.",
        "ne": "टोकियोको अहिलेको मौसम बताउनुहोस्।",
        "expect_tool": "get_weather",
        "expect_args": {"city": "Tokyo"},
    },
    {
        "id": "add_7_5",
        "en": "Add 7 and 5 for me.",
        "ne": "मेरो लागि ७ र ५ जोड्नुहोस्।",
        "expect_tool": "add_numbers",
        "expect_args": {"a": 7, "b": 5},
    },
    {
        "id": "add_42_8",
        "en": "What is 42 plus 8?",
        "ne": "४२ जोड ८ कति हुन्छ?",
        "expect_tool": "add_numbers",
        "expect_args": {"a": 42, "b": 8},
    },
    {
        "id": "stock_aapl",
        "en": "What is the latest stock price of AAPL?",
        "ne": "AAPL को पछिल्लो सेयर मूल्य कति छ?",
        "expect_tool": "get_stock_price",
        "expect_args": {"ticker": "AAPL"},
    },
    {
        "id": "stock_tsla",
        "en": "Get me the current stock price for TSLA.",
        "ne": "TSLA को अहिलेको सेयर मूल्य ल्याउनुहोस्।",
        "expect_tool": "get_stock_price",
        "expect_args": {"ticker": "TSLA"},
    },
    {
        "id": "translate_hello_fr",
        "en": "Translate the text 'hello' into French.",
        "ne": "'hello' भन्ने पाठलाई फ्रेन्चमा अनुवाद गर्नुहोस्।",
        "expect_tool": "translate_text",
        "expect_args": {"text": "hello", "target_language": "French"},
    },
    {
        "id": "translate_thanks_spanish",
        "en": "Translate the text 'thank you' into Spanish.",
        "ne": "'thank you' भन्ने पाठलाई स्पेनिसमा अनुवाद गर्नुहोस्।",
        "expect_tool": "translate_text",
        "expect_args": {"text": "thank you", "target_language": "Spanish"},
    },
    {
        "id": "timer_10",
        "en": "Set a timer for 10 minutes.",
        "ne": "१० मिनेटको टाइमर सेट गर्नुहोस्।",
        "expect_tool": "set_timer",
        "expect_args": {"minutes": 10},
    },
    {
        "id": "timer_5",
        "en": "Start a 5 minute timer please.",
        "ne": "कृपया ५ मिनेटको टाइमर सुरु गर्नुहोस्।",
        "expect_tool": "set_timer",
        "expect_args": {"minutes": 5},
    },
]


def prompt_for(task: dict, lang: str) -> str:
    """The user prompt for a task in the given language ('en' | 'ne')."""
    return task["en"] if lang == "en" else task["ne"]
