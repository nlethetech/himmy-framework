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
        description="Set a countdown timer for a number of minutes, with a label.",
        args_json_schema={
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Duration in minutes."},
                "label": {"type": "string", "description": "What the timer is for."},
            },
            "required": ["minutes", "label"],
        },
        read_only=False,
    ),
    BoundTool(
        name="convert_currency",
        description="Convert an amount of money into a target currency.",
        args_json_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount of money."},
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code, e.g. USD.",
                },
            },
            "required": ["amount", "to_currency"],
        },
        read_only=True,
    ),
    BoundTool(
        name="schedule_reminder",
        description="Schedule a reminder with a message at a given time.",
        args_json_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Reminder text."},
                "time": {"type": "string", "description": "Time, e.g. 3pm."},
            },
            "required": ["message", "time"],
        },
        read_only=False,
    ),
]


# --------------------------------------------------------------------------- #
# Tasks. ``expect_tool`` = the right tool (selection). ``expect_args`` = the args the
# call must carry (argument-correctness). ``en`` / ``ne`` are the user prompts; only the
# prompt is translated — tool names + arg keys stay English.
# --------------------------------------------------------------------------- #
#: The suite is weighted toward 2-ARG tools (add_numbers, translate_text, set_timer,
#: convert_currency, schedule_reminder) because the exemplar all-args fix targets exactly
#: those: a pre-boost exemplar that demonstrates only ONE argument can never satisfy a
#: 2-arg tool's arg contract (the documented ~50% ceiling). The 1-arg tools (get_weather,
#: get_stock_price) are the control: the boost should not regress them. EN n=18.
TASKS: list[dict] = [
    # --- 1-arg control tasks (get_weather, get_stock_price) ---
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
        "id": "weather_london",
        "en": "How is the weather in London?",
        "ne": "लन्डनको मौसम कस्तो छ?",
        "expect_tool": "get_weather",
        "expect_args": {"city": "London"},
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
        "id": "stock_msft",
        "en": "What's the share price of MSFT?",
        "ne": "MSFT को सेयर मूल्य कति छ?",
        "expect_tool": "get_stock_price",
        "expect_args": {"ticker": "MSFT"},
    },
    # --- 2-arg tasks: add_numbers (a + b) ---
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
        "id": "add_13_29",
        "en": "Add 13 and 29 together.",
        "ne": "१३ र २९ जोड्नुहोस्।",
        "expect_tool": "add_numbers",
        "expect_args": {"a": 13, "b": 29},
    },
    # --- 2-arg tasks: translate_text (text + target_language) ---
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
        "id": "translate_goodbye_german",
        "en": "Translate the text 'goodbye' into German.",
        "ne": "'goodbye' भन्ने पाठलाई जर्मनमा अनुवाद गर्नुहोस्।",
        "expect_tool": "translate_text",
        "expect_args": {"text": "goodbye", "target_language": "German"},
    },
    # --- 2-arg tasks: set_timer (minutes + label) ---
    {
        "id": "timer_10_pasta",
        "en": "Set a timer for 10 minutes labeled pasta.",
        "ne": "pasta नाम दिएर १० मिनेटको टाइमर सेट गर्नुहोस्।",
        "expect_tool": "set_timer",
        "expect_args": {"minutes": 10, "label": "pasta"},
    },
    {
        "id": "timer_5_tea",
        "en": "Start a 5 minute timer called tea.",
        "ne": "tea नामको ५ मिनेटको टाइमर सुरु गर्नुहोस्।",
        "expect_tool": "set_timer",
        "expect_args": {"minutes": 5, "label": "tea"},
    },
    # --- 2-arg tasks: convert_currency (amount + to_currency) ---
    {
        "id": "convert_100_usd",
        "en": "Convert 100 into USD.",
        "ne": "१०० लाई USD मा रूपान्तरण गर्नुहोस्।",
        "expect_tool": "convert_currency",
        "expect_args": {"amount": 100, "to_currency": "USD"},
    },
    {
        "id": "convert_50_eur",
        "en": "Convert 50 into EUR.",
        "ne": "५० लाई EUR मा रूपान्तरण गर्नुहोस्।",
        "expect_tool": "convert_currency",
        "expect_args": {"amount": 50, "to_currency": "EUR"},
    },
    # --- 2-arg tasks: schedule_reminder (message + time) ---
    {
        "id": "remind_meeting_3pm",
        "en": "Schedule a reminder 'meeting' at 3pm.",
        "ne": "'meeting' भन्ने रिमाइन्डर ३ बजे राख्नुहोस्।",
        "expect_tool": "schedule_reminder",
        "expect_args": {"message": "meeting", "time": "3pm"},
    },
    {
        "id": "remind_call_9am",
        "en": "Schedule a reminder 'call mom' at 9am.",
        "ne": "'call mom' भन्ने रिमाइन्डर ९ बजे राख्नुहोस्।",
        "expect_tool": "schedule_reminder",
        "expect_args": {"message": "call mom", "time": "9am"},
    },
]


def prompt_for(task: dict, lang: str) -> str:
    """The user prompt for a task in the given language ('en' | 'ne')."""
    return task["en"] if lang == "en" else task["ne"]
