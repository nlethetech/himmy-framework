"""Reproducible runtime registration of OpenRouter-backed models into BFCL.

``bfcl-eval`` has no plugin / entry-point mechanism: handlers are resolved from
``bfcl_eval.constants.model_config.MODEL_CONFIG_MAPPING`` (a module-level dict)
which every BFCL module imports *by reference*. So we can register custom models
WITHOUT editing site-packages by mutating that dict before the CLI uses it.

``scripts/bfcl/sitecustomize.py`` imports this module at interpreter startup
(via ``PYTHONPATH`` pointing at ``scripts/bfcl/``), which guarantees the models
are present before ``bfcl_eval.__main__`` reads the mapping.

Models registered here
-----------------------
RAW arm (this file's responsibility):
  * ``or-raw-gpt-4o-mini``               -> openai/gpt-4o-mini, native FC
  * ``or-raw-gpt-4o-mini-prompting``     -> openai/gpt-4o-mini, prompting mode
  * ``or-raw-qwen2.5-7b``                -> qwen/qwen-2.5-7b-instruct, native FC
  * ``or-raw-qwen2.5-7b-prompting``      -> qwen/qwen-2.5-7b-instruct, prompting

The "-prompting" variants exist so the HIMMY arm (which routes tool-calling
through himmy's text/prose format registry) can be compared apples-to-apples
against a RAW prompting baseline that uses BFCL's default prose decoding. The
HIMMY-arm registrations are added by a separate module and are intentionally not
defined here, keeping the RAW path himmy-free.

All registrations are idempotent.
"""

from __future__ import annotations

from bfcl_eval.constants import model_config as _mc

from handlers import OpenRouterRawHandler

# OpenRouter list/usage pricing (USD per 1M tokens) as published on
# openrouter.ai/models at time of writing. Used only for BFCL's cost display;
# the authoritative spend is computed from real usage in the lift tooling.
_PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
    "meta-llama/llama-3.2-3b-instruct": (0.015, 0.025),
    "meta-llama/llama-3.2-1b-instruct": (0.005, 0.01),
}


def _cfg(
    registry_name: str,
    provider_model: str,
    display_name: str,
    handler,
    is_fc_model: bool,
):
    in_price, out_price = _PRICES.get(provider_model, (None, None))
    # underscore_to_dot mirrors BFCL's official OpenAI configs: native-FC entries
    # send tool names with '.' replaced by '_' (the OpenAI tools schema forbids
    # dots), so the checker must convert back -> underscore_to_dot=True. Prompting
    # entries keep dotted names verbatim -> False. Getting this wrong produces
    # spurious "Function name 'math.factorial' not found" failures.
    return _mc.ModelConfig(
        model_name=provider_model,
        display_name=display_name,
        url="https://openrouter.ai/" + provider_model,
        org="OpenRouter",
        license="Proprietary/Various",
        model_handler=handler,
        input_price=in_price,
        output_price=out_price,
        is_fc_model=is_fc_model,
        underscore_to_dot=is_fc_model,
    )


# (registry_name, provider_model, display_name, handler, is_fc_model)
_RAW_MODELS = [
    (
        "or-raw-gpt-4o-mini",
        "openai/gpt-4o-mini",
        "gpt-4o-mini RAW (OpenRouter, native FC)",
        OpenRouterRawHandler,
        True,
    ),
    (
        "or-raw-gpt-4o-mini-prompting",
        "openai/gpt-4o-mini",
        "gpt-4o-mini RAW (OpenRouter, prompting)",
        OpenRouterRawHandler,
        False,
    ),
    (
        "or-raw-qwen2.5-7b",
        "qwen/qwen-2.5-7b-instruct",
        "Qwen2.5-7B-Instruct RAW (OpenRouter, native FC)",
        OpenRouterRawHandler,
        True,
    ),
    (
        "or-raw-qwen2.5-7b-prompting",
        "qwen/qwen-2.5-7b-instruct",
        "Qwen2.5-7B-Instruct RAW (OpenRouter, prompting)",
        OpenRouterRawHandler,
        False,
    ),
    # Small open model where himmy lift is expected (emits messy/prose
    # tool-calls in prompting mode). Prompting variant only (the apples-to-apples
    # control for the HIMMY arm). FC variant kept for diagnostic comparison.
    (
        "or-raw-llama3.2-3b",
        "meta-llama/llama-3.2-3b-instruct",
        "Llama-3.2-3B-Instruct RAW (OpenRouter, native FC)",
        OpenRouterRawHandler,
        True,
    ),
    (
        "or-raw-llama3.2-3b-prompting",
        "meta-llama/llama-3.2-3b-instruct",
        "Llama-3.2-3B-Instruct RAW (OpenRouter, prompting)",
        OpenRouterRawHandler,
        False,
    ),
    (
        "or-raw-llama3.2-1b-prompting",
        "meta-llama/llama-3.2-1b-instruct",
        "Llama-3.2-1B-Instruct RAW (OpenRouter, prompting)",
        OpenRouterRawHandler,
        False,
    ),
]


def register() -> list[str]:
    """Register the RAW OpenRouter models into BFCL's mapping. Idempotent.

    Returns the list of registry names that were registered.
    """
    registered = []
    for registry_name, provider_model, display_name, handler, is_fc in _RAW_MODELS:
        cfg = _cfg(registry_name, provider_model, display_name, handler, is_fc)
        _mc.MODEL_CONFIG_MAPPING[registry_name] = cfg
        # Keep the sub-maps coherent in case anything reads them directly.
        _mc.api_inference_model_map[registry_name] = cfg
        registered.append(registry_name)
    return registered


if __name__ == "__main__":
    names = register()
    print("Registered RAW OpenRouter models:")
    for n in names:
        print(f"  {n}")
