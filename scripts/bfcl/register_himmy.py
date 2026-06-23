"""Reproducible runtime registration of the HIMMY-arm OpenRouter models into BFCL.

Companion to ``register.py`` (the RAW arm). Registers the SAME OpenRouter models,
but with ``HimmyOpenRouterHandler`` (tool-calls routed through himmy) so the only
difference between the two ``MODEL_CONFIG_MAPPING`` entries is himmy in the loop.

Both arms register in **prompting mode** (``is_fc_model=False``) so the comparison
is apples-to-apples (same model, tasks, decoding); the only variables are himmy's
manifest render + tolerant parse + multi-turn result threading vs BFCL's defaults.

Per-model himmy format
----------------------
Each himmy model pins a himmy format via a tiny handler subclass that sets
``HIMMY_FORMAT_OVERRIDE`` (a class attribute, so it is per-registration and not a
process-wide env var). The format is the model-family-native tool-call grammar:

  * gpt-4o-mini      -> "generic"            (no open-weight grammar; expect ~0 lift,
                         the honest frontier-ceiling result)
  * qwen2.5-7b       -> "hermes_chatml_xml"  (Qwen2.5/Hermes ChatML <tools> grammar;
                         where himmy's native render + tolerant parse should add lift)

Models registered here
-----------------------
  * ``or-himmy-gpt-4o-mini``  -> openai/gpt-4o-mini, prompting, himmy generic format
  * ``or-himmy-qwen2.5-7b``   -> qwen/qwen-2.5-7b-instruct, prompting, hermes_chatml_xml

All registrations are idempotent.
"""

from __future__ import annotations

from bfcl_eval.constants import model_config as _mc

from handlers_himmy import HimmyOpenRouterHandler

# OpenRouter list pricing (USD per 1M tokens); same source as register.py.
_PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
    "meta-llama/llama-3.2-3b-instruct": (0.015, 0.025),
    "meta-llama/llama-3.2-1b-instruct": (0.005, 0.01),
}


class HimmyGenericHandler(HimmyOpenRouterHandler):
    """himmy arm pinned to the GENERIC text format (frontier baseline)."""

    HIMMY_FORMAT_OVERRIDE = "generic"


class HimmyHermesHandler(HimmyOpenRouterHandler):
    """himmy arm pinned to the Hermes/Qwen ChatML-XML format (open-weight grammar)."""

    HIMMY_FORMAT_OVERRIDE = "hermes_chatml_xml"


class HimmyLlama3Handler(HimmyOpenRouterHandler):
    """himmy arm pinned to the Llama3-JSON tool-call grammar."""

    HIMMY_FORMAT_OVERRIDE = "llama3_json"


def _cfg(provider_model: str, display_name: str, handler):
    in_price, out_price = _PRICES.get(provider_model, (None, None))
    # HIMMY arm runs in prompting mode (is_fc_model=False) so the model emits prose
    # we decode with himmy's parser -> underscore_to_dot stays False (dotted names
    # are decoded verbatim, matching the RAW prompting baseline).
    return _mc.ModelConfig(
        model_name=provider_model,
        display_name=display_name,
        url="https://openrouter.ai/" + provider_model,
        org="OpenRouter+himmy",
        license="Proprietary/Various",
        model_handler=handler,
        input_price=in_price,
        output_price=out_price,
        is_fc_model=False,
        underscore_to_dot=False,
    )


# (registry_name, provider_model, display_name, handler)
_HIMMY_MODELS = [
    (
        "or-himmy-gpt-4o-mini",
        "openai/gpt-4o-mini",
        "gpt-4o-mini under himmy (OpenRouter, prompting, generic format)",
        HimmyGenericHandler,
    ),
    (
        "or-himmy-qwen2.5-7b",
        "qwen/qwen-2.5-7b-instruct",
        "Qwen2.5-7B-Instruct under himmy (OpenRouter, prompting, hermes_chatml_xml)",
        HimmyHermesHandler,
    ),
    (
        "or-himmy-llama3.2-3b",
        "meta-llama/llama-3.2-3b-instruct",
        "Llama-3.2-3B-Instruct under himmy (OpenRouter, prompting, llama3_json)",
        HimmyLlama3Handler,
    ),
    (
        "or-himmy-llama3.2-1b",
        "meta-llama/llama-3.2-1b-instruct",
        "Llama-3.2-1B-Instruct under himmy (OpenRouter, prompting, llama3_json)",
        HimmyLlama3Handler,
    ),
]


def register() -> list[str]:
    """Register the HIMMY OpenRouter models into BFCL's mapping. Idempotent."""
    registered = []
    for registry_name, provider_model, display_name, handler in _HIMMY_MODELS:
        cfg = _cfg(provider_model, display_name, handler)
        _mc.MODEL_CONFIG_MAPPING[registry_name] = cfg
        _mc.api_inference_model_map[registry_name] = cfg
        registered.append(registry_name)
    return registered


if __name__ == "__main__":
    names = register()
    print("Registered HIMMY OpenRouter models:")
    for n in names:
        print(f"  {n}")
