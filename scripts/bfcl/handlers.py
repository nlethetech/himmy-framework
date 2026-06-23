"""BFCL handlers for OpenRouter-backed models.

This module defines the model handlers that BFCL (Berkeley Function Calling
Leaderboard, ``bfcl-eval``) uses to drive models hosted on OpenRouter
(https://openrouter.ai), an OpenAI-compatible aggregator.

It currently provides ONE handler:

    OpenRouterRawHandler
        The RAW (baseline) arm. A thin subclass of BFCL's stock
        ``OpenAICompletionsHandler`` pointed at the OpenRouter base URL with the
        key read from the environment. It deliberately imports NO himmy code and
        exercises only BFCL's normal code paths:
          * Function-Calling (FC) mode -> native ``tools=[...]`` field, decoded
            from ``message.tool_calls`` (BFCL default).
          * Prompting mode -> system-injected manifest, decoded with BFCL's
            ``default_decode_ast_prompting`` (BFCL default).
        This is the apples-to-apples baseline against which the HIMMY arm
        (added separately) is measured for same-model tool-calling lift.

Design notes
------------
* Modeled on ``bfcl_eval/model_handler/api_inference/novita.py`` and
  ``openai_completion.py`` (the OpenAI-compatible templates). The only changes
  vs. the OpenAI handler are the ``base_url`` and the API-key env var.
* The handler is registered into BFCL's ``MODEL_CONFIG_MAPPING`` at interpreter
  startup by ``scripts/bfcl/sitecustomize.py`` (no site-packages edits). See
  ``scripts/bfcl/README.md`` for the reproducible setup.
* PURITY GUARANTEE for the RAW arm: this module must never import ``himmy``.
  There is a regression check in ``scripts/bfcl/README.md`` (and the lift
  tooling) that greps this file path to prove the RAW path is himmy-free.

The OpenRouter API key is read from the ``OPENROUTER_API_KEY`` environment
variable. ``scripts/bfcl/sitecustomize.py`` loads it from the himmy repo ``.env``
before any handler is constructed; the secret is never logged.
"""

from __future__ import annotations

import os

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)

# OpenAI-compatible base URL for OpenRouter.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Optional ranking/identification headers OpenRouter recommends. Harmless if
# absent; they do not affect routing or determinism.
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/himmy-agent/bfcl-harness",
    "X-Title": "himmy BFCL same-model lift harness",
}


def _openrouter_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. scripts/bfcl/sitecustomize.py loads it "
            "from the himmy repo .env; ensure that file exists and contains the key."
        )
    return key


class OpenRouterRawHandler(OpenAICompletionsHandler):
    """RAW baseline arm: the model on OpenRouter via BFCL's stock OpenAI path.

    No himmy code is imported or invoked anywhere in this class. This is the
    control arm for the same-model lift comparison.
    """

    def _build_client_kwargs(self):
        """Return OpenRouter client kwargs.

        The parent ``__init__`` calls this to build ``self.client``. By
        overriding it (rather than rebuilding the client after ``super().__init__``)
        we ensure valid credentials are present before the OpenAI SDK validates
        them — the SDK raises on a missing key at construction time. This mirrors
        NovitaHandler's hardcoded-base-url pattern but routes through the parent's
        client-construction seam.
        """
        return {
            "base_url": OPENROUTER_BASE_URL,
            "api_key": _openrouter_api_key(),
            "default_headers": _OPENROUTER_HEADERS,
        }

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        # Parent builds self.client = OpenAI(**self._build_client_kwargs()),
        # which our override above points at OpenRouter.
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        # OpenRouter speaks the OpenAI chat-completions dialect.
        self.model_style = ModelStyle.OPENAI_COMPLETIONS

    #### Robust usage extraction ####

    @staticmethod
    def _usage_tokens(api_response) -> tuple[int, int]:
        """Return ``(input_token, output_token)`` tolerating a missing usage block.

        OpenRouter occasionally returns a 200 with ``usage = None`` (some upstream
        providers omit it on certain routes). BFCL's stock OpenAI handler accesses
        ``api_response.usage.prompt_tokens`` unconditionally and raises
        ``'NoneType' object has no attribute 'prompt_tokens'``, which BFCL catches and
        records as ``Error during inference: ...`` — turning a perfectly good
        completion into a permanent task FAILURE. Since the missing-usage event is a
        transport artifact unrelated to tool-calling quality, we default the counts to
        0 so the (real) model reply is still graded. This override lives in the RAW
        base handler so BOTH arms inherit identical transport behavior (apples-to-apples).
        """
        usage = getattr(api_response, "usage", None)
        if usage is None:
            return 0, 0
        return (
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    def _parse_query_response_FC(self, api_response):
        # Reconstructs BFCL's stock FC parse (openai_completion.py) but reads tokens
        # via the tolerant helper instead of ``api_response.usage.prompt_tokens``
        # (which raises on a null usage block). We cannot delegate to super() because
        # super() itself touches ``.usage.prompt_tokens`` and would raise first.
        try:
            model_responses = [
                {fc.function.name: fc.function.arguments}
                for fc in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                fc.id for fc in api_response.choices[0].message.tool_calls
            ]
        except Exception:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []
        in_tok, out_tok = self._usage_tokens(api_response)
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": (
                api_response.choices[0].message
            ),
            "tool_call_ids": tool_call_ids,
            "input_token": in_tok,
            "output_token": out_tok,
        }

    def _parse_query_response_prompting(self, api_response):
        in_tok, out_tok = self._usage_tokens(api_response)
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": in_tok,
            "output_token": out_tok,
        }

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        # The BFCL registry strips the BFCL "model_name" registry key down to the
        # actual provider model id (see ModelConfig.model_name). For OpenRouter
        # the model id is e.g. "openai/gpt-4o-mini" or "qwen/qwen-2.5-7b-instruct".
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }

        kwargs = {
            "messages": message,
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"])
        }

        return self.generate_with_backoff(
            messages=inference_data["message"],
            model=self.model_name,
            temperature=self.temperature,
        )
