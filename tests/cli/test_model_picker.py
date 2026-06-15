"""The ``/model`` (no-args) picker: pure render helper + the REPL slash command.

:func:`himmy.cli.model_picker.render_model_picker` is exercised with a SYNTHETIC catalog
and a stub ``price_for`` so the assertions need no live model and no network. A second
test drives ``/model`` through :class:`ChatRepl`'s injectable ``input_fn`` and asserts
the printed picker, end-to-end through the REPL slash dispatch.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from himmy.cli.model_picker import provider_aware_price_for, render_model_picker
from himmy.cli.repl import ChatRepl
from himmy.config.agent_spec import AgentSpec
from himmy.services.inference import pricing
from himmy.services.inference.models import ModelPrice

# ----------------------------------------------------------------- render helper

#: A plain (no-ANSI) palette — assertions match on bare text.
_PLAIN = {
    k: "" for k in ("dim", "snow", "gold", "faint", "green", "reset", "bold", "ink")
}

#: A synthetic LOCAL catalog (shape of build_model_catalog): an available claude-cli,
#: an UNavailable ollama (must be skipped — no "ollama" header for it).
_CATALOG = [
    {"provider": "ollama", "available": False, "models": []},
    {
        "provider": "claude-cli",
        "available": True,
        "models": [{"name": "haiku"}, {"name": "sonnet"}, {"name": "opus"}],
    },
]


def _price_for(model: str) -> ModelPrice:
    """Stub price table: gpt-4o-mini is priced; everything else is unknown ($0)."""
    if "gpt-4o-mini" in model:
        return ModelPrice(input_per_1k=0.00015, output_per_1k=0.0006)
    return ModelPrice()


def test_render_lists_local_provider_with_active_marker() -> None:
    out = render_model_picker(
        _CATALOG,
        active_provider="claude-cli",
        active_model="sonnet",
        color=_PLAIN,
        price_for=_price_for,
        cloud_available={"openrouter": False, "openai": False, "anthropic": False},
        environ={},
    )
    # The active line names the active model + provider and a no-cost note.
    assert "active:" in out
    assert "sonnet · claude-cli" in out
    assert "local · no API cost" in out
    # The available local provider is listed with switch hints + a free tag.
    assert "claude-cli" in out
    assert "/model haiku claude-cli" in out
    assert "free · local" in out
    # The active model line carries the (active) marker — and only it.
    active_line = next(ln for ln in out.splitlines() if "/model sonnet claude-cli" in ln)
    assert "(active)" in active_line
    haiku_line = next(ln for ln in out.splitlines() if "/model haiku claude-cli" in ln)
    assert "(active)" not in haiku_line
    # The UNavailable ollama provider is not rendered as a header.
    assert not any(ln.strip() == "ollama" for ln in out.splitlines())


def test_render_includes_available_cloud_provider_with_cost_figure() -> None:
    out = render_model_picker(
        _CATALOG,
        active_provider="claude-cli",
        active_model="haiku",
        color=_PLAIN,
        price_for=_price_for,
        cloud_available={"openrouter": True, "openai": False, "anthropic": False},
        environ={"OPENROUTER_MODEL": "meta-llama/llama-3.1-8b-instruct"},
    )
    # The available cloud provider is listed with its configured default model first.
    assert "openrouter" in out
    assert "/model meta-llama/llama-3.1-8b-instruct openrouter" in out
    # A priced curated model shows a real $/1M figure (not a fake $0).
    gpt_line = next(
        ln for ln in out.splitlines() if "/model openai/gpt-4o-mini openrouter" in ln
    )
    assert "per 1M" in gpt_line
    assert "$0.15" in gpt_line  # 0.00015 * 1000 per 1M, two-decimals
    # An unpriced curated model shows "price n/a", never "$0.00".
    llama_line = next(
        ln
        for ln in out.splitlines()
        if "/model meta-llama/llama-3.3-70b-instruct openrouter" in ln
    )
    assert "price n/a" in llama_line
    # Footer hints how to enable the still-unavailable providers, once.
    assert "set OPENAI_API_KEY to enable openai" in out
    assert "set ANTHROPIC_API_KEY to enable anthropic" in out
    assert "set OPENROUTER_API_KEY" not in out  # openrouter IS available


# ------------------------------------------------- live openrouter pricing (no net)

#: A fake OpenRouter ``/models`` payload (per-token USD strings), enough for the picker.
_OR_PAYLOAD = {
    "data": [
        {
            "id": "openai/gpt-4o-mini",
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "anthropic/claude-sonnet-4.5",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
    ]
}


def test_picker_renders_live_dollar_for_openrouter_model(monkeypatch) -> None:
    """Given a live payload, an openrouter curated model renders a real positive $."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pricing.reload_openrouter()

    def fetch(url: str, headers: dict, timeout: float) -> dict:
        return _OR_PAYLOAD

    # The picker wires a provider-aware price fn; openrouter targets resolve LIVE.
    price_for = provider_aware_price_for(fetch=fetch, now=lambda: 0.0)
    out = render_model_picker(
        _CATALOG,
        active_provider="claude-cli",
        active_model="haiku",
        color=_PLAIN,
        price_for=price_for,
        cloud_available={"openrouter": True, "openai": False, "anthropic": False},
        environ={},
    )
    pricing.reload_openrouter()

    # claude-sonnet-4.5 has NO static price, but the LIVE payload prices it → real $.
    sonnet_line = next(
        ln
        for ln in out.splitlines()
        if "/model anthropic/claude-sonnet-4.5 openrouter" in ln
    )
    assert "per 1M" in sonnet_line
    assert "$3.00" in sonnet_line  # 0.000003 * 1e6 = $3 / 1M input
    assert "price n/a" not in sonnet_line


def test_picker_offline_fallback_does_not_crash(monkeypatch) -> None:
    """When the live fetch fails, openrouter models degrade to static/price n/a — no crash."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pricing.reload_openrouter()

    def boom(url: str, headers: dict, timeout: float) -> dict:
        raise RuntimeError("offline")

    price_for = provider_aware_price_for(fetch=boom, now=lambda: 0.0)
    out = render_model_picker(
        _CATALOG,
        active_provider="claude-cli",
        active_model="haiku",
        color=_PLAIN,
        price_for=price_for,
        cloud_available={"openrouter": True, "openai": False, "anthropic": False},
        environ={},
    )
    pricing.reload_openrouter()

    # gpt-4o-mini is in the bundled static table → still a real $ even with no live data.
    gpt_line = next(
        ln for ln in out.splitlines() if "/model openai/gpt-4o-mini openrouter" in ln
    )
    assert "$0.15" in gpt_line
    # An unpriced-everywhere model degrades to "price n/a", never a crash or fake $0.
    llama_line = next(
        ln
        for ln in out.splitlines()
        if "/model meta-llama/llama-3.3-70b-instruct openrouter" in ln
    )
    assert "price n/a" in llama_line


def test_render_degrades_when_catalog_empty() -> None:
    out = render_model_picker(
        [],
        active_provider=None,
        active_model=None,
        color=_PLAIN,
        price_for=_price_for,
        cloud_available={"openrouter": False, "openai": False, "anthropic": False},
        environ={},
    )
    assert "active:" in out
    assert "auto" in out  # no explicit provider/model
    assert "no models detected" in out  # nothing available anywhere


# ------------------------------------------------------------------ REPL command


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "file": None,
        "name": "t",
        "instruction": None,
        "provider": "stub",
        "model": None,
        "message": None,
        "session": None,
        "yolo": False,
        "safe": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_model_no_args_prints_the_picker(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Offline-first: clear cloud keys so the live OpenRouter price fetch is skipped
    # (no real network call inside the unit test on a machine where keys are set).
    for _var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(_var, raising=False)
    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(provider="stub"), input_fn=lambda _p: "")

    result = repl._slash("/model", thread=object())

    # The slash returns the (placeholder) thread unchanged and prints to stderr.
    assert result is not None
    err = capsys.readouterr().err
    assert "active:" in err
    assert "available:" in err
    # The active provider (stub) is local → no API cost, and switch help is shown.
    assert "no API cost" in err
    assert "/model <name> <provider>" in err


def test_model_no_args_degrades_when_picker_raises(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(provider="stub"), input_fn=lambda _p: "")

    def boom() -> str:
        raise RuntimeError("render blew up")

    monkeypatch.setattr(repl, "_render_model_picker", boom)

    # The try-guard must keep the REPL alive: returns the thread, prints a fallback line.
    result = repl._slash("/model", thread=object())
    assert result is not None
    err = capsys.readouterr().err
    assert "couldn't render model list" in err
    assert "model:" in err
    assert "active:" not in err  # NOT the full picker


def test_model_with_args_still_switches(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(provider="stub"), input_fn=lambda _p: "")
    rebuilt: list[bool] = []
    monkeypatch.setattr(repl, "rebuild", lambda: rebuilt.append(True))

    repl._slash("/model haiku claude-cli", thread=object())

    # Switching path is unchanged: args mutated + rebuild called + short label line.
    assert repl.args.model == "haiku"
    assert repl.args.provider == "claude-cli"
    assert rebuilt == [True]
    err = capsys.readouterr().err
    assert "model:" in err
    assert "active:" not in err  # NOT the picker
