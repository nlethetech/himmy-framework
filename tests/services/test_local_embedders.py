"""Tests for the real local embedders + the embedder factory (offline)."""

from __future__ import annotations

import pytest

from himmy.core import HimmyError
from himmy.services.knowledge import local_embedders
from himmy.services.knowledge.embedder import DeterministicEmbedder
from himmy.services.knowledge.local_embedders import (
    FastEmbedEmbedder,
    OllamaEmbedder,
    build_embedder,
    default_dim_for,
    resolve_auto_backend,
)
from tests.conftest import run_async


def _fake_transport(vec: list[float]):
    def transport(path: str, payload: dict) -> dict:
        assert path == "/api/embeddings"
        assert "prompt" in payload
        return {"embedding": vec}

    return transport


def test_ollama_embedder_query_and_documents() -> None:
    """OllamaEmbedder posts to /api/embeddings via the injected transport."""
    emb = OllamaEmbedder(dim=3, transport=_fake_transport([0.1, 0.2, 0.3]))
    assert run_async(emb.embed_query("hi")) == [0.1, 0.2, 0.3]
    docs = run_async(emb.embed_documents(["a", "b"]))
    assert docs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_ollama_embedder_rejects_empty() -> None:
    """An empty embedding from the server is a clear error."""
    emb = OllamaEmbedder(transport=_fake_transport([]))
    with pytest.raises(HimmyError):
        run_async(emb.embed_query("x"))


def test_factory_selects_backends() -> None:
    """build_embedder maps names to embedder types with the right dim."""
    assert isinstance(build_embedder("deterministic", dim=16), DeterministicEmbedder)
    assert build_embedder("deterministic", dim=16).dim == 16
    assert isinstance(build_embedder("ollama", dim=768), OllamaEmbedder)
    assert isinstance(build_embedder("fastembed", dim=384), FastEmbedEmbedder)


def test_factory_unknown_raises() -> None:
    with pytest.raises(HimmyError):
        build_embedder("nope")


def test_default_dims() -> None:
    assert default_dim_for("ollama") == 4096  # qwen3-embedding's native dim
    assert default_dim_for("fastembed") == 384
    assert default_dim_for("deterministic") == 64


def test_fastembed_lazy_import_or_works() -> None:
    """fastembed embeds when installed; otherwise raises a clear extra error."""
    emb = FastEmbedEmbedder(dim=384)
    try:
        import fastembed  # noqa: F401
    except ImportError:
        with pytest.raises(HimmyError):
            run_async(emb.embed_query("hello"))
    else:  # pragma: no cover - only when the extra is installed
        vec = run_async(emb.embed_query("hello"))
        assert len(vec) == 384


def test_config_build_embedder_and_dim() -> None:
    """ToolkitConfig builds the configured embedder + dim."""
    from himmy.toolkit.config import ToolkitConfig

    emb, dim = ToolkitConfig(embedder="ollama").build_embedder_and_dim()
    assert isinstance(emb, OllamaEmbedder)
    assert dim == 4096
    emb2, dim2 = ToolkitConfig(
        embedder="deterministic", embedder_dim=32
    ).build_embedder_and_dim()
    assert dim2 == 32


# ---- "auto": prefer a real local embedder, fall back to deterministic ----


def test_auto_prefers_fastembed_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """When fastembed is importable, "auto" selects the fastembed backend (no network)."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: True)
    # The Ollama probe must never run when fastembed already won the selection.
    monkeypatch.setattr(
        local_embedders,
        "ollama_reachable",
        lambda *a, **k: pytest.fail("ollama probe should not run when fastembed wins"),
    )
    assert resolve_auto_backend() == "fastembed"
    emb = build_embedder("auto")
    assert isinstance(emb, FastEmbedEmbedder)
    assert emb.dim == 384  # fastembed's native dim, not the deterministic default


def test_auto_prefers_ollama_when_reachable_and_no_fastembed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fastembed but a reachable Ollama WITH the embed model pulled -> "auto" = ollama."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: True)
    monkeypatch.setattr(
        local_embedders, "ollama_embed_model_available", lambda *a, **k: True
    )
    assert resolve_auto_backend() == "ollama"
    emb = build_embedder("auto")
    assert isinstance(emb, OllamaEmbedder)
    assert emb.dim == 4096  # qwen3-embedding's native dim


def test_auto_skips_ollama_when_embed_model_not_pulled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reachable Ollama with NO embed model pulled -> "auto" degrades to deterministic.

    This is the robustness fix: auto-selecting a reachable-but-embed-less Ollama would
    404 at embed time, so the embed-model gate sends "auto" to the offline fallback.
    """
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: True)
    monkeypatch.setattr(
        local_embedders, "ollama_embed_model_available", lambda *a, **k: False
    )
    assert resolve_auto_backend() == "deterministic"


def test_auto_falls_back_to_deterministic_when_nothing_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fastembed and no reachable Ollama -> "auto" degrades to deterministic."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    assert resolve_auto_backend() == "deterministic"
    emb = build_embedder("auto")
    assert isinstance(emb, DeterministicEmbedder)
    # An offline embedder still embeds with no network.
    assert len(run_async(emb.embed_query("hello world"))) == 64


def test_auto_default_dim_tracks_resolved_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """default_dim_for("auto") reports the dim of the backend it actually resolves to."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: True)
    assert default_dim_for("auto") == 384
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    # The decision is memoised per process; the environment genuinely changed mid-test, so
    # re-probe (mirrors a runtime reconfig — never happens in a real single-env process).
    local_embedders.reset_auto_backend_cache()
    assert default_dim_for("auto") == 64


def test_config_auto_resolves_embedder_and_matching_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ToolkitConfig(embedder="auto") builds a real embedder + a matching dim."""
    from himmy.toolkit.config import ToolkitConfig

    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: True)
    monkeypatch.setattr(
        local_embedders, "ollama_embed_model_available", lambda *a, **k: True
    )
    emb, dim = ToolkitConfig(embedder="auto").build_embedder_and_dim()
    assert isinstance(emb, OllamaEmbedder)
    assert dim == 4096  # the dim matches the resolved backend, not the "auto" alias

    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    # Environment flipped mid-test → clear the per-process decision memo to re-probe.
    local_embedders.reset_auto_backend_cache()
    emb2, dim2 = ToolkitConfig(embedder="auto").build_embedder_and_dim()
    assert isinstance(emb2, DeterministicEmbedder)
    assert dim2 == 64


def test_ollama_reachable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ollama probe returns False on any transport error (never raises)."""
    import httpx

    def _boom(*_a: object, **_k: object) -> object:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    # An unreachable server must fail closed, not propagate the connection error.
    assert local_embedders.ollama_reachable("http://localhost:11434") is False


class _ProbeCounter:
    """Counts calls to the Ollama probes so a test can assert they fire at most once."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reachable_calls = 0
        self.model_calls = 0

        def _reachable(*_a: object, **_k: object) -> bool:
            self.reachable_calls += 1
            return True

        def _model(*_a: object, **_k: object) -> bool:
            self.model_calls += 1
            return True

        monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
        monkeypatch.setattr(local_embedders, "ollama_reachable", _reachable)
        monkeypatch.setattr(
            local_embedders, "ollama_embed_model_available", _model
        )


def test_resolve_auto_backend_probes_ollama_at_most_once_per_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blocking Ollama probes fire exactly once per base_url, not per call.

    ``build_runtime_for_spec`` resolves ``"auto"`` 1-4x per turn; each uncached resolve
    does two blocking HTTP probes (~0.25s + ~0.5s). Memoising per base_url makes the probe
    fire once for the process while returning a byte-identical decision on every call.
    """
    counter = _ProbeCounter(monkeypatch)

    results = [resolve_auto_backend() for _ in range(20)]

    assert results == ["ollama"] * 20  # identical decision every call
    assert counter.reachable_calls == 1
    assert counter.model_calls == 1


def test_resolve_auto_backend_reset_forces_a_reprobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset_auto_backend_cache() drops the memo so the next resolve re-probes Ollama."""
    counter = _ProbeCounter(monkeypatch)

    assert resolve_auto_backend() == "ollama"
    assert counter.reachable_calls == 1

    local_embedders.reset_auto_backend_cache()
    assert resolve_auto_backend() == "ollama"
    assert counter.reachable_calls == 2  # re-probed after the reset


def test_reset_embedder_cache_also_clears_auto_backend_memo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The general reset hook also forces an auto-backend re-probe."""
    counter = _ProbeCounter(monkeypatch)

    assert resolve_auto_backend() == "ollama"
    assert counter.reachable_calls == 1

    local_embedders.reset_embedder_cache()
    assert resolve_auto_backend() == "ollama"
    assert counter.reachable_calls == 2


def test_resolve_auto_backend_caches_per_base_url_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different base_urls memoise separately: each is probed once, neither shares state."""
    seen: list[str] = []

    def _reachable(base: str = local_embedders._DEFAULT_OLLAMA_URL, **_k: object) -> bool:
        seen.append(base)
        return True

    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", _reachable)
    monkeypatch.setattr(
        local_embedders, "ollama_embed_model_available", lambda *a, **k: True
    )

    url_a = "http://a.local:11434"
    url_b = "http://b.local:11434"
    for _ in range(5):
        assert resolve_auto_backend(ollama_base_url=url_a) == "ollama"
        assert resolve_auto_backend(ollama_base_url=url_b) == "ollama"

    # Each distinct base_url is probed exactly once despite 5 rounds of both.
    assert sorted(seen) == [url_a, url_b]


def test_backend_probe_cache_off_reprobes_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HIMMY_BACKEND_PROBE_CACHE=off restores the pre-cache re-probe-every-call behaviour."""
    monkeypatch.setenv("HIMMY_BACKEND_PROBE_CACHE", "off")
    counter = _ProbeCounter(monkeypatch)

    for _ in range(3):
        assert resolve_auto_backend() == "ollama"

    assert counter.reachable_calls == 3  # no memoisation while the kill-switch is set


def test_resolve_auto_backend_concurrent_probes_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under concurrent first-resolves the memo stays consistent and results identical.

    A benign race may probe more than once (the probe runs outside the lock so a slow
    Ollama can't block unrelated base_urls), but the cache must converge to one entry and
    every caller must get the same decision — never a torn/None result.
    """
    import threading

    barrier = threading.Barrier(16)
    results: list[str] = []
    results_lock = threading.Lock()

    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: True)
    monkeypatch.setattr(
        local_embedders, "ollama_embed_model_available", lambda *a, **k: True
    )

    def _worker() -> None:
        barrier.wait()
        r = resolve_auto_backend()
        with results_lock:
            results.append(r)

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == ["ollama"] * 16  # every concurrent caller saw the same decision
    # Exactly one converged cache entry for the default base_url (key is now a
    # ``(base, embed_model)`` tuple; value is a ``(decision, deadline)`` tuple —
    # only the decision is asserted here).
    key = (
        local_embedders._DEFAULT_OLLAMA_URL,
        local_embedders.default_ollama_embed_model(),
    )
    assert list(local_embedders._AUTO_BACKEND_CACHE) == [key]
    assert local_embedders._AUTO_BACKEND_CACHE[key][0] == "ollama"


# ---------------------------------------- red-team r1: staleness TTL + normalisation


class _FlippableProbe:
    """Ollama probes whose reachability can be flipped between resolves + call-counted."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, reachable: bool) -> None:
        self.reachable = reachable
        self.reachable_calls = 0

        def _reachable(*_a: object, **_k: object) -> bool:
            self.reachable_calls += 1
            return self.reachable

        monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
        monkeypatch.setattr(local_embedders, "ollama_reachable", _reachable)
        monkeypatch.setattr(
            local_embedders, "ollama_embed_model_available", lambda *a, **k: True
        )


def test_non_terminal_decision_reprobes_after_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``deterministic`` decision made before Ollama was up UPGRADES after the TTL.

    Regression: the memo pinned the first probe for the process lifetime, so a server that
    started before Ollama came up served non-semantic ``deterministic`` embeddings forever.
    With a TTL the stale non-terminal decision is re-probed and upgrades to ``ollama``.
    """
    monkeypatch.setenv("HIMMY_BACKEND_PROBE_TTL", "5")
    probe = _FlippableProbe(monkeypatch, reachable=False)
    fake_now = [100.0]
    monkeypatch.setattr(local_embedders.time, "monotonic", lambda: fake_now[0])

    # Ollama not up yet -> deterministic, cached with a deadline at now+5.
    assert resolve_auto_backend() == "deterministic"
    # Within the TTL window: no re-probe, still deterministic.
    fake_now[0] = 103.0
    assert resolve_auto_backend() == "deterministic"
    assert probe.reachable_calls == 1
    # Past the deadline + Ollama now up: re-probe upgrades to ollama.
    fake_now[0] = 200.0
    probe.reachable = True
    assert resolve_auto_backend() == "ollama"
    assert probe.reachable_calls == 2


def test_ttl_zero_caches_forever(monkeypatch: pytest.MonkeyPatch) -> None:
    """HIMMY_BACKEND_PROBE_TTL=0 pins the decision (prior cache-forever behaviour)."""
    monkeypatch.setenv("HIMMY_BACKEND_PROBE_TTL", "0")
    probe = _FlippableProbe(monkeypatch, reachable=False)
    assert resolve_auto_backend() == "deterministic"
    probe.reachable = True
    # No TTL: the pinned deterministic decision is served regardless of clock/liveness.
    for _ in range(5):
        assert resolve_auto_backend() == "deterministic"
    assert probe.reachable_calls == 1  # never re-probed


def test_fastembed_decision_is_terminal_never_reprobes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``fastembed`` decision is cached forever even with a TTL (importability is stable)."""
    monkeypatch.setenv("HIMMY_BACKEND_PROBE_TTL", "1")
    calls = {"n": 0}

    def _fastembed() -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr(local_embedders, "fastembed_available", _fastembed)
    monkeypatch.setattr(local_embedders.time, "monotonic", lambda: 0.0)
    assert resolve_auto_backend() == "fastembed"
    monkeypatch.setattr(local_embedders.time, "monotonic", lambda: 10_000.0)
    assert resolve_auto_backend() == "fastembed"
    assert calls["n"] == 1  # terminal: no re-probe despite the elapsed TTL


def test_equivalent_base_urls_share_one_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing-slash / host-case spellings of the same server probe once, not per spelling.

    Regression: the memo keyed on the RAW base_url, so ``.../11434`` and ``.../11434/`` each
    paid their own blocking probe and stored a separate entry despite hitting one server.
    """
    counter = _ProbeCounter(monkeypatch)
    for url in (
        "http://localhost:11434",
        "http://localhost:11434/",
        "http://LOCALHOST:11434",
        "http://localhost:11434///",
    ):
        assert resolve_auto_backend(ollama_base_url=url) == "ollama"
    # All four spellings collapsed to ONE normalised key and ONE probe.
    assert counter.reachable_calls == 1
    assert list(local_embedders._AUTO_BACKEND_CACHE) == [
        ("http://localhost:11434", local_embedders.default_ollama_embed_model())
    ]


def test_normalise_base_url_leaves_path_and_port() -> None:
    """Normalisation only touches trailing slashes + scheme/host case, not path/port."""
    n = local_embedders._normalise_base_url
    assert n("http://localhost:11434/") == "http://localhost:11434"
    assert n("HTTP://LocalHost:11434") == "http://localhost:11434"
    assert n("http://host:8080/v1/") == "http://host:8080/v1"
    assert n("not-a-url/") == "not-a-url"


# --------------------------------------- red-team r2: model-keyed memo + guard TOCTOU


def test_auto_backend_memo_reprobes_when_embed_model_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different HIMMY_OLLAMA_EMBED_MODEL re-probes: the memo is keyed on the model too.

    The Ollama leg is model-dependent (it checks the *configured* model is pulled), so a
    decision made while ``nomic-embed-text`` is pulled must NOT be served when an un-pulled
    ``qwen3-embedding`` is later configured against the SAME server — that would build an
    OllamaEmbedder that 404s at first embed instead of degrading to deterministic.
    """
    local_embedders.reset_auto_backend_cache()
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: True)

    pulled = {"nomic-embed-text"}
    probed_models: list[str | None] = []

    def _avail(model: str | None = None, base_url: str = "", **_k: object) -> bool:
        probed_models.append(model)
        return (model or "").split(":", 1)[0] in pulled

    monkeypatch.setattr(local_embedders, "ollama_embed_model_available", _avail)

    monkeypatch.setenv("HIMMY_OLLAMA_EMBED_MODEL", "nomic-embed-text")
    assert resolve_auto_backend(ollama_base_url="http://localhost:11434") == "ollama"

    # Switch to an UN-pulled model on the SAME server: must re-probe (new cache key) and
    # degrade to deterministic rather than serving the stale ``ollama`` decision.
    monkeypatch.setenv("HIMMY_OLLAMA_EMBED_MODEL", "qwen3-embedding")
    assert (
        resolve_auto_backend(ollama_base_url="http://localhost:11434")
        == "deterministic"
    )
    # The second probe actually ran and asked about the NEW model.
    assert "qwen3-embedding" in probed_models


def test_impl_and_guard_reads_cache_flag_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime env flip between the two reads can never pair a SHARED session + nullcontext.

    ``_impl_and_guard`` must read ``_embed_cache_enabled`` ONCE and thread it into
    ``_model_impl`` so the guard decision (shared -> lock) and the impl decision
    (shared -> process-wide session) are made from the SAME observation. Previously the
    two independent reads could diverge (off at read #1 -> nullcontext; on at read #2 ->
    shared session), returning the shared session under a no-op guard.
    """
    reads = {"n": 0}

    def _counting_enabled() -> bool:
        reads["n"] += 1
        return True  # cache ON: the shared session must be paired with the LOCK, never nullcontext

    monkeypatch.setattr(local_embedders, "_embed_cache_enabled", _counting_enabled)

    shared_session = object()
    monkeypatch.setattr(
        local_embedders, "_load_text_embedding", lambda _model: shared_session
    )

    emb = FastEmbedEmbedder(model="stub")
    impl, guard = emb._impl_and_guard()

    # Exactly ONE flag read backed BOTH the guard decision and the impl decision, so a
    # runtime env flip landing between two independent reads can no longer pair the shared
    # session with a no-op guard.
    assert reads["n"] == 1
    # cache ON -> the SHARED process-wide session, serialised under the real lock.
    assert impl is shared_session
    assert guard is local_embedders._TEXT_EMBED_LOCK


def test_impl_and_guard_private_session_uses_nullcontext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single flag read of OFF yields a PRIVATE session under nullcontext (no shared race)."""
    reads = {"n": 0}

    def _counting_disabled() -> bool:
        reads["n"] += 1
        return False

    monkeypatch.setattr(local_embedders, "_embed_cache_enabled", _counting_disabled)

    private = object()
    emb = FastEmbedEmbedder(model="stub")
    emb._impl = private  # a pre-set (private) session wins over any cache path
    impl, guard = emb._impl_and_guard()

    assert reads["n"] == 1
    assert impl is private
    import contextlib

    assert isinstance(guard, contextlib.nullcontext)
