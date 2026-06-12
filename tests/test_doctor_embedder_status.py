"""``himmy doctor`` surfaces the auto-embedder cascade.

The probes are monkeypatched so these tests never import fastembed's heavy module
or touch the network — only the cascade-resolution and rendering logic is exercised.
"""

from __future__ import annotations

import argparse

import pytest

from himmy.cli import commands
from himmy.runtime import diagnostics


def test_report_selects_fastembed_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fastembed importable → selected = fastembed; all three statuses present."""
    monkeypatch.setattr(
        "himmy.services.knowledge.local_embedders.fastembed_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "himmy.services.knowledge.local_embedders.ollama_reachable",
        lambda *a, **k: False,
    )
    report = diagnostics.collect_doctor_report()
    assert report.embedder_selected == "fastembed"
    names = [s.name for s in report.embedder_statuses]
    assert names == ["fastembed", "ollama", "deterministic"]
    by_name = {s.name: s for s in report.embedder_statuses}
    assert by_name["fastembed"].available is True
    assert by_name["ollama"].available is False
    assert by_name["deterministic"].available is True


def test_report_falls_back_to_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fastembed, no Ollama → selected = deterministic with a clear reason."""
    monkeypatch.setattr(
        "himmy.services.knowledge.local_embedders.fastembed_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "himmy.services.knowledge.local_embedders.ollama_reachable",
        lambda *a, **k: False,
    )
    report = diagnostics.collect_doctor_report()
    assert report.embedder_selected == "deterministic"
    by_name = {s.name: s for s in report.embedder_statuses}
    assert by_name["fastembed"].available is False
    assert by_name["fastembed"].reason is not None
    assert by_name["ollama"].reason is not None
    assert by_name["deterministic"].available is True


def test_report_to_dict_includes_embedder_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The JSON-friendly dict carries the new embedder fields (Studio parity)."""
    monkeypatch.setattr(
        "himmy.services.knowledge.local_embedders.fastembed_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "himmy.services.knowledge.local_embedders.ollama_reachable",
        lambda *a, **k: False,
    )
    data = diagnostics.collect_doctor_report().to_dict()
    assert data["embedder_selected"] == "deterministic"
    assert isinstance(data["embedder_statuses"], list)
    assert {"name", "available", "reason"} <= set(data["embedder_statuses"][0])


def test_install_model_next_step_points_to_quickstart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brand-new user (no model, no agent) is pointed at the 5-minute quickstart.

    The ``install_model`` next-step is the first onboarding hook a fresh install
    hits (surfaced by ``himmy doctor`` and Studio's Doctor screen), so it carries
    the quickstart path for a non-technical user.
    """
    # No local CLI provider on PATH and no provider keys → the install_model branch.
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _binary: None)
    for key in diagnostics._KEYS:
        monkeypatch.delenv(key, raising=False)
    report = diagnostics.collect_doctor_report()
    assert report.has_real_model is False
    assert report.next_step is not None
    assert report.next_step.kind == "install_model"
    assert "docs/QUICKSTART.md" in report.next_step.message


def test_cmd_doctor_renders_embedder_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``cmd_doctor`` prints the embedders section marking the selected backend."""
    monkeypatch.setattr(
        "himmy.services.knowledge.local_embedders.fastembed_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "himmy.services.knowledge.local_embedders.ollama_reachable",
        lambda *a, **k: False,
    )
    rc = commands.cmd_doctor(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "embedders" in out
    assert "fastembed (selected)" in out
    assert "[-- ] ollama" in out
    assert "deterministic (fallback)" in out
