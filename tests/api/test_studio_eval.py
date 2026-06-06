"""Tests for Studio Evaluation suite discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_eval as se


def test_discover_suites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "demo.eval.yaml").write_text(
        "name: demo\n"
        "cases:\n"
        "  - input: { prompt: hi }\n"
        "    expected_output: { contains: hello }\n"
        "    metric_weights: { accuracy: 1.0 }\n"
    )
    (tmp_path / "not-a-suite.yaml").write_text("name: x\n")  # ignored (wrong glob)
    monkeypatch.chdir(tmp_path)
    suites = se.discover_suites()
    assert [s.name for s in suites] == ["demo"]
    assert suites[0].cases == 1
    assert suites[0].path == "demo.eval.yaml"
