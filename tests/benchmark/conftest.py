"""Benchmark-test fixtures.

Keep the suite hermetic: the benchmark runner appends to the repo's
``benchmarks/history.jsonl`` by default, which tests must never touch. This autouse
fixture disables history globally for benchmark tests; the dedicated history tests
opt back in by passing an explicit ``path=`` (a tmp file) or ``enabled=True`` to the
history API, which overrides the env switch.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_repo_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_BENCH_NO_HISTORY", "1")
