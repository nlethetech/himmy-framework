"""Tests for benchmark statistics (Wilson interval, percentiles, mean)."""

from __future__ import annotations

from himmy.benchmark.stats import mean, percentile, wilson_interval


def test_wilson_empty() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_bounds_stay_in_unit_interval() -> None:
    lo, hi = wilson_interval(10, 10)
    # All-pass: interval is high but the upper bound is < 1 (real uncertainty).
    assert 0.6 < lo < hi <= 1.0 and hi > 0.9
    lo, hi = wilson_interval(0, 10)
    # All-fail: lower bound clamps at 0, upper is small but positive.
    assert lo == 0.0 and 0.0 < hi < 0.4


def test_wilson_brackets_the_point_estimate() -> None:
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi
    # More trials → tighter interval.
    lo2, hi2 = wilson_interval(50, 100)
    assert (hi2 - lo2) < (hi - lo)


def test_percentile() -> None:
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 0.0) == 1.0
    assert percentile(vals, 1.0) == 4.0
    assert percentile(vals, 0.5) == 2.5  # interpolated median
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.9) == 7.0


def test_mean() -> None:
    assert mean([2.0, 4.0]) == 3.0
    assert mean([]) == 0.0
