"""Tests for the deterministic benchmark graders."""

from __future__ import annotations

import pytest

from himmy.benchmark.graders import grade


def test_contains_case_insensitive() -> None:
    assert grade({"type": "contains", "value": "Paris"}, "the capital is paris")
    assert not grade({"type": "contains", "value": "London"}, "paris")


def test_not_contains() -> None:
    assert grade({"type": "not_contains", "value": "error"}, "all good")
    assert not grade({"type": "not_contains", "value": "error"}, "an Error happened")


def test_regex() -> None:
    assert grade({"type": "regex", "value": r"\b391\b"}, "the answer is 391.0")
    assert not grade({"type": "regex", "value": r"\b392\b"}, "391")


def test_exact_normalized() -> None:
    assert grade({"type": "exact", "value": "Paris"}, "  paris  ")
    assert not grade({"type": "exact", "value": "Paris"}, "paris, france")


def test_numeric_with_tolerance_and_separators() -> None:
    assert grade({"type": "numeric", "value": 391}, "result: 391.0")
    assert grade({"type": "numeric", "value": 1234567, "tolerance": 0}, "रू 1,234,567")
    assert grade({"type": "numeric", "value": 20, "tolerance": 1.5}, "about 21 C")
    assert not grade({"type": "numeric", "value": 20, "tolerance": 0.5}, "25 C")


def test_all_of_and_any_of() -> None:
    spec = {
        "type": "all_of",
        "of": [
            {"type": "contains", "value": "nepal"},
            {"type": "contains", "value": "capital"},
        ],
    }
    assert grade(spec, "Kathmandu is the capital of Nepal")
    assert not grade(spec, "Kathmandu is in Nepal")  # missing 'capital'
    any_spec = {
        "type": "any_of",
        "of": [
            {"type": "contains", "value": "x"},
            {"type": "contains", "value": "paris"},
        ],
    }
    assert grade(any_spec, "paris")


def test_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        grade({"type": "telepathy", "value": "x"}, "answer")
