"""The broad phone PII rule must redact real phone numbers but NOT over-catch the
year ranges, ISO dates, and coordinates that legitimately appear in user/tool text."""

from __future__ import annotations

import pytest

from himmy.services.guardrails.builtins import PIIGuardrail


def _redact(text: str) -> str:
    return PIIGuardrail().inspect(text, context={}).text


@pytest.mark.parametrize(
    "text",
    [
        "Colombia (1916-2016)",          # the regression that motivated this
        "the period 2011-2019",
        "data from 1990 - 2020",
        "between 1850-1999",
    ],
)
def test_year_ranges_are_not_redacted_as_phone(text: str) -> None:
    out = _redact(text)
    assert "[REDACTED-PHONE]" not in out, out
    assert out == text  # left completely intact


@pytest.mark.parametrize(
    "text",
    [
        "the date 2026-06-12 works",     # ISO-8601 date
        "coordinates 27.7172453 north",  # decimal / lat-long
        "version 12.4567 shipped",
    ],
)
def test_dates_and_decimals_still_not_redacted(text: str) -> None:
    assert "[REDACTED-PHONE]" not in _redact(text)


@pytest.mark.parametrize(
    "text",
    [
        # The broad phone class includes . ( ) space, so it used to MERGE several numbers in
        # financial/statistical text into one fake phone. These must all stay intact.
        "NABIL is down -2.00 (-0.376647834%) today",
        "change of -2.0 (-0.376647834%)",
        "NAV 1,009,816.69 (-1.03%) on the day",
        "the stock moved from 531.0 to 529.0 (a 0.38 point drop)",
        "p = 0.0031 (95% CI 0.0012 to 0.0049)",
    ],
)
def test_financial_and_stat_numbers_are_not_redacted_as_phone(text: str) -> None:
    out = _redact(text)
    assert "[REDACTED-PHONE]" not in out, out


@pytest.mark.parametrize(
    "text",
    [
        "call me at +1 415 555 0132 today",
        "office: (415) 555-0132",
        "ring 9779812345678 anytime",
        "local line 1234-5678 please",   # NOT a year pair → still a phone-shaped run
    ],
)
def test_real_phone_numbers_are_still_redacted(text: str) -> None:
    assert "[REDACTED-PHONE]" in _redact(text), text
