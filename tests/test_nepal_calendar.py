"""Tests for the Bikram Sambat calendar kernel."""

from __future__ import annotations

import datetime

from himmy.nepal import (
    BikramDate,
    ad_to_bs,
    bs_to_ad,
    nepali_fiscal_year,
    today_bs,
)


def test_ad_to_bs_known_date() -> None:
    """A known AD date converts to the right BS date."""
    bs = ad_to_bs(datetime.date(2024, 6, 1))
    assert (bs.year, bs.month, bs.day) == (2081, 2, 19)
    assert str(bs) == "2081-02-19 BS"


def test_round_trips_both_directions() -> None:
    """BS <-> AD round-trips exactly."""
    ad = datetime.date(2024, 6, 1)
    assert bs_to_ad(2081, 2, 19) == ad
    assert ad_to_bs(ad).to_ad() == ad


def test_month_and_weekday_names() -> None:
    """Month and weekday names render in English and Devanagari."""
    bs = BikramDate(year=2081, month=2, day=19)
    assert bs.month_name("en") == "Jestha"
    assert bs.month_name("ne") == "जेठ"
    # 2024-06-01 is a Saturday.
    assert bs.weekday_name("en") == "Saturday"
    assert bs.weekday_name("ne") == "शनिबार"
    assert bs.weekday() == 6  # 0 = Sunday


def test_fiscal_year_boundaries() -> None:
    """The Nepal FY starts on Shrawan 1 (month 4); months 1-3 are the prior FY."""
    # Shrawan (month 4) -> FY starts this BS year.
    assert BikramDate(year=2081, month=4, day=1).fiscal_year() == "2081/82"
    assert BikramDate(year=2081, month=12, day=30).fiscal_year() == "2081/82"
    # Baishakh/Jestha/Ashadh (months 1-3) -> prior FY.
    assert BikramDate(year=2081, month=1, day=1).fiscal_year() == "2080/81"
    assert BikramDate(year=2081, month=2, day=19).fiscal_year() == "2080/81"


def test_today_and_helper() -> None:
    """today_bs() returns a plausible current BS date and the FY helper works."""
    today = today_bs()
    assert today.year >= 2080  # sanity: we're well past BS 2080
    assert 1 <= today.month <= 12
    assert nepali_fiscal_year(datetime.date(2024, 6, 1)) == "2080/81"
