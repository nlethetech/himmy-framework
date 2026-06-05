"""Nepal kernel: the Bikram Sambat (BS) calendar + Nepal fiscal year.

A typed, deterministic wrapper over the authoritative ``nepali-datetime`` library
(the ``nepal`` extra), so every Himmy agent can reason in the calendar Nepal
actually uses — convert AD <-> BS, name months/weekdays (English or Devanagari),
and resolve the Nepal fiscal year (which starts on Shrawan 1, the 4th BS month).
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel

from himmy.core.errors import HimmyError

#: BS month names, index 0 = Baishakh (month 1).
NEPALI_MONTHS_EN = [
    "Baishakh",
    "Jestha",
    "Ashadh",
    "Shrawan",
    "Bhadra",
    "Ashwin",
    "Kartik",
    "Mangsir",
    "Poush",
    "Magh",
    "Falgun",
    "Chaitra",
]
NEPALI_MONTHS_NE = [
    "बैशाख",
    "जेठ",
    "असार",
    "साउन",
    "भदौ",
    "असोज",
    "कात्तिक",
    "मंसिर",
    "पुस",
    "माघ",
    "फागुन",
    "चैत",
]
#: Weekday names, index 0 = Sunday (the Nepali week starts on Sunday).
NEPALI_WEEKDAYS_EN = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]
NEPALI_WEEKDAYS_NE = [
    "आइतबार",
    "सोमबार",
    "मंगलबार",
    "बुधबार",
    "बिहीबार",
    "शुक्रबार",
    "शनिबार",
]

#: The first month of the Nepal fiscal year (Shrawan).
_FISCAL_START_MONTH = 4


def _require_nepali_datetime():  # noqa: ANN202 - returns the module
    try:
        import nepali_datetime
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise HimmyError(
            "Bikram Sambat support requires the [nepal] extra "
            "(pip install 'himmy[nepal]')."
        ) from exc
    return nepali_datetime


class BikramDate(BaseModel):
    """A Bikram Sambat date (year/month/day), convertible to/from Gregorian."""

    year: int
    month: int
    day: int

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d} BS"

    def month_name(self, lang: str = "en") -> str:
        """The BS month name in English (``en``) or Devanagari (``ne``)."""
        names = NEPALI_MONTHS_NE if lang == "ne" else NEPALI_MONTHS_EN
        return names[self.month - 1]

    def to_ad(self) -> datetime.date:
        """Convert this BS date to its Gregorian (AD) ``datetime.date``."""
        nd = _require_nepali_datetime()
        return nd.date(self.year, self.month, self.day).to_datetime_date()

    def weekday(self) -> int:
        """Day of week, 0 = Sunday (the Nepali week start)."""
        return (self.to_ad().weekday() + 1) % 7

    def weekday_name(self, lang: str = "en") -> str:
        """The weekday name in English (``en``) or Devanagari (``ne``)."""
        names = NEPALI_WEEKDAYS_NE if lang == "ne" else NEPALI_WEEKDAYS_EN
        return names[self.weekday()]

    def fiscal_year(self) -> str:
        """The Nepal fiscal year label (e.g. ``"2081/82"``).

        The Nepal FY runs Shrawan 1 (month 4) of one BS year through Ashadh end of
        the next, so months 1–3 belong to the prior fiscal year.
        """
        start = self.year if self.month >= _FISCAL_START_MONTH else self.year - 1
        return f"{start}/{(start + 1) % 100:02d}"

    @classmethod
    def from_ad(cls, value: datetime.date) -> BikramDate:
        """Build a BS date from a Gregorian (AD) ``datetime.date``."""
        nd = _require_nepali_datetime()
        bs = nd.date.from_datetime_date(value)
        return cls(year=bs.year, month=bs.month, day=bs.day)

    @classmethod
    def today(cls) -> BikramDate:
        """Today's date in Bikram Sambat."""
        nd = _require_nepali_datetime()
        today = nd.date.today()
        return cls(year=today.year, month=today.month, day=today.day)


def ad_to_bs(value: datetime.date) -> BikramDate:
    """Convert a Gregorian date to Bikram Sambat."""
    return BikramDate.from_ad(value)


def bs_to_ad(year: int, month: int, day: int) -> datetime.date:
    """Convert a Bikram Sambat date to Gregorian."""
    return BikramDate(year=year, month=month, day=day).to_ad()


def today_bs() -> BikramDate:
    """Today's Bikram Sambat date."""
    return BikramDate.today()


def nepali_fiscal_year(value: datetime.date | None = None) -> str:
    """The Nepal fiscal-year label for an AD date (default: today)."""
    bs = BikramDate.from_ad(value) if value is not None else BikramDate.today()
    return bs.fiscal_year()


__all__ = [
    "BikramDate",
    "ad_to_bs",
    "bs_to_ad",
    "today_bs",
    "nepali_fiscal_year",
    "NEPALI_MONTHS_EN",
    "NEPALI_MONTHS_NE",
    "NEPALI_WEEKDAYS_EN",
    "NEPALI_WEEKDAYS_NE",
]
