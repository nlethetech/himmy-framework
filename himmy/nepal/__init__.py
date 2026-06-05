"""Nepal kernel: Nepal-domain primitives (Bikram Sambat calendar, …)."""

from __future__ import annotations

from himmy.nepal.calendar import (
    NEPALI_MONTHS_EN,
    NEPALI_MONTHS_NE,
    NEPALI_WEEKDAYS_EN,
    NEPALI_WEEKDAYS_NE,
    BikramDate,
    ad_to_bs,
    bs_to_ad,
    nepali_fiscal_year,
    today_bs,
)
from himmy.nepal.language import (
    NepaliEmbedder,
    is_devanagari,
    normalize_nepali,
    transliterate,
)

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
    "is_devanagari",
    "transliterate",
    "normalize_nepali",
    "NepaliEmbedder",
]
