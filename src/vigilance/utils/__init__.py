"""Utility helpers shared by comparison modules."""

from vigilance.utils.footnotes_utils import footnotes_list_to_dict
from vigilance.utils.genai import get_openai_api_key
from vigilance.utils.indicator_cleaner import (
    is_trailing_number_semantic,
    normalize_indicator_for_comparison,
    normalize_indicator_variants,
    strip_dates_from_indicator_label,
    strip_dates_from_table_title,
    strip_trailing_note_or_column_value,
    strip_units_currency_from_indicator_label,
)
from vigilance.utils.matching_normalizer import (
    is_generic_title,
    is_non_indicator_line,
    normalize_for_matching,
    normalize_label,
)
from vigilance.utils.text_normalizer import TextNormalizer
from vigilance.utils.type_metier import compute_type_metier

__all__ = [
    "TextNormalizer",
    "compute_type_metier",
    "footnotes_list_to_dict",
    "get_openai_api_key",
    "is_generic_title",
    "is_non_indicator_line",
    "is_trailing_number_semantic",
    "normalize_for_matching",
    "normalize_indicator_for_comparison",
    "normalize_indicator_variants",
    "normalize_label",
    "strip_dates_from_indicator_label",
    "strip_dates_from_table_title",
    "strip_trailing_note_or_column_value",
    "strip_units_currency_from_indicator_label",
]
