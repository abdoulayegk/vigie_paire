"""Utilitaires partages par les modules de comparaison."""

from vigie.support.utils.footnotes_utils import (
    footnotes_list_to_dict,
    normalize_footnotes_to_canonical,
)
from vigie.support.utils.genai import get_openai_api_key
from vigie.support.utils.indicator_cleaner import (
    clean_spaced_out_text,
    clean_table_title_contamination,
    dedupe_indicators,
    is_table_title_contaminated,
    is_trailing_number_semantic,
    normalize_indicator_for_comparison,
    normalize_indicator_variants,
    singularize_words,
    strip_dates_from_indicator_label,
    strip_dates_from_table_title,
    strip_trailing_note_or_column_value,
    strip_units_currency_from_indicator_label,
)
from vigie.support.utils.matching_normalizer import (
    is_generic_title,
    is_non_indicator_line,
    normalize_for_matching,
    normalize_label,
)
from vigie.support.utils.type_metier import compute_type_metier

__all__ = [
    "clean_spaced_out_text",
    "clean_table_title_contamination",
    "dedupe_indicators",
    "is_table_title_contaminated",
    "compute_type_metier",
    "footnotes_list_to_dict",
    "normalize_footnotes_to_canonical",
    "get_openai_api_key",
    "is_generic_title",
    "is_non_indicator_line",
    "is_trailing_number_semantic",
    "normalize_for_matching",
    "normalize_indicator_for_comparison",
    "normalize_indicator_variants",
    "normalize_label",
    "singularize_words",
    "strip_dates_from_indicator_label",
    "strip_dates_from_table_title",
    "strip_trailing_note_or_column_value",
    "strip_units_currency_from_indicator_label",
]
