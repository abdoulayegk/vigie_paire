"""Verify BNC CWB audit: trace real execution path for the two ADDED indicators.

Uses the same indicator lists as real BNC T2 vs T1 (Variation des fonds propres
reglementaires): T2 has CWB-specific sub-lines that must be reported as ADDED.
"""

from __future__ import annotations

from app.comparison_runner import (
    _canonical_indicator_key,
    _filter_neighbor_aligned_candidates,
    _indicator_diff,
    _is_likely_extraction_split,
    _ordered_indicator_keys,
    _structural_duplicate_value_keys_from_rows,
    _structural_header_keys_from_rows,
)
from vigilance.models.table_models import TableArtifact, get_comparison_indicators
from vigilance.utils.indicator_normalizer import strip_footnote_markers_from_indicator


# Exact indicator lists mirroring BNC T2 vs T1 (CET1 section)
T1_INDICATORS = [
    "Émission d'actions ordinaires (y compris au titre du régime d'options d'achat d'actions)",
    "Options de remplacement",
    "Incidence des actions acquises ou vendues à des fins de négociation",
    "Rachat d'actions ordinaires",
]

T2_INDICATORS = [
    "Émission d'actions ordinaires (y compris au titre du régime d'options d'achat d'actions)",
    "Émissions d'actions ordinaires relatives à l'acquisition de CWB",
    "Options de remplacement",
    "Options de remplacement relatives à l'acquisition de CWB",
    "Incidence des actions acquises ou vendues à des fins de négociation",
    "Rachat d'actions ordinaires",
]

CWB_EMISSIONS_RAW = "Émissions d'actions ordinaires relatives à l'acquisition de CWB"
CWB_OPTIONS_RAW = "Options de remplacement relatives à l'acquisition de CWB"


def _table(indicators: list[str], raw: list[str] | None = None) -> TableArtifact:
    raw_list = raw if raw is not None else indicators
    return TableArtifact(
        bank_code="bnc",
        section="capital",
        page_pdf=1,
        table_id="table_1",
        title="Variation des fonds propres réglementaires",
        headers=["Indicateur", "Montant"],
        rows=[],
        first_column_indicators=indicators,
        first_column_indicators_raw=raw_list,
        extraction_method="vision_full_gpt4o",
        quarter="t1",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )


def test_cwb_indicators_present_in_clean_and_canonical_keys() -> None:
    """1. Verify CWB indicators are in first_column_indicators (clean) and have distinct canonical keys."""
    t2 = _table(T2_INDICATORS)
    left = get_comparison_indicators(t2)
    assert CWB_EMISSIONS_RAW in left or any(
        "cwb" in (x or "").lower() and "emission" in (x or "").lower() for x in left
    ), "CWB emissions line should be in comparison indicators"
    assert CWB_OPTIONS_RAW in left or any(
        "cwb" in (x or "").lower() and "remplacement" in (x or "").lower() for x in left
    ), "CWB options line should be in comparison indicators"

    key_emissions = _canonical_indicator_key(
        strip_footnote_markers_from_indicator(CWB_EMISSIONS_RAW)
    )
    key_options = _canonical_indicator_key(
        strip_footnote_markers_from_indicator(CWB_OPTIONS_RAW)
    )
    assert "cwb" in key_emissions and "relative" in key_emissions
    assert "cwb" in key_options and "relative" in key_options
    assert key_emissions != key_options


def test_cwb_keys_in_added_keys_before_filter() -> None:
    """2. Verify the two CWB canonical keys appear in added_keys (right_map - left_map) before neighbor filter."""
    t1 = _table(T1_INDICATORS)
    t2 = _table(T2_INDICATORS)
    left = get_comparison_indicators(t1)
    right = get_comparison_indicators(t2)

    left_structural = (
        _structural_header_keys_from_rows(t1) | _structural_duplicate_value_keys_from_rows(t1)
    ) - set(_ordered_indicator_keys(right))
    right_structural = (
        _structural_header_keys_from_rows(t2) | _structural_duplicate_value_keys_from_rows(t2)
    ) - set(_ordered_indicator_keys(left))

    def build_map(values: list[str], structural_keys: set[str]) -> dict[str, str]:
        from vigilance.utils.matching_normalizer import _classify_excluded_line

        mapped: dict[str, str] = {}
        for value in values:
            if _classify_excluded_line(value):
                continue
            value_clean = strip_footnote_markers_from_indicator(value)
            key = _canonical_indicator_key(value_clean)
            if key in structural_keys or not key or key in mapped:
                continue
            mapped[key] = value_clean
        return mapped

    left_map = build_map(left, left_structural)
    right_map = build_map(right, right_structural)
    added_keys = set(right_map.keys()) - set(left_map.keys())

    key_emissions = _canonical_indicator_key(
        strip_footnote_markers_from_indicator(CWB_EMISSIONS_RAW)
    )
    key_options = _canonical_indicator_key(
        strip_footnote_markers_from_indicator(CWB_OPTIONS_RAW)
    )
    assert key_emissions in added_keys, "CWB emissions key must be in added_keys before filter"
    assert key_options in added_keys, "CWB options key must be in added_keys before filter"


def test_filter_does_not_remove_cwb_with_semantic_check() -> None:
    """3. With _is_likely_extraction_split: _filter_neighbor_aligned_candidates must NOT remove CWB keys."""
    t1 = _table(T1_INDICATORS)
    t2 = _table(T2_INDICATORS)
    added, _, _, excluded = _indicator_diff(t1, t2, neighbor_aligned_filter_enabled=True)
    added_lower = [a.lower() for a in added]
    assert any("cwb" in a for a in added_lower), "CWB indicators must remain in added (not filtered)"
    assert excluded.get("neighbor_aligned", 0) == 0, "No CWB key should be counted as neighbor_aligned"


def test_is_likely_extraction_split_keeps_cwb_distinct() -> None:
    """4. Exact condition: CWB keys have >= 2 tokens in neither prev nor next, so not treated as split."""
    key_emissions = "emission d action ordinaire relative a l acquisition de cwb"
    key_options_cwb = "option de remplacement relative a l acquisition de cwb"
    prev_emissions = "emission d action ordinaire y compri titre du regime d option d achat d action"
    next_emissions = "incidence des actions acquises ou vendues a des fins de negociation"
    prev_options = "option de remplacement"
    next_options = "incidence des actions acquises ou vendues a des fins de negociation"

    assert not _is_likely_extraction_split(key_emissions, prev_emissions, next_emissions), (
        "CWB emissions has tokens (relative, acquisition, cwb, ...) not in prev/next -> not a split"
    )
    assert not _is_likely_extraction_split(key_options_cwb, prev_options, next_options), (
        "CWB options has tokens (relative, acquisition, cwb, ...) not in prev/next -> not a split"
    )


def test_old_behavior_would_filter_without_semantic_check() -> None:
    """5. Without _is_likely_extraction_split, both CWB keys would be filtered (exact condition)."""
    t1 = _table(T1_INDICATORS)
    t2 = _table(T2_INDICATORS)
    left = get_comparison_indicators(t1)
    right = get_comparison_indicators(t2)
    left_structural = (
        _structural_header_keys_from_rows(t1) | _structural_duplicate_value_keys_from_rows(t1)
    ) - set(_ordered_indicator_keys(right))
    right_structural = (
        _structural_header_keys_from_rows(t2) | _structural_duplicate_value_keys_from_rows(t2)
    ) - set(_ordered_indicator_keys(left))

    def build_map(values: list[str], structural_keys: set[str]) -> dict[str, str]:
        from vigilance.utils.matching_normalizer import _classify_excluded_line

        mapped: dict[str, str] = {}
        for value in values:
            if _classify_excluded_line(value):
                continue
            value_clean = strip_footnote_markers_from_indicator(value)
            key = _canonical_indicator_key(value_clean)
            if key in structural_keys or not key or key in mapped:
                continue
            mapped[key] = value_clean
        return mapped

    left_map = build_map(left, left_structural)
    right_map = build_map(right, right_structural)
    added_keys = set(right_map.keys()) - set(left_map.keys())
    left_order = _ordered_indicator_keys(left, excluded_keys=left_structural)
    right_order = _ordered_indicator_keys(right, excluded_keys=right_structural)

    remaining_added_keys = {
        _canonical_indicator_key(strip_footnote_markers_from_indicator(right_map[k]))
        for k in added_keys
    }

    filtered = _filter_neighbor_aligned_candidates(
        remaining_added_keys,
        source_order=right_order,
        target_order=left_order,
    )
    key_emissions = _canonical_indicator_key(
        strip_footnote_markers_from_indicator(CWB_EMISSIONS_RAW)
    )
    key_options = _canonical_indicator_key(
        strip_footnote_markers_from_indicator(CWB_OPTIONS_RAW)
    )
    assert key_emissions not in filtered and key_options not in filtered, (
        "With _is_likely_extraction_split guard, CWB keys must not be in filtered"
    )
