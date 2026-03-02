from __future__ import annotations

from vigilance.utils.indicator_cleaner import merge_line_split_indicators
from vigilance.utils.indicator_line_merge import (
    IndicatorLineMergeConfig,
    merge_indicator_lines,
)


def test_merge_known_split_categorie_supplementaires() -> None:
    raw = [
        "Billets subordonnés à 4,800 % aux termes des fonds propres de catégorie 1",
        "supplémentaires¹",
    ]

    merged, merge_count = merge_indicator_lines(raw)

    assert merge_count == 1
    assert merged == [
        "Billets subordonnés à 4,800 % aux termes des fonds propres de catégorie 1 supplémentaires¹"
    ]
    assert merged[0].endswith("¹")


def test_no_merge_when_next_is_total_header() -> None:
    raw = ["Billets subordonnés à 4,800 %", "Total"]

    merged, merge_count = merge_indicator_lines(raw)

    assert merge_count == 0
    assert merged == raw


def test_no_merge_when_next_is_numbered_item() -> None:
    raw = ["Billets subordonnés à 4,800 %", "1. Instruments émis"]

    merged, merge_count = merge_indicator_lines(raw)

    assert merge_count == 0
    assert merged == raw


def test_wrapper_uses_shared_rules_and_token_k_default() -> None:
    raw = [
        "Billets subordonnés à 4,800 % aux termes des fonds propres de catégorie 1",
        "supplémentaires¹",
    ]

    merged, merge_count = merge_line_split_indicators(raw)

    assert merge_count == 1
    assert len(merged) == 1


def test_no_merge_uppercase_long_new_item() -> None:
    raw = [
        "Billets subordonnés à 4,800 %",
        "PASSIFS LIÉS AUX ACTIFS FINANCIERS TRANSFÉRÉS",
    ]
    cfg = IndicatorLineMergeConfig(max_next_tokens=6)

    merged, merge_count = merge_indicator_lines(raw, config=cfg)

    assert merge_count == 0
    assert merged == raw
