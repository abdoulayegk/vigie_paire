"""Tests des extensions dynamiques de recadrage de tableaux."""

from __future__ import annotations

import pytest

from vigie.support.utils.page_layout_context import (
    DynamicCropExtensions,
    build_page_table_map,
    compute_dynamic_extensions,
)


def _page_map(page_num: int, entries: list[tuple[int, list[float]]]) -> dict[int, list[tuple[int, list[float]]]]:
    return {page_num: entries}


def test_tight_gap_uses_reduced_footnote_margin() -> None:
    """Gap 1.3% doit produire une extension bas > 0 (pas zero comme avant)."""
    table_bbox = [0.05, 0.20, 0.95, 0.387]  # bottom at 38.7%
    next_bbox = [0.05, 0.400, 0.95, 0.55]  # gap = 1.3%

    result = compute_dynamic_extensions(
        table_idx=7,
        page_num=29,
        table_bbox=table_bbox,
        page_table_map=_page_map(29, [(7, table_bbox), (8, next_bbox)]),
        default_bottom=0.12,
        default_top=0.03,
        tight_gap_threshold=0.03,
        tight_gap_footnote_margin=0.005,
    )

    assert isinstance(result, DynamicCropExtensions)
    assert result.tight_inter_table_gap is True
    assert result.inter_table_gap == pytest.approx(0.013, abs=0.001)
    assert result.bottom_extension == pytest.approx(0.008, abs=0.001)
    assert result.bottom_extension > 0.0


def test_normal_gap_unchanged_with_standard_margin() -> None:
    table_bbox = [0.05, 0.20, 0.95, 0.35]
    next_bbox = [0.05, 0.50, 0.95, 0.70]  # gap = 15%

    result = compute_dynamic_extensions(
        table_idx=1,
        page_num=10,
        table_bbox=table_bbox,
        page_table_map=_page_map(10, [(1, table_bbox), (2, next_bbox)]),
        default_bottom=0.12,
        default_top=0.03,
    )

    assert result.tight_inter_table_gap is False
    assert result.inter_table_gap == pytest.approx(0.15, abs=0.001)
    assert result.bottom_extension == pytest.approx(0.13, abs=0.001)


def test_last_table_on_page_extends_to_bottom_margin() -> None:
    table_bbox = [0.05, 0.60, 0.95, 0.80]

    result = compute_dynamic_extensions(
        table_idx=3,
        page_num=12,
        table_bbox=table_bbox,
        page_table_map=_page_map(12, [(3, table_bbox)]),
        default_bottom=0.12,
        default_top=0.03,
    )

    assert result.inter_table_gap is None
    assert result.tight_inter_table_gap is False
    assert result.bottom_extension == pytest.approx(0.15, abs=0.001)


def test_build_page_table_map_sorts_by_top_coordinate() -> None:
    items = [
        (2, 5, [0.1, 0.55, 0.9, 0.7], "t2", None),
        (1, 5, [0.1, 0.15, 0.9, 0.3], "t1", None),
    ]
    page_map = build_page_table_map(items)

    assert [entry[0] for entry in page_map[5]] == [1, 2]
