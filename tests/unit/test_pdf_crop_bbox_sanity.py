"""Tests des garde-fous geometriques appliques aux crops de tableaux."""

from vigilance.utils.pdf_crop import bbox_sanity_profile, is_bbox_sane


def test_wide_financial_table_is_not_rejected_as_full_page() -> None:
    bbox = [0.04, 0.20, 0.94, 0.45]

    sane, reason, profile = is_bbox_sane(
        bbox,
        {"bbox_near_full_page_threshold": 0.90},
    )

    assert sane is True
    assert reason is None
    assert profile["width_norm"] == 0.90
    assert profile["height_norm"] == 0.25


def test_tall_narrow_table_is_not_rejected_as_full_page() -> None:
    sane, reason, _profile = is_bbox_sane(
        [0.20, 0.04, 0.55, 0.96],
        {"bbox_near_full_page_threshold": 0.90},
    )

    assert sane is True
    assert reason is None


def test_actual_near_full_page_bbox_is_rejected() -> None:
    bbox = [0.02, 0.02, 0.98, 0.98]

    sane, reason, profile = is_bbox_sane(
        bbox,
        {"bbox_near_full_page_threshold": 0.90},
    )

    assert sane is False
    assert reason == "bbox_near_full_page"
    assert profile["is_near_full_page"] is True


def test_default_profile_marks_only_actual_full_page_geometry() -> None:
    wide = bbox_sanity_profile([0.02, 0.20, 0.98, 0.45])
    full = bbox_sanity_profile([0.02, 0.02, 0.98, 0.98])

    assert wide["is_near_full_page"] is False
    assert full["is_near_full_page"] is True
