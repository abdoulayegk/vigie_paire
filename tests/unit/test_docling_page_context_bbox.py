"""Tests de persistance de la geometrie verifiee au niveau de la page."""

from __future__ import annotations

from pathlib import Path

import pytest

from vigie.extraction.docling import (
    DoclingProcessor,
    ExtractedTable,
)
from vigie.extraction.locator_merge_reconciliation import (
    _is_locator_merge_conflict,
    _reconcile_on_demand_locator_merges,
)
from vigie.extraction.vision_full import VisionFullResult
from vigie.support.utils import page_layout_context


def test_page_context_inventory_padding_includes_section_boundaries() -> None:
    processor = DoclingProcessor()

    padded = processor._pad_page_ranges([(6, 8), (29, 47), (47, 52)], 2)

    assert padded == [(4, 10), (27, 49), (45, 54)]
    assert processor._is_page_in_ranges(9, padded) is True
    assert processor._is_page_in_ranges(10, padded) is True
    assert processor._is_page_in_ranges(11, padded) is False


def test_page_context_inventory_padding_never_creates_page_zero() -> None:
    processor = DoclingProcessor()

    assert processor._pad_page_ranges([(1, 2)], 2) == [(1, 4)]


def test_locator_merge_conflict_preserves_distinct_docling_blocks() -> None:
    first_original = [0.06, 0.36, 0.96, 0.61]
    second_original = [0.06, 0.63, 0.96, 0.87]
    merged_locator = [0.06, 0.36, 0.96, 0.87]

    assert (
        _is_locator_merge_conflict(
            first_original,
            second_original,
            merged_locator,
            merged_locator,
        )
        is True
    )


def test_locator_same_region_is_not_conflict_for_duplicate_docling_boxes() -> None:
    original = [0.06, 0.18, 0.96, 0.75]
    corrected = [0.05, 0.17, 0.97, 0.76]

    assert (
        _is_locator_merge_conflict(
            original,
            original,
            corrected,
            corrected,
        )
        is False
    )


def _locator_table(
    *,
    table_id: str,
    original_bbox: list[float],
    final_bbox: list[float],
    indicators: list[str],
    headers: list[str],
    title: str = "T25 Groupes d’actifs liquides",
    bbox_source: str = "page_context_locator",
) -> ExtractedTable:
    return ExtractedTable(
        table_id=table_id,
        page_number=38,
        title=title,
        headers=headers,
        rows=[],
        first_column_indicators=indicators,
        first_column_indicators_raw=indicators,
        bbox=final_bbox,
        extraction_method="vision_full_gpt4o",
        extraction_status="rescued",
        debug_metrics={
            "bbox_original": original_bbox,
            "bbox_final": final_bbox,
            "bbox_source": bbox_source,
            "bbox_verified": bbox_source == "page_context_locator",
        },
    )


def test_on_demand_locator_collapses_same_semantic_table_split_by_docling() -> None:
    final_bbox = [0.061, 0.527, 0.939, 0.944]
    first = _locator_table(
        table_id="tableau_1",
        original_bbox=[0.061, 0.512, 0.940, 0.721],
        final_bbox=final_bbox,
        indicators=[
            "Trésorerie et dépôts auprès de banques centrales",
            "Dépôts auprès d’autres institutions financières",
            "Métaux précieux",
            "Titres",
            "Total",
        ],
        headers=["Actifs liquides", "Total", "Donnés en garantie"],
    )
    second = _locator_table(
        table_id="tableau_2",
        original_bbox=[0.060, 0.737, 0.940, 0.941],
        final_bbox=final_bbox,
        indicators=[
            "Trésorerie et dépôts auprès de banques centrales",
            "Dépôts auprès d’autres institutions financières",
            "Métaux précieux",
            "Titres :",
            "Total",
        ],
        headers=[
            "Actifs liquides",
            "Total",
            "Actifs liquides grevés",
            "Donnés en garantie",
        ],
    )

    reconciled = _reconcile_on_demand_locator_merges([first, second])

    assert len(reconciled) == 1
    assert reconciled[0].table_id == "tableau_1"
    assert len(reconciled[0].headers) == 4
    assert reconciled[0].debug_metrics["locator_merge_collapsed"] is True
    assert reconciled[0].debug_metrics["locator_merged_table_ids"] == [
        "tableau_1",
        "tableau_2",
    ]
    assert reconciled[0].debug_metrics["locator_original_bboxes"] == [
        [0.061, 0.512, 0.94, 0.721],
        [0.06, 0.737, 0.94, 0.941],
    ]


def test_on_demand_locator_preserves_distinct_semantic_tables() -> None:
    final_bbox = [0.05, 0.10, 0.95, 0.90]
    first = _locator_table(
        table_id="tableau_1",
        original_bbox=[0.05, 0.10, 0.95, 0.40],
        final_bbox=final_bbox,
        title="Ratios de capital",
        indicators=["Ratio CET1", "Ratio de levier", "Ratio TLAC"],
        headers=["Mesure", "Valeur"],
    )
    second = _locator_table(
        table_id="tableau_2",
        original_bbox=[0.05, 0.60, 0.95, 0.90],
        final_bbox=final_bbox,
        title="Financement de gros",
        indicators=["Dépôts", "Titres de créance", "Total"],
        headers=["Source", "Échéance"],
    )

    reconciled = _reconcile_on_demand_locator_merges([first, second])

    assert reconciled == [first, second]


def test_on_demand_locator_collapses_contained_docling_copy() -> None:
    locator = _locator_table(
        table_id="tableau_1",
        original_bbox=[0.057, 0.558, 0.942, 0.637],
        final_bbox=[0.055, 0.566, 0.947, 0.736],
        title="TABLEAU 25",
        indicators=["Canada", "États-Unis", "Total"],
        headers=["Moins de 5 ans", "De 6 à 10 ans", "Plus de 35 ans"],
    )
    contained_docling = _locator_table(
        table_id="tableau_2",
        original_bbox=[0.057, 0.650, 0.941, 0.729],
        final_bbox=[0.057, 0.650, 0.941, 0.729],
        title="",
        indicators=["Canada", "États-Unis", "Total"],
        headers=["Moins de 5 ans", "De 6 à 10 ans", "Plus de 35 ans"],
        bbox_source="docling",
    )

    reconciled = _reconcile_on_demand_locator_merges([locator, contained_docling])

    assert len(reconciled) == 1
    assert reconciled[0].bbox == locator.bbox
    assert reconciled[0].debug_metrics["bbox_source"] == "page_context_locator"
    assert reconciled[0].debug_metrics["locator_merge_collapsed"] is True


def test_page_context_bbox_replaces_docling_bbox_and_preserves_suspect_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_bbox = [0.10, 0.22, 0.90, 0.39]
    corrected_bbox = [0.08, 0.18, 0.92, 0.52]

    monkeypatch.setattr(
        "vigie.support.utils.pdf_crop.is_bbox_sane",
        lambda *_args, **_kwargs: (True, None, {}),
    )
    monkeypatch.setattr(
        "vigie.support.utils.pdf_crop.crop_table_region_to_bytes",
        lambda *_args, **_kwargs: b"crop",
    )
    monkeypatch.setattr(
        "vigie.support.utils.page_layout_context.compute_dynamic_extensions",
        lambda **_kwargs: page_layout_context.DynamicCropExtensions(
            top_extension=0.0,
            bottom_extension=0.0,
            inter_table_gap=None,
            tight_inter_table_gap=False,
        ),
    )

    class FakeVisionExtractor:
        def extract_with_quality_pass(self, **_kwargs) -> VisionFullResult:
            return VisionFullResult(
                table_title="",
                table_summary="",
                headers=[],
                indicators=[],
                footnotes_content=[],
                no_table_detected=True,
                extraction_status="confirmed_no_table",
            )

    processor = DoclingProcessor()
    _, table, _ = processor._vision_extract_one_table(
        (7, 32, corrected_bbox, "tableau_7", None),
        {
            "pdf_path": tmp_path / "report.pdf",
            "bank_code": "rbc",
            "quarter": "t2",
            "year": 2026,
            "pdf_sha": "pdf-sha",
            "vision_extraction_cfg": {},
            "bottom_extension_footnotes": 0.0,
            "top_extension_title": 0.0,
            "horizontal_padding": 0.0,
            "vision_extractor": FakeVisionExtractor(),
            "page_table_locator": None,
            "schema_failure_flag": [False],
            "vision_schema_error_cls": RuntimeError,
            "schema_failure_policy": "fail_fast",
            "labels_only": False,
            "page_table_map": {},
            "page_context_seed": {
                7: {
                    "bbox_original": original_bbox,
                    "bbox_norm": corrected_bbox,
                    "bbox_source": "page_context_inventory",
                    "confidence": 0.97,
                    "title_text": "Prêts douteux bruts",
                    "continuation": False,
                    "table_count": 1,
                }
            },
        },
    )

    assert table.bbox == corrected_bbox
    assert table.title == "Prêts douteux bruts"
    assert table.extraction_status == "suspect_unresolved"
    assert table.tables_on_page == 1
    assert table.debug_metrics["bbox_original"] == original_bbox
    assert table.debug_metrics["bbox_final"] == corrected_bbox
    assert table.debug_metrics["bbox_source"] == "page_context_inventory"
    assert table.debug_metrics["bbox_confidence"] == 0.97
    assert table.debug_metrics["bbox_verified"] is True


def test_unresolved_near_full_page_bbox_is_preserved_as_suspect(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_bbox = [0.01, 0.01, 0.99, 0.99]

    monkeypatch.setattr(
        "vigie.support.utils.pdf_crop.is_bbox_sane",
        lambda *_args, **_kwargs: (
            False,
            "bbox_near_full_page",
            {"width": 0.98, "height": 0.98},
        ),
    )

    processor = DoclingProcessor()
    _, table, _ = processor._vision_extract_one_table(
        (7, 32, original_bbox, "tableau_7", None),
        {
            "pdf_path": tmp_path / "report.pdf",
            "bank_code": "rbc",
            "quarter": "t2",
            "year": 2026,
            "pdf_sha": "pdf-sha",
            "vision_extraction_cfg": {},
            "bottom_extension_footnotes": 0.0,
            "top_extension_title": 0.0,
            "horizontal_padding": 0.0,
            "vision_extractor": object(),
            "page_table_locator": None,
            "schema_failure_flag": [False],
            "vision_schema_error_cls": RuntimeError,
            "schema_failure_policy": "fail_fast",
            "labels_only": False,
            "page_table_map": {},
            "page_context_seed": {
                7: {
                    "bbox_original": original_bbox,
                    "bbox_source": "near_full_page_unresolved",
                    "table_count": 2,
                    "bbox_verification_reason": ("near_full_page_multiple_regions"),
                }
            },
        },
    )

    assert table.bbox == original_bbox
    assert table.extraction_status == "suspect_unresolved"
    assert table.debug_metrics["bbox_source"] == "near_full_page_unresolved"
    assert table.debug_metrics["bbox_verified"] is False
    assert table.debug_metrics["page_context_table_count"] == 2
    assert table.debug_metrics["bbox_verification_reason"] == "near_full_page_multiple_regions"


def test_proactive_page_context_crop_used_when_tight_inter_table_gap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    table_bbox = [0.05, 0.20, 0.95, 0.387]
    proactive_bbox = [0.08, 0.18, 0.92, 0.42]
    received: dict = {}

    monkeypatch.setattr(
        "vigie.support.utils.pdf_crop.is_bbox_sane",
        lambda *_args, **_kwargs: (True, None, {}),
    )
    monkeypatch.setattr(
        "vigie.support.utils.pdf_crop.crop_table_region_to_bytes",
        lambda *_args, **_kwargs: b"docling-crop",
    )
    monkeypatch.setattr(
        "vigie.extraction.docling.vision_pass.render_pdf_page",
        lambda *_args, **_kwargs: b"page-image",
    )
    monkeypatch.setattr(
        "vigie.support.utils.page_layout_context.compute_dynamic_extensions",
        lambda **_kwargs: page_layout_context.DynamicCropExtensions(
            top_extension=0.03,
            bottom_extension=0.008,
            inter_table_gap=0.013,
            tight_inter_table_gap=True,
        ),
    )

    class FakePageTableLocator:
        min_confidence = 0.85

        def locate_page(self, *_args, **_kwargs):
            from vigie.extraction.page_table_locator import _parse_page_layout

            payload = {
                "tables": [
                    {
                        "table_bbox": proactive_bbox,
                        "title_bbox": [0.08, 0.14, 0.70, 0.17],
                        "footnotes_bbox": [0.08, 0.43, 0.90, 0.48],
                        "title_text": "Ratios",
                        "continuation": False,
                        "confidence": 0.96,
                    }
                ],
                "table_count": 1,
            }
            return _parse_page_layout(payload, 29)

    class FakeVisionExtractor:
        def extract_with_quality_pass(self, **kwargs) -> VisionFullResult:
            received.update(kwargs)
            return VisionFullResult(
                table_title="Ratios",
                table_summary="",
                headers=[],
                indicators=["CET1"],
                footnotes_content=[{"id": "1", "text": "(1) Note"}],
                extraction_status="ok",
            )

    processor = DoclingProcessor()
    _, table, _ = processor._vision_extract_one_table(
        (7, 29, table_bbox, "tableau_7", None),
        {
            "pdf_path": tmp_path / "report.pdf",
            "bank_code": "bnc",
            "quarter": "t1",
            "year": 2025,
            "pdf_sha": "pdf-sha",
            "vision_extraction_cfg": {},
            "bottom_extension_footnotes": 0.12,
            "top_extension_title": 0.03,
            "horizontal_padding": 0.02,
            "vision_extractor": FakeVisionExtractor(),
            "page_table_locator": FakePageTableLocator(),
            "schema_failure_flag": [False],
            "vision_schema_error_cls": RuntimeError,
            "schema_failure_policy": "fail_fast",
            "labels_only": False,
            "page_table_map": {29: [(7, table_bbox), (8, [0.05, 0.40, 0.95, 0.55])]},
            "page_context_seed": {},
        },
    )

    assert received["crop_bytes"] == b"docling-crop"
    assert received["bbox_norm"] != table_bbox
    assert received["bbox_norm"][1] == pytest.approx(0.18)
    assert received["bbox_norm"][3] > table_bbox[3]
    assert table.extraction_status == "ok"
    assert table.first_column_indicators == ["cet1"]
    assert table.title == "Ratios"
