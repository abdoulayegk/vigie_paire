"""Tests de persistance de la geometrie verifiee au niveau de la page."""

from __future__ import annotations

from pathlib import Path

from vigilance.extraction.docling_processor import DoclingProcessor
from vigilance.extraction.vision_full_extractor import VisionFullResult


def test_page_context_bbox_replaces_docling_bbox_and_preserves_suspect_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_bbox = [0.10, 0.22, 0.90, 0.39]
    corrected_bbox = [0.08, 0.18, 0.92, 0.52]

    monkeypatch.setattr(
        "vigilance.utils.pdf_crop.is_bbox_sane",
        lambda *_args, **_kwargs: (True, None, {}),
    )
    monkeypatch.setattr(
        "vigilance.utils.pdf_crop.crop_table_region_to_bytes",
        lambda *_args, **_kwargs: b"crop",
    )
    monkeypatch.setattr(
        "vigilance.utils.page_layout_context.compute_dynamic_extensions",
        lambda **_kwargs: (0.0, 0.0),
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
