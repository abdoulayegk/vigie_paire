"""E2E integration test for Vision extraction pipeline.

Verifies extract_tables_docling_by_sections with use_vision_extraction=True
produces tables that write_footnotes_json outputs correctly (real text in
footnotes_content, no dict repr).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from vigie.extraction.docling import (
    ExtractedDocument,
    ExtractedTable,
    extract_tables_docling_by_sections,
)
from vigie.extraction.vision_extraction_writer import write_footnotes_json


def test_vision_pipeline_produces_footnotes_with_real_text(tmp_path: Path) -> None:
    """Run pipeline with mocked extract_pdf; assert footnotes.json has real text."""
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    vision_table = ExtractedTable(
        table_id="TABLE_1",
        page_number=1,
        title="LCR",
        headers=["Indicator", "Value"],
        rows=[["LCR Ratio", "125%"], ["NSFR", "110%"]],
        first_column_indicators=["LCR Ratio", "NSFR"],
        footnotes=[{"id": "1", "text": "LCR calcule conformement aux normes BSIF."}],
        extraction_method="vision_full_gpt4o",
    )

    def mock_extract_pdf(
        path: str,
        bank_code: str,
        quarter: str,
        year: int,
        *,
        page_ranges: object = None,
        use_vision_extraction: bool | None = None,
    ) -> ExtractedDocument:
        assert use_vision_extraction is True
        return ExtractedDocument(
            file_path=path,
            bank_code=bank_code,
            quarter=quarter,
            year=year,
            total_pages=1,
            all_tables=[vision_table],
        )

    with patch(
        "vigie.extraction.docling.processor.extract_pdf",
        side_effect=mock_extract_pdf,
    ):
        tables = extract_tables_docling_by_sections(
            pdf_path=str(pdf_path),
            bank_code="bnc",
            quarter="t1",
            year=2025,
            section_ranges=[{"section": "test_section", "start": 1, "end": 1}],
            use_vision_extraction=True,
        )

    assert tables
    write_footnotes_json(tables, [], out_dir, "bnc", "vision_e2e_test")
    fn_path = out_dir / "footnotes.json"
    assert fn_path.exists()
    data = json.loads(fn_path.read_text(encoding="utf-8"))

    assert "tables" in data
    assert "bank_code" in data
    assert "meta" in data
    assert data["meta"]["tables_total"] >= 1
    assert data["meta"]["footnote_entries_total"] >= 1
    assert data["meta"]["repr_suspect_count"] == 0
    for entry in data["tables"]:
        assert "table_id" in entry
        assert "title" in entry
        assert "page" in entry
        assert "source" in entry
        assert "footnotes_content" in entry
        for k, v in entry["footnotes_content"].items():
            assert isinstance(v, str)
            assert v
            assert "{" not in v
            assert "'" not in v
            assert "dict" not in v.lower()


def test_docling_only_flags_stay_strict_when_disabled(tmp_path: Path) -> None:
    """Docling-only run must not implicitly activate Vision extraction flags."""
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"")

    docling_table = ExtractedTable(
        table_id="TABLE_1",
        page_number=1,
        title="LCR",
        headers=["Indicator", "Value"],
        rows=[["LCR Ratio", "125%"]],
        first_column_indicators=["LCR Ratio"],
        footnotes=[{"id": "1", "text": "Docling note"}],
        extraction_method="docling",
    )

    def mock_extract_pdf(
        path: str,
        bank_code: str,
        quarter: str,
        year: int,
        *,
        page_ranges: object = None,
        use_vision_extraction: bool | None = None,
    ) -> ExtractedDocument:
        assert use_vision_extraction is False
        return ExtractedDocument(
            file_path=path,
            bank_code=bank_code,
            quarter=quarter,
            year=year,
            total_pages=1,
            all_tables=[docling_table],
        )

    with patch(
        "vigie.extraction.docling.processor.extract_pdf",
        side_effect=mock_extract_pdf,
    ):
        tables = extract_tables_docling_by_sections(
            pdf_path=str(pdf_path),
            bank_code="bnc",
            quarter="t1",
            year=2025,
            section_ranges=[{"section": "test_section", "start": 1, "end": 1}],
            use_vision_extraction=False,
        )

    assert tables
    assert tables[0].extraction_method == "docling"
