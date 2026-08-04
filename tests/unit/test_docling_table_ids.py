from __future__ import annotations

from vigie.extraction.docling import (
    ExtractedTable,
    extract_tables_docling_by_sections,
)


def _table(*, table_id: str, page: int, bbox: list[float]) -> ExtractedTable:
    return ExtractedTable(
        table_id=table_id,
        page_number=page,
        title=None,
        headers=[],
        rows=[],
        bbox=bbox,
    )


def test_extract_tables_docling_by_sections_assigns_page_local_table_ids(
    monkeypatch,
) -> None:
    raw_tables = [
        _table(table_id="legacy_b", page=12, bbox=[0.1, 0.55, 0.9, 0.85]),
        _table(table_id="legacy_a", page=12, bbox=[0.1, 0.10, 0.9, 0.40]),
        _table(table_id="legacy_c", page=13, bbox=[0.1, 0.20, 0.9, 0.50]),
    ]

    def fake_extract_tables_docling_priority(**kwargs):
        return raw_tables

    monkeypatch.setattr(
        "vigie.extraction.docling.processor.extract_tables_docling_priority",
        fake_extract_tables_docling_priority,
    )

    tables = extract_tables_docling_by_sections(
        pdf_path="dummy.pdf",
        bank_code="bnc",
        quarter="t1",
        year=2025,
        section_ranges=[
            {"section": "risk_management", "start": 12, "end": 13},
        ],
    )

    assert [table.table_id for table in tables] == [
        "tbl_p012_i02",
        "tbl_p012_i01",
        "tbl_p013_i01",
    ]
    assert [table.table_index_on_page for table in tables] == [2, 1, 1]
    assert [table.tables_on_page for table in tables] == [2, 2, 1]
    assert [table.page_local_role for table in tables] == ["last", "first", "single"]
    assert all(table.section == "risk_management" for table in tables)
