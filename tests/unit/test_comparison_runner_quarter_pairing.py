from __future__ import annotations

from app import comparison_runner as cr
from vigilance.models.table_models import TableArtifact


def _mk_table(table_id: str, page: int) -> TableArtifact:
    return TableArtifact(
        bank_code="bnc",
        section="capital_management",
        page_pdf=page,
        table_id=table_id,
        title="Capital",
        headers=["metric", "value"],
        rows=[["ratio", "1"]],
        first_column_indicators=["ratio"],
        extraction_method="vision_primary",
        bbox=[0.1, 0.1, 0.8, 0.5],
        pdf_path="/tmp/fake.pdf",
    )


def test_run_comparison_uses_current_vs_previous_quarter_context(monkeypatch) -> None:
    calls: list[dict] = []

    previous_table = _mk_table("prev", 10)
    current_table = _mk_table("curr", 12)

    def _fake_extract_tables(**kwargs):
        calls.append(dict(kwargs))
        quarter = kwargs.get("quarter")
        if quarter == "t3":
            return [previous_table]
        if quarter == "t1":
            return [current_table]
        raise AssertionError(f"Unexpected quarter passed to extraction: {quarter!r}")

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "merge_table_fragments", lambda tables, **_kw: (tables, []))
    monkeypatch.setattr(
        cr,
        "run_strict_intra_section_compare",
        lambda **_kw: {"pairs": [], "added_tables": [], "removed_tables": []},
    )
    monkeypatch.setattr(cr, "get_matching_thresholds", lambda **_kw: {})
    monkeypatch.setattr(cr, "get_quality_gate_config", lambda **_kw: {"enabled": False})
    monkeypatch.setattr(
        cr,
        "get_validation_config",
        lambda **_kw: {
            "vision_pair_validation": False,
            "rename_validator_enabled": False,
        },
    )

    result = cr.run_comparison_with_sections(
        pdf_path_previous="/tmp/q3_2025.pdf",
        pdf_path_current="/tmp/q1_2026.pdf",
        bank_code="bnc",
        sections_previous=[
            {"section": "capital_management", "start_page": 1, "end_page": 2}
        ],
        sections_current=[
            {"section": "capital_management", "start_page": 1, "end_page": 2}
        ],
        current_quarter="Q1-2026",
        current_year=2026,
        api_key="test-key",
    )

    assert len(calls) == 2
    assert calls[0]["quarter"] == "t3"
    assert calls[0]["year"] == 2025
    assert calls[1]["quarter"] == "t1"
    assert calls[1]["year"] == 2026

    assert result["current_quarter"] == "Q1-2026"
    assert result["previous_quarter"] == "Q3-2025"
    assert result["comparison_direction"] == "current_vs_previous"
    assert result["quarter_to"] == "Q1-2026"
    assert result["quarter_from"] == "Q3-2025"
