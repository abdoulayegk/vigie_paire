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
        first_column_indicators_raw=["ratio"],
        extraction_method="vision_full_gpt4o",
        bbox=[0.1, 0.1, 0.8, 0.5],
        pdf_path="/tmp/fake.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
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
    assert result["meta"]["extraction_sources"]["previous"]["quarter"] == "t3"
    assert result["meta"]["extraction_sources"]["previous"]["mode"] == "unknown"
    assert result["meta"]["extraction_sources"]["previous"]["tables_path"].endswith(
        "outputs/extractions/bnc/2025/t3/tables.json"
    )


def test_run_comparison_surfaces_pairing_metrics_from_strict_matcher(monkeypatch) -> None:
    previous_table = _mk_table("prev", 10)
    current_table = _mk_table("curr", 12)

    def _fake_extract_tables(**kwargs):
        return [previous_table] if kwargs.get("quarter") == "t1" else [current_table]

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "merge_table_fragments", lambda tables, **_kw: (tables, []))
    monkeypatch.setattr(
        cr,
        "run_strict_intra_section_compare",
        lambda **_kw: {
            "pairs": [],
            "added_tables": [],
            "removed_tables": [],
            "tables_comparable_t1": 1,
            "tables_comparable_t2": 1,
            "pairing_coverage": 0.0,
            "ambiguous_pairs": [
                {"t2_uid": "capital_management|curr|p12", "candidate_t1_uids": ["capital_management|prev|p10"]}
            ],
            "ambiguous_tables": [
                {
                    "side": "current",
                    "uid": "capital_management|curr|p12",
                    "table_id": "curr",
                    "title": "Capital",
                    "page": 12,
                    "section": "capital_management",
                    "reason": "ambiguous_candidate",
                }
            ],
        },
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
        pdf_path_t1="/tmp/q1_2025.pdf",
        pdf_path_t2="/tmp/q2_2025.pdf",
        bank_code="bnc",
        sections_t1=[{"section": "capital_management", "start_page": 1, "end_page": 2}],
        sections_t2=[{"section": "capital_management", "start_page": 1, "end_page": 2}],
        api_key="test-key",
    )

    assert result["summary"]["tables_comparable_t1"] == 1
    assert result["summary"]["tables_comparable_t2"] == 1
    assert result["summary"]["pairing_coverage"] == 0.0
    assert result["summary"]["ambiguous_pairs"] == 1
    assert result["summary"]["pairing_low_confidence"] is True
    assert result["meta"]["validation_summary"]["strict_matcher"]["pairing_coverage"] == 0.0


def test_run_comparison_includes_extraction_source_provenance(monkeypatch) -> None:
    previous_table = _mk_table("prev", 10)
    current_table = _mk_table("curr", 12)

    def _fake_extract_tables(**kwargs):
        quarter = kwargs.get("quarter")
        provenance = {
            "mode": "stored" if quarter == "t1" else "fresh",
            "artifact_dir": f"/tmp/{quarter}",
            "tables_path": f"/tmp/{quarter}/tables.json",
            "indicators_path": f"/tmp/{quarter}/indicators.json",
            "footnotes_path": f"/tmp/{quarter}/footnotes.json",
            "meta_path": f"/tmp/{quarter}/meta.json",
            "artifacts_present": {
                "tables": True,
                "meta": True,
                "indicators": True,
                "footnotes": True,
            },
        }
        tables = [previous_table] if quarter == "t1" else [current_table]
        if kwargs.get("return_provenance"):
            return tables, provenance
        return tables

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
        pdf_path_t1="/tmp/q1_2025.pdf",
        pdf_path_t2="/tmp/q2_2025.pdf",
        bank_code="bnc",
        sections_t1=[{"section": "capital_management", "start_page": 1, "end_page": 2}],
        sections_t2=[{"section": "capital_management", "start_page": 1, "end_page": 2}],
        api_key="test-key",
    )

    prev = result["meta"]["extraction_sources"]["previous"]
    curr = result["meta"]["extraction_sources"]["current"]
    assert prev["mode"] == "stored"
    assert curr["mode"] == "fresh"
    assert prev["tables_path"].endswith("/t1/tables.json")
    assert curr["indicators_path"].endswith("/t2/indicators.json")


def test_empty_result_includes_structured_extraction_sources() -> None:
    result = cr._empty_result(
        "bnc",
        2025,
        "Aucune section valide fournie.",
        quarter_context={
            "previous": {"code": "t1", "label": "Q1-2025", "year": 2025},
            "current": {"code": "t2", "label": "Q2-2025", "year": 2025},
        },
    )

    prev = result["meta"]["extraction_sources"]["previous"]
    curr = result["meta"]["extraction_sources"]["current"]
    assert prev["quarter"] == "t1"
    assert curr["quarter"] == "t2"
    assert prev["mode"] == "unknown"
    assert curr["artifacts_present"]["tables"] is False
