"""Tests for validation pipeline order and observability in comparison_runner."""

from __future__ import annotations

import json
from pathlib import Path

from vigilance.models.table_models import TableArtifact


def _mk_table(
    *,
    bank: str,
    section: str,
    page: int,
    table_id: str,
    indicators: list[str],
    bbox_left: float = 0.1,
) -> TableArtifact:
    return TableArtifact(
        bank_code=bank,
        section=section,
        page_pdf=page,
        table_id=table_id,
        title="Table",
        headers=["col1", "col2"],
        rows=[[ind, "1"] for ind in indicators],
        first_column_indicators=indicators,
        extraction_method="vision_primary",
        bbox=[bbox_left, 0.1, min(0.9, bbox_left + 0.3), 0.6],
        pdf_path="/tmp/fake.pdf",
    )


def test_vision_pair_rejection_happens_before_indicator_diff(monkeypatch) -> None:
    """When Vision pair rejects, indicator diff/rename pipeline is skipped for that pair."""
    from app import comparison_runner as cr

    t1 = _mk_table(
        bank="bnc",
        section="capital_management",
        page=1,
        table_id="t1a",
        indicators=["alpha", "beta"],
        bbox_left=0.1,
    )
    t2 = _mk_table(
        bank="bnc",
        section="capital_management",
        page=2,
        table_id="t2a",
        indicators=["gamma", "delta"],
        bbox_left=0.5,
    )

    def _fake_extract_tables(**kwargs):
        return [t1] if kwargs.get("quarter") == "t1" else [t2]

    def _fake_strict(**_kwargs):
        return {
            "pairs": [
                {
                    "t1_uid": "capital_management|t1a|p1",
                    "t2_uid": "capital_management|t2a|p2",
                    "score": 0.82,
                    "rescue_type": "title_structure",
                }
            ],
            "added_tables": [],
            "removed_tables": [],
        }

    def _raise_if_diff(*_args, **_kwargs):
        raise AssertionError("_indicator_diff must not run after Vision pair rejection")

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "run_strict_intra_section_compare", _fake_strict)
    monkeypatch.setattr(cr, "_indicator_diff", _raise_if_diff)
    monkeypatch.setattr(cr, "merge_table_fragments", lambda tables, **_kw: (tables, []))
    monkeypatch.setattr(cr, "get_matching_thresholds", lambda **_kw: {})
    monkeypatch.setattr(cr, "get_quality_gate_config", lambda **_kw: {"enabled": False})
    monkeypatch.setattr(
        cr,
        "get_validation_config",
        lambda **_kw: {
            "vision_pair_validation": True,
            "vision_pair_confidence_min": 0.75,
            "rename_validator_enabled": True,
            "rename_validator_confidence_min": 0.8,
            "rename_validator_batch_size": 10,
        },
    )

    from vigilance import config as cfg_mod
    from vigilance.extraction import vision_pair_validator as vp

    monkeypatch.setattr(
        cfg_mod,
        "get_vision_extraction_config",
        lambda **_kw: {"save_indicators_footnotes_json": False},
    )
    monkeypatch.setattr(vp, "validate_pair_same_concept", lambda *a, **k: (False, 0.99))

    result = cr.run_comparison_with_sections(
        pdf_path_t1="/tmp/t1.pdf",
        pdf_path_t2="/tmp/t2.pdf",
        bank_code="bnc",
        sections_t1=[{"section_key": "capital_management", "start_page": 1, "end_page": 1}],
        sections_t2=[{"section_key": "capital_management", "start_page": 1, "end_page": 1}],
        api_key="test-key",
    )

    summary = result["meta"]["validation_summary"]
    assert summary["vision_pair"]["rejected"] == 1
    assert summary["rename_validator"]["pairs_validated"] == 0
    assert result["table_comparisons"] == []


def test_rename_validator_uncertain_band_is_applied(monkeypatch) -> None:
    """Only in-band rename candidates are sent to GenAI validator; out-of-band are auto-accepted."""
    from app import comparison_runner as cr

    t1 = _mk_table(
        bank="bmo",
        section="risk_management",
        page=3,
        table_id="t1b",
        indicators=["old_a", "old_b"],
        bbox_left=0.1,
    )
    t2 = _mk_table(
        bank="bmo",
        section="risk_management",
        page=4,
        table_id="t2b",
        indicators=["new_a", "new_b"],
        bbox_left=0.5,
    )

    def _fake_extract_tables(**kwargs):
        return [t1] if kwargs.get("quarter") == "t1" else [t2]

    def _fake_strict(**_kwargs):
        return {
            "pairs": [
                {
                    "t1_uid": "risk_management|t1b|p3",
                    "t2_uid": "risk_management|t2b|p4",
                    "score": 0.87,
                }
            ],
            "added_tables": [],
            "removed_tables": [],
        }

    def _fake_indicator_diff(*_args, **_kwargs):
        return ["new_a", "new_b"], ["old_a", "old_b"], False, {
            "total": 0,
            "unit": 0,
            "date": 0,
        }

    def _fake_hungarian(*_args, **_kwargs):
        return [], [], [("old_a", "new_a"), ("old_b", "new_b")], {
            "rename_pair_debug": [
                {"final_score": 90.0},
                {"final_score": 99.0},
            ]
        }

    called: dict[str, list[tuple[str, str]]] = {"pairs": []}

    def _fake_validate_rename_pairs(pairs, **_kwargs):
        called["pairs"] = list(pairs)
        return [], [("old_a", "new_a")], {
            "calls": 1,
            "pairs_validated": len(pairs),
            "accepted": 0,
            "rejected": len(pairs),
            "errors": 0,
        }

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "run_strict_intra_section_compare", _fake_strict)
    monkeypatch.setattr(cr, "_indicator_diff", _fake_indicator_diff)
    monkeypatch.setattr(cr, "_hungarian_pair_added_removed", _fake_hungarian)
    monkeypatch.setattr(cr, "merge_table_fragments", lambda tables, **_kw: (tables, []))
    monkeypatch.setattr(cr, "get_matching_thresholds", lambda **_kw: {"indicator_hungarian_enabled": True})
    monkeypatch.setattr(cr, "get_quality_gate_config", lambda **_kw: {"enabled": False})
    monkeypatch.setattr(
        cr,
        "get_validation_config",
        lambda **_kw: {
            "vision_pair_validation": False,
            "rename_validator_enabled": True,
            "rename_validator_confidence_min": 0.8,
            "rename_validator_batch_size": 10,
            "rename_validator_uncertain_score_band": [0.85, 0.95],
        },
    )

    from vigilance import config as cfg_mod
    from vigilance import genai as genai_mod

    monkeypatch.setattr(
        cfg_mod,
        "get_vision_extraction_config",
        lambda **_kw: {"save_indicators_footnotes_json": False},
    )
    monkeypatch.setattr(genai_mod, "validate_rename_pairs", _fake_validate_rename_pairs)

    result = cr.run_comparison_with_sections(
        pdf_path_t1="/tmp/t1.pdf",
        pdf_path_t2="/tmp/t2.pdf",
        bank_code="bmo",
        sections_t1=[{"section_key": "risk_management", "start_page": 1, "end_page": 1}],
        sections_t2=[{"section_key": "risk_management", "start_page": 1, "end_page": 1}],
        api_key="test-key",
    )

    assert called["pairs"] == [("old_a", "new_a")]

    comp = result["table_comparisons"][0]
    assert comp["renamed_indicators"] == [{"from": "old_b", "to": "new_b"}]

    summary = result["meta"]["validation_summary"]["rename_validator"]
    assert summary["pairs_validated"] == 1
    assert summary["candidates_in_band"] == 1
    assert summary["auto_accepted_out_of_band"] == 1


def test_validation_log_injects_run_id(tmp_path: Path, monkeypatch) -> None:
    """Validation JSONL records must include run_id correlation when provided."""
    from app import comparison_runner as cr

    log_path = tmp_path / "validation.jsonl"
    monkeypatch.setattr(cr, "_VALIDATION_LOG", log_path)

    cr._write_validation_log({"validator": "x", "value": 1}, run_id="run-123")
    cr._write_validation_log(
        {"validator": "x", "value": 2, "run_id": "already-present"},
        run_id="run-123",
    )

    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["run_id"] == "run-123"
    assert rows[1]["run_id"] == "already-present"
