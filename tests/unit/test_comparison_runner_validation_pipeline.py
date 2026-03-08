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
        first_column_indicators_raw=indicators,
        extraction_method="vision_full_gpt4o",
        bbox=[bbox_left, 0.1, min(0.9, bbox_left + 0.3), 0.6],
        pdf_path="/tmp/fake.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
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


def test_vision_unmatched_rescue_consumes_ambiguous_candidates(monkeypatch) -> None:
    from app import comparison_runner as cr

    t1 = _mk_table(
        bank="bmo",
        section="risk_management",
        page=10,
        table_id="t1c",
        indicators=["alpha", "beta", "gamma"],
        bbox_left=0.1,
    )
    t2 = _mk_table(
        bank="bmo",
        section="risk_management",
        page=11,
        table_id="t2c",
        indicators=["alpha", "beta", "gamma", "delta"],
        bbox_left=0.5,
    )
    t1_uid = "risk_management|t1c|p10"
    t2_uid = "risk_management|t2c|p11"

    def _fake_extract_tables(**kwargs):
        return [t1] if kwargs.get("quarter") == "t1" else [t2]

    def _fake_strict(**_kwargs):
        unmatched_prev = {
            "t1_uid": t1_uid,
            "t1_table_id": "t1c",
            "section": "risk_management",
            "page_t1": 10,
            "title_t1": "Table A",
            "reason": "ambiguous_candidate",
            "unmatched_status": "ambiguous",
            "suspicion_flags": ["prefix_bias"],
        }
        unmatched_curr = {
            "t2_uid": t2_uid,
            "t2_table_id": "t2c",
            "section": "risk_management",
            "page_t2": 11,
            "title_t2": "Table B",
            "reason": "ambiguous_candidate",
            "unmatched_status": "ambiguous",
            "suspicion_flags": ["prefix_bias"],
        }
        row_candidate = {
            "t2_uid": t2_uid,
            "score": 0.71,
            "decision_level": "probable",
            "reason": "prefix_bias",
            "indicator_overlap": 0.55,
            "indicator_containment": 0.55,
            "coverage_min": 0.48,
            "coverage_gap": 0.12,
            "top_overlap": 0.9,
            "tail_overlap": 0.35,
            "distinctive_overlap_score": 0.42,
            "row_count_ratio": 1.2,
            "suspicion_flags": ["prefix_bias"],
        }
        col_candidate = {
            "t1_uid": t1_uid,
            "score": 0.71,
            "decision_level": "probable",
            "reason": "prefix_bias",
            "indicator_overlap": 0.55,
            "indicator_containment": 0.55,
            "coverage_min": 0.48,
            "coverage_gap": 0.12,
            "top_overlap": 0.9,
            "tail_overlap": 0.35,
            "distinctive_overlap_score": 0.42,
            "row_count_ratio": 1.2,
            "suspicion_flags": ["prefix_bias"],
        }
        return {
            "pairs": [],
            "probable_pairs": [],
            "added_tables": [],
            "removed_tables": [],
            "suspicious_pairs": [
                {
                    "t1_uid": t1_uid,
                    "t2_uid": t2_uid,
                    "score": 0.71,
                    "coverage_min": 0.48,
                    "coverage_gap": 0.12,
                    "top_overlap": 0.9,
                    "tail_overlap": 0.35,
                    "distinctive_overlap_score": 0.42,
                    "row_count_ratio": 1.2,
                    "suspicion_flags": ["prefix_bias"],
                    "row_candidates": [row_candidate],
                    "column_candidates": [col_candidate],
                }
            ],
            "debug_unmatched_candidates": [{"t1_uid": t1_uid, "candidates": [row_candidate]}],
            "debug_unmatched_candidates_t2": [{"t2_uid": t2_uid, "candidates": [col_candidate]}],
            "unmatched_t1": [unmatched_prev],
            "unmatched_t2": [unmatched_curr],
            "unmatched_confirmed_t1": [],
            "unmatched_confirmed_t2": [],
            "unmatched_ambiguous_t1": [unmatched_prev],
            "unmatched_ambiguous_t2": [unmatched_curr],
            "ambiguous_unmatched_previous": [unmatched_prev],
            "ambiguous_unmatched_current": [unmatched_curr],
            "matching_diagnostics": {},
            "rescued_matches_count": 0,
            "split_merge_rescues_count": 0,
        }

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "run_strict_intra_section_compare", _fake_strict)
    monkeypatch.setattr(cr, "merge_table_fragments", lambda tables, **_kw: (tables, []))
    monkeypatch.setattr(cr, "get_matching_thresholds", lambda **_kw: {})
    monkeypatch.setattr(cr, "get_quality_gate_config", lambda **_kw: {"enabled": False})
    monkeypatch.setattr(
        cr,
        "get_validation_config",
        lambda **_kw: {
            "vision_pair_validation": False,
            "vision_unmatched_rescue_enabled": True,
            "vision_unmatched_rescue_confidence_min": 0.75,
            "vision_unmatched_rescue_max_pairs": 10,
            "vision_unmatched_rescue_max_candidates_per_table": 2,
            "vision_unmatched_rescue_max_tables_per_run": 10,
            "rename_validator_enabled": False,
            "added_table_validator_enabled": False,
            "indicator_validator_enabled": False,
        },
    )

    from vigilance import config as cfg_mod
    from vigilance.extraction import vision_pair_validator as vp

    monkeypatch.setattr(
        cfg_mod,
        "get_vision_extraction_config",
        lambda **_kw: {"save_indicators_footnotes_json": False},
    )
    monkeypatch.setattr(
        vp,
        "validate_pair_full",
        lambda *a, **k: vp.VisionDecision(
            decision=vp.DECISION_MATCH,
            confidence=0.91,
            reason_code="vision_ok",
        ),
    )

    result = cr.run_comparison_with_sections(
        pdf_path_t1="/tmp/t1.pdf",
        pdf_path_t2="/tmp/t2.pdf",
        bank_code="bmo",
        sections_t1=[{"section_key": "risk_management", "start_page": 1, "end_page": 1}],
        sections_t2=[{"section_key": "risk_management", "start_page": 1, "end_page": 1}],
        api_key="test-key",
    )

    assert result["summary"]["tables_matched"] == 1
    assert result["summary"]["vision_rescued_pairs"] == 1
    assert result["summary"]["tables_added"] == 0
    assert result["summary"]["tables_removed"] == 0
    assert result["summary"]["ambiguous_tables"] == 0
    assert len(result["vision_rescued_pairs"]) == 1
    rescue_summary = result["meta"]["validation_summary"]["vision_unmatched_rescue"]
    assert rescue_summary["candidate_pairs_tested"] == 1
    assert rescue_summary["rescued_pairs"] == 1


def test_vision_unmatched_rescue_unknown_keeps_tables_ambiguous(monkeypatch) -> None:
    from app import comparison_runner as cr

    t1 = _mk_table(
        bank="bns",
        section="capital_management",
        page=20,
        table_id="t1d",
        indicators=["cet1", "at1"],
        bbox_left=0.1,
    )
    t2 = _mk_table(
        bank="bns",
        section="capital_management",
        page=21,
        table_id="t2d",
        indicators=["cet1", "at1", "tier2"],
        bbox_left=0.5,
    )
    t1_uid = "capital_management|t1d|p20"
    t2_uid = "capital_management|t2d|p21"

    def _fake_extract_tables(**kwargs):
        return [t1] if kwargs.get("quarter") == "t1" else [t2]

    def _fake_strict(**_kwargs):
        unmatched_prev = {
            "t1_uid": t1_uid,
            "t1_table_id": "t1d",
            "section": "capital_management",
            "page_t1": 20,
            "title_t1": "Capitaux",
            "reason": "weak_signals",
            "unmatched_status": "confirmed",
            "suspicion_flags": [],
        }
        unmatched_curr = {
            "t2_uid": t2_uid,
            "t2_table_id": "t2d",
            "section": "capital_management",
            "page_t2": 21,
            "title_t2": "Capitaux",
            "reason": "unmatched",
            "unmatched_status": "confirmed",
            "suspicion_flags": [],
        }
        row_candidate = {
            "t2_uid": t2_uid,
            "score": 0.67,
            "decision_level": "probable",
            "reason": "weak_signals",
            "indicator_overlap": 0.5,
            "indicator_containment": 0.5,
            "coverage_min": 0.4,
            "coverage_gap": 0.2,
            "top_overlap": 0.8,
            "tail_overlap": 0.3,
            "distinctive_overlap_score": 0.35,
            "row_count_ratio": 1.4,
            "suspicion_flags": ["weak_tail"],
        }
        col_candidate = dict(row_candidate)
        col_candidate["t1_uid"] = t1_uid
        return {
            "pairs": [],
            "probable_pairs": [],
            "added_tables": [
                {
                    "t2_uid": t2_uid,
                    "t2_table_id": "t2d",
                    "section": "capital_management",
                    "page_t2": 21,
                    "title_t2": "Capitaux",
                    "reason": "added_table",
                    "source_reason": "unmatched",
                    "first_column_indicators": ["cet1", "at1", "tier2"],
                    "first_column_indicators_raw": ["cet1", "at1", "tier2"],
                }
            ],
            "removed_tables": [
                {
                    "t1_uid": t1_uid,
                    "t1_table_id": "t1d",
                    "section": "capital_management",
                    "page_t1": 20,
                    "title_t1": "Capitaux",
                    "reason": "removed_table",
                    "source_reason": "weak_signals",
                    "first_column_indicators": ["cet1", "at1"],
                    "first_column_indicators_raw": ["cet1", "at1"],
                }
            ],
            "suspicious_pairs": [],
            "debug_unmatched_candidates": [{"t1_uid": t1_uid, "candidates": [row_candidate]}],
            "debug_unmatched_candidates_t2": [{"t2_uid": t2_uid, "candidates": [col_candidate]}],
            "unmatched_t1": [unmatched_prev],
            "unmatched_t2": [unmatched_curr],
            "unmatched_confirmed_t1": [unmatched_prev],
            "unmatched_confirmed_t2": [unmatched_curr],
            "unmatched_ambiguous_t1": [],
            "unmatched_ambiguous_t2": [],
            "ambiguous_unmatched_previous": [],
            "ambiguous_unmatched_current": [],
            "matching_diagnostics": {},
            "rescued_matches_count": 0,
            "split_merge_rescues_count": 0,
        }

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "run_strict_intra_section_compare", _fake_strict)
    monkeypatch.setattr(cr, "merge_table_fragments", lambda tables, **_kw: (tables, []))
    monkeypatch.setattr(cr, "get_matching_thresholds", lambda **_kw: {})
    monkeypatch.setattr(cr, "get_quality_gate_config", lambda **_kw: {"enabled": False})
    monkeypatch.setattr(
        cr,
        "get_validation_config",
        lambda **_kw: {
            "vision_pair_validation": False,
            "vision_unmatched_rescue_enabled": True,
            "vision_unmatched_rescue_confidence_min": 0.75,
            "vision_unmatched_rescue_max_pairs": 10,
            "vision_unmatched_rescue_max_candidates_per_table": 2,
            "vision_unmatched_rescue_max_tables_per_run": 10,
            "rename_validator_enabled": False,
            "added_table_validator_enabled": False,
            "indicator_validator_enabled": False,
        },
    )

    from vigilance import config as cfg_mod
    from vigilance.extraction import vision_pair_validator as vp

    monkeypatch.setattr(
        cfg_mod,
        "get_vision_extraction_config",
        lambda **_kw: {"save_indicators_footnotes_json": False},
    )
    monkeypatch.setattr(
        vp,
        "validate_pair_full",
        lambda *a, **k: vp.VisionDecision(
            decision=vp.DECISION_UNKNOWN,
            confidence=0.0,
            reason_code="api_error",
        ),
    )

    result = cr.run_comparison_with_sections(
        pdf_path_t1="/tmp/t1.pdf",
        pdf_path_t2="/tmp/t2.pdf",
        bank_code="bns",
        sections_t1=[{"section_key": "capital_management", "start_page": 1, "end_page": 1}],
        sections_t2=[{"section_key": "capital_management", "start_page": 1, "end_page": 1}],
        api_key="test-key",
    )

    assert result["summary"]["tables_matched"] == 0
    assert result["summary"]["tables_added"] == 0
    assert result["summary"]["tables_removed"] == 0
    assert result["summary"]["ambiguous_tables"] == 2
    assert len(result["ambiguous_tables"]) == 2
    rescue_summary = result["meta"]["validation_summary"]["vision_unmatched_rescue"]
    assert rescue_summary["candidate_pairs_tested"] == 1
    assert rescue_summary["vision_unresolved_pairs"] == 1


def test_vision_unmatched_rescue_builds_local_same_section_candidates(monkeypatch) -> None:
    from app import comparison_runner as cr

    t1 = _mk_table(
        bank="td",
        section="risk_management",
        page=14,
        table_id="t1e",
        indicators=["liquidite", "nsfr", "lcr"],
        bbox_left=0.1,
    )
    t2 = _mk_table(
        bank="td",
        section="risk_management",
        page=16,
        table_id="t2e",
        indicators=["liquidite", "nsfr", "lcr", "hqla"],
        bbox_left=0.5,
    )
    other_t2 = _mk_table(
        bank="td",
        section="capital_management",
        page=30,
        table_id="t2f",
        indicators=["cet1", "tier1"],
        bbox_left=0.5,
    )
    t1_uid = "risk_management|t1e|p14"
    t2_uid = "risk_management|t2e|p16"

    def _fake_extract_tables(**kwargs):
        if kwargs.get("quarter") == "t1":
            return [t1]
        return [t2, other_t2]

    def _fake_strict(**_kwargs):
        unmatched_prev = {
            "t1_uid": t1_uid,
            "t1_table_id": "t1e",
            "section": "risk_management",
            "page_t1": 14,
            "title_t1": "Liquidité",
            "reason": "weak_signals",
            "unmatched_status": "confirmed",
            "suspicion_flags": [],
        }
        unmatched_curr = {
            "t2_uid": t2_uid,
            "t2_table_id": "t2e",
            "section": "risk_management",
            "page_t2": 16,
            "title_t2": "Liquidité",
            "reason": "unmatched",
            "unmatched_status": "confirmed",
            "suspicion_flags": [],
        }
        return {
            "pairs": [],
            "probable_pairs": [],
            "added_tables": [
                {
                    "t2_uid": t2_uid,
                    "t2_table_id": "t2e",
                    "section": "risk_management",
                    "page_t2": 16,
                    "title_t2": "Liquidité",
                    "reason": "added_table",
                    "source_reason": "unmatched",
                    "first_column_indicators": ["liquidite", "nsfr", "lcr", "hqla"],
                    "first_column_indicators_raw": ["liquidite", "nsfr", "lcr", "hqla"],
                }
            ],
            "removed_tables": [
                {
                    "t1_uid": t1_uid,
                    "t1_table_id": "t1e",
                    "section": "risk_management",
                    "page_t1": 14,
                    "title_t1": "Liquidité",
                    "reason": "removed_table",
                    "source_reason": "weak_signals",
                    "first_column_indicators": ["liquidite", "nsfr", "lcr"],
                    "first_column_indicators_raw": ["liquidite", "nsfr", "lcr"],
                }
            ],
            "suspicious_pairs": [],
            "debug_unmatched_candidates": [],
            "debug_unmatched_candidates_t2": [],
            "unmatched_t1": [unmatched_prev],
            "unmatched_t2": [unmatched_curr],
            "unmatched_confirmed_t1": [unmatched_prev],
            "unmatched_confirmed_t2": [unmatched_curr],
            "unmatched_ambiguous_t1": [],
            "unmatched_ambiguous_t2": [],
            "ambiguous_unmatched_previous": [],
            "ambiguous_unmatched_current": [],
            "matching_diagnostics": {},
            "rescued_matches_count": 0,
            "split_merge_rescues_count": 0,
        }

    monkeypatch.setattr(cr, "_extract_tables", _fake_extract_tables)
    monkeypatch.setattr(cr, "run_strict_intra_section_compare", _fake_strict)
    monkeypatch.setattr(cr, "merge_table_fragments", lambda tables, **_kw: (tables, []))
    monkeypatch.setattr(cr, "get_matching_thresholds", lambda **_kw: {})
    monkeypatch.setattr(cr, "get_quality_gate_config", lambda **_kw: {"enabled": False})
    monkeypatch.setattr(
        cr,
        "get_validation_config",
        lambda **_kw: {
            "vision_pair_validation": False,
            "vision_unmatched_rescue_enabled": True,
            "vision_unmatched_rescue_confidence_min": 0.75,
            "vision_unmatched_rescue_max_pairs": 10,
            "vision_unmatched_rescue_max_candidates_per_table": 3,
            "vision_unmatched_rescue_max_tables_per_run": 10,
            "rename_validator_enabled": False,
            "added_table_validator_enabled": False,
            "indicator_validator_enabled": False,
        },
    )

    from vigilance import config as cfg_mod
    from vigilance.extraction import vision_pair_validator as vp

    monkeypatch.setattr(
        cfg_mod,
        "get_vision_extraction_config",
        lambda **_kw: {"save_indicators_footnotes_json": False},
    )

    seen_pairs: list[tuple[int, int]] = []

    def _fake_validate(*args, **kwargs):
        seen_pairs.append((args[1], args[4]))
        return vp.VisionDecision(
            decision=vp.DECISION_MATCH,
            confidence=0.88,
            reason_code="vision_ok",
        )

    monkeypatch.setattr(vp, "validate_pair_full", _fake_validate)

    result = cr.run_comparison_with_sections(
        pdf_path_t1="/tmp/t1.pdf",
        pdf_path_t2="/tmp/t2.pdf",
        bank_code="td",
        sections_t1=[{"section_key": "risk_management", "start_page": 1, "end_page": 1}],
        sections_t2=[{"section_key": "risk_management", "start_page": 1, "end_page": 1}],
        api_key="test-key",
    )

    assert seen_pairs == [(14, 16)]
    assert result["summary"]["vision_rescued_pairs"] == 1
    rescue_summary = result["meta"]["validation_summary"]["vision_unmatched_rescue"]
    assert rescue_summary["candidate_tables_considered"] >= 1
    assert rescue_summary["candidate_pairs_considered"] >= 1
