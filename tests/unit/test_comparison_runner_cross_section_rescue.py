"""Tests for cross-section vision rescue in comparison_runner."""

from __future__ import annotations

from vigilance.models.table_models import TableArtifact


def _mk_table(
    *,
    bank: str,
    section: str,
    page: int,
    table_id: str,
    indicators: list[str],
    title: str = "Table",
    bbox_left: float = 0.1,
) -> TableArtifact:
    return TableArtifact(
        bank_code=bank,
        section=section,
        page_pdf=page,
        table_id=table_id,
        title=title,
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


def test_collect_vision_rescue_candidates_includes_cross_section_when_enabled() -> None:
    """With cross_section_rescue_enabled=True, confirmed_unmatched from different sections produce candidates with is_cross_section=True."""
    from app.comparison_runner import _collect_vision_rescue_candidates

    t1_uid = "risk_management|t1_x|p10"
    t2_uid = "regulatory_updates|t2_x|p39"
    t1 = _mk_table(
        bank="bmo",
        section="risk_management",
        page=10,
        table_id="t1_x",
        indicators=["alpha", "beta", "gamma"],
        title="Prêts garantis",
    )
    t2 = _mk_table(
        bank="bmo",
        section="regulatory_updates",
        page=39,
        table_id="t2_x",
        indicators=["alpha", "beta", "gamma"],
        title="Prêts garantis",
    )
    strict = {
        "pairs": [],
        "removed_tables": [
            {
                "t1_uid": t1_uid,
                "t1_table_id": "t1_x",
                "section": "risk_management",
                "page_t1": 10,
                "title_t1": "Prêts garantis",
                "reason": "no_candidate_same_section",
                "unmatched_status": "confirmed",
            }
        ],
        "added_tables": [
            {
                "t2_uid": t2_uid,
                "t2_table_id": "t2_x",
                "section": "regulatory_updates",
                "page_t2": 39,
                "title_t2": "Prêts garantis",
                "reason": "unmatched",
                "unmatched_status": "confirmed",
            }
        ],
        "ambiguous_unmatched_previous": [],
        "ambiguous_unmatched_current": [],
        "suspicious_pairs": [],
        "debug_unmatched_candidates": [],
        "debug_unmatched_candidates_t2": [],
    }
    t1_by_uid = {t1_uid: t1}
    t2_by_uid = {t2_uid: t2}

    candidates, _ = _collect_vision_rescue_candidates(
        strict=strict,
        t1_by_uid=t1_by_uid,
        t2_by_uid=t2_by_uid,
        max_candidates_per_table=3,
        max_tables_per_run=10,
        cross_section_rescue_enabled=True,
        cross_section_rerank_min=0.30,
    )

    assert len(candidates) >= 1
    cross_candidates = [c for c in candidates if c.get("is_cross_section")]
    assert len(cross_candidates) >= 1
    assert cross_candidates[0].get("t1_uid") == t1_uid
    assert cross_candidates[0].get("t2_uid") == t2_uid


def test_collect_vision_rescue_candidates_excludes_cross_section_when_disabled() -> None:
    """With cross_section_rescue_enabled=False, tables from different sections do not produce candidates."""
    from app.comparison_runner import _collect_vision_rescue_candidates

    t1_uid = "risk_management|t1_y|p10"
    t2_uid = "regulatory_updates|t2_y|p39"
    t1 = _mk_table(
        bank="bmo",
        section="risk_management",
        page=10,
        table_id="t1_y",
        indicators=["alpha", "beta"],
        title="Table A",
    )
    t2 = _mk_table(
        bank="bmo",
        section="regulatory_updates",
        page=39,
        table_id="t2_y",
        indicators=["alpha", "beta"],
        title="Table A",
    )
    strict = {
        "pairs": [],
        "removed_tables": [
            {
                "t1_uid": t1_uid,
                "t1_table_id": "t1_y",
                "section": "risk_management",
                "page_t1": 10,
                "title_t1": "Table A",
                "reason": "removed_table",
                "unmatched_status": "confirmed",
            }
        ],
        "added_tables": [
            {
                "t2_uid": t2_uid,
                "t2_table_id": "t2_y",
                "section": "regulatory_updates",
                "page_t2": 39,
                "title_t2": "Table A",
                "reason": "added_table",
                "unmatched_status": "confirmed",
            }
        ],
        "ambiguous_unmatched_previous": [],
        "ambiguous_unmatched_current": [],
        "suspicious_pairs": [],
        "debug_unmatched_candidates": [],
        "debug_unmatched_candidates_t2": [],
    }
    t1_by_uid = {t1_uid: t1}
    t2_by_uid = {t2_uid: t2}

    candidates, _ = _collect_vision_rescue_candidates(
        strict=strict,
        t1_by_uid=t1_by_uid,
        t2_by_uid=t2_by_uid,
        max_candidates_per_table=3,
        max_tables_per_run=10,
        cross_section_rescue_enabled=False,
        cross_section_rerank_min=0.30,
    )

    cross_candidates = [c for c in candidates if c.get("is_cross_section")]
    assert len(cross_candidates) == 0


def test_get_validation_config_cross_section_defaults() -> None:
    """get_validation_config applies cross_section_rescue defaults when keys are missing."""
    from vigilance.config import get_validation_config

    cfg = get_validation_config(bank_code="bmo")
    assert "cross_section_rescue_enabled" in cfg
    assert "cross_section_rescue_rerank_min" in cfg
    assert "cross_section_rescue_vision_confidence_min" in cfg
    assert "cross_section_rescue_max_candidates_per_table" in cfg
    assert cfg["cross_section_rescue_rerank_min"] == 0.30
    assert cfg["cross_section_rescue_vision_confidence_min"] == 0.85
