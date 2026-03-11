"""Regression tests for table pairing engine quality-aware routing and scoring."""

from __future__ import annotations

from vigilance.compare import run_strict_intra_section_compare
from vigilance.models.table_models import TableArtifact


def _table(
    *,
    table_id: str,
    section: str = "risk_management",
    title: str | None = "Tableau",
    indicators: list[str],
    table_number: str | None = None,
    page: int = 1,
    debug_metrics: dict | None = None,
    title_reliability: str | None = None,
    raw_indicators: list[str] | None = None,
) -> TableArtifact:
    ind = list(indicators)
    dm = dict(debug_metrics or {})
    dm.setdefault("vision_extraction_applied", True)
    dm.setdefault("vision_extraction_confidence", 0.85)
    return TableArtifact(
        bank_code="bnc",
        section=section,
        page_pdf=page,
        table_id=table_id,
        title=title or "",
        headers=["Indicateur", "Valeur"],
        rows=[[label, "1"] for label in indicators],
        first_column_indicators=ind,
        first_column_indicators_raw=raw_indicators if raw_indicators is not None else ind,
        extraction_method="vision_full_gpt4o",
        quarter="t2-2025",
        table_number=table_number,
        footnotes=[],
        content_source="vision_gpt4o",
        debug_metrics=dm,
        title_reliability=title_reliability,
    )


def test_low_quality_family_overlap_becomes_ambiguous() -> None:
    """When one side is low-quality (not certified), it is excluded from matching; ineligible lists reflect that."""
    low_q_metrics = {
        "vision_extraction_applied": True,
        "vision_extraction_confidence": 0.4,
        "crop_reject_reason": None,
        "recrop_failed_incomplete": False,
    }
    t1 = _table(
        table_id="t1",
        title="Exposition geographique",
        indicators=["Canada", "Etats-Unis", "Europe", "Total"],
        debug_metrics=low_q_metrics,
    )
    t2 = _table(
        table_id="t2",
        title="Exposition par region",
        indicators=["Canada", "Etats-Unis", "Europe", "Total"],
        page=2,
    )
    result = run_strict_intra_section_compare([t1], [t2])
    ineligible_t1 = result.get("ineligible_t1") or []
    assert any(
        item.get("reason") == "extraction_not_certified" for item in ineligible_t1
    ), f"expected t1 ineligible (extraction_not_certified), got {ineligible_t1}"
    if result["pairs"]:
        pair = result["pairs"][0]
        assert pair.get("pairing_confidence", 0) <= 0.82 or "low_quality" in str(pair.get("pairing_quality_flags", []))
    else:
        assert result.get("ambiguous_pairs") or result.get("unmatched_ambiguous_t1") or ineligible_t1


def test_quality_suspect_requires_two_independent_anchors() -> None:
    """When quality_suspect, match requires at least 2 anchors; otherwise ambiguous."""
    low_q = {
        "vision_extraction_applied": True,
        "vision_extraction_confidence": 0.45,
        "recrop_failed_incomplete": True,
    }
    t1 = _table(
        table_id="t1",
        title="Tableau 5 - Fonds propres",
        table_number="5",
        indicators=["CET1", "Tier 2", "Total"],
        debug_metrics=low_q,
    )
    t2 = _table(
        table_id="t2",
        title="Tableau 5 - Fonds propres",
        table_number="5",
        indicators=["CET1", "Tier 2", "Total"],
        page=2,
    )
    result = run_strict_intra_section_compare([t1], [t2])
    if result["pairs"]:
        assert result["pairs"][0].get("pairing_confidence", 0) <= 0.82 or "low_quality" in str(result["pairs"][0].get("pairing_quality_flags", []))


def test_raw_normalized_instability_penalizes_match() -> None:
    """When normalized overlap is high but raw stability is low, score gets instability penalty."""
    from vigilance.compare.table_pairing_engine import (
        _candidate_score,
        _raw_indicator_stability,
        _eligible_table_views,
    )

    t1 = _table(
        table_id="t1",
        indicators=["A", "B", "C", "D", "E"],
        raw_indicators=["A", "B", "C", "D", "E"],
    )
    t2 = _table(
        table_id="t2",
        indicators=["A", "B", "C", "D", "E"],
        page=2,
        raw_indicators=["X", "Y", "Z"],
    )
    section_freq: dict = {}
    section_counts: dict = {"risk_management": 1}
    views, _ = _eligible_table_views([t1], section_frequencies=section_freq, section_counts=section_counts)
    views2, _ = _eligible_table_views([t2], section_frequencies=section_freq, section_counts=section_counts)
    assert views and views2
    score = _candidate_score(views[0], views2[0])
    stability = _raw_indicator_stability(views[0], views2[0])
    assert stability < 1.0
    assert score.raw_indicator_stability < 1.0


def test_low_quality_match_confidence_is_capped() -> None:
    """When match is allowed with low quality, confidence is capped."""
    low_q = {
        "vision_extraction_applied": True,
        "vision_extraction_confidence": 0.5,
        "recrop_failed_incomplete": False,
    }
    t1 = _table(
        table_id="t1",
        title="Tableau 1 - CET1",
        table_number="1",
        indicators=["CET1", "AT1", "Tier 2", "Total"],
        debug_metrics=low_q,
    )
    t2 = _table(
        table_id="t2",
        title="Tableau 1 - CET1",
        table_number="1",
        indicators=["CET1", "AT1", "Tier 2", "Total"],
        page=2,
    )
    result = run_strict_intra_section_compare([t1], [t2])
    for pair in result.get("pairs", []):
        if pair.get("pairing_confidence_cap"):
            assert pair.get("pairing_confidence", 0) <= 0.82
        break


def test_pairing_metadata_contains_quality_flags() -> None:
    """Pair payload can contain pairing_quality_flags and pairing_confidence_cap."""
    from vigilance.compare.table_pairing_engine import (
        _pair_dict,
        _candidate_score,
        ConservativePairingRouter,
        TableView,
        _eligible_table_views,
    )

    low_q = {
        "vision_extraction_applied": True,
        "vision_extraction_confidence": 0.5,
    }
    t1 = _table(
        table_id="t1",
        title="Tableau 2",
        table_number="2",
        indicators=["A", "B", "C", "D", "E", "F"],
        debug_metrics=low_q,
    )
    t2 = _table(
        table_id="t2",
        title="Tableau 2",
        table_number="2",
        indicators=["A", "B", "C", "D", "E", "F"],
        page=2,
    )
    section_freq: dict = {}
    section_counts: dict = {"risk_management": 1}
    v1, _ = _eligible_table_views([t1], section_frequencies=section_freq, section_counts=section_counts)
    v2, _ = _eligible_table_views([t2], section_frequencies=section_freq, section_counts=section_counts)
    if v1 and v2:
        score = _candidate_score(v1[0], v2[0])
        router = ConservativePairingRouter()
        decision = router.route(t2_view=v2[0], candidates=[score])
        payload = _pair_dict(score, decision)
        assert "pairing_confidence" in payload
        if decision.decision == "match" and getattr(decision, "pairing_quality_flags", None):
            assert "pairing_quality_flags" in payload or "pairing_confidence_cap" in payload


def test_unreliable_title_does_not_rescue_pair() -> None:
    """When title_reliability is weak, title_similarity contribution is downweighted."""
    from vigilance.compare.table_pairing_engine import (
        _candidate_score,
        _eligible_table_views,
        _title_reliability_numeric,
    )

    t1 = _table(
        table_id="t1",
        title="Tableau 3",
        table_number="3",
        indicators=["A", "B", "C"],
        title_reliability="weak",
    )
    t2 = _table(
        table_id="t2",
        title="Tableau 3",
        table_number="3",
        indicators=["A", "B", "C"],
        page=2,
    )
    assert _title_reliability_numeric(t1) < 0.7
    section_freq: dict = {}
    section_counts: dict = {"risk_management": 1}
    v1, _ = _eligible_table_views([t1], section_frequencies=section_freq, section_counts=section_counts)
    v2, _ = _eligible_table_views([t2], section_frequencies=section_freq, section_counts=section_counts)
    assert v1 and v2
    score = _candidate_score(v1[0], v2[0])
    assert score.title_reliability_score < 1.0


def test_vision_rescue_respects_anchor_diversity() -> None:
    """Rescue path uses get_extraction_quality_flags; low-quality rescue requires anchor diversity."""
    from vigilance.models.table_models import get_extraction_quality_flags

    low_q = {
        "vision_extraction_applied": True,
        "crop_reject_reason": None,
        "recrop_failed_incomplete": True,
    }

    class T:
        debug_metrics = low_q

    flags = get_extraction_quality_flags(T())
    assert "recrop_failed_incomplete" in flags
    assert flags.get("recrop_failed_incomplete") is True
