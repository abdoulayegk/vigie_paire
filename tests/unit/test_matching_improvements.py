"""Non-regression tests for matching improvements (plan: matching_tableaux)."""

from __future__ import annotations

from vigilance.compare.indicator_comparator import (
    MatchDecision,
    _extract_table_label,
    _indicator_set,
    _load_compare_thresholds,
    match_decision,
    match_tables_intra_section,
    run_strict_intra_section_compare,
)
from vigilance.models.table_models import TableArtifact


def _art(
    *,
    table_id: str = "t1",
    section: str = "capital_management",
    title: str = "Tableau 28 – Ratios",
    rows: list[list[str]] | None = None,
    first_column_indicators: list[str] | None = None,
    table_number: str | None = None,
    page: int = 1,
) -> TableArtifact:
    return TableArtifact(
        bank_code="rbc",
        section=section,
        page_pdf=page,
        table_id=table_id,
        title=title,
        headers=["Indicateur", "T2 2025"],
        rows=rows or [],
        first_column_indicators=first_column_indicators or [],
        extraction_method="docling",
        table_number=table_number,
        quarter="t1-2025",
        pdf_path="test.pdf",
    )


# ---------- 1.1 _indicator_set fallback to first_column_indicators ----------

class TestIndicatorSetFallback:
    def test_uses_rows_when_available(self) -> None:
        t = _art(rows=[["CET1", "13"], ["Tier1", "15"]], first_column_indicators=["Other"])
        result = _indicator_set(t)
        assert result == {"cet1", "tier1"}

    def test_falls_back_to_first_column_indicators_when_rows_empty(self) -> None:
        t = _art(rows=[], first_column_indicators=["CET1", "Tier 1", "Total Capital"])
        result = _indicator_set(t)
        assert result == {"cet1", "tier 1", "total capital"}

    def test_empty_when_both_empty(self) -> None:
        t = _art(rows=[], first_column_indicators=[])
        result = _indicator_set(t)
        assert result == set()

    def test_jaccard_works_with_first_column_indicators(self) -> None:
        a = _art(rows=[], first_column_indicators=["CET1", "Tier 1", "Total"])
        b = _art(rows=[], first_column_indicators=["CET1", "Tier 1", "RWA"])
        from vigilance.compare.indicator_comparator import _jaccard
        score = _jaccard(a, b)
        assert 0.4 < score < 0.6  # 2/4 = 0.5


# ---------- 1.2 table_number on TableArtifact ----------

class TestTableNumberField:
    def test_table_number_field_exists(self) -> None:
        t = _art(table_number="28")
        assert t.table_number == "28"

    def test_table_number_defaults_to_none(self) -> None:
        t = _art()
        assert t.table_number is None


# ---------- 2.4 _extract_table_label uses table_number ----------

class TestExtractTableLabelWithTableNumber:
    def test_prefers_table_number_over_title(self) -> None:
        t = _art(title="Tableau 99 – Something", table_number="42")
        label = _extract_table_label(t)
        assert label is not None
        assert label.base == "42"

    def test_falls_back_to_title_when_no_table_number(self) -> None:
        t = _art(title="Tableau 28 – Ratios", table_number=None)
        label = _extract_table_label(t)
        assert label is not None
        assert label.base == "28"

    def test_table_number_with_suffix(self) -> None:
        t = _art(table_number="14a")
        label = _extract_table_label(t)
        assert label is not None
        assert label.base == "14"
        assert label.suffix == "a"

    def test_table_number_match_between_tables(self) -> None:
        a = _art(table_id="a", table_number="28", title="Something")
        b = _art(table_id="b", table_number="28", title="Something else")
        la = _extract_table_label(a)
        lb = _extract_table_label(b)
        assert la is not None and lb is not None
        assert la.full == lb.full

    def test_extracts_header_footer_standalone_number(self) -> None:
        t = _art(
            title="30 Banque Royale du Canada Deuxième trimestre de 2025",
            table_number=None,
        )
        label = _extract_table_label(t)
        assert label is not None
        assert label.base == "30"


# ---------- 2.1 overlap_floor_min ----------

class TestOverlapFloorMin:
    def test_default_thresholds_loaded(self) -> None:
        th = _load_compare_thresholds()
        assert th["overlap_floor_min"] == 0.35
        assert th["overlap_threshold"] == 0.55

    def test_very_low_overlap_blocked(self) -> None:
        a = _art(
            table_id="a", section="capital_management",
            title="Tableau 28 – Ratios", table_number="28",
            rows=[["CET1", "1"], ["Tier1", "2"]],
        )
        b = _art(
            table_id="b", section="capital_management",
            title="Tableau 28 – Ratios", table_number="28",
            rows=[["RWA", "3"], ["Leverage", "4"]],
        )
        d = match_decision(a, b)
        assert d.indicator_overlap == 0.0
        assert d.is_match is False
        assert d.reason in ("low_containment", "low_label_overlap_reject")


# ---------- 2.2 indicator_overlap_match needs title_similarity ----------

class TestIndicatorOverlapMatchNeedsTitleSimilarity:
    def test_high_overlap_but_no_title_similarity_and_no_label_match(self) -> None:
        shared = [["CET1", "1"], ["Tier1", "2"], ["Total", "3"]]
        a = _art(table_id="a", title="Alpha report", rows=shared)
        b = _art(table_id="b", title="Beta different", rows=shared)
        d = match_decision(a, b, overlap_threshold=0.5)
        assert d.indicator_overlap == 1.0
        assert d.title_similarity < 0.50
        assert d.reason != "indicator_overlap_match" or d.table_label_base_match

    def test_high_overlap_with_good_title_matches(self) -> None:
        shared = [["CET1", "1"], ["Tier1", "2"], ["Total", "3"]]
        a = _art(table_id="a", title="Risque de credit", rows=shared)
        b = _art(table_id="b", title="Risque de crédit", rows=shared)
        d = match_decision(a, b, overlap_threshold=0.5)
        assert d.is_match is True


# ---------- 2.3 anti-greedy margin 0.10 ----------

class TestAntiGreedyMargin:
    def test_margin_value_is_010(self) -> None:
        th = _load_compare_thresholds()
        assert th["margin_threshold"] == 0.10


# ---------- 3.1 borderline logging ----------

class TestBorderlineThreshold:
    def test_borderline_score_threshold_loaded(self) -> None:
        th = _load_compare_thresholds()
        assert th["borderline_score_threshold"] == 0.65


# ---------- 3.2 unmatched contains score ----------

class TestUnmatchedScore:
    def test_unmatched_t1_contains_best_score(self) -> None:
        a = _art(
            table_id="a", section="capital_management",
            title="Tableau 28 – Ratios",
            rows=[["CET1", "1"]],
        )
        b = _art(
            table_id="b", section="capital_management",
            title="Tableau 99 – Something else",
            rows=[["RWA", "2"]],
        )
        result = match_tables_intra_section([a], [b])
        assert len(result["unmatched_t1"]) == 1
        entry = result["unmatched_t1"][0]
        assert "best_score" in entry
        assert "best_indicator_overlap" in entry
        assert isinstance(entry["best_score"], float)


# ---------- 3.3 full pipeline non-regression ----------

class TestFullPipelineRegression:
    def test_exact_table_number_match_still_works(self) -> None:
        a = _art(
            table_id="a", section="risk_management",
            title="Tableau 32 – Exposition crédit",
            table_number="32",
            rows=[["Prêts hypo", "1"], ["Cartes", "2"]],
        )
        b = _art(
            table_id="b", section="risk_management",
            title="Tableau 32 – Exposition au crédit",
            table_number="32",
            rows=[["Prêts hypo", "3"], ["Cartes", "4"]],
        )
        d = match_decision(a, b)
        assert d.is_match is True
        assert d.reason in (
            "table_number_match",
            "indicator_overlap_match",
            "indicator_set_hash_exact",
        )

    def test_cross_section_still_blocked(self) -> None:
        a = _art(table_id="a", section="capital_management", title="T28", rows=[["X", "1"]])
        b = _art(table_id="b", section="risk_management", title="T28", rows=[["X", "2"]])
        d = match_decision(a, b)
        assert d.is_match is False
        assert d.reason == "cross_section_forbidden"

    def test_table_number_conflict_still_blocks(self) -> None:
        a = _art(
            table_id="a", section="risk_management",
            title="Tableau 23 – Prêts",
            rows=[["Atlantique", "1"], ["Québec", "2"]],
        )
        b = _art(
            table_id="b", section="risk_management",
            title="Tableau 24 – Marges",
            rows=[["Atlantique", "3"], ["Québec", "4"]],
        )
        d = match_decision(a, b)
        assert d.is_match is False
        assert d.reason == "table_number_conflict"

    def test_run_strict_returns_expected_keys(self) -> None:
        a = _art(table_id="a", section="risk_management", title="T32", rows=[["X", "1"]])
        b = _art(table_id="b", section="risk_management", title="T33", rows=[["Y", "2"]])
        result = run_strict_intra_section_compare([a], [b])
        assert "pairs" in result
        assert "added_tables" in result
        assert "removed_tables" in result
        assert "unmatched_t1" in result
        assert "unmatched_t2" in result
        assert "reasons" in result

    def test_labels_only_mode_matching(self) -> None:
        """Tables with empty rows but matching first_column_indicators should match."""
        indicators = ["CET1", "Tier 1", "Total Capital", "Leverage Ratio"]
        a = _art(
            table_id="a", section="capital_management",
            title="Tableau 28 – Ratios de capital",
            table_number="28",
            rows=[], first_column_indicators=indicators,
        )
        b = _art(
            table_id="b", section="capital_management",
            title="Tableau 28 – Ratios de capital",
            table_number="28",
            rows=[], first_column_indicators=indicators,
        )
        d = match_decision(a, b)
        assert d.is_match is True
        assert d.indicator_overlap == 1.0
