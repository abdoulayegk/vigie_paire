"""Tests for the Adaptive Table Pairing Engine features."""

from __future__ import annotations

import pytest

from vigilance.compare.table_pairing_engine import (
    CandidateScore,
    PairingDecision,
    ScoringProfile,
    TableView,
    _adapt_weights,
    _candidate_score,
    _hungarian_assignment,
    _indicator_order_similarity,
    _rescue_unmatched,
)
from vigilance.models.table_models import TableArtifact
from vigilance.utils.matching_normalizer import header_literal_fingerprint


def _table(
    tid: str,
    *,
    section: str = "risk_management",
    title: str = "Tableau",
    indicators: list[str] | None = None,
    page: int = 1,
    headers: list[str] | None = None,
    table_number: str | None = None,
    title_reliability: str | None = None,
) -> TableArtifact:
    indicators = indicators or ["A", "B", "C"]
    return TableArtifact(
        bank_code="rbc",
        section=section,
        page_pdf=page,
        table_id=tid,
        title=title,
        headers=headers or ["Indicateur", "Valeur"],
        rows=[[label, "1"] for label in indicators],
        first_column_indicators=list(indicators),
        first_column_indicators_raw=list(indicators),
        extraction_method="vision_full_gpt4o",
        quarter="t2-2025",
        table_number=table_number,
        footnotes=[],
        content_source="vision_gpt4o",
        title_reliability=title_reliability,
    )


def _view(t: TableArtifact, sc: int = 1) -> TableView:
    return TableView.from_table(t, section_frequencies={}, section_table_count=sc)


# ---------------------------------------------------------------------------
# header_literal_fingerprint
# ---------------------------------------------------------------------------
class TestHeaderLiteralFingerprint:
    def test_identical_headers(self) -> None:
        assert header_literal_fingerprint(["A", "B"], ["A", "B"]) == 1.0

    def test_disjoint_headers(self) -> None:
        assert header_literal_fingerprint(["A", "B"], ["C", "D"]) == 0.0

    def test_partial_overlap(self) -> None:
        score = header_literal_fingerprint(["A", "B", "C"], ["B", "C", "D"])
        assert 0.3 < score < 0.8

    def test_empty_headers(self) -> None:
        assert header_literal_fingerprint([], ["A"]) == 0.0
        assert header_literal_fingerprint(None, None) == 0.0


# ---------------------------------------------------------------------------
# _indicator_order_similarity
# ---------------------------------------------------------------------------
class TestIndicatorOrderSimilarity:
    def test_identical_order(self) -> None:
        assert _indicator_order_similarity(["a", "b", "c", "d"], ["a", "b", "c", "d"]) == 1.0

    def test_reversed_order(self) -> None:
        score = _indicator_order_similarity(["a", "b", "c", "d"], ["d", "c", "b", "a"])
        assert score < 0.5

    def test_few_common_returns_zero(self) -> None:
        assert _indicator_order_similarity(["a"], ["a"]) == 0.0

    def test_partial_overlap_preserves_order(self) -> None:
        score = _indicator_order_similarity(
            ["a", "b", "c", "d", "e"], ["b", "c", "d", "e", "f"]
        )
        assert score >= 0.8


# ---------------------------------------------------------------------------
# ScoringProfile
# ---------------------------------------------------------------------------
class TestScoringProfile:
    def test_default_profile(self) -> None:
        p = ScoringProfile()
        assert p.w_distinctive_overlap > 0
        assert p.adaptive_mode == "default"

    def test_from_thresholds_maps_legacy_keys(self) -> None:
        th = {"weight_label_overlap": 0.5, "weight_title": 0.2}
        p = ScoringProfile.from_thresholds(th)
        assert p.w_distinctive_overlap == 0.5
        assert p.w_title == 0.2

    def test_from_thresholds_maps_new_keys(self) -> None:
        th = {"w_header_fingerprint": 0.15, "w_indicator_ordering": 0.06}
        p = ScoringProfile.from_thresholds(th)
        assert p.w_header_fingerprint == 0.15
        assert p.w_indicator_ordering == 0.06


# ---------------------------------------------------------------------------
# _adapt_weights
# ---------------------------------------------------------------------------
class TestAdaptWeights:
    def test_table_number_not_used_for_adaptation(self) -> None:
        """Zero-trust: table number is not used as positive signal; no table_number_anchor mode."""
        p = ScoringProfile()
        adapted = _adapt_weights(p, title_reliability=0.5, title_sim=0.3, n_indicators=10, has_table_number=True)
        assert adapted.adaptive_mode == "default"

    def test_few_indicators_mode(self) -> None:
        p = ScoringProfile()
        adapted = _adapt_weights(p, title_reliability=0.5, title_sim=0.3, n_indicators=2, has_table_number=False)
        assert adapted.adaptive_mode == "few_indicators"

    def test_strong_title_mode(self) -> None:
        p = ScoringProfile()
        adapted = _adapt_weights(p, title_reliability=0.9, title_sim=0.85, n_indicators=10, has_table_number=False)
        assert adapted.adaptive_mode == "strong_title"

    def test_default_mode_unchanged(self) -> None:
        p = ScoringProfile()
        adapted = _adapt_weights(p, title_reliability=0.5, title_sim=0.5, n_indicators=10, has_table_number=False)
        assert adapted.adaptive_mode == "default"


# ---------------------------------------------------------------------------
# Hungarian assignment
# ---------------------------------------------------------------------------
class TestHungarianAssignment:
    def test_hungarian_resolves_cascade_collision(self) -> None:
        t1_a = _table("t1_a", indicators=["X", "Y", "Z", "W", "V"], title="Alpha")
        t1_b = _table("t1_b", indicators=["X", "Y", "Z", "W", "V"], title="Beta", page=2)
        t2_a = _table("t2_a", indicators=["X", "Y", "Z", "W", "V"], title="Alpha", page=3)
        t2_b = _table("t2_b", indicators=["X", "Y", "Z", "W", "V"], title="Beta", page=4)
        v1a, v1b = _view(t1_a), _view(t1_b)
        v2a, v2b = _view(t2_a), _view(t2_b)

        sa_a = _candidate_score(v1a, v2a)
        sa_b = _candidate_score(v1b, v2a)
        sb_a = _candidate_score(v1a, v2b)
        sb_b = _candidate_score(v1b, v2b)
        cmap = {v2a.uid: [sa_a, sa_b], v2b.uid: [sb_a, sb_b]}

        accepted, ambiguous, no_match = _hungarian_assignment([v2a, v2b], [v1a, v1b], cmap)
        matched_t1 = {a[1].t1_view.uid for a in accepted}
        assert len(accepted) == 2
        assert len(matched_t1) == 2

    def test_hungarian_rejects_low_score_pairs(self) -> None:
        t1 = _table("t1", indicators=["A", "B", "C"])
        t2 = _table("t2", indicators=["X", "Y", "Z"], page=5)
        v1, v2 = _view(t1), _view(t2)
        score = _candidate_score(v1, v2)
        cmap = {v2.uid: [score]}
        accepted, _, _ = _hungarian_assignment([v2], [v1], cmap)
        assert len(accepted) == 0


# ---------------------------------------------------------------------------
# Rescue pass
# ---------------------------------------------------------------------------
class TestRescuePass:
    def test_rescue_recovers_unmatched_with_good_signals(self) -> None:
        t1 = _table(
            "t1_r",
            section="credit_risk",
            title="Credit exposure",
            indicators=["Loans", "Securities", "Derivatives", "Other", "Total"],
            page=40,
            title_reliability="reliable",
        )
        t2 = _table(
            "t2_r",
            section="credit_risk",
            title="Credit exposure",
            indicators=["Loans", "Securities", "Derivatives", "Other", "Total", "Net"],
            page=42,
            title_reliability="reliable",
        )
        v1, v2 = _view(t1), _view(t2)
        rescued = _rescue_unmatched(unmatched_t1_views=[v1], unmatched_t2_views=[v2])
        assert len(rescued) == 1


# ---------------------------------------------------------------------------
# VaR family disambiguation
# ---------------------------------------------------------------------------
class TestVaRFamilyDisambiguation:
    def test_var_tables_with_different_headers_are_not_confused(self) -> None:
        shared_indicators = [
            "Taux interet",
            "Ecart de taux",
            "Actions",
            "Change",
            "Marchandises",
            "Diversification",
            "VAR total",
        ]
        t1_summary = _table(
            "t1_var_sum",
            title="VAR par facteur de risque",
            indicators=shared_indicators,
            headers=["Facteur", "Moyenne", "Maximum", "Minimum", "Fin periode"],
            page=20,
        )
        t1_detail = _table(
            "t1_var_det",
            title="VAR par facteur de risque",
            indicators=shared_indicators,
            headers=["Facteur", "T2-2025", "T1-2025", "T4-2024", "T3-2024"],
            page=21,
        )
        t2_summary = _table(
            "t2_var_sum",
            title="VAR par facteur de risque",
            indicators=shared_indicators,
            headers=["Facteur", "Moyenne", "Maximum", "Minimum", "Fin periode"],
            page=25,
        )
        t2_detail = _table(
            "t2_var_det",
            title="VAR par facteur de risque",
            indicators=shared_indicators,
            headers=["Facteur", "T3-2025", "T2-2025", "T1-2025", "T4-2024"],
            page=26,
        )

        v1s = _view(t1_summary)
        v1d = _view(t1_detail)
        v2s = _view(t2_summary)
        v2d = _view(t2_detail)

        score_sum = _candidate_score(v1s, v2s)
        score_det = _candidate_score(v1d, v2d)
        score_cross1 = _candidate_score(v1s, v2d)
        score_cross2 = _candidate_score(v1d, v2s)

        assert score_sum.header_fingerprint > score_cross1.header_fingerprint
        assert score_det.header_fingerprint > score_cross2.header_fingerprint


# ---------------------------------------------------------------------------
# New signals in CandidateScore
# ---------------------------------------------------------------------------
class TestNewSignalsInCandidateScore:
    def test_candidate_score_includes_new_fields(self) -> None:
        t1 = _table("t1", indicators=["A", "B", "C", "D", "E"])
        t2 = _table("t2", indicators=["A", "B", "C", "D", "E"], page=2)
        v1, v2 = _view(t1), _view(t2)
        score = _candidate_score(v1, v2)
        d = score.as_feature_dict()
        assert "header_fingerprint" in d
        assert "indicator_ordering" in d
        assert "adaptive_mode" in d
        assert 0.0 <= d["header_fingerprint"] <= 1.0
        assert 0.0 <= d["indicator_ordering"] <= 1.0
