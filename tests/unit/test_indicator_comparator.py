"""Tests for strict intra-section indicator comparator."""

from __future__ import annotations

from unittest.mock import patch

from vigilance.compare.indicator_comparator import (
    MatchDecision,
    _compute_pair_score_with_guard_rails,
    _detect_split_diagnostic,
    _get_table_features,
    _indicator_set,
    explain_match,
    match_decision,
    match_tables_intra_section,
    run_strict_intra_section_compare,
)
from vigilance.models.table_models import TableArtifact


def _table(
    *,
    table_id: str,
    section: str,
    title: str,
    rows: list[list[str]],
    table_number: str | None = None,
    first_column_indicators: list[str] | None = None,
    page_pdf: int = 1,
    headers: list[str] | None = None,
) -> TableArtifact:
    return TableArtifact(
        bank_code="rbc",
        section=section,
        page_pdf=page_pdf,
        table_id=table_id,
        title=title,
        headers=headers or ["Indicateur", "Valeur"],
        rows=rows,
        first_column_indicators=first_column_indicators or [row[0] for row in rows if row],
        extraction_method="docling",
        quarter="t1-2025",
        pdf_path="dummy.pdf",
        table_number=table_number,
    )


def test_cross_section_forbidden_even_with_table_number_match() -> None:
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="TABLEAU 28: Fonds propres",
        rows=[["CET1", "13.2%"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="TABLEAU 28: Risque de credit",
        rows=[["CET1", "13.2%"]],
    )

    decision = match_decision(t1, t2)
    assert decision.is_match is False
    assert decision.reason == "cross_section_forbidden"
    assert decision.table_number_match is True


def test_match_when_same_section_and_indicator_overlap() -> None:
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Analyse risques",
        rows=[["Risque de credit", "100"], ["Risque de marche", "50"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Gestion des risques",
        rows=[["Risque de credit", "120"], ["Risque de marche", "60"]],
    )

    decision = match_decision(t1, t2, overlap_threshold=0.5)
    assert decision.is_match is True
    assert decision.reason in {"indicator_overlap_match", "table_number_match"}


def test_no_table_number_none_none_is_not_number_match() -> None:
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="Situation du capital",
        rows=[["CET1", "13.2%"]],
    )
    t2 = _table(
        table_id="t2",
        section="capital_management",
        title="Profil de capital",
        rows=[["RWA", "100"]],
    )

    decision = match_decision(t1, t2, overlap_threshold=0.5)
    assert decision.is_match is False
    assert decision.table_number_match is False
    assert decision.reason == "low_containment"


def test_table_number_conflict_blocks_false_positive_even_with_overlap() -> None:
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="TABLEAU 23 - Prêts hypothécaires à l'habitation",
        rows=[["Atlantique", "1"], ["Québec", "2"], ["Ontario", "3"]],
    )
    t2 = _table(
        table_id="t2",
        section="capital_management",
        title="TABLEAU 24 - Marges de crédit sur valeur domiciliaire",
        rows=[["Atlantique", "4"], ["Québec", "5"], ["Ontario", "6"]],
    )

    decision = match_decision(t1, t2, overlap_threshold=0.5)
    assert decision.is_match is False
    assert decision.reason == "table_number_conflict"


def test_official_facade_returns_added_and_removed_tables() -> None:
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="TABLEAU 55 - Risque de credit",
        rows=[["Risque de crédit", "1"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="TABLEAU 56 - Risque de liquidité",
        rows=[["Risque de liquidité", "2"]],
    )

    payload = run_strict_intra_section_compare([t1], [t2], overlap_threshold=0.9)
    assert payload["pairs"] == []
    assert payload["removed_tables"][0]["t1_table_id"] == "t1"
    assert payload["removed_tables"][0]["reason"] == "removed_table"
    assert payload["added_tables"][0]["t2_table_id"] == "t2"
    assert payload["added_tables"][0]["reason"] == "added_table"


def test_multi_signal_match_when_title_structure_and_position_align() -> None:
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Risque de crédit - exposition géographique",
        rows=[["Atlantique", "1"], ["Québec", "2"], ["Ontario", "3"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Risque de crédit exposition géographique",
        rows=[["Atlantique", "8"], ["Québec", "9"], ["États-Unis", "10"]],
    )
    t2.page_pdf = 2

    decision = match_decision(t1, t2, overlap_threshold=0.95)
    assert decision.is_match is True
    assert decision.reason in {"multi_signal_match", "indicator_overlap_match"}
    assert decision.coverage_min >= 2 / 3
    assert decision.title_similarity >= 0.72


def test_same_base_different_suffix_not_auto_but_can_match_with_overlap() -> None:
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="TABLEAU T14A - Fonds propres réglementaires",
        rows=[["CET1", "1"], ["AT1", "2"], ["Total", "3"]],
    )
    t2 = _table(
        table_id="t2",
        section="capital_management",
        title="TABLEAU T14B - Fonds propres réglementaires",
        rows=[["CET1", "4"], ["AT1", "5"], ["Total", "6"]],
    )

    decision = match_decision(t1, t2, overlap_threshold=0.5)
    assert decision.is_match is True
    assert decision.table_number_match is False
    assert decision.table_label_base_match is True
    assert decision.table_label_suffix_diff is True
    assert decision.reason in {
        "indicator_overlap_match",
        "indicator_set_hash_exact",
        "multi_signal_match",
    }


def test_anti_greedy_margin_marks_uncertain_competition() -> None:
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Analyse du risque de crédit portefeuille",
        rows=[["Canada", "1"], ["États-Unis", "2"], ["Europe", "3"]],
    )
    t2a = _table(
        table_id="t2a",
        section="risk_management",
        title="Analyse risque crédit portefeuille",
        rows=[["Canada", "9"], ["États-Unis", "8"], ["Europe", "7"]],
    )
    t2b = _table(
        table_id="t2b",
        section="risk_management",
        title="Analyse du risque de crédit du portefeuille",
        rows=[["Canada", "5"], ["États-Unis", "4"], ["Europe", "3"]],
    )
    t2a.page_pdf = 10
    t2b.page_pdf = 11

    payload = run_strict_intra_section_compare([t1], [t2a, t2b], overlap_threshold=0.5)
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0]["rescue_type"] == "single_rescue"
    assert payload["added_tables"] == []
    assert payload["removed_tables"] == []
    assert payload["unmatched_t1"] == []
    assert all(x["reason"] == "ambiguous_candidate" for x in payload["unmatched_t2"])
    assert all(x["unmatched_status"] == "ambiguous" for x in payload["unmatched_t2"])


def test_ambiguous_unmatched_do_not_become_added_removed() -> None:
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Analyse A",
        rows=[["alpha", "1"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Analyse B",
        rows=[["beta", "2"]],
    )

    base_payload = {
        "pairs": [],
        "probable_pairs": [],
        "unmatched_t1": [
            {
                "t1_uid": "risk_management|t1|p1",
                "t1_table_id": "t1",
                "section": "risk_management",
                "page_t1": 1,
                "title_t1": "Analyse A",
                "reason": "ambiguous_candidate",
                "unmatched_status": "ambiguous",
            }
        ],
        "unmatched_t2": [
            {
                "t2_uid": "risk_management|t2|p1",
                "t2_table_id": "t2",
                "section": "risk_management",
                "page_t2": 1,
                "title_t2": "Analyse B",
                "reason": "ambiguous_candidate",
                "unmatched_status": "ambiguous",
            }
        ],
        "debug_unmatched_candidates": [],
    }

    with patch(
        "vigilance.compare.indicator_comparator.match_tables_intra_section",
        return_value=base_payload,
    ):
        payload = run_strict_intra_section_compare([t1], [t2], overlap_threshold=0.9)

    assert payload["added_tables"] == []
    assert payload["removed_tables"] == []
    assert len(payload["unmatched_ambiguous_t1"]) == 1
    assert len(payload["unmatched_ambiguous_t2"]) == 1


def test_bidirectional_single_rescue_recovers_pair_missing_from_t1_pass() -> None:
    t1a = _table(
        table_id="t1a",
        section="risk_management",
        title="Tableau A",
        rows=[["alpha", "1"]],
    )
    t1b = _table(
        table_id="t1b",
        section="risk_management",
        title="Tableau B",
        rows=[["beta", "2"]],
    )
    t2a = _table(
        table_id="t2a",
        section="risk_management",
        title="Tableau A courant",
        rows=[["alpha", "3"]],
    )
    t2b = _table(
        table_id="t2b",
        section="risk_management",
        title="Tableau B courant",
        rows=[["beta", "4"]],
    )

    base_payload = {
        "pairs": [],
        "probable_pairs": [],
        "unmatched_t1": [
            {
                "t1_uid": "risk_management|t1a|p1",
                "t1_table_id": "t1a",
                "section": "risk_management",
                "page_t1": 1,
                "title_t1": "Tableau A",
                "reason": "weak_signals",
                "unmatched_status": "confirmed",
            },
            {
                "t1_uid": "risk_management|t1b|p1",
                "t1_table_id": "t1b",
                "section": "risk_management",
                "page_t1": 1,
                "title_t1": "Tableau B",
                "reason": "weak_signals",
                "unmatched_status": "confirmed",
            },
        ],
        "unmatched_t2": [
            {
                "t2_uid": "risk_management|t2a|p1",
                "t2_table_id": "t2a",
                "section": "risk_management",
                "page_t2": 1,
                "title_t2": "Tableau A courant",
                "reason": "unmatched",
                "unmatched_status": "confirmed",
            },
            {
                "t2_uid": "risk_management|t2b|p1",
                "t2_table_id": "t2b",
                "section": "risk_management",
                "page_t2": 1,
                "title_t2": "Tableau B courant",
                "reason": "unmatched",
                "unmatched_status": "confirmed",
            },
        ],
        "debug_unmatched_candidates": [],
    }

    def _decision(table_t1: TableArtifact, table_t2: TableArtifact, **_: object) -> MatchDecision:
        scores = {
            ("t1a", "t2a"): 0.95,
            ("t1a", "t2b"): 0.70,
            ("t1b", "t2a"): 0.94,
            ("t1b", "t2b"): 0.89,
        }
        score = scores[(table_t1.table_id, table_t2.table_id)]
        return MatchDecision(
            is_match=score >= 0.7,
            reason="indicator_overlap_match" if score >= 0.7 else "weak_signals",
            score=score,
            section_match=True,
            table_number_match=False,
            table_label_base_match=False,
            table_label_suffix_diff=False,
            indicator_overlap=score,
            title_similarity=0.75,
            structure_similarity=0.70,
            context_heading_similarity=0.40,
            position_proximity=0.50,
            t1_uid=f"risk_management|{table_t1.table_id}|p1",
            t2_uid=f"risk_management|{table_t2.table_id}|p1",
            t1_table_id=table_t1.table_id,
            t2_table_id=table_t2.table_id,
            indicator_containment=score,
            header_schema_similarity=0.75,
            section_state="same_known",
            decision_level="match",
            coverage_previous=score,
            coverage_current=score,
            coverage_min=score,
            coverage_gap=0.0,
        )

    with patch(
        "vigilance.compare.indicator_comparator.match_tables_intra_section",
        return_value=base_payload,
    ), patch(
        "vigilance.compare.indicator_comparator.match_decision",
        side_effect=_decision,
    ):
        payload = run_strict_intra_section_compare(
            [t1a, t1b],
            [t2a, t2b],
            overlap_threshold=0.55,
        )

    rescue_pairs = [pair for pair in payload["pairs"] if pair.get("rescue_type") == "single_rescue"]
    assert len(rescue_pairs) == 2
    assert {pair["t1_uid"] for pair in rescue_pairs} == {
        "risk_management|t1a|p1",
        "risk_management|t1b|p1",
    }
    assert {pair["t2_uid"] for pair in rescue_pairs} == {
        "risk_management|t2a|p1",
        "risk_management|t2b|p1",
    }


def test_table_number_conflict_bypass_with_identical_title() -> None:
    """Renumbered tables with identical titles should match via title override."""
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="TABLEAU 3 - STRUCTURE DE FONDS PROPRES ET RATIOS",
        table_number="3",
        rows=[["CET1", "13.2%"], ["AT1", "1.5%"], ["Tier 2", "2.0%"]],
    )
    t2 = _table(
        table_id="t2",
        section="capital_management",
        title="TABLEAU 2 - STRUCTURE DE FONDS PROPRES ET RATIOS",
        table_number="2",
        rows=[["CET1", "13.5%"], ["AT1", "1.6%"], ["Tier 2", "2.1%"]],
    )

    decision = match_decision(t1, t2)
    assert decision.is_match is True
    assert decision.table_number_match is False
    assert decision.table_label_base_match is False
    assert decision.title_similarity >= 0.85


def test_table_number_conflict_still_blocks_different_titles() -> None:
    """Different titles with conflicting numbers must still be rejected."""
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="TABLEAU 23 - Prêts hypothécaires à l'habitation",
        table_number="23",
        rows=[["Atlantique", "1"], ["Québec", "2"], ["Ontario", "3"]],
    )
    t2 = _table(
        table_id="t2",
        section="capital_management",
        title="TABLEAU 24 - Marges de crédit sur valeur domiciliaire",
        table_number="24",
        rows=[["Atlantique", "4"], ["Québec", "5"], ["Ontario", "6"]],
    )

    decision = match_decision(t1, t2)
    assert decision.is_match is False
    assert decision.reason == "table_number_conflict"


def test_title_override_match_with_empty_indicators() -> None:
    """High title similarity + structure match when both tables have no indicators (no label reject)."""
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Risque de marché - VaR globale",
        rows=[],
        first_column_indicators=[],
        headers=["Mesure", "T1", "T2", "T3", "T4"],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Risque de marché - VaR globale",
        rows=[],
        first_column_indicators=[],
        headers=["Mesure", "T1", "T2", "T3", "T4"],
    )

    decision = match_decision(t1, t2)
    assert decision.is_match is True
    assert decision.reason in ("title_override_match", "multi_signal_match", "table_number_low_overlap_rescue")
    assert decision.indicator_overlap == 0.0
    assert decision.title_similarity >= 0.88


def test_indicator_set_fallback_to_first_column_indicators() -> None:
    """When rows are truthy but yield no indicators after filtering, fall back."""
    t = _table(
        table_id="t1",
        section="capital_management",
        title="Test table",
        rows=[["", "100"], ["", "200"]],
        first_column_indicators=["CET1", "Tier 2", "Total"],
    )

    result = _indicator_set(t)
    assert "cet1" in result
    assert "tier 2" in result
    assert "total" in result
    assert len(result) == 3


def test_indicator_set_strips_trailing_footnote_numbers() -> None:
    t = _table(
        table_id="t1",
        section="capital_management",
        title="Test table",
        rows=[
            ["Actions ordinaires 2", "100"],
            ["Actions privilégiées de catégorie B 3", "200"],
        ],
    )

    result = _indicator_set(t)
    assert "action ordinaire" in result
    assert "action privilegiee de categorie b" in result
    assert "action ordinaire 2" not in result
    assert "action privilegiee de categorie b 3" not in result


def test_title_matching_strips_date_and_notes() -> None:
    """Titres '31 janvier 2025 1, 2' et '30 avril 2025 1, 2' matchent (meme tableau CIBC)."""
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="31 janvier 2025 1, 2",
        rows=[["LCR", "150%"], ["NSFR", "110%"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="30 avril 2025 1, 2",
        rows=[["LCR", "155%"], ["NSFR", "112%"]],
    )
    decision = match_decision(t1, t2, bank_code="cibc")
    assert decision.is_match is True
    assert decision.indicator_overlap >= 0.5


def test_few_indicators_lower_threshold_rbc() -> None:
    """RBC: few-indicators branch is detected even if V2 score remains below MATCH threshold."""
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="24 Banque Royale du Canada Premier trimestre de 2025",
        rows=[["A", "1"], ["B", "2"], ["C", "3"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="30 Banque Royale du Canada Deuxieme trimestre de 2025",
        rows=[["A", "10"], ["B", "20"], ["D", "40"]],
    )
    decision = match_decision(
        t1, t2, bank_code="rbc", overlap_threshold=0.5
    )
    assert decision.indicator_overlap == 2 / 4
    assert decision.is_match is False
    assert decision.reason == "few_indicators_header_footer_match"


def test_table_number_low_overlap_rescue_for_rbc_header_footer_titles() -> None:
    """With content-only decision, low label overlap is rejected even with strong title/structure."""
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="30 Banque Royale du Canada Premier trimestre de 2025",
        rows=[["Portefeuille A", "100"], ["Portefeuille B", "200"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="30 Banque Royale du Canada Deuxième trimestre de 2025",
        rows=[["Segment X", "300"], ["Segment Y", "400"]],
        page_pdf=2,
    )

    decision = match_decision(t1, t2, bank_code="rbc")
    assert decision.is_match is False
    assert decision.reason == "low_label_overlap_reject"
    assert decision.indicator_overlap == 0.0


def test_date_title_structure_rescue_for_cibc() -> None:
    """Low label overlap is rejected even with date-title rescue (anti-false-match rule)."""
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="31 janvier 2025 1, 2",
        rows=[["LCR", "150%"], ["NSFR", "110%"], ["Actifs liquides", "90"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="30 avril 2025 1, 2",
        rows=[["Engagements", "120"], ["Financement", "115"], ["Contreparties", "95"]],
        page_pdf=2,
    )

    decision = match_decision(t1, t2, bank_code="cibc")
    assert decision.is_match is False
    assert decision.reason == "low_label_overlap_reject"


def test_indicator_set_keeps_semantic_trailing_numbers() -> None:
    t = _table(
        table_id="t1",
        section="capital_management",
        title="Test table",
        rows=[
            ["Série 2", "100"],
            ["Série 3", "200"],
            ["Fonds propres de catégorie 1", "300"],
        ],
    )

    result = _indicator_set(t)
    assert "serie 2" in result
    assert "serie 3" in result
    assert "fonds propre de categorie 1" in result


def test_legacy_flag_and_explicit_true_same_result_when_clear_matches() -> None:
    """Le flag legacy use_hungarian=False route vers le meme moteur symetrique."""
    t1 = [
        _table(table_id="t1", section="risk", title="Risques A", rows=[["A", "1"], ["B", "2"]]),
        _table(table_id="t2", section="risk", title="Risques B", rows=[["C", "3"], ["D", "4"]]),
    ]
    t2 = [
        _table(table_id="t1", section="risk", title="Risques A", rows=[["A", "10"], ["B", "20"]], page_pdf=2),
        _table(table_id="t2", section="risk", title="Risques B", rows=[["C", "30"], ["D", "40"]], page_pdf=2),
    ]
    legacy_flag = match_tables_intra_section(t1, t2, use_hungarian=False)
    explicit_true = match_tables_intra_section(t1, t2, use_hungarian=True)
    assert len(legacy_flag["pairs"]) == len(explicit_true["pairs"]) == 2
    legacy_uids = {(p["t1_uid"], p["t2_uid"]) for p in legacy_flag["pairs"]}
    explicit_uids = {(p["t1_uid"], p["t2_uid"]) for p in explicit_true["pairs"]}
    assert legacy_uids == explicit_uids


def test_hungarian_handles_more_t1_than_t2() -> None:
    """Hungarian gere le cas ou il y a plus de T1 que de T2."""
    t1 = [
        _table(table_id="a", section="risk", title="A", rows=[["X", "1"]]),
        _table(table_id="b", section="risk", title="B", rows=[["Y", "2"]]),
        _table(table_id="c", section="risk", title="C", rows=[["Z", "3"]]),
    ]
    t2 = [
        _table(table_id="a", section="risk", title="A", rows=[["X", "10"]], page_pdf=2),
        _table(table_id="b", section="risk", title="B", rows=[["Y", "20"]], page_pdf=2),
    ]
    result = match_tables_intra_section(t1, t2, use_hungarian=True)
    assert len(result["pairs"]) == 2
    assert len(result["unmatched_t1"]) == 1
    assert len(result["unmatched_t2"]) == 0


def test_hungarian_handles_more_t2_than_t1() -> None:
    """Hungarian gere le cas ou il y a plus de T2 que de T1."""
    t1 = [
        _table(table_id="a", section="risk", title="A", rows=[["X", "1"]]),
        _table(table_id="b", section="risk", title="B", rows=[["Y", "2"]]),
    ]
    t2 = [
        _table(table_id="a", section="risk", title="A", rows=[["X", "10"]], page_pdf=2),
        _table(table_id="b", section="risk", title="B", rows=[["Y", "20"]], page_pdf=2),
        _table(table_id="c", section="risk", title="C", rows=[["Z", "30"]], page_pdf=2),
    ]
    result = match_tables_intra_section(t1, t2, use_hungarian=True)
    assert len(result["pairs"]) == 2
    assert len(result["unmatched_t1"]) == 0
    assert len(result["unmatched_t2"]) == 1


def test_default_matcher_uses_symmetric_engine() -> None:
    """Le moteur par defaut reste le matcher symetrique."""
    t1 = [
        _table(table_id="t1", section="risk", title="Risques", rows=[["A", "1"], ["B", "2"]]),
    ]
    t2 = [
        _table(table_id="t1", section="risk", title="Risques", rows=[["A", "10"], ["B", "20"]], page_pdf=2),
    ]
    r = match_tables_intra_section(t1, t2)
    assert len(r["pairs"]) == 1
    assert r["pairs"][0]["t1_uid"] == "risk|t1|p1"
    assert r["pairs"][0]["t2_uid"] == "risk|t1|p2"


# ---------- Plan implementation matching (Phases 1-6) ----------


def test_get_table_features_returns_anchors_and_hash() -> None:
    """_get_table_features returns coherent anchors and indicator_set_hash."""
    t = _table(
        table_id="a",
        section="capital",
        title="T28",
        rows=[["CET1", "1"], ["Tier1", "2"], ["Total", "3"]],
    )
    anchors, hash_val = _get_table_features(t)
    assert len(anchors) == 3
    assert "cet1" in anchors or "CET1" in [a.lower() for a in anchors]
    assert hash_val.startswith("sha1:")
    assert len(hash_val) == 45


def test_fast_path_hash_same_indicators_match_score_one() -> None:
    """Same indicator set yields match; fast path returns score 1.0 when hash matches."""
    indicators = ["Pret hypo", "Cartes"]
    a = _table(
        table_id="a",
        section="capital",
        title="Tableau 32 Exposition credit",
        rows=[[indicators[0], "1"], [indicators[1], "2"]],
        first_column_indicators=indicators,
    )
    b = _table(
        table_id="b",
        section="capital",
        title="Tableau 32 Exposition au credit",
        rows=[[indicators[0], "3"], [indicators[1], "4"]],
        first_column_indicators=indicators,
    )
    d = match_decision(a, b)
    assert d.is_match is True
    assert d.indicator_overlap == 1.0
    assert d.reason in ("indicator_set_hash_exact", "indicator_overlap_match")


def test_size_mismatch_reject_when_ratio_over_threshold() -> None:
    """Garde-fou taille: size_ratio > 0.60 rejects match."""
    from vigilance.compare.indicator_comparator import _load_compare_thresholds

    th = _load_compare_thresholds()
    th["size_mismatch_reject_threshold"] = 0.50
    a = _table(
        table_id="a",
        section="capital",
        title="Ratios",
        rows=[
            ["CET1", "1"], ["Tier1", "2"], ["RWA", "3"],
            ["Leverage", "4"], ["Total", "5"],
        ],
    )
    b = _table(
        table_id="b",
        section="capital",
        title="Ratios",
        rows=[["CET1", "10"], ["Tier1", "20"]],
    )
    d = match_decision(a, b, thresholds=th)
    assert d.is_match is False
    assert d.reason == "size_mismatch_reject"


def test_explain_match_returns_top5_and_subscores() -> None:
    """explain_match returns top5_common_labels, top5_missing, subscores."""
    a = _table(
        table_id="a",
        section="capital",
        title="T28",
        rows=[["CET1", "1"], ["Tier1", "2"], ["RWA", "3"], ["Leverage", "4"]],
    )
    b = _table(
        table_id="b",
        section="capital",
        title="T28",
        rows=[["CET1", "10"], ["Tier1", "20"], ["RWA", "30"]],
    )
    decision = match_decision(a, b)
    expl = explain_match(a, b, decision.score)
    assert "top5_common_labels" in expl
    assert "top5_missing_in_t2" in expl
    assert "top5_missing_in_t1" in expl
    assert "subscores" in expl
    assert "labels" in expl["subscores"]
    assert "anchors" in expl["subscores"]
    assert expl["n_indicators_t1"] == 4
    assert expl["n_indicators_t2"] == 3


def test_detect_split_diagnostic_returns_true_when_coverage_high() -> None:
    """_detect_split_diagnostic returns candidates when coverage > 0.80."""
    t1 = _table(
        table_id="t1",
        section="capital",
        title="T10",
        rows=[
            ["A", "1"], ["B", "2"], ["C", "3"], ["D", "4"], ["E", "5"],
        ],
    )
    t2a = _table(
        table_id="t2a",
        section="capital",
        title="Fragment 1",
        rows=[["A", "1"], ["B", "2"], ["C", "3"]],
    )
    t2b = _table(
        table_id="t2b",
        section="capital",
        title="Fragment 2",
        rows=[["D", "4"], ["E", "5"]],
    )
    candidates = [
        (t2a, 0.45),
        (t2b, 0.45),
    ]
    result = _detect_split_diagnostic(t1, candidates)
    assert result is not None
    assert len(result) == 2


# ---------- Hungarian post-threshold (Phase 1-3) ----------


def test_compute_pair_score_with_guard_rails_blocks_cross_section() -> None:
    """_compute_pair_score_with_guard_rails returns is_blocked for cross_section_forbidden."""
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="T28",
        rows=[["CET1", "1"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="T28",
        rows=[["CET1", "2"]],
    )
    sr = _compute_pair_score_with_guard_rails(t1, t2)
    assert sr.is_blocked is True
    assert sr.block_reason == "cross_section_forbidden"
    assert sr.score < -1e8


def test_compute_pair_score_with_guard_rails_blocks_table_number_conflict() -> None:
    """_compute_pair_score_with_guard_rails returns is_blocked for table_number_conflict."""
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="TABLEAU 23 - Pret hypotechaires",
        rows=[["A", "1"], ["B", "2"], ["C", "3"]],
    )
    t2 = _table(
        table_id="t2",
        section="capital_management",
        title="TABLEAU 24 - Marges credit",
        rows=[["A", "4"], ["B", "5"], ["C", "6"]],
    )
    sr = _compute_pair_score_with_guard_rails(t1, t2, overlap_threshold=0.5)
    assert sr.is_blocked is True
    assert sr.block_reason == "table_number_conflict"


def test_compute_pair_score_with_guard_rails_returns_real_score_when_allowed() -> None:
    """_compute_pair_score_with_guard_rails returns real score for admissible pairs."""
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Risques",
        rows=[["Risque credit", "1"], ["Risque marche", "2"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Risques",
        rows=[["Risque credit", "10"], ["Risque marche", "20"]],
        page_pdf=2,
    )
    sr = _compute_pair_score_with_guard_rails(t1, t2, overlap_threshold=0.5)
    assert sr.is_blocked is False
    assert sr.block_reason is None
    assert sr.score > 0.5
    assert sr.decision.indicator_overlap > 0


def test_post_hungarian_threshold_mode_produces_pairs() -> None:
    """With use_post_hungarian_threshold=True, Hungarian sees all scores then applies threshold."""
    from vigilance.compare.indicator_comparator import _DEFAULTS

    t1 = [
        _table(table_id="a", section="risk", title="A", rows=[["X", "1"], ["Y", "2"]]),
        _table(table_id="b", section="risk", title="B", rows=[["Z", "3"], ["W", "4"]]),
    ]
    t2 = [
        _table(table_id="a", section="risk", title="A", rows=[["X", "10"], ["Y", "20"]], page_pdf=2),
        _table(table_id="b", section="risk", title="B", rows=[["Z", "30"], ["W", "40"]], page_pdf=2),
    ]
    th = dict(_DEFAULTS)
    th["use_post_hungarian_threshold"] = 1.0
    with patch("vigilance.compare.indicator_comparator._load_compare_thresholds", return_value=th):
        result = match_tables_intra_section(t1, t2, use_hungarian=True)
    assert len(result["pairs"]) >= 1


def test_non_regression_post_threshold_disabled_by_default() -> None:
    """With use_post_hungarian_threshold=False (default), behavior unchanged."""
    t1 = [
        _table(table_id="t1", section="risk", title="Risques", rows=[["A", "1"], ["B", "2"]]),
    ]
    t2 = [
        _table(table_id="t1", section="risk", title="Risques", rows=[["A", "10"], ["B", "20"]], page_pdf=2),
    ]
    r = match_tables_intra_section(t1, t2, use_hungarian=True)
    assert len(r["pairs"]) == 1
    assert r["pairs"][0]["t1_uid"] == "risk|t1|p1"
    assert r["pairs"][0]["t2_uid"] == "risk|t1|p2"
