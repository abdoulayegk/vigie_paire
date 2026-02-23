"""Tests for strict intra-section indicator comparator."""

from __future__ import annotations

from vigilance.compare.indicator_comparator import (
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
    assert decision.is_match is False
    assert decision.decision_level in ("probable", "no_match")
    assert decision.indicator_overlap >= 0.30
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
    assert decision.reason in {"indicator_overlap_match", "multi_signal_match"}


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
    assert all(x["reason"] == "uncertain_competition" for x in payload["unmatched_t2"])


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
    assert "actions ordinaires" in result
    assert "actions privilegiees de categorie b" in result
    assert "actions ordinaires 2" not in result
    assert "actions privilegiees de categorie b 3" not in result


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
    assert "fonds propres de categorie 1" in result


def test_hungarian_vs_greedy_same_result_when_clear_matches() -> None:
    """Hungarian et greedy produisent le meme nombre de paires quand les matches sont clairs."""
    t1 = [
        _table(table_id="t1", section="risk", title="Risques A", rows=[["A", "1"], ["B", "2"]]),
        _table(table_id="t2", section="risk", title="Risques B", rows=[["C", "3"], ["D", "4"]]),
    ]
    t2 = [
        _table(table_id="t1", section="risk", title="Risques A", rows=[["A", "10"], ["B", "20"]], page_pdf=2),
        _table(table_id="t2", section="risk", title="Risques B", rows=[["C", "30"], ["D", "40"]], page_pdf=2),
    ]
    greedy = match_tables_intra_section(t1, t2, use_hungarian=False)
    hungarian = match_tables_intra_section(t1, t2, use_hungarian=True)
    assert len(greedy["pairs"]) == len(hungarian["pairs"]) == 2
    greedy_uids = {(p["t1_uid"], p["t2_uid"]) for p in greedy["pairs"]}
    hungarian_uids = {(p["t1_uid"], p["t2_uid"]) for p in hungarian["pairs"]}
    assert greedy_uids == hungarian_uids


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


def test_greedy_default_unchanged() -> None:
    """Avec use_hungarian=False (defaut), le comportement reste inchange."""
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
