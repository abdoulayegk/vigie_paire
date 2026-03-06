"""Tests for V2 rescue logic (single + split/merge) and section soft-gating."""

from __future__ import annotations

from vigilance.compare.indicator_comparator import (
    _allow_rbc_split_merge_rescue,
    match_decision,
    run_strict_intra_section_compare,
)
from vigilance.models.table_models import TableArtifact


def _table(
    *,
    table_id: str,
    section: str,
    title: str,
    rows: list[list[str]],
    headers: list[str] | None = None,
    page_pdf: int = 1,
    table_number: str | None = None,
    bank_code: str = "rbc",
) -> TableArtifact:
    return TableArtifact(
        bank_code=bank_code,
        section=section,
        page_pdf=page_pdf,
        table_id=table_id,
        title=title,
        headers=headers or ["Indicateur", "T2 2025"],
        rows=rows,
        first_column_indicators=[row[0] for row in rows if row and str(row[0]).strip()],
        extraction_method="docling",
        quarter="t1-2025",
        pdf_path="dummy.pdf",
        table_number=table_number,
    )


def test_table_id_like_tableau_number_does_not_force_number_conflict() -> None:
    t1 = _table(
        table_id="tableau_12",
        section="capital_management",
        title="",
        rows=[["CET1", "13.1"], ["Tier 1", "15.0"]],
    )
    t2 = _table(
        table_id="tableau_12",
        section="capital_management",
        title="",
        rows=[["CET1", "13.4"], ["Tier 1", "15.2"]],
    )

    decision = match_decision(t1, t2)
    assert decision.is_match is True
    assert decision.table_number_match is False
    assert decision.reason != "table_number_conflict"


def test_known_cross_section_is_hard_blocked() -> None:
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="Tableau 28 - Capital",
        rows=[["CET1", "13.1"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Tableau 28 - Credit",
        rows=[["CET1", "13.4"]],
    )

    decision = match_decision(t1, t2)
    assert decision.is_match is False
    assert decision.reason == "cross_section_forbidden"


def test_unknown_section_can_match_when_signals_are_strong() -> None:
    t1 = _table(
        table_id="t1",
        section="unknown_section",
        title="Tableau 30 - Ratios de liquidite",
        rows=[["LCR", "150"], ["NSFR", "120"]],
    )
    t2 = _table(
        table_id="t2",
        section="unknown_section",
        title="Tableau 30 - Ratios de liquidite",
        rows=[["LCR", "155"], ["NSFR", "121"]],
    )

    decision = match_decision(t1, t2)
    assert decision.section_state == "unknown_present"
    assert decision.is_match is True
    assert decision.score >= 0.74
    assert decision.indicator_containment >= 0.65


def test_split_rescue_one_to_two_fragments() -> None:
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Analyse globale des flux de paiement",
        rows=[
            ["segment a", "1"],
            ["segment b", "2"],
            ["segment c", "3"],
            ["segment d", "4"],
            ["segment e", "5"],
        ],
        page_pdf=10,
        bank_code="bmo",
    )
    t2a = _table(
        table_id="t2a",
        section="risk_management",
        title="En fonction des paiements contractuels",
        rows=[["segment a", "10"], ["segment b", "20"], ["segment c", "30"]],
        page_pdf=10,
        bank_code="bmo",
    )
    t2b = _table(
        table_id="t2b",
        section="risk_management",
        title="En fonction des paiements reels des clients",
        rows=[["segment d", "40"], ["segment e", "50"]],
        page_pdf=11,
        bank_code="bmo",
    )

    payload = run_strict_intra_section_compare([t1], [t2a, t2b], bank_code="bmo")
    assert payload["split_merge_rescues_count"] == 1
    assert payload["rescued_matches_count"] >= 1
    assert payload["added_tables"] == []
    assert payload["removed_tables"] == []
    rescued = [p for p in payload["pairs"] if p.get("rescue_type") == "split_merge_rescue"]
    assert len(rescued) == 1
    assert len(rescued[0].get("split_members_t2", [])) == 2


def test_merge_rescue_two_to_one_fragments() -> None:
    t1a = _table(
        table_id="t1a",
        section="risk_management",
        title="En fonction des paiements contractuels",
        rows=[["segment a", "1"], ["segment b", "2"]],
        page_pdf=10,
        bank_code="bmo",
    )
    t1b = _table(
        table_id="t1b",
        section="risk_management",
        title="En fonction des paiements reels des clients",
        rows=[["segment c", "3"], ["segment d", "4"], ["segment e", "5"]],
        page_pdf=11,
        bank_code="bmo",
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Analyse globale des flux de paiement",
        rows=[
            ["segment a", "10"],
            ["segment b", "20"],
            ["segment c", "30"],
            ["segment d", "40"],
            ["segment e", "50"],
        ],
        page_pdf=10,
        bank_code="bmo",
    )

    payload = run_strict_intra_section_compare([t1a, t1b], [t2], bank_code="bmo")
    assert payload["split_merge_rescues_count"] == 1
    rescued = [p for p in payload["pairs"] if p.get("rescue_type") == "split_merge_rescue"]
    assert len(rescued) == 1
    assert len(rescued[0].get("merge_members_t1", [])) == 2


def test_rbc_split_merge_guard_blocks_unreliable_title_when_signals_are_weak() -> None:
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Charges grevant les actifs",
        rows=[
            ["Titres hypothecaires", "1"],
            ["Prets hypothecaires", "2"],
            ["Autres prets", "3"],
            ["Derives", "4"],
            ["Autres", "5"],
        ],
        headers=["Actifs detenus par la banque", "Total de l'actif", "Actifs greves"],
        bank_code="rbc",
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Au 30 avril 2025",
        rows=[
            ["Titres hypothecaires", "10"],
            ["Autres prets", "30"],
        ],
        headers=["Montant", "Greve"],
        bank_code="rbc",
    )

    decision = match_decision(t1, t2, bank_code="rbc")
    assert (
        _allow_rbc_split_merge_rescue(
            bank_code="rbc",
            primary_table=t2,
            counterpart_table=t1,
            primary_decision=decision,
            union_containment=0.75,
            schema_score=0.68,
        )
        is False
    )


def test_dates_in_headers_and_titles_do_not_break_logical_match() -> None:
    t1 = _table(
        table_id="t1",
        section="risk_management",
        title="Risque de liquidite au 31 janvier 2025",
        headers=["Indicateur", "31 janvier 2025", "31 octobre 2024"],
        rows=[["LCR", "150"], ["NSFR", "120"]],
    )
    t2 = _table(
        table_id="t2",
        section="risk_management",
        title="Risque de liquidite au 30 avril 2025",
        headers=["Indicateur", "30 avril 2025", "31 janvier 2025"],
        rows=[["LCR", "155"], ["NSFR", "121"]],
    )

    decision = match_decision(t1, t2, bank_code="cibc")
    assert decision.is_match is True
    assert decision.header_schema_similarity >= 0.8
