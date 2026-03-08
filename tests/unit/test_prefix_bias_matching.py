from __future__ import annotations

from vigilance.compare.indicator_comparator import (
    match_decision,
    match_tables_intra_section,
    run_strict_intra_section_compare,
)
from vigilance.models.table_models import TableArtifact


def _table(
    *,
    table_id: str,
    title: str,
    rows: list[str],
    section: str = "risk_management",
    page_pdf: int = 1,
) -> TableArtifact:
    materialized_rows = [[label, str(idx)] for idx, label in enumerate(rows, start=1)]
    return TableArtifact(
        bank_code="bmo",
        section=section,
        page_pdf=page_pdf,
        table_id=table_id,
        title=title,
        headers=["Indicateur", "Valeur"],
        rows=materialized_rows,
        first_column_indicators=rows,
        first_column_indicators_raw=rows,
        extraction_method="vision_full_gpt4o",
        quarter="t2-2025",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )


def test_subset_superset_candidate_is_rejected_for_one_to_one() -> None:
    current = _table(
        table_id="current",
        title="Exposition de crédit détaillée",
        rows=[
            "canada",
            "etats unis",
            "europe",
            "asie",
            "detail entreprises",
            "detail particuliers",
            "detail garanties",
            "detail provisions",
        ],
    )
    previous_subset = _table(
        table_id="previous_subset",
        title="Exposition de crédit détaillée",
        rows=["canada", "etats unis", "europe", "asie"],
        page_pdf=2,
    )

    decision = match_decision(current, previous_subset, bank_code="bmo")
    assert decision.is_match is False
    assert decision.reason == "subset_superset_candidate"
    assert decision.coverage_min == 0.5
    assert decision.coverage_gap == 0.5


def test_prefix_only_candidate_becomes_ambiguous_not_added_removed() -> None:
    current = _table(
        table_id="current",
        title="Principales données détaillées",
        rows=[
            "encaisse",
            "depots",
            "titres",
            "prets",
            "engagements derives",
            "passifs negoc",
            "hors bilan",
            "actifs ponderes",
        ],
    )
    previous_wrong = _table(
        table_id="previous_wrong",
        title="Principales données détaillées",
        rows=[
            "encaisse",
            "depots",
            "titres",
            "prets",
            "capital privilegie",
            "capital ordinaire",
            "dividendes",
            "actions emises",
        ],
        page_pdf=2,
    )

    payload = run_strict_intra_section_compare([current], [previous_wrong], bank_code="bmo")
    assert payload["pairs"] == []
    assert payload["added_tables"] == []
    assert payload["removed_tables"] == []
    assert len(payload["suspicious_pairs"]) == 1
    assert len(payload["ambiguous_unmatched_previous"]) == 1
    assert len(payload["ambiguous_unmatched_current"]) == 1
    suspicious = payload["suspicious_pairs"][0]
    assert "prefix_bias" in suspicious["suspicion_flags"]


def test_true_match_beats_prefix_only_competitor() -> None:
    current = _table(
        table_id="current",
        title="Analyse détaillée du portefeuille",
        rows=[
            "canada",
            "etats unis",
            "europe",
            "asie",
            "industrie lourde",
            "services financiers",
            "immobilier commercial",
            "provisions attendues",
        ],
    )
    previous_wrong = _table(
        table_id="previous_wrong",
        title="Analyse détaillée du portefeuille",
        rows=[
            "canada",
            "etats unis",
            "europe",
            "asie",
            "autres actifs",
            "autres passifs",
            "capital",
            "dividendes",
        ],
        page_pdf=2,
    )
    previous_true = _table(
        table_id="previous_true",
        title="Analyse portefeuille détaillée mise à jour",
        rows=[
            "canada",
            "etats unis",
            "europe",
            "asie",
            "industrie lourde",
            "services financiers",
            "immobilier commercial",
            "provisions attendues",
        ],
        page_pdf=3,
    )

    result = match_tables_intra_section(
        [current],
        [previous_wrong, previous_true],
        bank_code="bmo",
    )
    assert len(result["pairs"]) == 1
    pair = result["pairs"][0]
    assert pair["t2_table_id"] == "previous_true"
    assert pair["match_stage"] in {"hungarian_main", "post_hungarian_rematch"}
