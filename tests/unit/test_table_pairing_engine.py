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
    headers: list[str] | None = None,
) -> TableArtifact:
    raw_title = title if title is not None else ""
    return TableArtifact(
        bank_code="bnc",
        section=section,
        page_pdf=page,
        table_id=table_id,
        title=raw_title,
        headers=headers or ["Indicateur", "Valeur"],
        rows=[[label, "1"] for label in indicators],
        first_column_indicators=list(indicators),
        first_column_indicators_raw=list(indicators),
        extraction_method="vision_full_gpt4o",
        quarter="t2-2025",
        table_number=table_number,
        footnotes=[],
        content_source="vision_gpt4o",
    )


def test_public_pairing_footnote_only_variants_full_overlap() -> None:
    """Footnote markers (*, (1), etc.) must not reduce indicator overlap for pairing."""
    t1 = _table(
        table_id="t1",
        title=None,
        table_number=None,
        indicators=["Total *", "CET1 (1)", "Tier 2 (2)"],
    )
    t2 = _table(
        table_id="t2",
        title=None,
        table_number=None,
        indicators=["Total", "CET1", "Tier 2"],
        page=2,
    )

    result = run_strict_intra_section_compare([t1], [t2])

    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["t1_table_id"] == "t1"
    assert result["pairs"][0]["t2_table_id"] == "t2"


def test_public_pairing_matches_without_title_or_number_when_indicators_are_distinctive() -> None:
    t1 = _table(
        table_id="t1",
        title=None,
        table_number=None,
        indicators=["Ajustements CVA", "Risque de marche standardise", "Levier total"],
    )
    t2 = _table(
        table_id="t2",
        title=None,
        table_number=None,
        indicators=["Ajustements CVA", "Risque de marche standardise", "Levier total"],
        page=2,
    )

    result = run_strict_intra_section_compare([t1], [t2])

    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["t1_table_id"] == "t1"
    assert result["pairs"][0]["t2_table_id"] == "t2"
    assert result["added_tables"] == []
    assert result["removed_tables"] == []


def test_public_pairing_same_number_does_not_win_without_content_alignment() -> None:
    t1 = _table(
        table_id="t1",
        section="capital_management",
        title="TABLEAU 12 - Fonds propres",
        table_number="12",
        indicators=["CET1", "AT1", "Tier 2"],
    )
    t2 = _table(
        table_id="t2",
        section="capital_management",
        title="TABLEAU 12 - Actifs ponderes",
        table_number="12",
        indicators=["RWA", "Levier", "LCR"],
        page=2,
    )

    result = run_strict_intra_section_compare([t1], [t2])

    assert result["pairs"] == []
    assert len(result["added_tables"]) == 0
    assert len(result["removed_tables"]) == 0
    assert len(result["unmatched_ambiguous_t1"]) == 1
    assert len(result["unmatched_ambiguous_t2"]) == 1


def test_public_pairing_high_global_overlap_but_generic_family_becomes_ambiguous() -> None:
    t1 = _table(
        table_id="t1",
        title="Exposition geographique",
        indicators=["Canada", "Etats-Unis", "Europe", "Total"],
    )
    t2 = _table(
        table_id="t2",
        title="Exposition par region",
        indicators=["Canada", "Etats-Unis", "Europe", "Total"],
        page=3,
    )

    result = run_strict_intra_section_compare([t1], [t2])

    assert result["pairs"] == []
    assert result["added_tables"] == []
    assert result["removed_tables"] == []
    assert len(result["ambiguous_pairs"]) == 1
    assert result["ambiguous_pairs"][0]["reason_codes"][0] == "family_similarity_without_distinctive_anchor"


def test_public_pairing_exposes_comparable_counts_and_coverage() -> None:
    comparable_t1 = _table(
        table_id="t1",
        indicators=["NSFR", "LCR"],
    )
    comparable_t2 = _table(
        table_id="t2",
        indicators=["NSFR", "LCR"],
        page=2,
    )
    ineligible_t1 = TableArtifact(
        bank_code="bnc",
        section="risk_management",
        page_pdf=4,
        table_id="legacy",
        title="Legacy",
        headers=["Indicateur", "Valeur"],
        rows=[["X", "1"]],
        first_column_indicators=["X"],
        extraction_method="docling",
        quarter="t1-2025",
        footnotes=[],
    )

    result = run_strict_intra_section_compare([comparable_t1, ineligible_t1], [comparable_t2])

    assert result["tables_comparable_t1"] == 1
    assert result["tables_comparable_t2"] == 1
    assert result["pairing_coverage"] == 1.0
    assert result["matching_diagnostics"]["tables_comparable_t1"] == 1
    assert result["matching_diagnostics"]["pairing_coverage"] == 1.0
