"""Regression tests for same-page multi-table matching (top vs lower, local order)."""

from __future__ import annotations

from vigilance.compare import run_strict_intra_section_compare
from vigilance.compare.table_pairing_engine import (
    _candidate_score,
    _eligible_table_views,
)
from vigilance.models.table_models import TableArtifact


def _table(
    *,
    table_id: str,
    section: str = "risk_management",
    title: str | None = "Tableau",
    indicators: list[str],
    table_number: str | None = None,
    page: int = 1,
    bbox: list[float] | None = None,
    table_index_on_page: int | None = None,
    tables_on_page: int | None = None,
    page_local_role: str | None = None,
) -> TableArtifact:
    return TableArtifact(
        bank_code="bnc",
        section=section,
        page_pdf=page,
        table_id=table_id,
        title=title or "",
        headers=["Indicateur", "Valeur"],
        rows=[[label, "1"] for label in indicators],
        first_column_indicators=list(indicators),
        first_column_indicators_raw=list(indicators),
        extraction_method="vision_full_gpt4o",
        quarter="t2-2025",
        table_number=table_number,
        footnotes=[],
        content_source="vision_gpt4o",
        bbox=bbox,
        table_index_on_page=table_index_on_page,
        tables_on_page=tables_on_page,
        page_local_role=page_local_role,
    )


def test_same_page_two_tables_pair_by_local_order() -> None:
    """Two tables on T1 page 1 and two on T2 page 1 should pair by index (top with top, lower with lower)."""
    t1_top = _table(
        table_id="t1_top",
        title="T29 Ratio structurel",
        indicators=["Fonds propres", "Dépôts stables", "Financement stable disponible"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t1_lower = _table(
        table_id="t1_lower",
        title="Total financement stable requis",
        indicators=["Actifs liquides HQLA", "Ratio niveau 1", "Ratio niveau 2", "Autres actifs"],
        page=1,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    t2_top = _table(
        table_id="t2_top",
        title="T29 Ratio structurel",
        indicators=["Fonds propres", "Dépôts stables", "Financement stable disponible"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t2_lower = _table(
        table_id="t2_lower",
        title="Total financement stable requis",
        indicators=["Actifs liquides HQLA", "Ratio niveau 1", "Ratio niveau 2", "Autres actifs"],
        page=1,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    result = run_strict_intra_section_compare([t1_top, t1_lower], [t2_top, t2_lower])
    assert len(result["pairs"]) == 2
    t1_ids = {p["t1_table_id"] for p in result["pairs"]}
    t2_ids = {p["t2_table_id"] for p in result["pairs"]}
    assert t1_ids == {"t1_top", "t1_lower"}
    assert t2_ids == {"t2_top", "t2_lower"}
    for p in result["pairs"]:
        if p["t1_table_id"] == "t1_top":
            assert p["t2_table_id"] == "t2_top"
        if p["t1_table_id"] == "t1_lower":
            assert p["t2_table_id"] == "t2_lower"


def test_same_page_different_index_penalized() -> None:
    """T2 has one table (index 1). T1 has two (index 1 and 2). Wrong-index T1 has higher indicator overlap with T2; correct index should still win."""
    t1_first = _table(
        table_id="t1_first",
        title="Ratio structurel",
        indicators=["Fonds propres", "Dépôts", "Total"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.4],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t1_second = _table(
        table_id="t1_second",
        title="Detail",
        indicators=["Ratio niveau 1", "Ratio niveau 2", "Liquidite", "Total actifs"],
        page=1,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    t2_only = _table(
        table_id="t2_only",
        title="Ratio structurel",
        indicators=["Fonds propres", "Dépôts", "Total"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.4],
        table_index_on_page=1,
        tables_on_page=1,
        page_local_role="single",
    )
    result = run_strict_intra_section_compare([t1_first, t1_second], [t2_only])
    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["t1_table_id"] == "t1_first"
    assert result["pairs"][0]["t2_table_id"] == "t2_only"


def test_near_page_same_index_still_preferred() -> None:
    """T1 top on page 49, T2 top on page 50 (drift). Correct top-top match should win."""
    t1_top = _table(
        table_id="t1_top",
        indicators=["A", "B", "C"],
        page=49,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t1_lower = _table(
        table_id="t1_lower",
        indicators=["X", "Y", "Z"],
        page=49,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    t2_top = _table(
        table_id="t2_top",
        indicators=["A", "B", "C"],
        page=50,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t2_lower = _table(
        table_id="t2_lower",
        indicators=["X", "Y", "Z"],
        page=50,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    result = run_strict_intra_section_compare([t1_top, t1_lower], [t2_top, t2_lower])
    assert len(result["pairs"]) == 2
    for p in result["pairs"]:
        if p["t2_table_id"] == "t2_top":
            assert p["t1_table_id"] == "t1_top"
        if p["t2_table_id"] == "t2_lower":
            assert p["t1_table_id"] == "t1_lower"


def test_three_tables_same_page_pair_1_to_1_by_order() -> None:
    """Three tables on same page T1 and T2 pair by index (same indicators per slot)."""
    indicators_slot1 = ["Fonds propres", "Dette"]
    indicators_slot2 = ["Actifs liquides", "HQLA"]
    indicators_slot3 = ["Exposition", "Risque"]
    t1_1 = _table(table_id="t1_1", indicators=indicators_slot1, page=1, bbox=[0.1, 0.1, 0.9, 0.3], table_index_on_page=1, tables_on_page=3, page_local_role="first")
    t1_2 = _table(table_id="t1_2", indicators=indicators_slot2, page=1, bbox=[0.1, 0.35, 0.9, 0.55], table_index_on_page=2, tables_on_page=3, page_local_role="middle")
    t1_3 = _table(table_id="t1_3", indicators=indicators_slot3, page=1, bbox=[0.1, 0.6, 0.9, 0.85], table_index_on_page=3, tables_on_page=3, page_local_role="last")
    t2_1 = _table(table_id="t2_1", indicators=indicators_slot1, page=1, bbox=[0.1, 0.1, 0.9, 0.3], table_index_on_page=1, tables_on_page=3, page_local_role="first")
    t2_2 = _table(table_id="t2_2", indicators=indicators_slot2, page=1, bbox=[0.1, 0.35, 0.9, 0.55], table_index_on_page=2, tables_on_page=3, page_local_role="middle")
    t2_3 = _table(table_id="t2_3", indicators=indicators_slot3, page=1, bbox=[0.1, 0.6, 0.9, 0.85], table_index_on_page=3, tables_on_page=3, page_local_role="last")
    result = run_strict_intra_section_compare([t1_1, t1_2, t1_3], [t2_1, t2_2, t2_3])
    assert len(result["pairs"]) == 3
    for p in result["pairs"]:
        t1_id = p["t1_table_id"]
        t2_id = p["t2_table_id"]
        idx1 = int(t1_id.split("_")[1])
        idx2 = int(t2_id.split("_")[1])
        assert idx1 == idx2


def test_page_order_bonus_in_candidate_score() -> None:
    """_candidate_score gives page_local_order_bonus and role bonus when structure matches."""
    section_freq: dict = {}
    section_counts: dict = {"risk_management": 2}

    t1 = _table(
        table_id="t1",
        indicators=["A", "B"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t2 = _table(
        table_id="t2",
        indicators=["A", "B"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    v1, _ = _eligible_table_views([t1], section_frequencies=section_freq, section_counts=section_counts)
    v2, _ = _eligible_table_views([t2], section_frequencies=section_freq, section_counts=section_counts)
    assert v1 and v2
    score = _candidate_score(v1[0], v2[0])
    assert score.page_local_order_bonus == 0.20
    assert score.page_local_role_bonus == 0.06

    t1_other = _table(
        table_id="t1_other",
        indicators=["X", "Y"],
        page=1,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    v1_other, _ = _eligible_table_views([t1_other], section_frequencies=section_freq, section_counts=section_counts)
    assert v1_other
    score_wrong = _candidate_score(v1_other[0], v2[0])
    assert score_wrong.page_local_order_bonus == -0.15

    t2_p2 = _table(
        table_id="t2_p2",
        indicators=["A", "B"],
        page=3,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=1,
        page_local_role="single",
    )
    v2_p2, _ = _eligible_table_views([t2_p2], section_frequencies=section_freq, section_counts=section_counts)
    assert v2_p2
    score_diff_page = _candidate_score(v1[0], v2_p2[0])
    assert score_diff_page.page_local_order_bonus == 0.0


def test_missing_title_still_pairs_by_local_structure_and_content() -> None:
    """When title is missing, pairing still succeeds by table_index_on_page and indicators."""
    t1_top = _table(
        table_id="t1_top",
        title=None,
        indicators=["Ratio CET1", "Ratio Tier 2"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t1_lower = _table(
        table_id="t1_lower",
        title=None,
        indicators=["LCR", "NSFR"],
        page=1,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    t2_top = _table(
        table_id="t2_top",
        title=None,
        indicators=["Ratio CET1", "Ratio Tier 2"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t2_lower = _table(
        table_id="t2_lower",
        title=None,
        indicators=["LCR", "NSFR"],
        page=1,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    result = run_strict_intra_section_compare([t1_top, t1_lower], [t2_top, t2_lower])
    assert len(result["pairs"]) == 2
    for p in result["pairs"]:
        if p["t1_table_id"] == "t1_top":
            assert p["t2_table_id"] == "t2_top"
        if p["t1_table_id"] == "t1_lower":
            assert p["t2_table_id"] == "t2_lower"


def test_wrong_title_on_lower_table_does_not_override_page_local_structure() -> None:
    """Lower table with wrong title (same as top) should still match lower by index."""
    t1_top = _table(
        table_id="t1_top",
        title="T29 Ratio structurel",
        indicators=["Fonds propres", "Dépôts"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t1_lower = _table(
        table_id="t1_lower",
        title="T29 Ratio structurel",
        indicators=["Actifs liquides", "Ratio niveau 1"],
        page=1,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    t2_top = _table(
        table_id="t2_top",
        title="T29 Ratio structurel",
        indicators=["Fonds propres", "Dépôts"],
        page=1,
        bbox=[0.1, 0.2, 0.9, 0.45],
        table_index_on_page=1,
        tables_on_page=2,
        page_local_role="first",
    )
    t2_lower = _table(
        table_id="t2_lower",
        title="Total financement stable",
        indicators=["Actifs liquides", "Ratio niveau 1"],
        page=1,
        bbox=[0.1, 0.5, 0.9, 0.85],
        table_index_on_page=2,
        tables_on_page=2,
        page_local_role="last",
    )
    result = run_strict_intra_section_compare([t1_top, t1_lower], [t2_top, t2_lower])
    assert len(result["pairs"]) == 2
    for p in result["pairs"]:
        if p["t2_table_id"] == "t2_top":
            assert p["t1_table_id"] == "t1_top"
        if p["t2_table_id"] == "t2_lower":
            assert p["t1_table_id"] == "t1_lower"
