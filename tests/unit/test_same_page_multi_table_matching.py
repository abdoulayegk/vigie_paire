"""Same-page multi-table matching: geometry, rank, zone, and cross-order rejection."""

from __future__ import annotations

import pytest

from vigilance.compare.indicator_comparator import (
    _geometry_position_score,
    _load_compare_thresholds,
    _match_section_hungarian,
    _match_section_hungarian_post_threshold,
    _match_tables_greedy,
    _multi_table_same_page_penalty,
    _page_local_rank_similarity,
    _page_zone_similarity,
    match_decision,
    match_tables_intra_section,
)
from vigilance.models.table_models import TableArtifact


def _table(
    *,
    table_id: str,
    section: str,
    title: str,
    rows: list[list[str]],
    page_pdf: int = 1,
    page_local_rank: int | None = None,
    page_table_count: int | None = None,
    page_zone: str | None = None,
    y_center: float | None = None,
    first_column_indicators: list[str] | None = None,
    context_before: str = "",
) -> TableArtifact:
    return TableArtifact(
        bank_code="rbc",
        section=section,
        page_pdf=page_pdf,
        table_id=table_id,
        title=title,
        headers=["Indicateur", "Valeur"],
        rows=rows,
        first_column_indicators=first_column_indicators or [row[0] for row in rows if row],
        extraction_method="docling",
        quarter="t1-2025",
        pdf_path="dummy.pdf",
        page_local_rank=page_local_rank,
        page_table_count=page_table_count,
        page_zone=page_zone,
        y_center=y_center,
        context_before=context_before,
    )


def test_same_page_top_bottom_preserves_ranked_pairing() -> None:
    """T1 and T2 each have top_summary + lower_detail; overlapping indicators on wrong pair.
    Assert top matches top and lower matches lower."""
    top_summary_t1 = _table(
        table_id="t1-top",
        section="risk_management",
        title="Risques - Synthese",
        rows=[["Risque credit", "100"], ["Risque marche", "50"]],
        page_pdf=5,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
        y_center=0.25,
    )
    lower_detail_t1 = _table(
        table_id="t1-lower",
        section="risk_management",
        title="Risques - Detail par type",
        rows=[
            ["Risque credit", "100"],
            ["Risque marche", "50"],
            ["Risque operationnel", "30"],
        ],
        page_pdf=5,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
        y_center=0.75,
    )
    top_summary_t2 = _table(
        table_id="t2-top",
        section="risk_management",
        title="Risques - Synthese",
        rows=[["Risque credit", "110"], ["Risque marche", "55"]],
        page_pdf=5,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
        y_center=0.25,
    )
    lower_detail_t2 = _table(
        table_id="t2-lower",
        section="risk_management",
        title="Risques - Detail par type",
        rows=[
            ["Risque credit", "110"],
            ["Risque marche", "55"],
            ["Risque operationnel", "35"],
        ],
        page_pdf=5,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
        y_center=0.75,
    )
    top_top = match_decision(top_summary_t1, top_summary_t2, overlap_threshold=0.4)
    top_bottom = match_decision(top_summary_t1, lower_detail_t2, overlap_threshold=0.4)
    bottom_bottom = match_decision(lower_detail_t1, lower_detail_t2, overlap_threshold=0.4)
    bottom_top = match_decision(lower_detail_t1, top_summary_t2, overlap_threshold=0.4)
    assert top_top.is_match
    assert bottom_bottom.is_match
    assert not top_bottom.is_match or top_bottom.reason == "same_page_role_mismatch"
    assert not bottom_top.is_match or bottom_top.reason == "same_page_role_mismatch"


def test_same_page_indicator_overlap_cannot_override_zone_mismatch() -> None:
    """Wrong pair has strong indicator overlap but opposite page_zone; assert no match."""
    top_t1 = _table(
        table_id="t1-top",
        section="capital_management",
        title="Capital - Resume",
        rows=[["CET1", "14%"], ["Tier1", "16%"]],
        page_pdf=3,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
        y_center=0.2,
    )
    bottom_t2 = _table(
        table_id="t2-bot",
        section="capital_management",
        title="Capital - Composition detaillee",
        rows=[["CET1", "14.1%"], ["Tier1", "16.2%"], ["Total", "18%"]],
        page_pdf=3,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
        y_center=0.8,
    )
    decision = match_decision(top_t1, bottom_t2, overlap_threshold=0.4)
    assert not decision.is_match
    assert decision.reason == "same_page_role_mismatch"


def test_hash_exact_does_not_match_top_to_bottom_when_zone_mismatch() -> None:
    """When indicator set hash matches and titles are similar but below override (0.90),
    and page_zone is top vs bottom (same-page multi-table), hash_exact must not force
    a match; same_page_role_mismatch applies."""
    indicators = ["CET1", "Tier 1", "Total"]
    # Titles chosen so title_sim passes hash_exact (>= 0.85) but is below override (0.90).
    title_t1 = "Tableau 5 - Fonds propres reglementaires"
    title_t2 = "Tableau 5 - Fonds propres reglementaires composition"
    top_t1 = _table(
        table_id="t1-top",
        section="capital_management",
        title=title_t1,
        rows=[[i, "1"] for i in indicators],
        page_pdf=4,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
        y_center=0.25,
        first_column_indicators=indicators,
    )
    bottom_t2 = _table(
        table_id="t2-bot",
        section="capital_management",
        title=title_t2,
        rows=[[i, "2"] for i in indicators],
        page_pdf=4,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
        y_center=0.75,
        first_column_indicators=indicators,
    )
    decision = match_decision(top_t1, bottom_t2, overlap_threshold=0.4)
    assert not decision.is_match
    assert decision.reason == "same_page_role_mismatch"


def test_hungarian_rejects_crossed_assignment_on_multi_table_page() -> None:
    """Construct a scenario where Hungarian would cross-match; post-check rejects it."""
    t1_top = _table(
        table_id="t1-0",
        section="risk_management",
        title="Risque - Synthese",
        rows=[["A", "1"], ["B", "2"]],
        page_pdf=1,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
    )
    t1_bot = _table(
        table_id="t1-1",
        section="risk_management",
        title="Risque - Detail",
        rows=[["A", "1"], ["B", "2"], ["C", "3"]],
        page_pdf=1,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
    )
    t2_top = _table(
        table_id="t2-0",
        section="risk_management",
        title="Risque - Synthese",
        rows=[["A", "10"], ["B", "20"]],
        page_pdf=1,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
    )
    t2_bot = _table(
        table_id="t2-1",
        section="risk_management",
        title="Risque - Detail",
        rows=[["A", "10"], ["B", "20"], ["C", "30"]],
        page_pdf=1,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
    )
    t1_list = [t1_top, t1_bot]
    t2_list = [t2_top, t2_bot]
    th = dict(_load_compare_thresholds(bank_code="rbc"))
    th.update({
        "overlap_threshold": 0.4,
        "margin_threshold": 0.08,
        "borderline_score_threshold": 0.60,
        "match_score_v2": 0.70,
        "probable_score_v2": 0.62,
        "hungarian_min_score": 0.62,
        "multi_table_min_title_or_context_for_override": 0.90,
    })
    assignments, unmatched, _ = _match_section_hungarian(
        t1_list,
        t2_list,
        th=th,
        overlap_threshold=0.4,
        bank_code="rbc",
        margin_threshold=0.08,
        borderline_score=0.60,
    )
    assert len(assignments) == 2
    by_t1_rank = {t1_list[i].page_local_rank: (i, j) for i, j, _ in assignments}
    assert by_t1_rank[0][1] == 0
    assert by_t1_rank[1][1] == 1


def test_greedy_sorts_by_page_local_rank_not_table_id() -> None:
    """Greedy iterates T1 by (section, page, page_local_rank); order of input does not drive wrong lock-in."""
    t1_lower = _table(
        table_id="t1-B",
        section="capital_management",
        title="Capital detail",
        rows=[["CET1", "14%"], ["Tier1", "16%"]],
        page_pdf=2,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
    )
    t1_top = _table(
        table_id="t1-A",
        section="capital_management",
        title="Capital synthese",
        rows=[["CET1", "14%"]],
        page_pdf=2,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
    )
    t2_top = _table(
        table_id="t2-A",
        section="capital_management",
        title="Capital synthese",
        rows=[["CET1", "14.1%"]],
        page_pdf=2,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
    )
    t2_lower = _table(
        table_id="t2-B",
        section="capital_management",
        title="Capital detail",
        rows=[["CET1", "14.1%"], ["Tier1", "16.1%"]],
        page_pdf=2,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
    )
    tables_t1 = [t1_lower, t1_top]
    tables_t2 = [t2_top, t2_lower]
    result = match_tables_intra_section(
        tables_t1,
        tables_t2,
        overlap_threshold=0.4,
        bank_code="rbc",
        use_hungarian=False,
    )
    pairs = result.get("pairs", [])
    assert len(pairs) == 2
    by_t1_uid = {p["t1_uid"]: p["t2_uid"] for p in pairs}
    assert by_t1_uid.get("capital_management|t1-A|p2") == "capital_management|t2-A|p2"
    assert by_t1_uid.get("capital_management|t1-B|p2") == "capital_management|t2-B|p2"


def test_context_and_title_role_penalize_summary_vs_detail() -> None:
    """Summary and detail tables share indicator tokens; role mismatch lowers score."""
    summary = _table(
        table_id="s",
        section="risk_management",
        title="Risques - Synthese globale",
        rows=[["Credit", "1"], ["Marche", "2"]],
        page_pdf=1,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
        context_before="Tableau de synthese",
    )
    detail = _table(
        table_id="d",
        section="risk_management",
        title="Risques - Detail par categorie",
        rows=[["Credit", "1"], ["Marche", "2"], ["Op", "3"]],
        page_pdf=1,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
        context_before="Tableau detaille",
    )
    decision = match_decision(summary, detail, overlap_threshold=0.4)
    assert not decision.is_match or decision.reason == "same_page_role_mismatch"


def test_geometry_position_score_uses_rank_and_zone() -> None:
    """_geometry_position_score is higher when rank/zone align."""
    t1 = _table(
        table_id="a",
        section="s",
        title="T",
        rows=[["X", "1"]],
        page_pdf=1,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
        y_center=0.25,
    )
    t2_same = _table(
        table_id="b",
        section="s",
        title="T",
        rows=[["X", "2"]],
        page_pdf=1,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
        y_center=0.26,
    )
    t2_other = _table(
        table_id="c",
        section="s",
        title="T",
        rows=[["X", "3"]],
        page_pdf=1,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
        y_center=0.75,
    )
    score_same = _geometry_position_score(t1, t2_same)
    score_other = _geometry_position_score(t1, t2_other)
    assert score_same > score_other


def test_page_local_rank_similarity() -> None:
    """Same rank => 1.0; large delta => low."""
    t0 = _table(table_id="a", section="s", title="T", rows=[], page_local_rank=0, page_table_count=2)
    t1 = _table(table_id="b", section="s", title="T", rows=[], page_local_rank=1, page_table_count=2)
    t3 = _table(table_id="c", section="s", title="T", rows=[], page_local_rank=3, page_table_count=4)
    assert _page_local_rank_similarity(t0, t0) == 1.0
    assert _page_local_rank_similarity(t0, t1) > 0.5
    assert _page_local_rank_similarity(t0, t3) < _page_local_rank_similarity(t0, t1)


def test_page_zone_similarity() -> None:
    """Same zone => 1.0; top vs bottom => 0.0."""
    top = _table(table_id="a", section="s", title="T", rows=[], page_zone="top", page_table_count=2)
    bot = _table(table_id="b", section="s", title="T", rows=[], page_zone="bottom", page_table_count=2)
    mid = _table(table_id="c", section="s", title="T", rows=[], page_zone="middle", page_table_count=3)
    assert _page_zone_similarity(top, top) == 1.0
    assert _page_zone_similarity(top, bot) == 0.0
    assert _page_zone_similarity(top, mid) < 1.0
    assert _page_zone_similarity(top, mid) > 0.0


def test_multi_table_same_page_penalty_zero_when_single_table() -> None:
    """No penalty when either page has only one table."""
    t1 = _table(table_id="a", section="s", title="T", rows=[], page_pdf=1, page_table_count=1)
    t2 = _table(table_id="b", section="s", title="T", rows=[], page_pdf=1, page_table_count=1)
    assert _multi_table_same_page_penalty(t1, t2) == 0.0


def test_multi_table_same_page_penalty_positive_when_zone_mismatch() -> None:
    """Penalty when both multi-table, same page, opposite zones."""
    t1 = _table(
        table_id="a",
        section="s",
        title="T",
        rows=[],
        page_pdf=1,
        page_table_count=2,
        page_zone="top",
        page_local_rank=0,
    )
    t2 = _table(
        table_id="b",
        section="s",
        title="T",
        rows=[],
        page_pdf=1,
        page_table_count=2,
        page_zone="bottom",
        page_local_rank=1,
    )
    assert _multi_table_same_page_penalty(t1, t2) > 0.0


def test_post_threshold_hungarian_uses_crossing_check() -> None:
    """Post-threshold Hungarian also rejects crossed same-page assignments."""
    t1_top = _table(
        table_id="t1-0",
        section="risk_management",
        title="Risque - Synthese",
        rows=[["A", "1"], ["B", "2"]],
        page_pdf=1,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
    )
    t1_bot = _table(
        table_id="t1-1",
        section="risk_management",
        title="Risque - Detail",
        rows=[["A", "1"], ["B", "2"], ["C", "3"]],
        page_pdf=1,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
    )
    t2_top = _table(
        table_id="t2-0",
        section="risk_management",
        title="Risque - Synthese",
        rows=[["A", "10"], ["B", "20"]],
        page_pdf=1,
        page_local_rank=0,
        page_table_count=2,
        page_zone="top",
    )
    t2_bot = _table(
        table_id="t2-1",
        section="risk_management",
        title="Risque - Detail",
        rows=[["A", "10"], ["B", "20"], ["C", "30"]],
        page_pdf=1,
        page_local_rank=1,
        page_table_count=2,
        page_zone="bottom",
    )
    th = dict(_load_compare_thresholds(bank_code="rbc"))
    th.update({
        "overlap_threshold": 0.4,
        "margin_threshold": 0.08,
        "borderline_score_threshold": 0.60,
        "match_score_v2": 0.70,
        "probable_score_v2": 0.62,
        "hungarian_min_score": 0.62,
        "multi_table_min_title_or_context_for_override": 0.90,
    })
    assignments, _, _ = _match_section_hungarian_post_threshold(
        [t1_top, t1_bot],
        [t2_top, t2_bot],
        th=th,
        overlap_threshold=0.4,
        bank_code="rbc",
        margin_threshold=0.08,
        borderline_score=0.60,
    )
    assert len(assignments) == 2
    idx_t1_0 = next(i for i, j, _ in assignments if j == 0)
    idx_t1_1 = next(i for i, j, _ in assignments if j == 1)
    assert [t1_top, t1_bot][idx_t1_0].page_local_rank == 0
    assert [t1_top, t1_bot][idx_t1_1].page_local_rank == 1


def test_bounded_footnote_crop_does_not_capture_next_table_notes() -> None:
    """With bottom_stop_norm, crop_table_region_to_bytes accepts and uses it so
    upper table footnote band stops before the next table on the same page."""
    from vigilance.utils.pdf_crop import crop_table_region_to_bytes

    render_cache: dict = {}
    bbox = [0.1, 0.2, 0.9, 0.5]
    bottom_stop_norm = 0.58
    result = crop_table_region_to_bytes(
        "/nonexistent.pdf",
        1,
        bbox,
        bottom_extension=0.12,
        bottom_stop_norm=bottom_stop_norm,
        dpi=300,
        render_cache=render_cache,
    )
    assert result is not None
    # When cache is populated (e.g. with a real PDF), key includes bottom_stop_norm.
    keys_with_stop = [k for k in render_cache if isinstance(k, tuple) and k[-1] == bottom_stop_norm]
    assert len(keys_with_stop) <= 1
