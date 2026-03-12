"""Tests for indicator diff normalization in comparison runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction_storage import load_stored_extractions
from app.comparison_runner import (
    _build_clean_to_raw_indicator_lookup,
    _canonical_indicator_key,
    _clean_values_to_raw_display,
    _detect_fusion_split,
    _fuzzy_pair_added_removed,
    _hungarian_pair_added_removed,
    _indicator_diff,
    _ordered_indicator_keys,
    _structural_header_keys_from_rows,
    rapidfuzz_fuzz,
)
from vigilance.compare import run_strict_intra_section_compare
from vigilance.utils.indicator_normalizer import strip_footnote_markers_from_indicator
from vigilance.models.table_models import TableArtifact, get_comparison_indicators
from vigilance.utils.matching_normalizer import _classify_excluded_line

_HAS_RAPIDFUZZ = rapidfuzz_fuzz is not None
_STORED_EXTRACTIONS_DIR = Path("outputs/extractions")


def _table(indicators: list[str]) -> TableArtifact:
    return TableArtifact(
        bank_code="bmo",
        section="capital_management",
        page_pdf=1,
        table_id="tableau_1",
        title="Montant",
        headers=["Indicateur", "Montant"],
        rows=[],
        first_column_indicators=indicators,
        first_column_indicators_raw=indicators,
        extraction_method="vision_full_gpt4o",
        quarter="t1",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )


def _table_with_rows(
    indicators: list[str],
    rows: list[list[str]],
    *,
    raw: list[str] | None = None,
) -> TableArtifact:
    return TableArtifact(
        bank_code="bmo",
        section="capital_management",
        page_pdf=1,
        table_id="tableau_rows",
        title="Montant",
        headers=["Indicateur", "Montant"],
        rows=rows,
        first_column_indicators=indicators,
        first_column_indicators_raw=raw or indicators,
        extraction_method="vision_full_gpt4o",
        quarter="t1",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )


def _load_stored_pair(
    bank_code: str,
    t1_uid: str,
    t2_uid: str,
) -> tuple[TableArtifact, TableArtifact]:
    loaded = load_stored_extractions(bank_code, 2025, _STORED_EXTRACTIONS_DIR)
    assert loaded is not None, f"missing stored extractions for {bank_code}"
    tables_t1, tables_t2, _, _ = loaded
    strict = run_strict_intra_section_compare(tables_t1, tables_t2)
    assert any(
        pair["t1_uid"] == t1_uid and pair["t2_uid"] == t2_uid
        for pair in strict["matched_pairs"]
    ), f"missing matched pair {t1_uid} -> {t2_uid}"

    t1_lookup = {f"{t.section}|{t.table_id}|p{t.page_pdf}": t for t in tables_t1}
    t2_lookup = {f"{t.section}|{t.table_id}|p{t.page_pdf}": t for t in tables_t2}
    return t1_lookup[t1_uid], t2_lookup[t2_uid]


def _old_neighbor_filter(
    candidate_keys: set[str],
    *,
    source_order: list[str],
    target_order: list[str],
) -> set[str]:
    target_pos = {key: idx for idx, key in enumerate(target_order)}
    filtered: set[str] = set()
    for idx, key in enumerate(source_order):
        if key not in candidate_keys:
            continue
        block_start = idx
        while block_start > 0 and source_order[block_start - 1] in candidate_keys:
            block_start -= 1
        block_end = idx
        while (
            block_end + 1 < len(source_order)
            and source_order[block_end + 1] in candidate_keys
        ):
            block_end += 1
        if block_end > block_start:
            continue
        prev_key = next(
            (
                source_order[j]
                for j in range(idx - 1, -1, -1)
                if source_order[j] not in candidate_keys
            ),
            None,
        )
        next_key = next(
            (
                source_order[j]
                for j in range(idx + 1, len(source_order))
                if source_order[j] not in candidate_keys
            ),
            None,
        )
        if (
            prev_key
            and next_key
            and prev_key in target_pos
            and next_key in target_pos
            and target_pos[prev_key] < target_pos[next_key]
        ):
            filtered.add(key)
    return filtered


def _indicator_diff_pre_fix_neighbor_filter(
    t1: TableArtifact,
    t2: TableArtifact,
) -> tuple[list[str], list[str]]:
    left = get_comparison_indicators(t1)
    right = get_comparison_indicators(t2)
    left_all_keys = set(_ordered_indicator_keys(left))
    right_all_keys = set(_ordered_indicator_keys(right))
    left_structural_keys = _structural_header_keys_from_rows(t1) - right_all_keys
    right_structural_keys = _structural_header_keys_from_rows(t2) - left_all_keys

    def _norm(
        values: list[str],
        *,
        structural_keys: set[str],
    ) -> dict[str, str]:
        mapped: dict[str, str] = {}
        for value in values:
            if _classify_excluded_line(value):
                continue
            value_clean = strip_footnote_markers_from_indicator(value)
            key = _canonical_indicator_key(value_clean)
            if key in structural_keys:
                continue
            if key and key not in mapped:
                mapped[key] = value_clean
        return mapped

    left_map = _norm(left, structural_keys=left_structural_keys)
    right_map = _norm(right, structural_keys=right_structural_keys)
    added_keys = set(right_map.keys() - left_map.keys())
    removed_keys = set(left_map.keys() - right_map.keys())
    left_order = _ordered_indicator_keys(left, excluded_keys=left_structural_keys)
    right_order = _ordered_indicator_keys(right, excluded_keys=right_structural_keys)
    added_keys -= _old_neighbor_filter(
        added_keys,
        source_order=right_order,
        target_order=left_order,
    )
    removed_keys -= _old_neighbor_filter(
        removed_keys,
        source_order=left_order,
        target_order=right_order,
    )

    added = sorted(right_map[key] for key in added_keys)
    removed = sorted(left_map[key] for key in removed_keys)
    added, removed, _ = _detect_fusion_split(added, removed)
    return added, removed


def test_clean_to_raw_display_mapping_prefers_raw_text() -> None:
    table = TableArtifact(
        bank_code="bmo",
        section="capital_management",
        page_pdf=1,
        table_id="tableau_raw",
        title="Montant",
        headers=["Indicateur", "Montant"],
        rows=[],
        first_column_indicators=["fonds propre de categorie 1", "autre ligne"],
        first_column_indicators_raw=["fonds propre de categorie 1¹", "autre ligne"],
        extraction_method="vision_full_gpt4o",
        quarter="t1",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )
    lookup = _build_clean_to_raw_indicator_lookup(table)
    display = _clean_values_to_raw_display(["fonds propre de categorie 1"], lookup)
    assert display == ["fonds propre de categorie 1"]


def test_clean_to_raw_display_strips_parenthesized_footnote_markers() -> None:
    table = TableArtifact(
        bank_code="bmo",
        section="capital_management",
        page_pdf=1,
        table_id="tableau_raw_paren",
        title="Montant",
        headers=["Indicateur", "Montant"],
        rows=[],
        first_column_indicators=["titres vendu a decouvert"],
        first_column_indicators_raw=["Titres vendus à découvert(4)"],
        extraction_method="vision_full_gpt4o",
        quarter="t1",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )
    lookup = _build_clean_to_raw_indicator_lookup(table)
    display = _clean_values_to_raw_display(["titres vendu a decouvert"], lookup)
    assert display == ["Titres vendus à découvert"]


def test_clean_to_raw_display_preserves_alignment_when_clean_contains_empty_slots() -> (
    None
):
    table = TableArtifact(
        bank_code="bnc",
        section="risk_management",
        page_pdf=1,
        table_id="tableau_alignment",
        title="LCR",
        headers=["Indicateur", "Valeur"],
        rows=[],
        first_column_indicators=[
            "actif liquide de haute qualite",
            "",
            "ratio de liquidite a court terme",
        ],
        first_column_indicators_raw=[
            "Actifs liquides de haute qualité",
            "Total des HQLA",
            "Ratio de liquidité à court terme (%) (4)",
        ],
        extraction_method="vision_full_gpt4o",
        quarter="t2",
        pdf_path="dummy.pdf",
        footnotes=[],
        content_source="vision_gpt4o",
    )
    lookup = _build_clean_to_raw_indicator_lookup(table)
    display = _clean_values_to_raw_display(["ratio de liquidite a court terme"], lookup)
    assert display == ["Ratio de liquidité à court terme (%)"]


def test_strip_footnote_markers_handles_chained_parenthesized_suffixes() -> None:
    value = "À dividende non cumulatif, série BW (3), (4),"
    assert (
        strip_footnote_markers_from_indicator(value)
        == "À dividende non cumulatif, série BW"
    )


def test_indicator_diff_footnote_markers_no_false_add_remove() -> None:
    """Footnote markers (*, (1), dagger) must not create false added/removed indicators."""
    t1 = _table(
        [
            "Total des fonds propres *",
            "CET1 (1)",
            "Actifs ponderes (2)",
        ]
    )
    t2 = _table(
        [
            "Total des fonds propres",
            "CET1",
            "Actifs ponderes",
        ]
    )

    added, removed, _, _, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []


def test_indicator_diff_order_aware_resolves_aligned_similar_keys() -> None:
    """With order-aware alignment enabled, positionally aligned similar keys are excluded as order_aware_stable."""
    t1 = _table(["Ligne A", "Ligne B", "Ratio liquidite"])
    t2 = _table(["Ligne A", "Ligne B", "Ratio de liquidite"])
    th = {
        "indicator_order_aware_alignment_enabled": True,
        "indicator_order_aware_min_ratio": 0.85,
    }
    added, removed, _, excluded, _ = _indicator_diff(t1, t2, th=th)
    assert excluded.get("order_aware_stable", 0) >= 1
    assert added == [] and removed == []


def test_indicator_diff_parent_child_not_renamed() -> None:
    """Parent/child indicators must not be matched as rename (stay add+remove)."""
    t1 = _table(
        [
            "Total actifs",
            "Autre ligne",
        ]
    )
    t2 = _table(
        [
            "Total actifs ponderes en fonction des risques",
            "Autre ligne",
        ]
    )
    added, removed, _, _, _ = _indicator_diff(t1, t2)
    assert "Total actifs ponderes en fonction des risques" in added
    assert "Total actifs" in removed
    added_restant, removed_restant, renamed_pairs, _ = _hungarian_pair_added_removed(
        removed, added, th={}
    )
    assert not any(
        r == "Total actifs" and a == "Total actifs ponderes en fonction des risques"
        for (r, a) in renamed_pairs
    )
    assert "Total actifs ponderes en fonction des risques" in added_restant
    assert "Total actifs" in removed_restant


def test_indicator_diff_ignores_trailing_note_numbers() -> None:
    t1 = _table(
        [
            "Actions ordinaires 2",
            "Actions privilégiées de catégorie B 3",
            "Autres instruments de capitaux propres 3",
        ]
    )
    t2 = _table(
        [
            "Actions ordinaires",
            "Actions privilégiées de catégorie B 2",
            "Autres instruments de capitaux propres 2",
        ]
    )

    added, removed, _, _, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []


def test_indicator_diff_keeps_semantic_trailing_numbers() -> None:
    t1 = _table(["Série 2"])
    t2 = _table(["Série 3"])

    added, removed, _, _, _ = _indicator_diff(t1, t2)
    assert added == ["Série 3"]
    assert removed == ["Série 2"]


def test_indicator_diff_excludes_totals_and_pure_numbers() -> None:
    t1 = _table(["Actif A", "Total du passif et des capitaux propres", "1"])
    t2 = _table(["Actif A", "Total du passif et des capitaux propres", "26"])
    added, removed, _, excluded, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert excluded.get("total", 0) >= 1
    assert excluded.get("number", 0) >= 1


def test_indicator_diff_keeps_regulatory_totals() -> None:
    """Regulatory indicators like 'Total des fonds propres' must NOT be excluded."""
    regulatory_indicators = [
        "Total des fonds propres réglementaires",
        "Total des actifs pondérés en fonction des risques",
        "Total des expositions",
        "Total des provisions",
        "Total des prêts",
        "Total des dépôts",
        "Total des revenus",
    ]
    t1 = _table(regulatory_indicators)
    t2 = _table([])  # all removed in T2

    added, removed, _, excluded, _ = _indicator_diff(t1, t2)
    # These should appear as removed (not silently excluded as "total")
    assert len(removed) == len(regulatory_indicators)
    assert excluded.get("total", 0) == 0


def test_indicator_diff_excludes_structural_empty_row_headers() -> None:
    t1 = _table_with_rows(
        ["Actions ordinaires", "Série 33"],
        rows=[
            ["Actions ordinaires", "100"],
            ["Série 33", "10"],
        ],
    )
    t2 = _table_with_rows(
        ["Actions ordinaires", "Actions privilégiées de catégorie B", "Série 33"],
        rows=[
            ["Actions ordinaires", "100"],
            ["Actions privilégiées de catégorie B", ""],
            ["Série 33", "10"],
        ],
    )

    added, removed, _, excluded, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert excluded.get("structural", 0) >= 1


def test_neighbor_filter_disabled_by_config() -> None:
    """When neighbor_aligned_filter_enabled=False, singleton add between shared neighbors stays in added."""
    t1 = _table(["Ligne A", "Ligne B", "Ligne C"])
    t2 = _table(["Ligne A", "Nouvelle ligne centrale", "Ligne C"])
    added, removed, _, _, _ = _indicator_diff(
        t1, t2, neighbor_aligned_filter_enabled=False
    )
    assert "Nouvelle ligne centrale" in added
    assert "Ligne B" in removed


def test_indicator_diff_filters_neighbor_aligned_singleton_extraction_miss() -> None:
    t1 = _table(
        [
            "Options sur actions",
            "Droits non acquis",
        ]
    )
    t2 = _table(
        [
            "Options sur actions",
            "Droits acquis",
            "Droits non acquis",
        ]
    )

    added, removed, _, excluded, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert excluded.get("neighbor_aligned", 0) >= 1


def test_indicator_diff_keeps_semantically_distinct_additions_between_neighbors() -> (
    None
):
    """BNC CWB case: sub-lines with distinct tokens (e.g. CWB, acquisition) must stay ADDED."""
    t1 = _table(
        [
            "Émission d'actions ordinaires (y compris au titre du régime d'options d'achat d'actions)",
            "Options de remplacement",
            "Rachat d'actions ordinaires",
        ]
    )
    t2 = _table(
        [
            "Émission d'actions ordinaires (y compris au titre du régime d'options d'achat d'actions)",
            "Émissions d'actions ordinaires relatives à l'acquisition de CWB",
            "Options de remplacement",
            "Options de remplacement relatives à l'acquisition de CWB",
            "Rachat d'actions ordinaires",
        ]
    )

    added, removed, _, excluded, _ = _indicator_diff(t1, t2)
    assert len(added) >= 2
    added_lower = [a.lower() for a in added]
    assert any("cwb" in a for a in added_lower), "CWB-specific indicators must be ADDED"
    assert any("acquisition" in a for a in added_lower)


def test_bnc_stored_capital_pair_keeps_only_true_cwb_additions() -> None:
    t1, t2 = _load_stored_pair(
        "bnc",
        "capital_management|tableau_2|p24",
        "capital_management|tableau_2|p28",
    )

    added, removed, _, excluded, _ = _indicator_diff(t1, t2)
    assert any("cwb" in value for value in added)
    assert not any("actif net des regime" in value for value in removed)
    assert not any("actif des regime de retraite" in value for value in added)
    assert not any(
        "autre element de fonds propre de categorie 1" in value for value in removed
    )
    assert excluded.get("page_reference_table", 0) == 0


def test_bnc_stored_lcr_pair_maps_added_ratio_to_correct_raw_label() -> None:
    t1, t2 = _load_stored_pair(
        "bnc",
        "risk_management|tableau_17|p36",
        "risk_management|tableau_16|p39",
    )

    added, removed, _, excluded, _ = _indicator_diff(t1, t2)
    if added == ["ratio de liquidite a court terme"] and removed == []:
        raw_display = _clean_values_to_raw_display(
            added,
            _build_clean_to_raw_indicator_lookup(t2),
        )
        assert raw_display == ["Ratio de liquidité à court terme (%)"]
    else:
        assert removed == []
        assert excluded.get("near_stable", 0) >= 1 or "ratio de liquidite" in str(added)


def test_page_reference_tables_suppress_indicator_level_noise_on_stored_pair() -> None:
    t1, t2 = _load_stored_pair(
        "bnc",
        "risk_management|tableau_25|p44",
        "risk_management|tableau_24|p47",
    )

    added, removed, had_fusion_split, excluded, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert had_fusion_split is False
    assert excluded.get("page_reference_table", 0) == 1


@pytest.mark.parametrize(
    ("bank_code", "t1_uid", "t2_uid", "expected_fragment"),
    [
        (
            "bnc",
            "capital_management|tableau_2|p24",
            "capital_management|tableau_2|p28",
            "cwb",
        ),
        (
            "rbc",
            "risk_management|tableau_16|p34",
            "risk_management|tableau_17|p40",
            "75 milliard",
        ),
        (
            "td",
            "risk_management|tableau_15|p48",
            "risk_management|tableau_14|p50",
            "court terme",
        ),
        (
            "bmo",
            "risk_management|tableau_27|p47",
            "risk_management|tableau_26|p51",
            "garanti",
        ),
        (
            "cibc",
            "risk_management|tableau_11|p36",
            "risk_management|tableau_10|p43",
            "caraibe",
        ),
        (
            "bns",
            "risk_management|tableau_0|p31",
            "risk_management|tableau_0|p37",
            "titre pri",
        ),
    ],
)
def test_stored_extractions_preserve_real_singleton_additions_systemically(
    bank_code: str,
    t1_uid: str,
    t2_uid: str,
    expected_fragment: str,
) -> None:
    t1, t2 = _load_stored_pair(bank_code, t1_uid, t2_uid)

    added_current, removed_current, _, _, _ = _indicator_diff(t1, t2)
    added_old, removed_old = _indicator_diff_pre_fix_neighbor_filter(t1, t2)

    assert any(expected_fragment in value for value in added_current)
    assert not any(expected_fragment in value for value in added_old)
    assert len(added_current) + len(removed_current) > len(added_old) + len(removed_old)


def test_indicator_diff_resolves_split_label_before_neighbor_filter() -> None:
    t1 = _table(
        [
            "Fonds propres",
            "CET1 catégorie 1 total",
            "Actif pondéré en fonction des risques",
        ]
    )
    t2 = _table(
        [
            "Fonds propres",
            "CET1 catégorie 1",
            "total",
            "Actif pondéré en fonction des risques",
        ]
    )

    added, removed, had_fusion_split, _, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert had_fusion_split is True


def test_indicator_diff_excludes_dont_group_with_duplicated_child_values() -> None:
    t1 = _table_with_rows(
        [
            "Prêts hypothécaires résidentielles productifs, dont :",
            "Prêts présentant un risque pondéré inférieur ou égal à 35 %",
            "Autres actifs",
        ],
        rows=[
            [
                "Prêts hypothécaires résidentielles productifs, dont :",
                "56 697",
                "56 547",
            ],
            [
                "Prêts présentant un risque pondéré inférieur ou égal à 35 %",
                "56 697",
                "56 547",
            ],
            ["Autres actifs", "17 247", "16 844"],
        ],
    )
    t2 = _table_with_rows(
        [
            "Prêts présentant un risque pondéré inférieur ou égal à 35 %",
            "Autres actifs",
        ],
        rows=[
            [
                "Prêts présentant un risque pondéré inférieur ou égal à 35 %",
                "56 697",
                "56 547",
            ],
            ["Autres actifs", "17 247", "16 844"],
        ],
    )

    added, removed, _, excluded, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert excluded.get("structural", 0) >= 1


def test_indicator_diff_keeps_added_block_without_two_shared_neighbors() -> None:
    t1 = _table(
        [
            "Série M – tranche 1",
            "Options sur actions",
        ]
    )
    t2 = _table(
        [
            "Série M – première tranche",
            "Série N – deuxième tranche",
            "Série N – première tranche",
            "Options sur actions",
        ]
    )

    added, removed, _, _, _ = _indicator_diff(t1, t2)
    assert added == ["Série N – deuxième tranche", "Série N – première tranche"]
    assert removed == []


def test_indicator_diff_fusion_split() -> None:
    """Une ligne T2 = concat de deux lignes T1 -> pas 1 add + 2 remove."""
    t1 = _table(["Ligne A", "Ligne B"])
    t2 = _table(["Ligne A Ligne B"])
    added, removed, had_fusion_split, _, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert had_fusion_split is True


def test_detect_fusion_split_returns_had_fusion_split() -> None:
    """_detect_fusion_split retourne had_fusion_split=True quand un merge est effectue."""
    added, removed, had_fusion_split = _detect_fusion_split(
        added=["Ligne A Ligne B"], removed=["Ligne A", "Ligne B"]
    )
    assert had_fusion_split is True
    assert added == []
    assert removed == []


def test_detect_fusion_split_no_merge_returns_false() -> None:
    """_detect_fusion_split retourne had_fusion_split=False quand aucun merge."""
    added, removed, had_fusion_split = _detect_fusion_split(
        added=["Nouvelle ligne"], removed=["Ancienne ligne"]
    )
    assert had_fusion_split is False
    assert added == ["Nouvelle ligne"]
    assert removed == ["Ancienne ligne"]


def test_fuzzy_pair_added_removed_empty_lists() -> None:
    """_fuzzy_pair_added_removed with empty added or removed returns unchanged and no renames."""
    added, removed, renames = _fuzzy_pair_added_removed([], ["x"], "td")
    assert added == []
    assert removed == ["x"]
    assert renames == []

    added, removed, renames = _fuzzy_pair_added_removed(["y"], [], "td")
    assert added == ["y"]
    assert removed == []
    assert renames == []


def test_fuzzy_pair_added_removed_without_rapidfuzz_returns_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When rapidfuzz is unavailable, added/removed are unchanged and renames is empty."""
    import app.comparison_runner as runner_mod

    monkeypatch.setattr(runner_mod, "rapidfuzz_fuzz", None)
    added_in = ["tresorerie et montants a recevoir"]
    removed_in = ["en millions de dollars canadiens et montants a recevoir"]
    added, removed, renames = _fuzzy_pair_added_removed(added_in, removed_in, "td")
    assert added == added_in
    assert removed == removed_in
    assert renames == []


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_fuzzy_pair_added_removed_pairs_reformulation() -> None:
    """Reformulation pair is matched as rename and removed from added/removed (TD threshold)."""
    # Pair with high token_set_ratio (one string extends the other) so it passes 0.85/0.88
    added_in = ["Capitaux propres (en millions de dollars)"]
    removed_in = ["Capitaux propres"]
    added, removed, renames = _fuzzy_pair_added_removed(added_in, removed_in, "td")
    assert added == []
    assert removed == []
    assert len(renames) == 1
    assert renames[0] == (removed_in[0], added_in[0])


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_fuzzy_pair_added_removed_greedy_one_to_one() -> None:
    """Only one-to-one pairing; unrelated items stay in added/removed."""
    # First pair matches (T2 extends T1); second pair does not
    added_in = ["Capitaux propres (en millions)", "autre ligne nouvelle"]
    removed_in = ["Capitaux propres", "autre ancienne"]
    added, removed, renames = _fuzzy_pair_added_removed(added_in, removed_in, "td")
    assert len(renames) == 1
    assert renames[0][0] == removed_in[0]
    assert renames[0][1] == added_in[0]
    assert "autre ligne nouvelle" in added
    assert "autre ancienne" in removed


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_hungarian_pair_added_removed_determinism() -> None:
    """Same input produces same output across multiple calls."""
    removed = ["Ratio CET1", "RWA total"]
    added = ["Ratio CET1 (Bâle III)", "Actifs pondérés aux risques"]
    out1 = _hungarian_pair_added_removed(removed, added, th={})
    out2 = _hungarian_pair_added_removed(removed, added, th={})
    a1, r1, pairs1, _ = out1
    a2, r2, pairs2, _ = out2
    assert set(a1) == set(a2)
    assert set(r1) == set(r2)
    assert set((x[0], x[1]) for x in pairs1) == set((x[0], x[1]) for x in pairs2)


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_hungarian_pair_added_removed_one_to_one() -> None:
    """Rename assignment is 1-to-1; no duplicate added or removed in pairs."""
    removed = ["Label A", "Label B"]
    added = ["Label A reformulated", "Label B extended"]
    _, _, pairs, _ = _hungarian_pair_added_removed(removed, added, th={})
    added_in_pairs = [p[1] for p in pairs]
    removed_in_pairs = [p[0] for p in pairs]
    assert len(added_in_pairs) == len(set(added_in_pairs))
    assert len(removed_in_pairs) == len(set(removed_in_pairs))
    assert len(pairs) <= min(len(removed), len(added))


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_hungarian_pair_added_removed_gating_rejects_absurd() -> None:
    """Gating rejects absurd matches (no token overlap, no shared acronym)."""
    removed = ["Total des actifs"]
    added = ["Ratio de levier au sens de Bâle III"]
    added_rest, removed_rest, pairs, _ = _hungarian_pair_added_removed(
        removed, added, th={}
    )
    assert len(pairs) == 0
    assert "Total des actifs" in removed_rest
    assert "Ratio de levier au sens de Bâle III" in added_rest
