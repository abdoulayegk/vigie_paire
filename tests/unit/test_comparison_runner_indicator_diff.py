"""Tests for indicator diff normalization in comparison runner."""

from __future__ import annotations

import pytest

from app.comparison_runner import (
    _build_clean_to_raw_indicator_lookup,
    _clean_values_to_raw_display,
    _detect_fusion_split,
    _fuzzy_pair_added_removed,
    _hungarian_pair_added_removed,
    _indicator_diff,
    _strip_footnote_markers_from_indicator,
    rapidfuzz_fuzz,
)
from vigilance.models.table_models import TableArtifact

_HAS_RAPIDFUZZ = rapidfuzz_fuzz is not None


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


def test_strip_footnote_markers_handles_chained_parenthesized_suffixes() -> None:
    value = "À dividende non cumulatif, série BW (3), (4),"
    assert _strip_footnote_markers_from_indicator(value) == "À dividende non cumulatif, série BW"


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

    added, removed, _, _ = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []


def test_indicator_diff_keeps_semantic_trailing_numbers() -> None:
    t1 = _table(["Série 2"])
    t2 = _table(["Série 3"])

    added, removed, _, _ = _indicator_diff(t1, t2)
    assert added == ["Série 3"]
    assert removed == ["Série 2"]


def test_indicator_diff_excludes_totals_and_pure_numbers() -> None:
    t1 = _table(
        ["Actif A", "Total du passif et des capitaux propres", "1"]
    )
    t2 = _table(
        ["Actif A", "Total du passif et des capitaux propres", "26"]
    )
    added, removed, _, excluded = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert excluded.get("total", 0) >= 1
    assert excluded.get("number", 0) >= 1


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

    added, removed, _, excluded = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert excluded.get("structural", 0) >= 1


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

    added, removed, _, excluded = _indicator_diff(t1, t2)
    assert added == []
    assert removed == []
    assert excluded.get("neighbor_aligned", 0) >= 1


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

    added, removed, _, _ = _indicator_diff(t1, t2)
    assert added == ["Série N – deuxième tranche", "Série N – première tranche"]
    assert removed == []


def test_indicator_diff_fusion_split() -> None:
    """Une ligne T2 = concat de deux lignes T1 -> pas 1 add + 2 remove."""
    t1 = _table(["Ligne A", "Ligne B"])
    t2 = _table(["Ligne A Ligne B"])
    added, removed, had_fusion_split, _ = _indicator_diff(t1, t2)
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
    added_rest, removed_rest, pairs, _ = _hungarian_pair_added_removed(removed, added, th={})
    assert len(pairs) == 0
    assert "Total des actifs" in removed_rest
    assert "Ratio de levier au sens de Bâle III" in added_rest
