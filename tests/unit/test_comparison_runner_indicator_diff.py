"""Tests for indicator diff normalization in comparison runner."""

from __future__ import annotations

from app.comparison_runner import _detect_fusion_split, _indicator_diff
from vigilance.models.table_models import TableArtifact


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
        extraction_method="docling",
        quarter="t1",
        pdf_path="dummy.pdf",
    )


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
