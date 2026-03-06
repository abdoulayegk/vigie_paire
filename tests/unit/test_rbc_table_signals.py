from __future__ import annotations

from types import SimpleNamespace

from app.comparison_runner import _table_to_artifact
from vigilance.utils.rbc_table_signals import (
    build_rbc_first_column_signals,
    classify_rbc_title_reliability,
)


def test_rbc_title_reliability_marks_date_and_unit_titles_unreliable() -> None:
    assert (
        classify_rbc_title_reliability("Au 30 avril 2025", bank_code="rbc")
        == "unreliable"
    )
    assert (
        classify_rbc_title_reliability(
            "(en millions de dollars canadiens)",
            bank_code="rbc",
        )
        == "unreliable"
    )
    assert (
        classify_rbc_title_reliability(
            "Lien entre le risque de marche et les principales donnees figurant au bilan",
            bank_code="rbc",
        )
        == "reliable"
    )


def test_build_rbc_first_column_signals_separates_groups_from_indicators() -> None:
    rows = [
        ["(en millions de dollars canadiens)", "", "", ""],
        ["Actifs exposes au risque de marche", "", "", ""],
        ["Tresorerie et montants a recevoir de banques", "71 200 $", "-", "71 200 $"],
        ["Prets", "", "", ""],
        ["  Prets de detail", "633 400", "-", "633 400"],
        ["  Prets de gros", "379 250", "2 825", "376 425"],
        ["Total de l'actif", "2 191 026 $", "609 885 $", "1 573 697 $"],
    ]

    signals = build_rbc_first_column_signals(rows=rows, raw_indicators=[])

    assert "Actifs exposes au risque de marche" in signals.groups_raw
    assert "Prets" in signals.groups_raw
    assert "Tresorerie et montants a recevoir de banques" in signals.indicators_raw
    assert "Prets de detail" in signals.indicators_raw
    assert "Prets > Prets de detail" in signals.hierarchical_indicator_signature
    assert "Prets > Prets de gros" in signals.hierarchical_indicator_signature
    assert "Total de l'actif" not in signals.indicators_raw


def test_rbc_table_to_artifact_filters_groups_and_sets_title_reliability() -> None:
    table = SimpleNamespace(
        page_number=31,
        table_id="tableau_31",
        title="Au 30 avril 2025",
        title_clean=None,
        title_raw="Au 30 avril 2025",
        headers=["Montant figurant au bilan", "Risque lie aux activites de negociation"],
        rows=[
            ["Actifs exposes au risque de marche", "", ""],
            ["Tresorerie et montants a recevoir de banques", "56 723 $", "56 723 $"],
            ["Prets", "", ""],
            ["  Prets de detail", "626 978", "626 978"],
            ["  Prets de gros", "360 439", "357 287"],
        ],
        first_column_indicators=[
            "Actifs exposes au risque de marche",
            "Tresorerie et montants a recevoir de banques",
            "Prets",
            "Prets de detail",
            "Prets de gros",
        ],
        first_column_indicators_raw=[
            "Actifs exposes au risque de marche",
            "Tresorerie et montants a recevoir de banques",
            "Prets",
            "Prets de detail",
            "Prets de gros",
        ],
        extraction_method="vision_full_gpt4o",
        section="risk_management",
        table_number=None,
        bbox=None,
        footnotes=None,
        fragmentation_detected=False,
        debug_metrics={},
    )

    artifact = _table_to_artifact(
        table,
        bank_code="rbc",
        quarter="t2",
        pdf_path="dummy.pdf",
    )

    assert artifact.title_reliability == "unreliable"
    assert artifact.first_column_groups == [
        "Actifs exposes au risque de marche",
        "Prets",
    ]
    assert "actifs expose au risque de marche" not in artifact.first_column_indicators
    assert "tresorerie et montant a recevoir de banque" in artifact.first_column_indicators
    assert "pret de detail" in artifact.first_column_indicators
    assert artifact.hierarchical_indicator_signature == [
        "Tresorerie et montants a recevoir de banques",
        "Prets > Prets de detail",
        "Prets > Prets de gros",
    ]
