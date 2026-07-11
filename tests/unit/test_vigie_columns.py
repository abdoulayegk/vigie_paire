from typing import get_args

from vigilance.amf_taxonomy import ThemeAMF
from vigilance.vigie_columns import (
    build_text_vigie_display_row,
    derive_secondary_labels,
    derive_vigie_category,
    summarize_change,
)


def test_vigie_category_prioritizes_explicit_model_risk() -> None:
    triage = {
        "themes_amf": ["RISQUE_EMERGENT", "GOUVERNANCE_RISQUES"],
    }

    category = derive_vigie_category(
        triage,
        text="La banque renforce la validation des modèles conformément à E-23.",
        section="Gestion des risques",
    )

    assert category == "14 — Risque de modèle"


def test_vigie_columns_keep_multilabels_and_a_factual_change_summary() -> None:
    triage = {
        "themes_amf": ["RISQUE_EMERGENT", "RISQUE_DONNEES", "GOUVERNANCE_RISQUES"],
    }

    labels = derive_secondary_labels(triage)
    summary = summarize_change(
        {"diff_type": "added"},
        current_text="La banque ajoute un cadre de gouvernance pour l'IA générative.",
    )

    assert "Risque émergent" in labels
    assert "Risque et gouvernance des données" in labels
    assert summary.startswith("Ajout : La banque ajoute un cadre")


def test_vigie_category_keeps_an_explicit_out_of_scope_emerging_subject() -> None:
    triage = {"themes_amf": ["SUJET_EMERGENT_HORS_GRILLE"]}

    category = derive_vigie_category(
        triage,
        text="La banque annonce une nouvelle stratégie internationale qui modifie son profil de risque.",
    )

    assert category == "À qualifier — sujet émergent hors grille"
    assert "hors grille" in derive_secondary_labels(triage)
    assert "SUJET_EMERGENT_HORS_GRILLE" in get_args(ThemeAMF)


def test_compact_display_row_exposes_eight_analyst_fields() -> None:
    reason = " ".join(f"raison{i}" for i in range(100))
    row = build_text_vigie_display_row(
        {
            "diff_type": "added",
            "subsection_heading": "Cyberrisque",
            "change_summary": "Ajout d’un contrôle contre les ransomwares.",
            "source_text_t2": "La banque ajoute un contrôle contre les ransomwares.",
            "genai_triage": {
                "is_relevant": True,
                "themes_amf": ["RISQUE_EMERGENT", "CONTROLE_CONFORMITE"],
                "nouvelle_idee": True,
                "relevance_reason": reason,
            },
        },
        section_title="Gestion des risques",
    )

    assert row["category"] == "7 — Cyberrisque"
    assert "Risque émergent" in row["secondary_labels"]
    assert row["section"] == "Gestion des risques"
    assert row["subsection"] == "Cyberrisque"
    assert row["change_type"] == "Ajout"
    assert row["what_changed"] == "Ajout d’un contrôle contre les ransomwares."
    assert row["nouvelle_idee_label"] == "Oui"
    assert row["relevance_reason"] == reason


def test_compact_display_row_reads_legacy_pertinence_reason() -> None:
    row = build_text_vigie_display_row(
        {
            "diff_type": "modified",
            "change_summary": "Précision du cadre de contrôle.",
            "genai_triage": {
                "themes_amf": ["CONTROLE_CONFORMITE"],
                "nouvelle_idee": False,
                "nouvelle_idee_justification": (
                    "NON — Nouvel élément à surveiller : Non.\n\n"
                    "Sujet détecté : Contrôle interne.\n\n"
                    "Ce qui change : Le cadre est précisé.\n\n"
                    "Pertinence métier : La précision facilite la comparaison "
                    "des dispositifs de contrôle entre les banques.\n\n"
                    "Point de surveillance : Contrôle interne."
                ),
            },
        },
        section_title="Gestion des risques",
    )

    assert row["relevance_reason"].startswith(
        "La précision facilite la comparaison"
    )
