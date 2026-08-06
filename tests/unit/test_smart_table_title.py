"""Tests pour la résolution et l'auto-titrage sémantique des tableaux Dash."""

from __future__ import annotations

from vigie.interface.components.table_title_resolver import resolve_display_table_title


def test_resolve_display_table_title_existing_valid_title() -> None:
    table = {
        "table_name": "Charges grevant les actifs",
        "table_id_t2": "tbl_p082_i01",
    }
    assert resolve_display_table_title(table) == "Charges grevant les actifs"


def test_resolve_display_table_title_raw_id_fallback_to_genai_sujet() -> None:
    table = {
        "table_name": "tbl_p082_i01",
        "table_id_t2": "tbl_p082_i01",
        "section": "Gestion des risques",
        "genai_analysis": {
            "nouvelle_idee_justification": "OUI — Nouvel élément à surveiller : Oui Sujet détecté : Risque émergent : IA, cybersécurité..."
        },
    }
    assert resolve_display_table_title(table) == "[Tableau : Risque émergent : IA, cybersécurité]"


def test_resolve_display_table_title_raw_id_fallback_to_changement_constate() -> None:
    table = {
        "table_name": "tbl_p076_i01",
        "table_id_t2": "tbl_p076_i01",
        "section": "Gestion des risques",
        "genai_analysis": {
            "changement_constate": "RBC supprime le tableau de synthèse des principaux facteurs de risque."
        },
    }
    assert resolve_display_table_title(table) == "[Tableau : le tableau de synthèse des principaux fact...]"


def test_resolve_display_table_title_fallback_to_section() -> None:
    table = {
        "table_name": "tbl_p090_i01",
        "table_id_t2": "tbl_p090_i01",
        "section": "Risque de crédit",
    }
    assert resolve_display_table_title(table) == "[Tableau : Risque de crédit]"


def test_resolve_display_table_title_fallback_to_raw_id_if_no_context() -> None:
    table = {
        "table_name": "tbl_p099_i01",
        "table_id_t2": "tbl_p099_i01",
    }
    assert resolve_display_table_title(table) == "tbl_p099_i01"
