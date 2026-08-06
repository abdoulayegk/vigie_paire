"""Tests unitaires pour l'extraction et l'éclatement des sous-éléments des chapitres supprimés."""

from __future__ import annotations

from vigie.analyse_texte.comparaison_sections import _format_sub_items_breakdown


def test_format_sub_items_breakdown_empty():
    assert _format_sub_items_breakdown("") == ""
    assert _format_sub_items_breakdown("   ") == ""


def test_format_sub_items_breakdown_with_list_items():
    text = (
        "Le chapitre sur l'engagement climat comprenait :\n"
        "- Participation à l'Alliance bancaire Net Zéro (NZBA).\n"
        "- Alignement sur la méthodologie PCAF pour les émissions.\n"
        "- Engagements relatifs à la biodiversité."
    )
    res = _format_sub_items_breakdown(text)
    assert "Sous-éléments et clauses spécifiques retirés :" in res
    assert "• Participation à l'Alliance bancaire Net Zéro (NZBA)." in res
    assert "• Alignement sur la méthodologie PCAF pour les émissions." in res


def test_format_sub_items_breakdown_with_sentences():
    text = (
        "La Banque a mis à jour les responsabilités du comité. "
        "Le comité des RH supervise directement la culture de risque de la Banque. "
        "Il valide également l'ensemble des programmes de rémunération incitative."
    )
    res = _format_sub_items_breakdown(text)
    assert "Sous-éléments et clauses spécifiques retirés :" in res
    assert "• La Banque a mis à jour les responsabilités du comité" in res
