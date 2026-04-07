from __future__ import annotations

from vigilance.text_analysis_pipeline import (
    _compute_conservative_new_idea,
    _is_new_major_or_allowed_moderate,
    _should_keep_for_expert_excel,
    _sanitize_semantic_text,
)


def test_sanitize_semantic_text_removes_numbers_and_regulatory_refs() -> None:
    raw = "En 2026, le ratio CET1 atteint 13,2 % selon OSFI et la banque renforce sa stratégie de capital."

    cleaned = _sanitize_semantic_text(raw)

    assert "2026" not in cleaned
    assert "13,2" not in cleaned
    assert "CET1" not in cleaned
    assert "OSFI" not in cleaned
    assert "stratégie de capital" in cleaned


def test_sanitize_semantic_text_rephrases_regulatory_frameworks() -> None:
    raw = "Le Groupe a mis en œuvre des réformes de III et une ligne directrice sur le levier avec un coussin de ratio de levier de %."

    cleaned = _sanitize_semantic_text(raw)

    assert "III" not in cleaned
    assert "ligne directrice" not in cleaned.lower()
    assert "%" not in cleaned
    assert "La banque a" in cleaned
    assert "exigences de levier" in cleaned


def test_sanitize_semantic_text_expands_residual_acronyms() -> None:
    raw = "L'approche fondée sur des indicateurs répartit les treize indicateurs en cinq catégories pour les BISM et améliore la VaR."

    cleaned = _sanitize_semantic_text(raw)

    assert "BISM" not in cleaned
    assert "VaR" not in cleaned
    assert "banques d'importance systémique" in cleaned
    assert "mesure de risque de marché" in cleaned


def test_keep_change_for_major_relevant() -> None:
    triage = {"is_relevant": True, "impact_level": "MAJEUR", "nouvelle_idee": False, "signals": {}}

    assert _is_new_major_or_allowed_moderate(triage) is True


def test_keep_change_for_new_moderate_signal() -> None:
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "nouvelle_idee": False,
        "signals": {"regulatory_reference_added": True, "methodology_change": False},
    }

    assert _is_new_major_or_allowed_moderate(triage) is True


def test_drop_editorial_moderate_change() -> None:
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "nouvelle_idee": False,
        "signals": {"regulatory_reference_added": False, "methodology_change": False},
    }

    assert _is_new_major_or_allowed_moderate(triage) is False


def test_expert_excel_keeps_relevant_moderate_change() -> None:
    triage = {"is_relevant": True, "impact_level": "MODERE"}

    assert _should_keep_for_expert_excel(triage) is True


def test_conservative_new_idea_is_false_for_moderate_methodology_change() -> None:
    change = {
        "diff_type": "modified",
        "semantic_text_t1": "La banque améliore progressivement sa méthode de mesure des risques.",
        "semantic_text_t2": "La banque améliore sa méthode de gestion des risques selon les meilleures pratiques.",
    }
    triage = {
        "is_relevant": True,
        "impact_level": "MODERE",
        "category": "RISQUE",
        "signals": {"regulatory_reference_added": False, "methodology_change": True},
    }

    assert _compute_conservative_new_idea(change, triage) is False


def test_conservative_new_idea_is_true_for_major_added_regulatory_change() -> None:
    change = {
        "diff_type": "added",
        "semantic_text_t1": "",
        "semantic_text_t2": "La banque introduit un nouveau dispositif de contrôle contre le crime financier.",
    }
    triage = {
        "is_relevant": True,
        "impact_level": "MAJEUR",
        "category": "RISQUE",
        "signals": {"regulatory_reference_added": True, "methodology_change": False},
    }

    assert _compute_conservative_new_idea(change, triage) is True
