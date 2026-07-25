from __future__ import annotations

import pytest

from vigilance.analyst_change_presentation import (
    build_change_presentation,
    business_relevance_paragraph,
    canonicalize_analyst_narrative,
    change_scope,
)


@pytest.mark.parametrize(
    ("bank", "source", "expected"),
    [
        (
            "td",
            (
                "Le T2 ajoute l’incapacité à atteindre les cibles financières "
                "parmi les facteurs pouvant créer un écart par rapport aux attentes "
                "des investisseurs et des analystes."
            ),
            (
                "TD ajoute l’incapacité à atteindre les cibles financières parmi "
                "les facteurs pouvant créer un écart par rapport aux attentes des "
                "investisseurs et des analystes."
            ),
        ),
        (
            "td",
            (
                "Le T2 précise que l’incidence de la résolution globale comprend "
                "celle de la limite imposée à l’actif de la Banque aux États-Unis."
            ),
            (
                "TD précise que l’incidence de la résolution globale comprend celle "
                "de la limite imposée à l’actif de la Banque aux États-Unis."
            ),
        ),
        (
            "bmo",
            (
                "Le T2 ajoute la surveillance des risques liés à l’intelligence "
                "artificielle à ses objectifs de gestion des risques."
            ),
            (
                "BMO ajoute la surveillance des risques liés à l’intelligence "
                "artificielle à ses objectifs de gestion des risques."
            ),
        ),
        (
            "bmo",
            (
                "Le rapport courant ajoute le renforcement de sa capacité à absorber "
                "les périodes de crise à son objectif relatif au capital et à la liquidité."
            ),
            (
                "BMO ajoute le renforcement de sa capacité à absorber les périodes de "
                "crise à son objectif relatif au capital et à la liquidité."
            ),
        ),
    ],
)
def test_build_change_presentation_uses_bank_subject(
    bank: str,
    source: str,
    expected: str,
) -> None:
    presentation = build_change_presentation(
        {"diff_type": "modified"},
        bank_code=bank,
        candidate_summary=source,
    )

    assert presentation.summary == expected
    assert presentation.quality_status == "ready"
    assert "T1" not in presentation.summary
    assert "T2" not in presentation.summary


def test_build_change_presentation_keeps_only_the_main_factual_sentence() -> None:
    presentation = build_change_presentation(
        {"diff_type": "added"},
        bank_code="bmo",
        candidate_summary=(
            "Le T2 ajoute la surveillance des risques liés à l’IA. "
            "Cette information est importante pour la comparaison entre pairs."
        ),
    )

    assert presentation.summary == (
        "BMO ajoute la surveillance des risques liés à l’IA."
    )


def test_business_relevance_removes_repeated_factual_sentence() -> None:
    summary = "CIBC introduit une nouvelle section sur les risques liés à l’IA."

    result = business_relevance_paragraph(
        (
            "Le rapport courant introduit une nouvelle section sur les risques "
            "liés à l’IA. Cet ajout permet de comparer la gouvernance et les "
            "pratiques de gestion de l’IA entre les banques."
        ),
        summary=summary,
        bank_code="cibc",
    )

    assert result == (
        "Cet ajout permet de comparer la gouvernance et les pratiques de gestion "
        "de l’IA entre les banques."
    )
    assert result.count("introduit une nouvelle section") == 0


def test_canonicalize_narrative_preserves_structured_sections() -> None:
    narrative = (
        "Ce qui change : Le T2 ajoute une précision absente du T1.\n\n"
        "Pertinence métier : La mention au T2 modifie la lecture du risque."
    )

    result = canonicalize_analyst_narrative(narrative, bank_code="bnc")

    assert "Ce qui change : BNC ajoute une précision absente du rapport précédent." in result
    assert "\n\nPertinence métier :" in result
    assert "T1" not in result
    assert "T2" not in result


@pytest.mark.parametrize(
    "reason",
    [
        "variation_numerique_propre_banque",
        "reformulation_mineure",
        "formatage_visuel",
        "mise_a_jour_calendrier",
        "deplacement_texte",
    ],
)
def test_change_scope_classifies_noise_as_secondary(reason: str) -> None:
    assert (
        change_scope(
            {
                "diff_type": "modified",
                "genai_triage": {
                    "is_relevant": False,
                    "nouvelle_idee": False,
                    "exclusion_reason": reason,
                },
            }
        )
        == "secondary"
    )


def test_change_scope_keeps_relevant_qualitative_change() -> None:
    assert (
        change_scope(
            {
                "diff_type": "modified",
                "genai_triage": {
                    "is_relevant": True,
                    "nouvelle_idee": True,
                },
            }
        )
        == "qualitative"
    )
