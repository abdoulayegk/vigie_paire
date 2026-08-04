from __future__ import annotations

import pytest

from vigie.comparaison.analyst_change_presentation import (
    build_analyst_narrative,
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


def test_business_relevance_keeps_three_complementary_sentences() -> None:
    summary = "CIBC ajoute la surveillance des risques liés à l’IA."

    result = business_relevance_paragraph(
        (
            "Le rapport courant ajoute la surveillance des risques liés à l’IA. "
            "Cet ajout fait passer l’intelligence artificielle d’un enjeu "
            "technologique implicite à une catégorie de risque reconnue. "
            "Il permet de comparer la gouvernance, les responsabilités et les "
            "contrôles déclarés par les banques. Le passage ne permet toutefois "
            "pas de conclure que ces mécanismes sont entièrement mis en œuvre."
        ),
        summary=summary,
        bank_code="cibc",
    )

    assert result == (
        "Cet ajout fait passer l’intelligence artificielle d’un enjeu "
        "technologique implicite à une catégorie de risque reconnue. "
        "Il permet de comparer la gouvernance, les responsabilités et les "
        "contrôles déclarés par les banques. Le passage ne permet toutefois "
        "pas de conclure que ces mécanismes sont entièrement mis en œuvre."
    )
    assert "CIBC ajoute" not in result


def test_business_relevance_removes_generic_lead_ins() -> None:
    result = business_relevance_paragraph(
        (
            "Cette information est importante pour la comparaison entre pairs. "
            "Pour la vigie, cet ajout rend le risque explicite. "
            "Dans le cadre de cette analyse, il permet de comparer les contrôles. "
            "Il convient de noter que le passage ne démontre pas leur mise en œuvre."
        ),
        summary="CIBC ajoute un risque émergent.",
        bank_code="cibc",
    )

    assert result == (
        "Cet ajout rend le risque explicite. "
        "Il permet de comparer les contrôles. "
        "Le passage ne démontre pas leur mise en œuvre."
    )
    assert "Pour la vigie" not in result
    assert "Cette information est importante" not in result
    assert "Il convient de noter que" not in result
    assert "Dans le cadre de cette analyse" not in result


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


def test_structured_narrative_has_priority_and_preserves_bmo_na() -> None:
    change = {
        "diff_type": "modified",
        "change_summary": "Résumé historique qui ne doit pas être publié.",
        "genai_triage": {
            "is_relevant": True,
            "changement_constate": (
                "Le rapport courant remplace BMO Harris Bank N.A. "
                "par BMO Bank N.A."
            ),
            "signification_metier": (
                "Cette mise à jour clarifie la dénomination juridique utilisée."
            ),
            "comparaison_interbanques": (
                "Elle permet de comparer les entités juridiques visées par les banques."
            ),
            "limite_interpretation": (
                "La divulgation ne démontre aucun changement de pratique."
            ),
            "motif_non_pertinence": "",
            "relevance_reason": (
                "RAISON LEGACY contradictoire qui ne doit jamais être reparsée."
            ),
        },
    }

    narrative = build_analyst_narrative(change, bank_code="bmo")

    assert narrative.source == "structured"
    assert narrative.changement_constate == (
        "BMO remplace BMO Harris Bank N.A. par BMO Bank N.A."
    )
    assert narrative.pertinence_metier == (
        "Cette mise à jour clarifie la dénomination juridique utilisée. "
        "Elle permet de comparer les entités juridiques visées par les banques. "
        "La divulgation ne démontre aucun changement de pratique."
    )
    assert narrative.business_relevance == narrative.pertinence_metier
    assert narrative.motif_non_pertinence == ""
    assert "LEGACY" not in narrative.business_relevance


def test_structured_secondary_narrative_uses_only_non_relevance_reason() -> None:
    change = {
        "diff_type": "modified",
        "genai_triage": {
            "is_relevant": False,
            "changement_constate": (
                "BMO reformule la dénomination BMO Bank N.A. sans changer le fond."
            ),
            "signification_metier": "",
            "comparaison_interbanques": "",
            "limite_interpretation": "",
            "motif_non_pertinence": (
                "Cette reformulation ne crée aucune nouvelle pratique comparable."
            ),
            "relevance_reason": (
                "BMO ajoute à tort un changement substantiel. "
                "Cette phrase legacy ne doit pas être affichée."
            ),
        },
    }

    narrative = build_analyst_narrative(change, bank_code="bmo")

    assert narrative.pertinence_metier == ""
    assert narrative.motif_non_pertinence == (
        "Cette reformulation ne crée aucune nouvelle pratique comparable."
    )
    assert narrative.business_relevance == narrative.motif_non_pertinence
    assert "legacy" not in narrative.business_relevance.lower()


def test_legacy_narrative_still_splits_factual_and_business_units() -> None:
    change = {
        "diff_type": "added",
        "genai_triage": {
            "is_relevant": True,
            "relevance_reason": (
                "Le rapport courant ajoute un contrôle annuel de cybersécurité. "
                "Cet ajout rend la fréquence du contrôle comparable entre les banques."
            ),
        },
    }

    narrative = build_analyst_narrative(change, bank_code="cibc")

    assert narrative.source == "legacy"
    assert narrative.changement_constate == (
        "CIBC ajoute un contrôle annuel de cybersécurité."
    )
    assert narrative.pertinence_metier == (
        "Cet ajout rend la fréquence du contrôle comparable entre les banques."
    )
    assert "ajoute un contrôle annuel" not in narrative.business_relevance


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


def test_build_analyst_narrative_combines_relevance_and_surveillance_details() -> None:
    change = {
        "diff_type": "modified",
        "genai_triage": {
            "is_relevant": True,
            "nouvelle_idee": True,
            "nouvelle_idee_justification": (
                "OUI — Nouvel élément à surveiller : Oui.\n\n"
                "Sujet détecté : Risque commercial et géopolitique, Cybersécurité.\n\n"
                "Ce qui change : BMO ajoute une sous-section sur la surveillance des tensions géopolitiques et des cybermenaces.\n\n"
                "Pertinence métier : Ce changement met l'accent sur la réduction de la transparence de la banque concernant les cybermenaces, un risque émergent prioritaire. Le retrait de cette divulgation modifie la lecture de l'exposition de la banque aux risques technologiques et à la sécurité de l'information.\n\n"
                "Point de surveillance : Risque commercial et géopolitique — Le changement indique que BMO renforce sa surveillance des tensions géopolitiques. Ce point permet de suivre l'évolution de la résilience de la banque face aux risques externes et la comparabilité de sa gestion des risques géopolitiques avec les pairs."
            ),
        },
    }

    narrative = build_analyst_narrative(change, bank_code="bmo")

    assert "cybermenaces" in narrative.pertinence_metier
    assert "résilience" in narrative.pertinence_metier
    assert "comparabilité" in narrative.pertinence_metier
    assert "tensions géopolitiques" in narrative.pertinence_metier
    assert "\n" not in narrative.pertinence_metier.strip()  # Paragraphe continu fluide
