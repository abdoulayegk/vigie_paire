"""Tests ciblés du schéma adaptatif de matérialité AMF."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vigilance.amf_taxonomy import (
    TriageAMFCompactLLMResultWithIndex,
    TriageAMFMaterialityLLMBatch,
    TriageAMFMaterialityLLMResultWithIndex,
    TriageAMFResult,
)
from vigilance.utils.openai_schema import build_strict_openai_response_format


def _compact_relevant_payload() -> dict:
    return {
        "change_index": 1,
        "is_relevant": True,
        "themes_amf": ["GOUVERNANCE_RISQUES"],
        "nouvelle_idee": False,
        "changement_constate": ("BMO remplace une unité de gouvernance et modifie son périmètre."),
        "signification_metier": ("Le changement peut modifier la responsabilité de surveillance."),
        "motif_non_pertinence": "",
    }


def _valid_explanation() -> str:
    return (
        "La banque modifie explicitement le périmètre d'une unité de gouvernance. "
        "Cette modification change la lecture des responsabilités de surveillance. "
        "La preuve justifie donc une évaluation directe de la matérialité."
    )


def _valid_non_new_idea_justification() -> str:
    return (
        "NON - Nouvel élément à surveiller : Non.\n\n"
        "Sujet détecté : Gouvernance des risques et responsabilité de surveillance.\n\n"
        "Ce qui change : La banque remplace l'unité responsable et élargit "
        "explicitement son périmètre de surveillance. Le changement est démontré "
        "dans les formulations avant et après.\n\n"
        "Pertinence métier : Cette modification demeure importante même si le "
        "sujet général de gouvernance existait déjà. Elle peut modifier le niveau "
        "d'autorité, la reddition de comptes et la comparaison entre banques.\n\n"
        "Point de surveillance : Le point à retenir est la modification démontrée "
        "du périmètre de responsabilité. L'absence de nouvelle idée ne détermine "
        "donc pas à elle seule le niveau de matérialité."
    )


def test_legacy_compact_payload_keeps_neutral_adaptive_defaults() -> None:
    triage = TriageAMFCompactLLMResultWithIndex(**_compact_relevant_payload())

    assert triage.materiality_level is None
    assert triage.change_nature == []
    assert triage.business_equivalence == "INDETERMINE"
    assert triage.materiality_confidence == "INDETERMINE"
    assert triage.evidence_sufficiency == "INDETERMINE"
    assert triage.decision_status == "PROVISOIRE"
    assert triage.review_required is False
    assert triage.supporting_evidence == []
    assert triage.counterarguments == []


def test_relevant_change_accepts_empty_optional_themes() -> None:
    payload = {
        **_compact_relevant_payload(),
        "themes_amf": [],
        "materiality_level": "MODERE",
        "change_nature": ["MODIFICATION_TERMINOLOGIE"],
        "business_equivalence": "NON_DEMONTREE",
        "materiality_confidence": "MOYENNE",
        "evidence_sufficiency": "PARTIELLE",
        "decision_status": "A_CONFIRMER",
        "review_required": True,
        "supporting_evidence": [
            "La terminologie change sans thème taxonomique suffisamment précis."
        ],
        "counterarguments": [],
    }

    triage = TriageAMFCompactLLMResultWithIndex(**payload)

    assert triage.is_relevant is True
    assert triage.themes_amf == []
    assert triage.materiality_level == "MODERE"


def test_compact_direct_major_is_independent_from_nouvelle_idee() -> None:
    payload = {
        **_compact_relevant_payload(),
        "materiality_level": "MAJEUR",
        "change_nature": [
            "MODIFICATION_GOUVERNANCE",
            "MODIFICATION_RESPONSABILITES",
        ],
        "business_equivalence": "REFUTEE",
        "materiality_confidence": "ELEVEE",
        "evidence_sufficiency": "SUFFISANTE",
        "decision_status": "CONFIRME",
        "review_required": False,
        "supporting_evidence": ["Le texte attribue la surveillance à une nouvelle unité décisionnelle."],
        "counterarguments": ["Le sujet général de gouvernance était déjà divulgué."],
    }

    triage = TriageAMFCompactLLMResultWithIndex(**payload)

    assert triage.nouvelle_idee is False
    assert triage.materiality_level == "MAJEUR"
    assert triage.change_nature == [
        "MODIFICATION_GOUVERNANCE",
        "MODIFICATION_RESPONSABILITES",
    ]
    assert triage.decision_status == "CONFIRME"


def test_uncertain_minor_requires_review() -> None:
    payload = {
        **_compact_relevant_payload(),
        "materiality_level": "MINEUR",
        "change_nature": ["MODIFICATION_TERMINOLOGIE"],
        "business_equivalence": "NON_DEMONTREE",
        "materiality_confidence": "MOYENNE",
        "evidence_sufficiency": "PARTIELLE",
        "decision_status": "A_CONFIRMER",
        "review_required": False,
        "supporting_evidence": ["Les deux formulations emploient des termes métier différents."],
        "counterarguments": [],
    }

    with pytest.raises(ValidationError, match="review_required=True"):
        TriageAMFCompactLLMResultWithIndex(**payload)

    payload["review_required"] = True
    triage = TriageAMFCompactLLMResultWithIndex(**payload)
    assert triage.review_required is True


def test_probable_equivalence_minor_also_requires_review() -> None:
    payload = {
        **_compact_relevant_payload(),
        "materiality_level": "MINEUR",
        "change_nature": ["MODIFICATION_TERMINOLOGIE"],
        "business_equivalence": "PROBABLE",
        "materiality_confidence": "MOYENNE",
        "evidence_sufficiency": "SUFFISANTE",
        "decision_status": "CONFIRME",
        "review_required": False,
        "supporting_evidence": [
            "Les formulations paraissent proches sans preuve d'identité."
        ],
        "counterarguments": [],
    }

    with pytest.raises(ValidationError, match="review_required=True"):
        TriageAMFCompactLLMResultWithIndex(**payload)


def test_provisional_direct_materiality_requires_review() -> None:
    payload = {
        **_compact_relevant_payload(),
        "materiality_level": "MINEUR",
        "change_nature": ["REFORMULATION_EQUIVALENTE"],
        "business_equivalence": "REFUTEE",
        "materiality_confidence": "MOYENNE",
        "evidence_sufficiency": "PARTIELLE",
        "decision_status": "PROVISOIRE",
        "review_required": False,
        "supporting_evidence": ["Le référent paraît inchangé."],
        "counterarguments": [],
    }

    with pytest.raises(ValidationError, match="review_required=True"):
        TriageAMFCompactLLMResultWithIndex(**payload)


def test_non_relevant_cannot_carry_major_materiality() -> None:
    payload = {
        **_compact_relevant_payload(),
        "is_relevant": False,
        "themes_amf": [],
        "nouvelle_idee": False,
        "signification_metier": "",
        "motif_non_pertinence": "La preuve démontre une équivalence complète.",
        "materiality_level": "MAJEUR",
        "change_nature": ["REFORMULATION_EQUIVALENTE"],
        "business_equivalence": "REFUTEE",
        "materiality_confidence": "ELEVEE",
        "evidence_sufficiency": "SUFFISANTE",
        "decision_status": "CONFIRME",
        "review_required": False,
        "supporting_evidence": ["Le mandat demeure explicitement identique."],
        "counterarguments": [],
    }

    with pytest.raises(
        ValidationError,
        match="is_relevant=False exige materiality_level=MINEUR",
    ):
        TriageAMFCompactLLMResultWithIndex(**payload)


def test_fresh_materiality_schema_rejects_null_direct_level() -> None:
    with pytest.raises(ValidationError, match="materiality_level"):
        TriageAMFMaterialityLLMResultWithIndex(
            **_compact_relevant_payload()
        )


def test_confirmed_decision_requires_sufficient_evidence() -> None:
    payload = {
        **_compact_relevant_payload(),
        "materiality_level": "MODERE",
        "change_nature": ["MODIFICATION_PERIMETRE"],
        "business_equivalence": "PROBABLE",
        "materiality_confidence": "MOYENNE",
        "evidence_sufficiency": "PARTIELLE",
        "decision_status": "CONFIRME",
        "review_required": False,
        "supporting_evidence": ["Le texte semble élargir le périmètre, sans en préciser les limites."],
        "counterarguments": [],
    }

    with pytest.raises(
        ValidationError,
        match="decision_status=CONFIRME exige evidence_sufficiency=SUFFISANTE",
    ):
        TriageAMFCompactLLMResultWithIndex(**payload)


@pytest.mark.parametrize("level", ["MODERE", "MAJEUR"])
def test_confirmed_equivalence_coerces_non_minor_level_to_mineur(
    level: str,
) -> None:
    payload = {
        **_compact_relevant_payload(),
        "materiality_level": level,
        "change_nature": ["MODIFICATION_TERMINOLOGIE"],
        "business_equivalence": "CONFIRMEE",
        "materiality_confidence": "ELEVEE",
        "evidence_sufficiency": "SUFFISANTE",
        "decision_status": "CONFIRME",
        "review_required": False,
        "supporting_evidence": [
            "Les formulations décrivent explicitement le même référent métier."
        ],
        "counterarguments": [],
    }

    triage = TriageAMFCompactLLMResultWithIndex(**payload)

    assert triage.materiality_level == "MINEUR"
    assert triage.business_equivalence == "CONFIRMEE"


def test_persisted_result_copies_direct_level_to_legacy_impact() -> None:
    triage = TriageAMFResult(
        is_relevant=True,
        themes_amf=["GOUVERNANCE_RISQUES"],
        materiality_level="MAJEUR",
        change_nature=[
            "MODIFICATION_GOUVERNANCE",
            "MODIFICATION_RESPONSABILITES",
        ],
        business_equivalence="REFUTEE",
        materiality_confidence="ELEVEE",
        evidence_sufficiency="SUFFISANTE",
        decision_status="CONFIRME",
        review_required=False,
        supporting_evidence=["Le texte transfère explicitement la surveillance à une nouvelle unité."],
        counterarguments=["La rubrique générale de gouvernance demeure la même."],
        nouvelle_idee=False,
        explanation=_valid_explanation(),
        nouvelle_idee_justification=_valid_non_new_idea_justification(),
    )

    assert triage.materiality_level == "MAJEUR"
    assert triage.impact_level == "MAJEUR"
    assert triage.nouvelle_idee is False


def test_explicit_legacy_and_direct_levels_must_match() -> None:
    with pytest.raises(
        ValidationError,
        match="materiality_level et impact_level doivent être identiques",
    ):
        TriageAMFResult(
            is_relevant=True,
            themes_amf=["GOUVERNANCE_RISQUES"],
            impact_level="MINEUR",
            materiality_level="MAJEUR",
            change_nature=["MODIFICATION_GOUVERNANCE"],
            business_equivalence="REFUTEE",
            materiality_confidence="ELEVEE",
            evidence_sufficiency="SUFFISANTE",
            decision_status="CONFIRME",
            supporting_evidence=["Le texte transfère explicitement une responsabilité de surveillance."],
            nouvelle_idee=False,
            explanation=_valid_explanation(),
            nouvelle_idee_justification=_valid_non_new_idea_justification(),
        )


def test_strict_compact_schema_exposes_all_adaptive_fields() -> None:
    response_format = build_strict_openai_response_format(
        TriageAMFMaterialityLLMBatch,
        name="triage_amf_compact",
    )
    item_schema = response_format["json_schema"]["schema"]["$defs"][
        "TriageAMFMaterialityLLMResultWithIndex"
    ]

    expected_fields = {
        "materiality_level",
        "change_nature",
        "business_equivalence",
        "materiality_confidence",
        "evidence_sufficiency",
        "decision_status",
        "review_required",
        "supporting_evidence",
        "counterarguments",
    }
    assert expected_fields <= set(item_schema["properties"])
    assert "comparaison_interbanques" not in item_schema["properties"]
    assert "limite_interpretation" not in item_schema["properties"]
    assert set(item_schema["required"]) == set(item_schema["properties"])
    assert item_schema["properties"]["materiality_level"]["type"] == "string"


def test_legacy_comparison_and_limit_fields_are_ignored() -> None:
    payload = {
        **_compact_relevant_payload(),
        "comparaison_interbanques": "Ancienne dimension de comparaison.",
        "limite_interpretation": "Ancienne limite d’interprétation.",
    }

    triage = TriageAMFCompactLLMResultWithIndex(**payload)

    assert "comparaison_interbanques" not in triage.model_dump()
    assert "limite_interpretation" not in triage.model_dump()
