from __future__ import annotations

import json
from typing import Any

from vigilance.amf_taxonomy import (
    TriageAMFCompactLLMBatch,
    TriageAMFCompactLLMResultWithIndex,
)
from vigilance.text_analysis.precedent_memory import (
    AnalystPrecedent,
    PrecedentMemory,
)
from vigilance.text_analysis.triage import (
    _ConsolidatedDossierAssessment,
    _annotate_triage_dossiers,
    _attach_consolidated_dossier_outcomes,
    _evaluate_consolidated_dossier_materiality,
    _persisted_triage_from_compact,
    _triage_section_changes,
)


def _parsed_response(parsed_obj: Any):
    message = type("FakeMessage", (), {"parsed": parsed_obj, "refusal": None})()
    choice = type(
        "FakeChoice",
        (),
        {"message": message, "finish_reason": "stop"},
    )()
    return type("FakeResponse", (), {"choices": [choice]})()


class _FakeCompletions:
    def __init__(self, side_effect) -> None:
        self.side_effect = side_effect
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if callable(self.side_effect):
            return self.side_effect(**kwargs)
        return self.side_effect


class _FakeClient:
    def __init__(self, side_effect) -> None:
        completions = _FakeCompletions(side_effect)
        chat = type("Chat", (), {"completions": completions})()
        self.beta = type("Beta", (), {"chat": chat})()
        self._completions = completions

    @property
    def call_count(self) -> int:
        return len(self._completions.calls)


def _semantic_fields(*, relevant: bool = True) -> dict[str, str]:
    if relevant:
        return {
            "changement_constate": (
                "BMO modifie le périmètre et les responsabilités déclarées."
            ),
            "signification_metier": (
                "Cette évolution change la compréhension du dispositif déclaré."
            ),
            "comparaison_interbanques": (
                "Elle permet de comparer le périmètre et les responsabilités."
            ),
            "limite_interpretation": (
                "La divulgation ne quantifie pas les effets de cette évolution."
            ),
            "motif_non_pertinence": "",
        }
    return {
        "changement_constate": "BMO modifie uniquement une présentation équivalente.",
        "signification_metier": "",
        "comparaison_interbanques": "",
        "limite_interpretation": "",
        "motif_non_pertinence": (
            "L'équivalence métier est démontrée par les textes fournis."
        ),
    }


def _direct_triage(
    *,
    level: str,
    relevant: bool = True,
    themes: list[str] | None = None,
    nature: list[str] | None = None,
    equivalence: str = "REFUTEE",
    confidence: str = "ELEVEE",
    evidence: str = "SUFFISANTE",
    status: str = "CONFIRME",
    review_required: bool = False,
) -> TriageAMFCompactLLMResultWithIndex:
    return TriageAMFCompactLLMResultWithIndex(
        change_index=1,
        is_relevant=relevant,
        themes_amf=(
            themes
            if themes is not None
            else (["GOUVERNANCE_RISQUES"] if relevant else [])
        ),
        nouvelle_idee=False,
        materiality_level=level,
        change_nature=nature or ["MODIFICATION_PERIMETRE"],
        business_equivalence=equivalence,
        materiality_confidence=confidence,
        evidence_sufficiency=evidence,
        decision_status=status,
        review_required=review_required,
        supporting_evidence=[
            "Le texte modifie explicitement une dimension métier."
        ],
        counterarguments=[],
        **_semantic_fields(relevant=relevant),
    )


def _change(
    *,
    change_id: str = "c1",
    before: str = "Le comité conseille la direction.",
    after: str = "Le comité approuve les limites.",
    summary: str = "BMO remplace un rôle de conseil par un rôle d'approbation.",
) -> dict[str, Any]:
    return {
        "change_id": change_id,
        "section_key": "gestion_risques",
        "subsection_heading": "Gouvernance",
        "diff_type": "modified",
        "alignment_decision": "same_disclosure",
        "alignment_confidence": "high",
        "source_text_t1": before,
        "source_text_t2": after,
        "semantic_text_t1": before,
        "semantic_text_t2": after,
        "change_summary": summary,
    }


def test_direct_materiality_is_independent_from_nouvelle_idee() -> None:
    compact = _direct_triage(level="MAJEUR").model_dump(
        exclude={"change_index"}
    )

    triage = _persisted_triage_from_compact(
        compact,
        change=_change(),
        bank_code="bmo",
    )

    assert triage["nouvelle_idee"] is False
    assert triage["impact_level"] == "MAJEUR"
    assert triage["materiality_level"] == "MAJEUR"
    assert triage["legacy_impact_level"] == "MINEUR"
    assert triage["materiality_decision_basis"] == "direct_materiality"


def test_acquisition_signal_does_not_veto_new_data_risk() -> None:
    parsed = TriageAMFCompactLLMBatch(
        triages=[
            _direct_triage(
                level="MAJEUR",
                themes=["RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"],
                nature=["AJOUT_INFORMATION", "MODIFICATION_CONTROLE"],
            )
        ]
    )
    client = _FakeClient(_parsed_response(parsed))
    change = _change(
        before="La banque décrit son risque de données.",
        after=(
            "Après l'acquisition de CWB, la banque ajoute le risque de cycle de "
            "vie des données, les fournisseurs externes et des contrôles de migration."
        ),
        summary=(
            "BMO ajoute des risques de données, de fournisseurs et de migration "
            "dans le contexte de l'acquisition de CWB."
        ),
    )

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        bank_code="bmo",
        changes=[change],
    )

    triage = result[0]["genai_triage"]
    assert client.call_count == 1
    assert triage["is_relevant"] is True
    assert triage["impact_level"] == "MAJEUR"
    assert "operation_interne_banque" in triage["advisory_signals"]
    assert triage.get("source_guardrail") is None


def test_calendar_status_reaches_materiality_judge() -> None:
    parsed = TriageAMFCompactLLMBatch(
        triages=[
            _direct_triage(
                level="MODERE",
                themes=["EXIGENCES_REGLEMENTAIRES"],
                nature=[
                    "MODIFICATION_EXIGENCE_REGLEMENTAIRE",
                    "MODIFICATION_STATUT_MISE_EN_OEUVRE",
                ],
            )
        ]
    )
    client = _FakeClient(_parsed_response(parsed))
    change = _change(
        before=(
            "Le BSIF prévoit une augmentation du coefficient de plancher en 2027."
        ),
        after=(
            "Le BSIF reporte toute augmentation du coefficient de plancher "
            "jusqu'à nouvel ordre."
        ),
        summary=(
            "BMO remplace une échéance déterminée du plancher par un report "
            "jusqu'à nouvel ordre."
        ),
    )

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_capital",
        bank_code="bmo",
        changes=[change],
    )

    triage = result[0]["genai_triage"]
    assert client.call_count == 1
    assert triage["impact_level"] == "MODERE"
    assert "mise_a_jour_calendrier" in triage["advisory_signals"]
    assert result[0].get("triage_prefilter") is None


def test_low_confidence_text_move_requires_review_instead_of_minor() -> None:
    client = _FakeClient(RuntimeError("aucun appel attendu"))
    change = _change()
    change.update(
        {
            "alignment_decision": "moved_text",
            "alignment_confidence": "low",
        }
    )

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        bank_code="bmo",
        changes=[change],
    )

    triage = result[0]["genai_triage"]
    assert client.call_count == 0
    assert triage["materiality_level"] is None
    assert triage["decision_status"] == "A_CONFIRMER"
    assert triage["review_required"] is True
    assert triage["source"] == "alignment_review_required"


def test_high_confidence_text_move_can_be_confirmed_minor() -> None:
    client = _FakeClient(RuntimeError("aucun appel attendu"))
    change = _change()
    change.update(
        {
            "alignment_decision": "moved_text",
            "alignment_confidence": "high",
        }
    )

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        bank_code="bmo",
        changes=[change],
    )

    triage = result[0]["genai_triage"]
    assert client.call_count == 0
    assert triage["materiality_level"] == "MINEUR"
    assert triage["decision_status"] == "CONFIRME"
    assert triage["review_required"] is False


def test_related_changes_are_all_classified_without_verdict_propagation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vigilance.text_analysis.triage._embed_texts",
        lambda client, texts, model="text-embedding-3-small": [
            [1.0, 0.0],
            [0.99, 0.01],
        ],
    )

    def response_for_prompt(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        payload = json.loads(prompt.rsplit("Changements :\n", 1)[1])
        level = (
            "MAJEUR"
            if "répartition des ressources"
            in payload[0]["source_snippet_t2"]
            else "MODERE"
        )
        return _parsed_response(
            TriageAMFCompactLLMBatch(
                triages=[_direct_triage(level=level)]
            )
        )

    client = _FakeClient(response_for_prompt)
    changes = [
        _change(
            change_id="allocation",
            after="Le processus guide la répartition des ressources.",
            summary="BMO ajoute la répartition des ressources.",
        ),
        _change(
            change_id="rendements",
            after="Le processus surveille les rendements.",
            summary="BMO ajoute la surveillance des rendements.",
        ),
    ]

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_capital",
        bank_code="bmo",
        changes=changes,
    )

    assert client.call_count == 2
    assert [item["genai_triage"]["impact_level"] for item in result] == [
        "MAJEUR",
        "MODERE",
    ]
    assert all(
        item["triage_dedup"]["propagated"] is False for item in result
    )
    assert all(
        item["genai_triage"]["consolidated_materiality_level"] == "MAJEUR"
        for item in result
    )
    assert all(
        item["genai_triage"]["triage_group_verdict_propagated"] is False
        for item in result
    )


def test_same_subsection_context_does_not_create_a_cumulative_group() -> None:
    changes = [
        {
            **_change(
                change_id="governance-role",
                summary="BMO modifie le rôle du comité.",
            ),
            "genai_triage": {"impact_level": "MAJEUR", "is_relevant": True},
        },
        {
            **_change(
                change_id="capital-calendar",
                before="La mesure prend effet en janvier.",
                after="La mesure prend effet en avril.",
                summary="BMO actualise une date d'application.",
            ),
            "genai_triage": {"impact_level": "MINEUR", "is_relevant": False},
        },
    ]

    annotated = _annotate_triage_dossiers(
        changes,
        section_key="gestion_risques",
        groups=[[0], [1]],
    )
    consolidated = _attach_consolidated_dossier_outcomes(annotated)

    assert all(
        item["triage_dossier"]["related_changes"] for item in consolidated
    )
    assert all(
        "group_id" not in item["triage_dossier"] for item in consolidated
    )
    assert all(
        "triage_group_id" not in item["genai_triage"]
        for item in consolidated
    )


def test_independent_dossier_assessment_can_raise_cumulative_level() -> None:
    assessment = _ConsolidatedDossierAssessment(
        materiality_level="MAJEUR",
        change_nature=[
            "MODIFICATION_PERIMETRE",
            "MODIFICATION_RESPONSABILITES",
        ],
        business_equivalence="REFUTEE",
        materiality_confidence="ELEVEE",
        evidence_sufficiency="SUFFISANTE",
        decision_status="CONFIRME",
        review_required=False,
        supporting_evidence=[
            "Le dossier combine un nouveau périmètre et une nouvelle finalité "
            "d'allocation du capital."
        ],
        counterarguments=[
            "Chaque remplacement terminologique pourrait sembler modéré isolément."
        ],
        materiality_rationale=(
            "Les changements reliés redéfinissent ensemble le périmètre et les "
            "finalités déclarées du processus de capital."
        ),
    )
    client = _FakeClient(_parsed_response(assessment))
    changes = [
        {
            **_change(change_id=f"c{index}"),
            "triage_dossier": {
                "group_id": "gestion_capital_subsection_dossier_001",
                "member_change_ids": ["c1", "c2"],
            },
            "genai_triage": {
                "is_relevant": True,
                "themes_amf": ["FONDS_PROPRES_REGLEMENTAIRES"],
                "impact_level": "MODERE",
                "materiality_level": "MODERE",
                "changement_constate": factual,
            },
        }
        for index, factual in (
            (1, "BMO remplace les groupes par des unités d'exploitation."),
            (2, "BMO ajoute une finalité de répartition des ressources."),
        )
    ]

    result = _evaluate_consolidated_dossier_materiality(
        client=client,
        model="gpt-4o",
        bank_code="bmo",
        section_key="gestion_capital",
        changes=changes,
    )

    assert client.call_count == 1
    assert all(
        item["genai_triage"]["impact_level"] == "MODERE"
        for item in result
    )
    assert all(
        item["genai_triage"]["consolidated_materiality_level"] == "MAJEUR"
        for item in result
    )
    assert all(
        item["genai_triage"]["consolidated_assessment_source"]
        == "independent_dossier_assessment"
        for item in result
    )
    assert all(
        item["genai_triage"]["consolidated_relevant"] is True
        for item in result
    )


def test_blind_challenger_promotes_sensitive_minor_for_review() -> None:
    primary = _direct_triage(
        level="MINEUR",
        themes=["GOUVERNANCE_RISQUES"],
        nature=["REFORMULATION_EQUIVALENTE"],
        equivalence="CONFIRMEE",
    )
    challenger = _direct_triage(
        level="MAJEUR",
        themes=["GOUVERNANCE_RISQUES"],
        nature=["MODIFICATION_RESPONSABILITES"],
        equivalence="REFUTEE",
    )
    responses = iter(
        (
            _parsed_response(TriageAMFCompactLLMBatch(triages=[primary])),
            _parsed_response(TriageAMFCompactLLMBatch(triages=[challenger])),
        )
    )
    client = _FakeClient(lambda **_kwargs: next(responses))

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        bank_code="bmo",
        changes=[_change()],
    )

    triage = result[0]["genai_triage"]
    assert client.call_count == 2
    assert triage["impact_level"] == "MAJEUR"
    assert triage["decision_status"] == "A_CONFIRMER"
    assert triage["review_required"] is True
    assert triage["materiality_challenge"]["blind"] is True
    assert triage["materiality_challenge"]["disagreement"] is True


def test_blind_challenger_reviews_missed_sensitive_non_relevance() -> None:
    primary = _direct_triage(
        level="MINEUR",
        relevant=False,
        themes=[],
        nature=["REFORMULATION_EQUIVALENTE"],
        equivalence="CONFIRMEE",
    )
    challenger = _direct_triage(
        level="MODERE",
        themes=["FONDS_PROPRES_REGLEMENTAIRES"],
        nature=["MODIFICATION_TERMINOLOGIE"],
        equivalence="NON_DEMONTREE",
        confidence="MOYENNE",
        evidence="PARTIELLE",
        status="A_CONFIRMER",
        review_required=True,
    )
    responses = iter(
        (
            _parsed_response(TriageAMFCompactLLMBatch(triages=[primary])),
            _parsed_response(TriageAMFCompactLLMBatch(triages=[challenger])),
        )
    )
    client = _FakeClient(lambda **_kwargs: next(responses))

    result = _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_capital",
        bank_code="bmo",
        changes=[
            _change(
                before="La suffisance du capital est évaluée par groupe.",
                after=(
                    "L'adéquation des fonds propres est évaluée par unité "
                    "d'exploitation."
                ),
            )
        ],
    )

    triage = result[0]["genai_triage"]
    assert client.call_count == 2
    assert triage["is_relevant"] is True
    assert triage["impact_level"] == "MODERE"
    assert triage["review_required"] is True
    assert triage["materiality_challenge"]["resolution"] == (
        "challenger_higher_materiality"
    )


def test_prompt_uses_validated_precedents_without_alignment_rationale() -> None:
    precedent = AnalystPrecedent(
        precedent_id="validated-governance-major",
        change_id="old-1",
        bank_code="td",
        section_key="gestion_risques",
        text_before="Le comité conseille.",
        text_after="Le comité approuve.",
        materiality_level="MAJEUR",
        change_nature="MODIFICATION_RESPONSABILITES",
        themes_amf=("GOUVERNANCE_RISQUES",),
        rationale="L'autorité décisionnelle change.",
    )
    memory = PrecedentMemory(precedents=(precedent,))
    parsed = TriageAMFCompactLLMBatch(
        triages=[_direct_triage(level="MAJEUR")]
    )
    client = _FakeClient(_parsed_response(parsed))
    change = _change()
    change["alignment_rationale"] = "ANCRAGE_INTERDIT_SANS_CHANGEMENT_DE_FOND"

    _triage_section_changes(
        client=client,
        model="gpt-4o",
        section_key="gestion_risques",
        bank_code="bmo",
        changes=[change],
        precedent_memory=memory,
    )

    prompt = client._completions.calls[0]["messages"][1]["content"]
    assert "validated-governance-major" in prompt
    assert "ANCRAGE_INTERDIT_SANS_CHANGEMENT_DE_FOND" not in prompt
    assert "change_summary_factuel_non_arbitre" in prompt
    assert "materiality_level" in prompt
