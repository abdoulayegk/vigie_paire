from __future__ import annotations

import json

from vigilance.text_analysis.precedent_memory import (
    AnalystPrecedent,
    PrecedentMemory,
    PrecedentQuery,
    load_validated_precedents,
)


def _write_json(path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_loader_keeps_only_validated_structured_text_decisions(tmp_path) -> None:
    path = tmp_path / "text_comparison.json"
    common_triage = {
        "impact_level": "MINEUR",
        "materiality_level": "MINEUR",
        "change_nature": ["REFORMULATION_EQUIVALENTE"],
        "business_equivalence": "CONFIRMEE",
        "materiality_confidence": "ELEVEE",
        "evidence_sufficiency": "SUFFISANTE",
        "decision_status": "CONFIRME",
        "review_required": False,
        "supporting_evidence": [
            "Le mandat et le périmètre demeurent identiques."
        ],
        "is_relevant": True,
        "themes_amf": ["GOUVERNANCE_RISQUES"],
        "signification_metier": "Le libellé demeure équivalent.",
    }
    payload = {
        "bank_code": "bmo",
        "section_comparisons": [
            {
                "section_key": "gestion_capital",
                "all_block_comparisons": [
                    {
                        "change_id": "approved",
                        "diff_type": "modified",
                        "semantic_text_t1": "Groupes d'exploitation.",
                        "semantic_text_t2": "Unités d'exploitation.",
                        "genai_triage": dict(common_triage),
                            "_analyst_review": {
                                "status": "approved",
                                "decision_scope": "materiality",
                                "schema_version": (
                                    "analyst_materiality_review_v1"
                                ),
                                "review_user": "alice",
                            "reviewed_at": "2026-07-26T12:00:00Z",
                        },
                    },
                    {
                        "change_id": "raw-model-only",
                        "diff_type": "added",
                        "semantic_text_t2": "Nouvelle responsabilité.",
                        "genai_triage": {
                            "impact_level": "MAJEUR",
                            "is_relevant": True,
                        },
                    },
                    {
                        "change_id": "skipped",
                        "diff_type": "added",
                        "semantic_text_t2": "Texte à revoir.",
                        "genai_triage": {
                            "impact_level": "MAJEUR",
                            "is_relevant": True,
                        },
                        "_analyst_review": {
                            "status": "skipped",
                            "comment": "À revoir.",
                        },
                    },
                    {
                        "change_id": "rejected-comment-only",
                        "diff_type": "modified",
                        "semantic_text_t1": "Ancien texte.",
                        "semantic_text_t2": "Nouveau texte.",
                        "genai_triage": {
                            "impact_level": "MINEUR",
                            "is_relevant": False,
                        },
                        "_analyst_review": {
                            "status": "rejected",
                            "comment": "Devrait être majeur.",
                        },
                    },
                    {
                        "change_id": "corrected",
                        "diff_type": "modified",
                        "semantic_text_t1": "Le comité conseille la direction.",
                        "semantic_text_t2": "Le comité approuve les limites.",
                        "genai_triage": {
                            "impact_level": "MINEUR",
                            "is_relevant": True,
                        },
                        "_analyst_review": {
                            "status": "corrected",
                            "decision_scope": "materiality",
                            "schema_version": (
                                "analyst_materiality_review_v1"
                            ),
                            "comment": "La responsabilité change.",
                            "corrected_materiality_level": "majeur",
                            "corrected_change_nature": (
                                "MODIFICATION_RESPONSABILITES"
                            ),
                            "corrected_business_equivalence": False,
                            "corrected_materiality_confidence": "ELEVEE",
                            "corrected_evidence_sufficiency": "suffisante",
                            "corrected_decision_status": "CONFIRME",
                            "corrected_review_required": False,
                            "corrected_is_relevant": True,
                            "corrected_themes_amf": ["GOUVERNANCE_RISQUES"],
                            "corrected_materiality_rationale": (
                                "Le comité acquiert une autorité décisionnelle."
                            ),
                            "corrected_supporting_evidence": [
                                "Le verbe approuve remplace conseille."
                            ],
                            "corrected_counterarguments": [
                                "Le passage demeure court."
                            ],
                            "review_user": "bob",
                        },
                    },
                ],
                # Le doublon historique ne doit pas produire un second précédent.
                "block_comparisons": [
                    {
                        "change_id": "approved",
                        "diff_type": "modified",
                        "semantic_text_t1": "Groupes d'exploitation.",
                        "semantic_text_t2": "Unités d'exploitation.",
                        "genai_triage": dict(common_triage),
                        "_analyst_review": {
                            "status": "approved",
                            "decision_scope": "materiality",
                            "schema_version": (
                                "analyst_materiality_review_v1"
                            ),
                        },
                    }
                ],
            }
        ],
    }
    _write_json(path, payload)

    memory = PrecedentMemory.from_paths(path)

    assert {item.change_id for item in memory.precedents} == {"approved", "corrected"}
    approved = next(item for item in memory.precedents if item.change_id == "approved")
    corrected = next(item for item in memory.precedents if item.change_id == "corrected")
    assert approved.materiality_level == "MINEUR"
    assert approved.change_nature == "REFORMULATION_EQUIVALENTE"
    assert approved.decision_origin == "analyst_approved"
    assert corrected.materiality_level == "MAJEUR"
    assert corrected.change_nature == "MODIFICATION_RESPONSABILITES"
    assert corrected.business_equivalence == "REFUTEE"
    assert corrected.materiality_confidence == "ELEVEE"
    assert corrected.rationale == "Le comité acquiert une autorité décisionnelle."
    assert corrected.supporting_evidence == (
        "Le verbe approuve remplace conseille.",
    )
    assert corrected.counterarguments == ("Le passage demeure court.",)
    assert corrected.decision_origin == "analyst_correction"
    assert memory.load_report.records_seen == 5
    assert memory.load_report.accepted_records == 2
    assert memory.load_report.rejected_records == 3


def test_loader_excludes_legacy_reviews_without_materiality_scope(tmp_path) -> None:
    path = tmp_path / "comparison.review_state.json"
    _write_json(
        path,
        {
            "schema_version": "review_state_v1",
            "bank_code": "rbc",
            "review_queue": [
                {
                    "table_key": "rbc::risk_management::pair",
                    "section": "risk_management",
                    "changes": [
                        {
                            "change_id": "validated-structured",
                            "change_type": "footnote_modified",
                            "validation_status": "approved",
                            "validated_by": "analyst",
                            "payload": {
                                "old_text": "Le risque de crédit est surveillé.",
                                "new_text": "Le risque de crédit est surveillé et escaladé.",
                                "materiality_level": "MODÉRÉ",
                                "business_equivalence": False,
                                "themes_amf": ["GESTION_RISQUE_CREDIT"],
                            },
                        },
                        {
                            "change_id": "validated-without-materiality",
                            "change_type": "footnote_removed",
                            "validation_status": "approved",
                            "payload": {
                                "old_text": "Une note est retirée.",
                                "new_text": "",
                            },
                        },
                        {
                            "change_id": "rejected-without-correction",
                            "change_type": "modified",
                            "validation_status": "rejected",
                            "validation_notes": "Le niveau proposé est incorrect.",
                            "payload": {
                                "old_text": "Avant.",
                                "new_text": "Après.",
                                "materiality_level": "MINEUR",
                            },
                        },
                        {
                            "change_id": "approved-but-not-final",
                            "change_type": "modified",
                            "validation_status": "approved",
                            "payload": {
                                "old_text": "Avant incertain.",
                                "new_text": "Après incertain.",
                                "materiality_level": "MODERE",
                                "decision_status": "À confirmer",
                                "review_required": True,
                            },
                        },
                        {
                            "change_id": "approved-but-provisional",
                            "change_type": "modified",
                            "validation_status": "approved",
                            "payload": {
                                "old_text": "Avant provisoire.",
                                "new_text": "Après provisoire.",
                                "materiality_level": "MODERE",
                                "decision_status": "PROVISOIRE",
                            },
                        },
                    ],
                }
            ],
        },
    )

    precedents = load_validated_precedents(path)

    assert precedents == []


def test_loader_preserves_adaptive_materiality_vocabulary(tmp_path) -> None:
    path = tmp_path / "text_comparison.json"
    _write_json(
        path,
        {
            "bank_code": "bmo",
            "section_comparisons": [
                {
                    "section_key": "gestion_capital",
                    "all_block_comparisons": [
                        {
                            "change_id": "adaptive-correction",
                            "semantic_text_t1": "Groupes d'exploitation.",
                            "semantic_text_t2": "Unités d'exploitation.",
                            "_analyst_review": {
                                "status": "corrected",
                                "decision_scope": "materiality",
                                "schema_version": (
                                    "analyst_materiality_review_v1"
                                ),
                                "structured_correction": {
                                    "materiality_level": "MODERE",
                                    "change_nature": [
                                        "MODIFICATION_TERMINOLOGIE",
                                        "MODIFICATION_PERIMETRE",
                                    ],
                                    "business_equivalence": "NON_DEMONTREE",
                                    "materiality_confidence": "MOYENNE",
                                    "evidence_sufficiency": "SUFFISANTE",
                                    "decision_status": "CONFIRME",
                                    "review_required": False,
                                    "is_relevant": True,
                                    "themes_amf": [
                                        "FONDS_PROPRES_REGLEMENTAIRES"
                                    ],
                                    "materiality_rationale": (
                                        "Le référent métier n'est pas démontré "
                                        "comme strictement équivalent."
                                    ),
                                    "supporting_evidence": [
                                        "Les référents métier diffèrent."
                                    ],
                                },
                            },
                        }
                    ],
                }
            ],
        },
    )

    precedents = load_validated_precedents(path)

    assert len(precedents) == 1
    assert precedents[0].change_nature == (
        "MODIFICATION_TERMINOLOGIE|MODIFICATION_PERIMETRE"
    )
    assert precedents[0].business_equivalence == "NON_DEMONTREE"
    assert precedents[0].materiality_confidence == "MOYENNE"


def test_lexical_packet_is_deterministic_and_contains_contrastive_case() -> None:
    major = AnalystPrecedent(
        precedent_id="major-governance",
        change_id="chg-major",
        bank_code="td",
        section_key="gouvernance_risques",
        text_before="Le comité conseille la haute direction.",
        text_after="Le comité approuve les limites de risque.",
        materiality_level="MAJEUR",
        change_nature="GOUVERNANCE",
        business_equivalence="REFUTEE",
        themes_amf=("GOUVERNANCE_RISQUES",),
        rationale="L'autorité décisionnelle change.",
    )
    minor = AnalystPrecedent(
        precedent_id="minor-wording",
        change_id="chg-minor",
        bank_code="bmo",
        section_key="gouvernance_risques",
        text_before="Le comité de gestion du risque conseille la direction.",
        text_after="Le comité du risque conseille la direction.",
        materiality_level="MINEUR",
        change_nature="RENOMMAGE",
        business_equivalence="CONFIRMEE",
        themes_amf=("GOUVERNANCE_RISQUES",),
        rationale="Le mandat et l'autorité demeurent identiques.",
    )
    moderate = AnalystPrecedent(
        precedent_id="moderate-capital",
        change_id="chg-moderate",
        bank_code="bns",
        section_key="capital",
        text_before="Suffisance du capital.",
        text_after="Adéquation des fonds propres.",
        materiality_level="MODERE",
        change_nature="TERMINOLOGIE",
        business_equivalence="INDETERMINE",
        themes_amf=("FONDS_PROPRES",),
    )
    query = PrecedentQuery(
        bank_code="rbc",
        section_key="gouvernance_risques",
        text_before="Le comité conseille la direction.",
        text_after="Le comité autorise les limites de risque.",
        change_nature="GOUVERNANCE",
        themes_amf=("GOUVERNANCE_RISQUES",),
        candidate_materiality_level="MAJEUR",
    )

    packet_a = PrecedentMemory(
        precedents=(minor, moderate, major)
    ).build_packet(query, positive_limit=1, contrastive_limit=1)
    packet_b = PrecedentMemory(
        precedents=(major, moderate, minor)
    ).build_packet(query, positive_limit=1, contrastive_limit=1)

    assert packet_a.to_dict() == packet_b.to_dict()
    assert packet_a.retrieval_method == "lexical"
    assert packet_a.anchor_materiality_level == "MAJEUR"
    assert packet_a.positive_precedents[0].precedent.precedent_id == "major-governance"
    assert len(packet_a.contrastive_precedents) == 1
    assert (
        packet_a.contrastive_precedents[0].precedent.materiality_level
        != "MAJEUR"
    )
    serialized = packet_a.to_prompt_json()
    assert '"query_fingerprint"' in serialized
    assert '"score_breakdown"' in serialized
    assert "comment" not in serialized


def test_lexical_retrieval_rejects_unrelated_text_even_in_same_section() -> None:
    precedent = AnalystPrecedent(
        precedent_id="same-section-unrelated",
        change_id="chg-unrelated",
        bank_code="bmo",
        section_key="gestion_risques",
        text_before="Le portefeuille hypothécaire est présenté par province.",
        text_after="Le portefeuille hypothécaire est présenté par région.",
        materiality_level="MINEUR",
        change_nature="MODIFICATION_TERMINOLOGIE",
        business_equivalence="CONFIRMEE",
        themes_amf=("GOUVERNANCE_RISQUES",),
    )
    query = PrecedentQuery(
        bank_code="bmo",
        section_key="gestion_risques",
        text_before="Le comité conseille la haute direction.",
        text_after="Le comité approuve les limites de risque.",
        change_nature="MODIFICATION_TERMINOLOGIE",
        themes_amf=("GOUVERNANCE_RISQUES",),
        candidate_materiality_level="MINEUR",
    )

    packet = PrecedentMemory(precedents=(precedent,)).build_packet(query)

    assert packet.positive_precedents == ()
    assert packet.contrastive_precedents == ()


def test_equal_priority_conflicting_precedents_are_quarantined() -> None:
    common = {
        "precedent_id": "conflicting-case",
        "change_id": "chg-conflict",
        "bank_code": "td",
        "section_key": "gestion_risques",
        "text_before": "Le comité conseille la direction.",
        "text_after": "Le comité approuve les limites.",
        "change_nature": "MODIFICATION_RESPONSABILITES",
        "business_equivalence": "REFUTEE",
        "decision_origin": "analyst_correction",
    }
    major = AnalystPrecedent(
        **common,
        materiality_level="MAJEUR",
    )
    moderate = AnalystPrecedent(
        **common,
        materiality_level="MODERE",
    )

    first_order = PrecedentMemory(precedents=(major, moderate))
    reverse_order = PrecedentMemory(precedents=(moderate, major))

    assert first_order.precedents == ()
    assert reverse_order.precedents == ()


def test_loader_quarantines_conflicting_registry_decisions(tmp_path) -> None:
    path = tmp_path / "analyst_precedents.json"
    common = {
        "precedent_id": "registry-conflict",
        "change_id": "chg-registry-conflict",
        "schema_version": "analyst_precedent_v1",
        "validation_status": "approved",
        "bank_code": "rbc",
        "section_key": "gestion_risques",
        "text_before": "Le comité conseille la direction.",
        "text_after": "Le comité approuve les limites.",
        "change_nature": ["MODIFICATION_RESPONSABILITES"],
        "business_equivalence": "REFUTEE",
        "materiality_confidence": "ELEVEE",
        "evidence_sufficiency": "SUFFISANTE",
        "decision_status": "CONFIRME",
        "review_required": False,
        "is_relevant": True,
        "themes_amf": ["GOUVERNANCE_RISQUES"],
        "materiality_rationale": (
            "Le passage modifie explicitement l'autorité décisionnelle."
        ),
        "supporting_evidence": [
            "Le verbe approuve remplace le verbe conseille."
        ],
    }
    _write_json(
        path,
        {
            "precedents": [
                {**common, "materiality_level": "MAJEUR"},
                {**common, "materiality_level": "MODERE"},
            ]
        },
    )

    memory = PrecedentMemory.from_paths(path)

    assert memory.precedents == ()
    assert memory.load_report.duplicate_records == 1
    assert any(
        "mis en quarantaine" in error
        for error in memory.load_report.errors
    )


def test_embedding_failure_falls_back_to_lexical_with_audit_reason() -> None:
    precedent = AnalystPrecedent(
        precedent_id="case-1",
        change_id="chg-1",
        bank_code="cibc",
        section_key="conformite",
        text_before="Le programme couvre la LBA.",
        text_after="Le programme couvre la LBA et les sanctions.",
        materiality_level="MAJEUR",
        change_nature="PERIMETRE_REGLEMENTAIRE",
    )

    class OfflineEmbeddingEngine:
        def embed(self, texts):
            raise RuntimeError("service indisponible")

    packet = PrecedentMemory(
        precedents=(precedent,),
        embedding_engine=OfflineEmbeddingEngine(),
    ).build_packet(
        PrecedentQuery(
            section_key="conformite",
            text_before="La politique vise la LBA.",
            text_after="La politique vise la LBA et les sanctions.",
            change_nature="PERIMETRE_REGLEMENTAIRE",
        ),
        positive_limit=1,
    )

    assert packet.retrieval_method == "lexical"
    assert packet.positive_precedents
    assert packet.fallback_reason.startswith("RuntimeError:")


def test_prompt_packet_truncates_large_source_texts() -> None:
    precedent = AnalystPrecedent(
        precedent_id="long-case",
        change_id="long",
        bank_code="bnc",
        section_key="donnees",
        text_before="alpha " * 400,
        text_after="beta " * 400,
        materiality_level="MAJEUR",
        change_nature="CONTROLE_DONNEES",
        rationale="r" * 1_000,
        supporting_evidence=("e" * 800,),
    )
    packet = PrecedentMemory(precedents=(precedent,)).build_packet(
        PrecedentQuery(
            text_before="alpha " * 20,
            text_after="beta " * 20,
            change_nature="CONTROLE_DONNEES",
        ),
        positive_limit=1,
    )

    item = packet.to_dict()["positive_precedents"][0]
    assert len(item["text_before"]) == 1_000
    assert len(item["text_after"]) == 1_000
    assert len(item["analyst_decision"]["rationale"]) == 600
    assert len(item["analyst_decision"]["supporting_evidence"][0]) == 400
