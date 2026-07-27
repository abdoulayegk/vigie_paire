from __future__ import annotations

import json

import pytest

from vigilance.dash_app.callbacks.text_flow import _structured_text_correction
from vigilance.dash_app.layouts.page_text_analysis import _text_review_progress
from vigilance.dash_app.services.text_review import (
    apply_text_review_decision,
    is_final_direct_triage,
    write_text_review_to_disk,
)


def _analyst_correction(
    *,
    level: str = "MAJEUR",
    is_relevant: bool = True,
    review_required: bool = False,
) -> dict:
    if is_relevant:
        themes = ["GOUVERNANCE_RISQUES"]
        nature = ["MODIFICATION_RESPONSABILITES"]
        equivalence = "REFUTEE"
    else:
        themes = []
        nature = ["REFORMULATION_EQUIVALENTE"]
        equivalence = "CONFIRMEE"
    return {
        "materiality_level": level,
        "change_nature": nature,
        "is_relevant": is_relevant,
        "nouvelle_idee": False,
        "themes_amf": themes,
        "business_equivalence": equivalence,
        "materiality_confidence": "MOYENNE",
        "evidence_sufficiency": (
            "PARTIELLE" if review_required else "SUFFISANTE"
        ),
        "decision_status": (
            "A_CONFIRMER" if review_required else "CONFIRME"
        ),
        "review_required": review_required,
        "materiality_rationale": (
            "Le changement modifie la lecture du rôle déclaré."
            if is_relevant
            else "Les deux formulations conservent le même sens métier."
        ),
        "supporting_evidence": [
            "La comparaison avant-après soutient explicitement ce jugement."
        ],
        "counterarguments": [],
    }


def test_apply_text_review_decision_updates_all_change_buckets() -> None:
    payload = {
        "section_comparisons": [
            {
                "all_block_comparisons": [
                    {
                        "change_id": "chg-1",
                        "genai_triage": {
                            "impact_level": "MINEUR",
                            "materiality_level": "MINEUR",
                            "is_relevant": True,
                            "themes_amf": ["GOUVERNANCE_RISQUES"],
                        },
                    }
                ],
                "block_comparisons": [
                    {
                        "change_id": "chg-1",
                        "genai_triage": {
                            "impact_level": "MINEUR",
                            "materiality_level": "MINEUR",
                            "is_relevant": True,
                            "themes_amf": ["GOUVERNANCE_RISQUES"],
                        },
                    }
                ],
            }
        ]
    }

    updated, found = apply_text_review_decision(
        payload,
        change_id="chg-1",
        status="rejected",
        comment="Pas une nouvelle idée.",
    )

    assert found is True
    for bucket in ("all_block_comparisons", "block_comparisons"):
        review = updated["section_comparisons"][0][bucket][0]["_analyst_review"]
        assert review["status"] == "rejected"
        assert review["comment"] == "Pas une nouvelle idée."
        assert review["nouvelle_idee_override"] is False


def test_write_text_review_to_disk_can_skip_excel_regeneration(tmp_path, monkeypatch) -> None:
    payload = {
        "bank_code": "bnc",
        "quarter_current": "2025_t3",
        "quarter_previous": "2025_t2",
        "section_comparisons": [
            {
                "all_block_comparisons": [
                    {
                        "change_id": "chg-1",
                        "_analyst_review": {"status": "skipped", "comment": "À revoir plus tard."},
                    }
                ]
            }
        ],
    }
    target_dir = tmp_path / "bnc" / "2025_t3_vs_2025_t2"
    target_dir.mkdir(parents=True)
    target_json = target_dir / "text_comparison.json"
    target_json.write_text("{}", encoding="utf-8")
    called = {"excel": False}

    def _fake_generate_text_comparison_excel(*args, **kwargs):
        called["excel"] = True

    monkeypatch.setattr("vigilance.dash_app.services.text_review.TEXT_COMPARISON_DIR", tmp_path)
    monkeypatch.setattr(
        "vigilance.dash_app.services.text_review.generate_text_comparison_excel",
        _fake_generate_text_comparison_excel,
    )

    assert write_text_review_to_disk(payload, regenerate_excel=False) is True

    saved = json.loads(target_json.read_text(encoding="utf-8"))
    review = saved["section_comparisons"][0]["all_block_comparisons"][0]["_analyst_review"]
    assert review["status"] == "skipped"
    assert called["excel"] is False


def test_apply_text_review_decision_persists_structured_correction() -> None:
    payload = {
        "section_comparisons": [
            {
                "all_block_comparisons": [
                    {
                        "change_id": "chg-1",
                        "genai_triage": {
                            "impact_level": "MINEUR",
                            "materiality_level": "MINEUR",
                            "is_relevant": True,
                            "themes_amf": ["GOUVERNANCE_RISQUES"],
                        },
                    }
                ],
                "block_comparisons": [
                    {
                        "change_id": "chg-1",
                        "genai_triage": {
                            "impact_level": "MINEUR",
                            "materiality_level": "MINEUR",
                            "is_relevant": True,
                            "themes_amf": ["GOUVERNANCE_RISQUES"],
                        },
                    }
                ],
            }
        ]
    }

    updated, found = apply_text_review_decision(
        payload,
        change_id="chg-1",
        status="rejected",
        reviewer="analyste-amf",
        structured_correction={
            "materiality_level": "majeur",
            "change_nature": ["modification_responsabilites"],
            "is_relevant": True,
            "nouvelle_idee": False,
            "themes_amf": ["GOUVERNANCE_RISQUES"],
            "business_equivalence": "REFUTEE",
            "materiality_confidence": "ELEVEE",
            "evidence_sufficiency": "SUFFISANTE",
            "decision_status": "CONFIRME",
            "review_required": False,
            "materiality_rationale": (
                "Le comité acquiert une autorité d'approbation."
            ),
            "supporting_evidence": [
                "Le verbe approuve remplace le verbe conseille."
            ],
            "counterarguments": [],
        },
    )

    assert found is True
    review = updated["section_comparisons"][0]["all_block_comparisons"][0][
        "_analyst_review"
    ]
    assert review["corrected_materiality_level"] == "MAJEUR"
    assert review["status"] == "corrected"
    assert review["workflow_status"] == "completed"
    assert review["decision_scope"] == "materiality"
    assert review["corrected_change_nature"] == [
        "MODIFICATION_RESPONSABILITES"
    ]
    assert review["structured_correction"]["materiality_rationale"].startswith(
        "Le comité"
    )
    assert "nouvelle_idee_override" not in review
    effective = updated["section_comparisons"][0][
        "all_block_comparisons"
    ][0]["genai_triage"]
    assert effective["impact_level"] == "MAJEUR"
    assert effective["materiality_level"] == "MAJEUR"
    assert effective["source"] == "analyst_correction"
    assert effective["materiality_decision_basis"] == "analyst_correction"
    assert effective["action_requise"] == "revue_prioritaire"
    assert review["original_genai_triage"]["impact_level"] == "MINEUR"


def test_structured_correction_requires_level_nature_and_rationale() -> None:
    with pytest.raises(ValueError, match="materiality_level"):
        apply_text_review_decision(
            {"section_comparisons": []},
            change_id="chg-1",
            status="rejected",
            structured_correction={"change_nature": ["AUTRE"]},
        )


def test_ui_correction_promotes_non_relevant_change_when_level_is_major() -> None:
    payload = {
        "section_comparisons": [
            {
                "all_block_comparisons": [
                    {
                        "change_id": "chg-1",
                        "genai_triage": {
                            "is_relevant": False,
                            "themes_amf": [],
                            "supporting_evidence": [
                                "Le rôle de conseil devient un rôle d'approbation."
                            ],
                        },
                    }
                ]
            }
        ]
    }

    correction = _structured_text_correction(
        payload,
        change_id="chg-1",
        comment="L'autorité décisionnelle change.",
        materiality="majeur",
        nature=["modification_responsabilites"],
        equivalence="refutee",
        themes=["GOUVERNANCE_RISQUES"],
        is_relevant=True,
        nouvelle_idee=False,
        confidence="ELEVEE",
        evidence_sufficiency="SUFFISANTE",
        supporting_evidence=(
            "Le verbe approuve remplace le verbe conseille."
        ),
        counterarguments="",
    )

    assert correction is not None
    assert correction["materiality_level"] == "MAJEUR"
    assert correction["is_relevant"] is True
    assert correction["change_nature"] == [
        "MODIFICATION_RESPONSABILITES"
    ]
    assert correction["business_equivalence"] == "REFUTEE"
    assert correction["themes_amf"] == ["GOUVERNANCE_RISQUES"]
    assert correction["decision_status"] == "CONFIRME"
    assert correction["review_required"] is False


def test_ui_correction_requires_explicit_rationale_and_nature() -> None:
    assert (
        _structured_text_correction(
            {"section_comparisons": []},
            change_id="chg-1",
            comment="",
            materiality="MODERE",
            nature=["MODIFICATION_TERMINOLOGIE"],
            equivalence="NON_DEMONTREE",
            themes=["FONDS_PROPRES_REGLEMENTAIRES"],
            is_relevant=True,
            nouvelle_idee=False,
            confidence="MOYENNE",
            evidence_sufficiency="PARTIELLE",
            supporting_evidence="La terminologie change.",
            counterarguments="",
        )
        is None
    )


def test_approval_requires_a_fully_final_direct_materiality_decision() -> None:
    final_triage = {
        "materiality_level": "MAJEUR",
        "change_nature": ["MODIFICATION_RESPONSABILITES"],
        "business_equivalence": "REFUTEE",
        "materiality_confidence": "ELEVEE",
        "evidence_sufficiency": "SUFFISANTE",
        "decision_status": "CONFIRME",
        "review_required": False,
        "supporting_evidence": [
            "Le verbe approuve remplace le verbe conseille."
        ],
        "is_relevant": True,
        "nouvelle_idee": False,
        "themes_amf": ["GOUVERNANCE_RISQUES"],
    }

    assert is_final_direct_triage(final_triage) is True
    assert (
        is_final_direct_triage(
            {**final_triage, "evidence_sufficiency": "PARTIELLE"}
        )
        is False
    )
    assert (
        is_final_direct_triage(
            {**final_triage, "business_equivalence": "CONFIRMEE"}
        )
        is False
    )

    payload = {
        "section_comparisons": [
            {
                "all_block_comparisons": [
                    {
                        "change_id": "provisional",
                        "genai_triage": {
                            **final_triage,
                            "decision_status": "A_CONFIRMER",
                            "review_required": True,
                        },
                    }
                ]
            }
        ]
    }
    with pytest.raises(
        ValueError,
        match="doit être corrigée de façon structurée",
    ):
        apply_text_review_decision(
            payload,
            change_id="provisional",
            status="approved",
        )


def test_provisional_correction_stays_in_the_remaining_workflow() -> None:
    payload = {
        "section_comparisons": [
            {
                "all_block_comparisons": [
                    {
                        "change_id": "pending-correction",
                        "diff_type": "modified",
                        "genai_triage": {
                            "materiality_level": "MINEUR",
                            "impact_level": "MINEUR",
                            "is_relevant": True,
                            "themes_amf": ["GOUVERNANCE_RISQUES"],
                        },
                    }
                ]
            }
        ]
    }

    updated, found = apply_text_review_decision(
        payload,
        change_id="pending-correction",
        status="corrected",
        structured_correction=_analyst_correction(
            level="MODERE",
            review_required=True,
        ),
    )

    review = updated["section_comparisons"][0][
        "all_block_comparisons"
    ][0]["_analyst_review"]
    progress = _text_review_progress(updated["section_comparisons"])
    assert found is True
    assert review["status"] == "corrected"
    assert review["workflow_status"] == "pending"
    assert progress["pending"] == 1
    assert progress["corrected"] == 0
    assert progress["decided"] == 0
    assert progress["remaining"] == 1


def test_correction_rebuilds_retained_scope_and_global_summaries() -> None:
    payload = {
        "bank_code": "bmo",
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "summary": {"retained_changes": 0, "all_changes": 1},
                "all_block_comparisons": [
                    {
                        "change_id": "promoted",
                        "diff_type": "modified",
                        "change_summary": "Le comité acquiert un pouvoir.",
                        "genai_triage": {
                            "impact_level": "MINEUR",
                            "materiality_level": "MINEUR",
                            "is_relevant": False,
                            "themes_amf": [],
                            "nouvelle_idee": False,
                        },
                    }
                ],
                "block_comparisons": [],
            }
        ],
    }

    updated, found = apply_text_review_decision(
        payload,
        change_id="promoted",
        status="corrected",
        structured_correction=_analyst_correction(),
    )

    section = updated["section_comparisons"][0]
    assert found is True
    assert len(section["block_comparisons"]) == 1
    assert section["summary"]["retained_changes"] == 1
    assert updated["global_summary"]["counts"]["total_relevant"] == 1
    assert updated["global_summary"]["counts"]["by_impact"]["MAJEUR"] == 1


def test_correction_can_make_a_change_non_relevant_and_remove_it_from_scope() -> None:
    payload = {
        "bank_code": "bmo",
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "all_block_comparisons": [
                    {
                        "change_id": "demoted",
                        "diff_type": "modified",
                        "genai_triage": {
                            "impact_level": "MAJEUR",
                            "materiality_level": "MAJEUR",
                            "is_relevant": True,
                            "themes_amf": ["GOUVERNANCE_RISQUES"],
                            "nouvelle_idee": True,
                        },
                    }
                ],
                "block_comparisons": [
                    {
                        "change_id": "demoted",
                        "diff_type": "modified",
                        "genai_triage": {
                            "impact_level": "MAJEUR",
                            "materiality_level": "MAJEUR",
                            "is_relevant": True,
                            "themes_amf": ["GOUVERNANCE_RISQUES"],
                            "nouvelle_idee": True,
                        },
                    }
                ],
            }
        ],
    }

    updated, found = apply_text_review_decision(
        payload,
        change_id="demoted",
        status="corrected",
        structured_correction=_analyst_correction(
            level="MINEUR",
            is_relevant=False,
        ),
    )

    section = updated["section_comparisons"][0]
    effective = section["all_block_comparisons"][0]["genai_triage"]
    assert found is True
    assert effective["is_relevant"] is False
    assert effective["nouvelle_idee"] is False
    assert effective["themes_amf"] == []
    assert section["block_comparisons"] == []
    assert updated["global_summary"]["counts"]["total_relevant"] == 0


def test_second_correction_preserves_initial_triage_and_decision_history() -> None:
    payload = {
        "bank_code": "bmo",
        "section_comparisons": [
            {
                "section_key": "gestion_risques",
                "all_block_comparisons": [
                    {
                        "change_id": "twice-corrected",
                        "diff_type": "modified",
                        "genai_triage": {
                            "impact_level": "MINEUR",
                            "materiality_level": "MINEUR",
                            "is_relevant": True,
                            "themes_amf": ["GOUVERNANCE_RISQUES"],
                        },
                    }
                ],
            }
        ],
    }
    first, _ = apply_text_review_decision(
        payload,
        change_id="twice-corrected",
        status="corrected",
        structured_correction=_analyst_correction(level="MAJEUR"),
    )
    second, _ = apply_text_review_decision(
        first,
        change_id="twice-corrected",
        status="corrected",
        structured_correction=_analyst_correction(level="MODERE"),
    )

    change = second["section_comparisons"][0][
        "all_block_comparisons"
    ][0]
    review = change["_analyst_review"]
    assert review["original_genai_triage"]["impact_level"] == "MINEUR"
    assert review["decision_history"][0]["corrected_materiality_level"] == (
        "MAJEUR"
    )
    assert change["genai_triage"]["materiality_level"] == "MODERE"

    with pytest.raises(
        ValueError,
        match="doit rester enregistrée comme correction",
    ):
        apply_text_review_decision(
            second,
            change_id="twice-corrected",
            status="approved",
        )
