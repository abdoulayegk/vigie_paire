from __future__ import annotations

from vigilance.text_comparison.consolidation import (
    build_observations_from_group_specs,
    candidate_batches_for_llm,
)


def _change(
    change_id: str,
    *,
    subsection: str,
    page: int,
    summary: str,
    impact: str = "MAJEUR",
    nouvelle_idee: bool = True,
) -> dict:
    return {
        "change_id": change_id,
        "section_key": "gestion_risques",
        "subsection_heading": subsection,
        "current_hierarchy_path": f"Gestion des risques > {subsection}",
        "diff_type": "modified",
        "pages_t2": [page],
        "source_text_t1": f"Avant {change_id}",
        "source_text_t2": f"Apres {change_id}",
        "change_summary": summary,
        "objective_matches": [{"label": "IA"}],
        "genai_triage": {
            "is_relevant": True,
            "themes_amf": ["RISQUE_EMERGENT"],
            "impact_level": impact,
            "nouvelle_idee": nouvelle_idee,
            "action_requise": "revue_prioritaire",
            "category": "RISQUE",
        },
    }


def test_build_observations_from_group_specs_uses_llm_group_decision() -> None:
    changes = [
        _change(
            "ia-1",
            subsection="Risque lie a l'intelligence artificielle",
            page=101,
            summary="La portee du risque IA est elargie.",
        ),
        _change(
            "ia-2",
            subsection="Risque lie a l'intelligence artificielle",
            page=101,
            summary="Une directive de gouvernance IA est ajoutee.",
        ),
    ]

    observations = build_observations_from_group_specs(
        changes,
        [
            {
                "source_change_ids": ["ia-1", "ia-2"],
                "observation_title": "Renforcement de la divulgation IA",
                "analyst_summary": "BMO renforce la divulgation sur l'IA.",
                "rationale": "Les deux changements décrivent le même renforcement du risque IA.",
                "impact_level": "MODERE",
                "action_requise": "investigation",
                "nouvelle_idee": True,
                "themes_amf": ["RISQUE_EMERGENT", "GOUVERNANCE"],
                "nouvelle_idee_justification": (
                    "OUI - La divulgation IA devient une observation consolidée à suivre."
                ),
            }
        ],
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation["observation_type"] == "consolidated_intra_section"
    assert observation["consolidation_method"] == "llm"
    assert observation["consolidated_change_count"] == 2
    assert observation["source_change_ids"] == ["ia-1", "ia-2"]
    assert observation["change_summary"] == "BMO renforce la divulgation sur l'IA."
    assert observation["genai_triage"]["impact_level"] == "MODERE"
    assert observation["genai_triage"]["action_requise"] == "investigation"
    assert observation["genai_triage"]["themes_amf"] == ["RISQUE_EMERGENT", "GOUVERNANCE"]
    assert observation["genai_triage"]["nouvelle_idee"] is True


def test_build_observations_from_group_specs_allows_cross_subsection_llm_group() -> None:
    changes = [
        _change("ia-1", subsection="Risque lie a l'intelligence artificielle", page=101, summary="IA."),
        _change("cyber-1", subsection="Risque lie a la cybersecurite", page=101, summary="Cyber."),
    ]

    observations = build_observations_from_group_specs(
        changes,
        [
            {
                "source_change_ids": ["ia-1", "cyber-1"],
                "observation_title": "Risque technologique émergent",
                "analyst_summary": "BMO relie l'IA et la cybersécurité au risque technologique.",
                "rationale": "Les deux changements appuient une même observation de vigie.",
                "impact_level": "MAJEUR",
                "action_requise": "revue_prioritaire",
                "nouvelle_idee": True,
                "themes_amf": ["RISQUE_EMERGENT"],
                "nouvelle_idee_justification": "OUI - L'observation croise IA et cybersécurité.",
            }
        ],
    )

    assert len(observations) == 1
    assert observations[0]["source_change_ids"] == ["ia-1", "cyber-1"]
    assert observations[0]["observation_type"] == "consolidated_intra_section"


def test_build_observations_from_group_specs_rejects_invented_ids() -> None:
    changes = [
        _change("ia-1", subsection="Risque lie a l'intelligence artificielle", page=101, summary="IA."),
        _change("ia-2", subsection="Risque lie a l'intelligence artificielle", page=102, summary="IA 2."),
    ]

    observations = build_observations_from_group_specs(
        changes,
        [
            {
                "source_change_ids": ["ia-1", "id-invente"],
                "observation_title": "Groupe invalide",
                "analyst_summary": "Ne doit pas être utilisé.",
                "rationale": "Contient un identifiant inventé.",
                "impact_level": "MAJEUR",
                "action_requise": "revue_prioritaire",
                "nouvelle_idee": True,
                "themes_amf": ["RISQUE_EMERGENT"],
                "nouvelle_idee_justification": "OUI - invalide.",
            }
        ],
    )

    assert len(observations) == 2
    assert {obs["change_id"] for obs in observations} == {"ia-1", "ia-2"}
    assert all(obs["observation_type"] == "atomic_change" for obs in observations)


def test_candidate_batches_for_llm_batches_by_size_not_subsection_or_page() -> None:
    changes = [
        _change("ia-1", subsection="Risque lie a l'intelligence artificielle", page=101, summary="IA 1."),
        _change("ia-2", subsection="Risque lie a l'intelligence artificielle", page=130, summary="IA 2."),
        _change("cyber-1", subsection="Risque lie a la cybersecurite", page=102, summary="Cyber."),
    ]

    batches = candidate_batches_for_llm(changes, max_changes_per_batch=10, max_batch_chars=100_000)

    assert [[change["change_id"] for change in batch] for batch in batches] == [
        ["ia-1", "ia-2", "cyber-1"]
    ]
