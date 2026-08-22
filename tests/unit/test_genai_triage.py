"""Tests for the GenAI batch triage module (vigie.comparaison.triage.genai_triage)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from vigie.comparaison.triage.genai_triage import (
    _TRIAGE_SYSTEM_PROMPT,
    _build_change_prompt,
    _has_meaningful_diff,
    _validate_summary_response,
    _validate_triage_response,
    enrich_comparison_with_genai_triage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pair(*, added=None, removed=None, renamed=None, fn_added=None, status="modifie"):
    return {
        "previous_table_id": "tbl_001",
        "current_table_id": "tbl_002",
        "previous_table": {"title": "Gestion du capital", "section": "gestion_capital"},
        "current_table": {"title": "Gestion du capital", "section": "gestion_capital"},
        "technical_diff": {
            "table_level_change": status,
            "indicators_added": added or [],
            "indicators_removed": removed or [],
            "indicators_renamed": renamed or [],
            "footnotes_added": fn_added or [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
        },
    }


def _make_comparison(pairs=None, added=None, removed=None):
    return {
        "schema_version": 2,
        "pair_comparisons": pairs or [],
        "matching": {
            "tables_added": added or [],
            "tables_removed": removed or [],
        },
    }


# ---------------------------------------------------------------------------
# _has_meaningful_diff
# ---------------------------------------------------------------------------


class TestHasMeaningfulDiff:
    def test_stable_no_changes(self):
        pair = _make_pair(status="inchange")
        assert _has_meaningful_diff(pair) is False

    def test_modified_status(self):
        pair = _make_pair(status="modifie")
        assert _has_meaningful_diff(pair) is True

    def test_stable_with_added_indicators(self):
        pair = _make_pair(
            status="inchange",
            added=[{"value": "CET1"}],
        )
        assert _has_meaningful_diff(pair) is True


# ---------------------------------------------------------------------------
# _validate_triage_response
# ---------------------------------------------------------------------------


def _valid_justification_oui() -> str:
    return (
        "OUI - Nouvel élément à surveiller : Oui.\n\n"
        "Sujet détecté : Ratio prudentiel, exigence réglementaire, information ajoutée.\n\n"
        "Ce qui change : Le ratio TLAC est ajouté au TABLEAU 11 et n'apparaissait "
        "pas au trimestre précédent. Le changement introduit une nouvelle "
        "information prudentielle dans la divulgation.\n\n"
        "Pertinence métier : Ce changement est pertinent pour la vigie bancaire "
        "parce qu'il touche les exigences réglementaires et la présentation des "
        "ratios prudentiels. Une nouvelle divulgation TLAC peut modifier la "
        "lecture de la capacité d'absorption des pertes et la comparabilité "
        "entre banques canadiennes.\n\n"
        "Point de surveillance : Le point à retenir est que la banque rend visible "
        "un indicateur réglementaire supplémentaire, ce qui enrichit la lecture "
        "du capital et du cadre prudentiel."
    )


def _valid_justification_non() -> str:
    return (
        "NON - Nouvel élément à surveiller : Non.\n\n"
        "Sujet détecté : Mise à jour rédactionnelle sans changement de fond.\n\n"
        "Ce qui change : Le changement observé est une simple mise à jour de "
        "date entre les deux trimestres. Le contenu de la divulgation demeure "
        "stable et aucun nouveau sujet réglementaire n'est ajouté.\n\n"
        "Pertinence métier : Ce changement n'est pas une nouvelle idée pour la "
        "vigie bancaire parce qu'il ne touche aucun risque, seuil prudentiel, "
        "méthode de calcul, gouvernance ou exigence de conformité. Il ne "
        "modifie pas la lecture métier du rapport.\n\n"
        "Point de surveillance : Le point à retenir est que la substance de la "
        "divulgation demeure inchangée; la ligne reflète une mise à jour "
        "rédactionnelle attendue plutôt qu'un nouveau signal de surveillance."
    )


class TestValidateTriageResponse:
    def test_system_prompt_requests_analyst_style_justification(self):
        assert "NOTE D'ANALYSTE" in _TRIAGE_SYSTEM_PROMPT
        assert "Pertinence métier" in _TRIAGE_SYSTEM_PROMPT
        assert "Sujet détecté" in _TRIAGE_SYSTEM_PROMPT
        assert "simple liste de codes AMF" in _TRIAGE_SYSTEM_PROMPT
        assert "COUVERTURE DONNÉES / TIERS / CLOUD" in _TRIAGE_SYSTEM_PROMPT
        assert "CHANGEMENT DE POSTURE" in _TRIAGE_SYSTEM_PROMPT
        assert "impact_it" in _TRIAGE_SYSTEM_PROMPT

    def test_valid_response(self):
        data = {
            "is_relevant": True,
            "themes_amf": ["DIVULGATION_AJOUT", "RATIOS_REGLEMENTAIRES"],
            "nouvelle_idee": True,
            "nouvelle_idee_justification": _valid_justification_oui(),
            "category": "REGLEMENTAIRE",
            "relevance_score": "ELEVEE",
            "risk_level": "ELEVE",
            "confidence": 0.92,
            "explanation": "Ajout lié à Bâle III.",
            "impact_type": "contenu",
            "project_phase": "pilier_3",
            "action_requise": "revue_prioritaire",
            "reference_reglementaire": "Bâle III — CET1",
            "impact_description": "Nouveau ratio prudentiel ajouté.",
        }
        result = _validate_triage_response(data)
        assert result["is_relevant"] is True
        assert result["themes_amf"] == ["DIVULGATION_AJOUT", "RATIOS_REGLEMENTAIRES"]
        assert result["nouvelle_idee"] is True
        assert result["nouvelle_idee_justification"].startswith("OUI")
        assert result["category"] == "REGLEMENTAIRE"
        assert result["relevance_score"] == "ELEVEE"
        assert result["risk_level"] == "ELEVE"
        assert result["confidence"] == 0.92
        assert result["impact_type"] == "contenu"
        assert result["project_phase"] == "pilier_3"
        assert result["action_requise"] == "revue_prioritaire"
        assert result["reference_reglementaire"] == "Bâle III — CET1"
        assert result["impact_description"] == "Nouveau ratio prudentiel ajouté."
        assert result["impact_it"] == "INDETERMINE"
        assert result["changement_posture"] == "INDETERMINE"
        assert result["statut_mise_en_oeuvre"] == "INDETERMINE"
        assert result["confiance_posture"] == "INDETERMINE"
        assert result["source"] == "llm"

    def test_none_response(self):
        with pytest.raises(ValueError, match="empty or invalid"):
            _validate_triage_response(None)

    def _base_payload(self) -> dict:
        """Helper : payload minimal valide (non pertinent avec justification)."""
        return {
            "is_relevant": False,
            "themes_amf": [],
            "nouvelle_idee": False,
            "nouvelle_idee_justification": _valid_justification_non(),
            "category": "NON_PERTINENT",
            "action_requise": "aucune",
        }

    def test_invalid_category_defaults(self):
        payload = self._base_payload()
        payload["category"] = "FAKE"
        result = _validate_triage_response(payload)
        # FAKE n'est pas dans VALID_CATEGORIES → fallback INCONNU
        # Mais comme is_relevant=False, l'invariant force NON_PERTINENT en aval ?
        # Non — l'invariant ne contraint pas category, donc reste INCONNU.
        assert result["category"] == "INCONNU"

    def test_invalid_relevance_defaults(self):
        payload = self._base_payload()
        payload["relevance_score"] = "SUPER"
        result = _validate_triage_response(payload)
        assert result["relevance_score"] == "FAIBLE"

    def test_invalid_risk_level_defaults(self):
        payload = self._base_payload()
        payload["risk_level"] = "CRITIQUE"
        result = _validate_triage_response(payload)
        assert result["risk_level"] == "FAIBLE"

    def test_irrelevant_forces_mineur_even_if_llm_returns_eleve(self):
        payload = self._base_payload()
        payload["risk_level"] = "ELEVE"
        result = _validate_triage_response(payload)
        assert result["is_relevant"] is False
        assert result["impact_level"] == "MINEUR"
        assert result["risk_level"] == "FAIBLE"

    def test_confidence_clamped(self):
        payload = self._base_payload()
        payload["confidence"] = 2.5
        result = _validate_triage_response(payload)
        assert result["confidence"] == 1.0
        payload["confidence"] = -1.0
        result2 = _validate_triage_response(payload)
        assert result2["confidence"] == 0.0

    def test_invalid_impact_type_defaults(self):
        payload = self._base_payload()
        payload["impact_type"] = "FAKE"
        result = _validate_triage_response(payload)
        assert result["impact_type"] == "non_substantif"

    def test_invalid_project_phase_defaults(self):
        payload = self._base_payload()
        payload["project_phase"] = "FAKE"
        result = _validate_triage_response(payload)
        assert result["project_phase"] == "autre"

    def test_invalid_action_defaults(self):
        payload = self._base_payload()
        payload["action_requise"] = "FAKE"
        result = _validate_triage_response(payload)
        assert result["action_requise"] == "aucune"

    def test_invalid_theme_codes_filtered_out(self):
        result = _validate_triage_response(
            {
                "is_relevant": True,
                "themes_amf": ["DIVULGATION_AJOUT", "FAKE_THEME", "RISQUE_EMERGENT"],
                "nouvelle_idee": True,
                "nouvelle_idee_justification": _valid_justification_oui(),
                "category": "RISQUE",
                "relevance_score": "ELEVEE",
                "risk_level": "ELEVE",
                "confidence": 0.8,
                "action_requise": "revue_prioritaire",
            }
        )
        assert result["themes_amf"] == ["DIVULGATION_AJOUT", "RISQUE_EMERGENT"]

    def test_data_and_third_party_cloud_themes_are_accepted(self):
        result = _validate_triage_response(
            {
                "is_relevant": True,
                "themes_amf": ["RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"],
                "nouvelle_idee": True,
                "nouvelle_idee_justification": _valid_justification_oui(),
                "category": "RISQUE",
                "relevance_score": "ELEVEE",
                "risk_level": "ELEVE",
                "impact_it": "ELEVE",
                "impact_it_justification": (
                    "Éléments observés : Le rapport décrit une migration des "
                    "données vers le cloud.\n\n"
                    "Conséquence probable : Cette migration exige des changements "
                    "d'architecture et de contrôle.\n\n"
                    "Limite de l'analyse : Le calendrier et le périmètre de la "
                    "migration ne sont pas précisés."
                ),
                "changement_posture": "RENFORCEMENT",
                "justification_posture": (
                    "Preuve : La banque renforce la surveillance et les contrôles "
                    "contractuels appliqués au fournisseur critique.\n\n"
                    "Effet sur la gestion du risque : Le niveau d'encadrement du "
                    "tiers critique augmente.\n\n"
                    "Justification du statut : Le rapport décrit des travaux en "
                    "cours, sans confirmer leur achèvement.\n\n"
                    "Justification de la confiance : Le renforcement est formulé "
                    "explicitement dans le texte."
                ),
                "statut_mise_en_oeuvre": "EN_COURS",
                "confiance_posture": "ELEVEE",
                "confidence": 0.8,
                "action_requise": "revue_prioritaire",
            }
        )
        assert result["themes_amf"] == ["RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"]
        assert result["impact_it"] == "ELEVE"
        assert result["changement_posture"] == "RENFORCEMENT"
        assert result["statut_mise_en_oeuvre"] == "EN_COURS"
        assert result["confiance_posture"] == "ELEVEE"

    def test_posture_without_evidence_falls_back_to_unknown(self):
        result = _validate_triage_response(
            {
                "is_relevant": True,
                "themes_amf": ["RISQUE_TIERS_CLOUD"],
                "nouvelle_idee": False,
                "nouvelle_idee_justification": _valid_justification_non(),
                "category": "RISQUE",
                "risk_level": "MODERE",
                "changement_posture": "RENFORCEMENT",
                "statut_mise_en_oeuvre": "MIS_EN_OEUVRE",
                "action_requise": "information",
            }
        )

        assert result["changement_posture"] == "INDETERMINE"
        assert result["justification_posture"] == ""
        assert result["statut_mise_en_oeuvre"] == "INDETERMINE"
        assert result["confiance_posture"] == "INDETERMINE"

    def test_it_impact_without_justification_falls_back_to_unknown(self):
        result = _validate_triage_response(
            {
                "is_relevant": True,
                "themes_amf": ["RISQUE_TIERS_CLOUD"],
                "nouvelle_idee": True,
                "nouvelle_idee_justification": _valid_justification_oui(),
                "category": "RISQUE",
                "risk_level": "ELEVE",
                "impact_it": "ELEVE",
                "impact_it_justification": "Trop court",
                "changement_posture": "INDETERMINE",
                "action_requise": "revue_prioritaire",
            }
        )
        assert result["impact_it"] == "INDETERMINE"
        assert result["impact_it_justification"] == ""

    def test_relevant_without_themes_raises(self):
        """Invariant : is_relevant=True sans themes_amf -> erreur."""
        with pytest.raises(ValueError, match="AMF invariants"):
            _validate_triage_response(
                {
                    "is_relevant": True,
                    "themes_amf": [],
                    "nouvelle_idee": True,
                    "nouvelle_idee_justification": _valid_justification_oui(),
                    "action_requise": "revue_prioritaire",
                }
            )

    def test_short_justification_raises(self):
        """Invariant : justification trop courte -> erreur."""
        with pytest.raises(ValueError, match="AMF invariants"):
            _validate_triage_response(
                {
                    "is_relevant": True,
                    "themes_amf": ["DIVULGATION_AJOUT"],
                    "nouvelle_idee": True,
                    "nouvelle_idee_justification": "OUI ratio TLAC ajoute.",
                    "action_requise": "revue_prioritaire",
                }
            )

    def test_justification_wrong_prefix_raises(self):
        """Invariant : prefixe incoherent -> erreur."""
        with pytest.raises(ValueError, match="AMF invariants"):
            _validate_triage_response(
                {
                    "is_relevant": True,
                    "themes_amf": ["DIVULGATION_AJOUT"],
                    "nouvelle_idee": True,
                    "nouvelle_idee_justification": (
                        "NON le ratio CET1 existait deja au T1 et seule sa valeur a change. "
                        "Variation chiffree propre a la banque."
                    ),
                    "action_requise": "revue_prioritaire",
                }
            )


# ---------------------------------------------------------------------------
# _validate_summary_response
# ---------------------------------------------------------------------------


class TestValidateSummaryResponse:
    def test_valid(self):
        data = {
            "executive_overview": "Résumé du trimestre.",
            "key_highlights": ["Point 1", "Point 2"],
            "pertinence_globale": "ELEVEE",
            "par_phase": {
                "rapport_gestion": {"count": 3, "resume": "Capital changes"},
                "pilier_3": {"count": 1, "resume": "New table"},
                "ifc": {"count": 0, "resume": ""},
                "autre": {"count": 0, "resume": ""},
            },
            "par_action": {
                "revue_prioritaire": 1,
                "investigation": 2,
                "confirmation": 1,
                "information": 0,
                "aucune": 0,
            },
        }
        result = _validate_summary_response(data)
        assert result["executive_overview"] == "Résumé du trimestre."
        assert len(result["key_highlights"]) == 2
        assert result["source"] == "llm"
        assert result["par_phase"]["rapport_gestion"]["count"] == 3
        assert result["par_action"]["revue_prioritaire"] == 1

    def test_none_raises(self):
        with pytest.raises(ValueError, match="empty or invalid"):
            _validate_summary_response(None)

    def test_invalid_par_phase_ignored(self):
        data = {
            "executive_overview": "Test",
            "pertinence_globale": "MOYENNE",
            "par_phase": "not_a_dict",
            "par_action": {"revue_prioritaire": "not_int"},
        }
        result = _validate_summary_response(data)
        assert result["par_phase"] == {}
        assert result["par_action"]["revue_prioritaire"] == 0


# ---------------------------------------------------------------------------
# _build_change_prompt
# ---------------------------------------------------------------------------


class TestBuildChangePrompt:
    def test_pair_prompt_includes_section(self):
        pair = _make_pair(added=[{"value": "Ratio CET1"}])
        prompt = _build_change_prompt(pair, "pair")
        assert "gestion_capital" in prompt
        assert "Ratio CET1" in prompt

    def test_added_table_prompt(self):
        tbl = {
            "title": "Nouveau Risque",
            "section": "risque",
            "indicators": [{"value": "LCR"}],
        }
        prompt = _build_change_prompt(tbl, "added")
        assert "NOUVEAU TABLEAU" in prompt
        assert "LCR" in prompt

    def test_removed_table_prompt(self):
        tbl = {"title": "Ancien Tableau", "section": "capital", "indicators": []}
        prompt = _build_change_prompt(tbl, "removed")
        assert "SUPPRIMÉ" in prompt


# ---------------------------------------------------------------------------
# enrich_comparison_with_genai_triage (integration, mocked LLM)
# ---------------------------------------------------------------------------


class TestEnrichComparison:
    def test_raises_when_llm_not_configured(self, tmp_path):
        comparison = _make_comparison(
            pairs=[_make_pair(added=[{"value": "CET1"}])],
        )
        path = tmp_path / "comparison.json"
        path.write_text(json.dumps(comparison), encoding="utf-8")

        with patch(
            "vigie.comparaison.triage.genai_triage.require_configured",
            side_effect=RuntimeError("OPENAI_API_KEY absent"),
        ):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY absent"):
                enrich_comparison_with_genai_triage(path)

    def test_skips_unchanged_pairs(self, tmp_path):
        comparison = _make_comparison(
            pairs=[_make_pair(status="inchange")],
        )
        path = tmp_path / "comparison.json"
        path.write_text(json.dumps(comparison), encoding="utf-8")

        with patch("vigie.comparaison.triage.genai_triage.require_configured"):
            with patch("vigie.comparaison.triage.genai_triage.get_async_client"):
                enrich_comparison_with_genai_triage(path)

        enriched = json.loads(path.read_text(encoding="utf-8"))
        triage = enriched["pair_comparisons"][0]["genai_triage"]
        assert triage["is_relevant"] is False
        assert triage["source"] == "skip"
