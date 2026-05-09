"""Tests for the GenAI batch triage module (vigilance.genai_triage)."""

from __future__ import annotations

import json
from unittest.mock import patch

from vigilance.genai_triage import (
    _build_change_prompt,
    _fallback_enrich,
    _has_meaningful_diff,
    _validate_summary_response,
    _validate_triage_response,
    enrich_comparison_with_genai_triage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pair(
    *, added=None, removed=None, renamed=None, fn_added=None, status="modifie"
):
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
        "OUI - le ratio TLAC est ajoute au TABLEAU 11 et n'apparaissait pas au "
        "trimestre precedent. Ce changement croise les themes AMF "
        "DIVULGATION_AJOUT et RATIOS_REGLEMENTAIRES, signalant une nouvelle "
        "exigence prudentielle BSIF substantielle. L'analyste doit comparer "
        "cette divulgation avec les pairs canadiens pour valider la conformite "
        "et la coherence du calcul TLAC."
    )


def _valid_justification_non() -> str:
    return (
        "NON - le changement observe est une simple mise a jour de date entre "
        "les deux trimestres, sans modification de fond. Aucun thème AMF n'est "
        "touche et aucun seuil reglementaire ne change. L'analyste peut "
        "considerer cette ligne comme une mise a jour rédactionnelle attendue, "
        "sans implication metier sur la divulgation prudentielle."
    )


class TestValidateTriageResponse:
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
            "action_requise": "escalade",
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
        assert result["action_requise"] == "escalade"
        assert result["reference_reglementaire"] == "Bâle III — CET1"
        assert result["impact_description"] == "Nouveau ratio prudentiel ajouté."
        assert result["source"] == "llm"

    def test_none_response(self):
        result = _validate_triage_response(None)
        assert result["is_relevant"] is False
        assert result["source"] == "heuristic"
        assert result["themes_amf"] == []
        assert result["nouvelle_idee"] is False
        # Justification désormais OBLIGATOIRE même pour le squelette par défaut
        assert result["nouvelle_idee_justification"].startswith("NON")
        assert len(result["nouvelle_idee_justification"]) >= 200
        assert result["category"] == "NON_PERTINENT"
        assert result["risk_level"] == "FAIBLE"
        assert result["confidence"] == 0.0
        assert result["impact_type"] == "non_substantif"
        assert result["project_phase"] == "autre"
        assert result["action_requise"] == "aucune"
        assert result["reference_reglementaire"] == ""
        assert result["impact_description"] == ""

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
                "action_requise": "escalade",
            }
        )
        assert result["themes_amf"] == ["DIVULGATION_AJOUT", "RISQUE_EMERGENT"]

    def test_relevant_without_themes_falls_back_to_skeleton(self):
        """Invariant : is_relevant=True sans themes_amf → forcé en NON_PERTINENT."""
        result = _validate_triage_response(
            {
                "is_relevant": True,
                "themes_amf": [],
                "nouvelle_idee": True,
                "nouvelle_idee_justification": _valid_justification_oui(),
                "action_requise": "escalade",
            }
        )
        assert result["is_relevant"] is False
        assert result["source"] == "invariant_violation"
        assert result["category"] == "NON_PERTINENT"

    def test_short_justification_falls_back_to_skeleton(self):
        """Invariant : justification < 2 phrases substantives → forcé en NON_PERTINENT."""
        result = _validate_triage_response(
            {
                "is_relevant": True,
                "themes_amf": ["DIVULGATION_AJOUT"],
                "nouvelle_idee": True,
                "nouvelle_idee_justification": "OUI ratio TLAC ajoute.",
                "action_requise": "escalade",
            }
        )
        assert result["is_relevant"] is False
        assert result["source"] == "invariant_violation"

    def test_justification_wrong_prefix_falls_back_to_skeleton(self):
        """Invariant : nouvelle_idee=True mais justification commence par NON → forcé en skeleton."""
        result = _validate_triage_response(
            {
                "is_relevant": True,
                "themes_amf": ["DIVULGATION_AJOUT"],
                "nouvelle_idee": True,
                "nouvelle_idee_justification": (
                    "NON le ratio CET1 existait deja au t1 et seule sa valeur a change. "
                    "Variation chiffree propre a la banque."
                ),
                "action_requise": "escalade",
            }
        )
        assert result["is_relevant"] is False
        assert result["source"] == "invariant_violation"


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
                "escalade": 1,
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
        assert result["par_action"]["escalade"] == 1

    def test_none_returns_defaults(self):
        result = _validate_summary_response(None)
        assert result["executive_overview"] == ""
        assert result["pertinence_globale"] == "FAIBLE"
        assert result["source"] == "heuristic"
        assert result["par_phase"] == {}
        assert result["par_action"] == {}

    def test_invalid_par_phase_ignored(self):
        data = {
            "executive_overview": "Test",
            "pertinence_globale": "MOYENNE",
            "par_phase": "not_a_dict",
            "par_action": {"escalade": "not_int"},
        }
        result = _validate_summary_response(data)
        assert result["par_phase"] == {}
        assert result["par_action"]["escalade"] == 0


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
# _fallback_enrich
# ---------------------------------------------------------------------------


class TestFallbackEnrich:
    def test_enriches_all_entries(self):
        """Mode hors-ligne : tous les triages reçoivent le squelette neutre.

        Sans clé API, on ne peut pas réellement classifier — on affiche un
        squelette honnête (is_relevant=False, NON_PERTINENT) plutôt que de
        deviner au risque d'induire l'analyste en erreur.
        """
        comparison = _make_comparison(
            pairs=[_make_pair(added=[{"value": "X"}])],
            added=[{"title": "New", "section": "risque"}],
            removed=[{"title": "Old", "section": "capital"}],
        )
        result = _fallback_enrich(comparison)

        pair_triage = result["pair_comparisons"][0]["genai_triage"]
        assert pair_triage["source"] == "heuristic"
        assert pair_triage["is_relevant"] is False
        assert pair_triage["category"] == "NON_PERTINENT"
        assert pair_triage["themes_amf"] == []
        assert pair_triage["nouvelle_idee"] is False
        # Justification désormais OBLIGATOIRE (≥ 200 chars, préfixe NON)
        assert pair_triage["nouvelle_idee_justification"].startswith("NON")
        assert len(pair_triage["nouvelle_idee_justification"]) >= 200
        assert pair_triage["impact_type"] == "non_substantif"
        assert pair_triage["action_requise"] == "aucune"
        assert pair_triage["confidence"] == 0.0

        added_triage = result["matching"]["tables_added"][0]["genai_triage"]
        assert added_triage["source"] == "heuristic"
        assert added_triage["is_relevant"] is False
        assert added_triage["category"] == "NON_PERTINENT"

        removed_triage = result["matching"]["tables_removed"][0]["genai_triage"]
        assert removed_triage["source"] == "heuristic"
        assert removed_triage["is_relevant"] is False
        assert removed_triage["category"] == "NON_PERTINENT"

        assert result["global_summary"]["source"] == "heuristic"
        assert result["global_summary"]["par_phase"] == {}
        assert result["global_summary"]["par_action"] == {}


# ---------------------------------------------------------------------------
# enrich_comparison_with_genai_triage (integration, mocked LLM)
# ---------------------------------------------------------------------------


class TestEnrichComparison:
    def test_fallback_when_no_api_key(self, tmp_path):
        comparison = _make_comparison(
            pairs=[_make_pair(added=[{"value": "CET1"}])],
        )
        path = tmp_path / "comparison.json"
        path.write_text(json.dumps(comparison), encoding="utf-8")

        with patch("vigilance.utils.genai.get_openai_api_key", return_value=None):
            result_path = enrich_comparison_with_genai_triage(path)

        enriched = json.loads(result_path.read_text(encoding="utf-8"))
        assert "global_summary" in enriched
        assert enriched["pair_comparisons"][0]["genai_triage"]["source"] == "heuristic"

    def test_skips_unchanged_pairs(self, tmp_path):
        comparison = _make_comparison(
            pairs=[_make_pair(status="inchange")],
        )
        path = tmp_path / "comparison.json"
        path.write_text(json.dumps(comparison), encoding="utf-8")

        with patch("vigilance.utils.genai.get_openai_api_key", return_value=None):
            enrich_comparison_with_genai_triage(path)

        enriched = json.loads(path.read_text(encoding="utf-8"))
        # Fallback mode marks unchanged pairs as not relevant with heuristic source
        triage = enriched["pair_comparisons"][0]["genai_triage"]
        assert triage["is_relevant"] is False
        assert triage["source"] == "heuristic"
