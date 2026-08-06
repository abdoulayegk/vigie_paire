"""Validation Pydantic et normalisation des invariants pour le triage GenAI.

Ce module garantit la conformité stricte des réponses retournées par les modèles LLM
avec la taxonomie AMF v2 et applique les invariants de sécurité (ex. plancher d'impact).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from vigie.comparaison.triage.amf_taxonomy import (
    IMPACT_IT_DETAIL_LABELS,
    POSTURE_DETAIL_LABELS,
    THEMES_AMF_PIPELINE_2,
    missing_labeled_analysis_sections,
)

logger = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset({"REGLEMENTAIRE", "RISQUE", "CAPITAL", "STRUCTURE", "NON_PERTINENT", "INCONNU"})
VALID_RELEVANCE = frozenset({"ELEVEE", "MOYENNE", "FAIBLE"})
VALID_RISK_LEVELS = frozenset({"ELEVE", "MODERE", "FAIBLE"})
VALID_IMPACT_TYPES = frozenset({"structurel", "contenu", "methodologique", "non_substantif"})
VALID_PROJECT_PHASES = frozenset({"rapport_gestion", "pilier_3", "ifc", "autre"})
VALID_ACTIONS = frozenset({"revue_prioritaire", "investigation", "confirmation", "information", "aucune"})
VALID_IMPACT_IT = frozenset({"ELEVE", "MOYEN", "FAIBLE", "INDETERMINE"})
VALID_CHANGEMENTS_POSTURE = frozenset(
    {"RENFORCEMENT", "ALLEGEMENT", "NOUVEAU_DISPOSITIF", "RETRAIT_DISPOSITIF", "AUCUN", "INDETERMINE"}
)
VALID_STATUTS_MISE_EN_OEUVRE = frozenset({"ANNONCE", "PLANIFIE", "EN_COURS", "MIS_EN_OEUVRE", "INDETERMINE"})
VALID_CONFIANCES_POSTURE = frozenset({"ELEVEE", "MOYENNE", "FAIBLE", "INDETERMINE"})
VALID_THEMES_AMF = frozenset(THEMES_AMF_PIPELINE_2)

_JUSTIFICATION_MIN_SENTENCES = 3
_JUSTIFICATION_MIN_SENTENCE_LENGTH = 20
_JUSTIFICATION_MIN_TOTAL_LENGTH = 200
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+")
_REQUIRED_JUSTIFICATION_SECTIONS = (
    "Nouvel élément à surveiller :",
    "Sujet détecté :",
    "Ce qui change :",
    "Pertinence métier :",
    "Point de surveillance :",
)
_LEGACY_SURVEILLANCE_SECTION = "Lecture de vigie :"


def _count_substantive_sentences(text: str) -> int:
    """Compte les phrases ayant au moins ``_JUSTIFICATION_MIN_SENTENCE_LENGTH`` caracteres."""
    if not text:
        return 0
    raw_sentences = _SENTENCE_BOUNDARY_RE.split(text)
    substantive = [s.strip() for s in raw_sentences if len(s.strip()) >= _JUSTIFICATION_MIN_SENTENCE_LENGTH]
    return len(substantive)


def _has_required_justification_sections(text: str) -> bool:
    """Verifie la presence des rubriques obligatoires."""
    if not text:
        return False
    lower_text = text.lower()
    for section in _REQUIRED_JUSTIFICATION_SECTIONS:
        section_clean = section.rstrip(":").strip().lower()
        if section_clean not in lower_text:
            return False
    return True


def _empty_triage_skeleton(*, source: str = "fallback") -> dict[str, Any]:
    """Retourne un dictionnaire de triage par defaut (non pertinent)."""
    return {
        "is_relevant": False,
        "themes_amf": [],
        "nouvelle_idee": False,
        "nouvelle_idee_justification": (
            "NON — Nouvel élément à surveiller : Non.\n\n"
            "Sujet détecté : Information secondaire.\n\n"
            "Ce qui change : Ce changement constitue un ajustement mineur sans "
            "nouveauté sémantique.\n\n"
            "Pertinence métier : Ce changement met l'accent sur une variation non-substantielle du texte. "
            "Il n'impacte ni la structure des risques ni les exigences prudentielles. "
            "La lisibilité globale reste inchangée par rapport aux trimestres précédents.\n\n"
            "Point de surveillance : Aucun suivi requis."
        ),
        "category": "NON_PERTINENT",
        "relevance_score": "FAIBLE",
        "risk_level": "FAIBLE",
        "impact_level": "MINEUR",
        "impact_it": "INDETERMINE",
        "impact_it_justification": "",
        "changement_posture": "AUCUN",
        "justification_posture": "",
        "statut_mise_en_oeuvre": "INDETERMINE",
        "confiance_posture": "INDETERMINE",
        "confidence": 0.0,
        "explanation": f"Triage non effectue ({source}).",
        "impact_type": "non_substantif",
        "project_phase": "autre",
        "action_requise": "aucune",
        "reference_reglementaire": "",
        "impact_description": "",
        "source": source,
    }


def _validate_amf_invariants(
    *,
    is_relevant: bool,
    themes_amf: list[str],
    category: str,
    nouvelle_idee: bool,
    nouvelle_idee_justification: str,
    action_requise: str,
) -> str | None:
    """Verifie les invariants de la taxonomie AMF v2."""
    if is_relevant and not themes_amf:
        return "is_relevant=True exige au moins un code dans themes_amf"

    if not is_relevant:
        if themes_amf:
            return "is_relevant=False interdit d'avoir des codes dans themes_amf"
        if nouvelle_idee:
            return "is_relevant=False interdit nouvelle_idee=True"
        if action_requise != "aucune":
            return "is_relevant=False exige action_requise='aucune'"

    if len(nouvelle_idee_justification) < _JUSTIFICATION_MIN_TOTAL_LENGTH:
        return f"nouvelle_idee_justification trop courte ({len(nouvelle_idee_justification)} chars < {_JUSTIFICATION_MIN_TOTAL_LENGTH})"

    substantive_count = _count_substantive_sentences(nouvelle_idee_justification)
    if substantive_count < _JUSTIFICATION_MIN_SENTENCES:
        return f"nouvelle_idee_justification contient trop peu de phrases ({substantive_count} < {_JUSTIFICATION_MIN_SENTENCES})"

    expected_prefix = "OUI" if nouvelle_idee else "NON"
    upper_just = nouvelle_idee_justification.upper()
    if not upper_just.startswith(expected_prefix):
        return f"nouvelle_idee_justification doit commencer par '{expected_prefix}' (nouvelle_idee={nouvelle_idee})"

    if not _has_required_justification_sections(nouvelle_idee_justification):
        return "nouvelle_idee_justification ne contient pas toutes les rubriques requises"

    return None


def _validate_triage_response(data: dict[str, Any] | None) -> dict[str, Any]:
    """Valide et normalise une reponse LLM de triage individuelle."""
    if not data or not isinstance(data, dict):
        return _empty_triage_skeleton(source="heuristic")

    is_relevant = bool(data.get("is_relevant", False))

    raw_themes = data.get("themes_amf") or []
    themes_amf: list[str] = []
    if isinstance(raw_themes, list):
        seen: set[str] = set()
        for theme in raw_themes:
            code = str(theme or "").upper()
            if code in VALID_THEMES_AMF and code not in seen:
                seen.add(code)
                themes_amf.append(code)

    nouvelle_idee = bool(data.get("nouvelle_idee", False))
    nouvelle_idee_justification = str(data.get("nouvelle_idee_justification") or "").strip()

    category = str(data.get("category") or "INCONNU").upper()
    if category not in VALID_CATEGORIES:
        category = "INCONNU"

    relevance = str(data.get("relevance_score") or "FAIBLE").upper()
    if relevance not in VALID_RELEVANCE:
        relevance = "FAIBLE"

    risk_level = str(data.get("risk_level") or "FAIBLE").upper()
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "FAIBLE"

    _RISK_TO_IMPACT = {"ELEVE": "MAJEUR", "MODERE": "MODERE", "FAIBLE": "MINEUR"}
    impact_level = _RISK_TO_IMPACT.get(risk_level, "MINEUR")

    if nouvelle_idee and impact_level == "MINEUR":
        impact_level = "MODERE"
        risk_level = "MODERE"

    impact_it = str(data.get("impact_it") or "INDETERMINE").upper()
    if impact_it not in VALID_IMPACT_IT:
        impact_it = "INDETERMINE"
    impact_it_justification = str(data.get("impact_it_justification") or "").strip()[:500]
    if impact_it == "INDETERMINE":
        impact_it_justification = ""
    elif len(impact_it_justification) < 20 or missing_labeled_analysis_sections(
        impact_it_justification,
        IMPACT_IT_DETAIL_LABELS,
    ):
        impact_it = "INDETERMINE"
        impact_it_justification = ""

    changement_posture = str(data.get("changement_posture") or "INDETERMINE").upper()
    if changement_posture not in VALID_CHANGEMENTS_POSTURE:
        changement_posture = "INDETERMINE"

    justification_posture = str(data.get("justification_posture") or "").strip()[:500]
    statut_mise_en_oeuvre = str(data.get("statut_mise_en_oeuvre") or "INDETERMINE").upper()
    if statut_mise_en_oeuvre not in VALID_STATUTS_MISE_EN_OEUVRE:
        statut_mise_en_oeuvre = "INDETERMINE"
    confiance_posture = str(data.get("confiance_posture") or "INDETERMINE").upper()
    if confiance_posture not in VALID_CONFIANCES_POSTURE:
        confiance_posture = "INDETERMINE"

    posture_evaluee = changement_posture in {
        "RENFORCEMENT",
        "ALLEGEMENT",
        "NOUVEAU_DISPOSITIF",
        "RETRAIT_DISPOSITIF",
    }
    if not posture_evaluee:
        justification_posture = ""
        statut_mise_en_oeuvre = "INDETERMINE"
        confiance_posture = "INDETERMINE"
    elif (
        len(justification_posture) < 20
        or confiance_posture == "INDETERMINE"
        or missing_labeled_analysis_sections(
            justification_posture,
            POSTURE_DETAIL_LABELS,
        )
    ):
        changement_posture = "INDETERMINE"
        justification_posture = ""
        statut_mise_en_oeuvre = "INDETERMINE"
        confiance_posture = "INDETERMINE"

    if not is_relevant:
        impact_it = "INDETERMINE"
        impact_it_justification = ""
        changement_posture = "AUCUN"
        justification_posture = ""
        statut_mise_en_oeuvre = "INDETERMINE"
        confiance_posture = "INDETERMINE"
        impact_level = "MINEUR"
        risk_level = "FAIBLE"

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    explanation = str(data.get("explanation") or "")[:1200]

    impact_type = str(data.get("impact_type") or "non_substantif").lower()
    if impact_type not in VALID_IMPACT_TYPES:
        impact_type = "non_substantif"

    project_phase = str(data.get("project_phase") or "autre").lower()
    if project_phase not in VALID_PROJECT_PHASES:
        project_phase = "autre"

    action_requise = str(data.get("action_requise") or "aucune").lower()
    if action_requise not in VALID_ACTIONS:
        action_requise = "aucune"

    reference_reglementaire = str(data.get("reference_reglementaire") or "")[:200]
    impact_description = str(data.get("impact_description") or "")[:500]

    invariant_error = _validate_amf_invariants(
        is_relevant=is_relevant,
        themes_amf=themes_amf,
        category=category,
        nouvelle_idee=nouvelle_idee,
        nouvelle_idee_justification=nouvelle_idee_justification,
        action_requise=action_requise,
    )
    if invariant_error:
        logger.warning(
            "Invariants AMF violés dans la sortie LLM (%s) — triage forcé en NON_PERTINENT",
            invariant_error,
        )
        return _empty_triage_skeleton(source="invariant_violation")

    return {
        "is_relevant": is_relevant,
        "themes_amf": themes_amf,
        "nouvelle_idee": nouvelle_idee,
        "nouvelle_idee_justification": nouvelle_idee_justification,
        "impact_level": impact_level,
        "impact_it": impact_it,
        "impact_it_justification": impact_it_justification,
        "changement_posture": changement_posture,
        "justification_posture": justification_posture,
        "statut_mise_en_oeuvre": statut_mise_en_oeuvre,
        "confiance_posture": confiance_posture,
        "category": category,
        "relevance_score": relevance,
        "risk_level": risk_level,
        "confidence": confidence,
        "explanation": explanation,
        "impact_type": impact_type,
        "project_phase": project_phase,
        "action_requise": action_requise,
        "reference_reglementaire": reference_reglementaire,
        "impact_description": impact_description,
        "source": "llm",
    }
