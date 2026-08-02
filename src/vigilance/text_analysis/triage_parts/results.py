"""Construction des resultats de triage et derivation des champs historiques.

Extrait de ``triage.py`` sans modification.
"""

from __future__ import annotations

import logging
from typing import Any

from vigilance.amf_taxonomy import TRIAGE_SOURCE_VERSION, empty_triage_skeleton
from vigilance.analyst_change_presentation import bank_subject as analyst_bank_subject
from vigilance.text_comparison.change_segments import build_change_segments
from vigilance.text_comparison.justification import build_compact_triage_justification

from .analyst_copy import (
    _analyst_exclusion_copy,
    _ensure_bank_subject,
    _secondary_analyst_justification,
    _semantic_reason_payload,
)
from .constants import _PROCESS_SIGNAL_RE
from .exclusions import _deterministic_bank_specific_exclusion

logger = logging.getLogger("vigilance.text_analysis.triage")

def _default_triage(bank_code: str = "") -> dict[str, Any]:
    """Retourne un triage par défaut conservateur (non pertinent).

    Schéma cible AMF v2 (``themes_amf``, ``exclusion_reason``) **plus** les
    champs hérités (``category``, ``signals``, ``confidence``, ...) maintenus
    avec valeurs par défaut pour préserver la compatibilité avec les
    consommateurs aval (review_export, review_models_v2, review_queue_normalizer)
    non encore migrés.
    """
    bank_subject = analyst_bank_subject(bank_code)
    analyst_copy = _semantic_reason_payload(
        is_relevant=False,
        changement_constate=(
            f"{bank_subject} ne dispose pas d’une qualification AMF exploitable "
            "pour ce changement."
        ),
        motif_non_pertinence=(
            "L’élément est conservé dans la file de revue sans être présenté "
            "comme une nouvelle idée, afin d’éviter une conclusion automatique "
            "non étayée par les informations disponibles."
        ),
    )
    triage = empty_triage_skeleton()
    triage["source"] = TRIAGE_SOURCE_VERSION
    triage.update(
        {
            "compact_schema_version": "analyst_compact_v2",
            "category": "NON_PERTINENT",
            "risk_type": "autre",
            "relevance_score": "FAIBLE",
            "risk_level": "FAIBLE",
            "impact_description": "",
            "reference_reglementaire": "",
            "confidence": 0.0,
            **analyst_copy,
            "nouvelle_idee_justification": _secondary_analyst_justification(
                subject_label="Élément non classifié",
                analyst_copy=analyst_copy,
                surveillance_note=(
                    "Une revue des preuves est requise avant toute conclusion."
                ),
            ),
            "signals": {
                "regulatory_reference_added": False,
                "methodology_change": False,
                "tone_changed": False,
                "forward_looking": False,
                "quantitative_changed": False,
            },
        }
    )
    return triage


def _prefilter_triage_result(
    change: dict[str, Any],
    exclusion_reason: str,
    *,
    bank_code: str = "",
) -> dict[str, Any]:
    triage = _default_triage(bank_code)
    factual, comparative, subject = _analyst_exclusion_copy(
        change,
        exclusion_reason,
        bank_code=bank_code,
    )
    analyst_copy = _semantic_reason_payload(
        is_relevant=False,
        changement_constate=factual,
        motif_non_pertinence=comparative,
    )
    triage.update(
        {
            "source": "deterministic_prefilter",
            "exclusion_reason": exclusion_reason,
            **analyst_copy,
            "nouvelle_idee_justification": (
                "NON — Nouvel élément à surveiller : Non.\n\n"
                f"Sujet détecté : {subject}.\n\n"
                f"Ce qui change : {factual}\n\n"
                f"Pertinence métier : {comparative}\n\n"
                "Point de surveillance : Aucun suivi prioritaire n'est requis."
            ),
            "change_segments": build_change_segments(change),
        }
    )
    enriched = dict(change)
    enriched["genai_triage"] = triage
    enriched["triage_prefilter"] = {
        "excluded": True,
        "exclusion_reason": exclusion_reason,
    }
    return enriched


def _cosmetic_triage_result(
    change: dict[str, Any],
    exclusion_reason: str,
    *,
    bank_code: str = "",
) -> dict[str, Any]:
    """Compatibilité : délégué au préfiltre généraliste."""
    return _prefilter_triage_result(
        change,
        exclusion_reason,
        bank_code=bank_code,
    )


def _derive_legacy_fields(triage_amf: dict[str, Any]) -> dict[str, Any]:
    """Dérive les champs hérités (category, signals, ...) depuis le schéma AMF v2.

    Permet aux consommateurs aval (review_export, review_models_v2, ...) qui
    lisent encore l'ancien schéma de continuer à fonctionner sans modification.
    À retirer une fois ces consommateurs migrés vers ``themes_amf``.
    """
    if not triage_amf.get("is_relevant"):
        return {
            "category": "NON_PERTINENT",
            "risk_type": "autre",
            "relevance_score": "FAIBLE",
            "risk_level": "FAIBLE",
            "impact_description": "",
            "reference_reglementaire": "",
            "confidence": 0.0,
            "signals": {
                "regulatory_reference_added": False,
                "methodology_change": False,
                "tone_changed": False,
                "forward_looking": False,
                "quantitative_changed": False,
            },
        }

    themes = set(triage_amf.get("themes_amf") or [])
    impact = str(triage_amf.get("impact_level") or "MINEUR").upper()

    if "SUJET_EMERGENT_HORS_GRILLE" in themes:
        category = "INCONNU"
        risk_type = "autre"
    elif themes & {"CAPITAL_REGLEMENTAIRE", "FONDS_PROPRES_REGLEMENTAIRES", "RATIOS_REGLEMENTAIRES"}:
        category = "CAPITAL"
        risk_type = "capital"
    elif "LIQUIDITE" in themes:
        category = "REGLEMENTAIRE"
        risk_type = "liquidite"
    elif themes & {"EXIGENCES_REGLEMENTAIRES", "NOUVELLE_MENTION_REGLEMENTAIRE"}:
        category = "REGLEMENTAIRE"
        risk_type = "conformite"
    elif themes & {"MODIFICATION_TEXTE_RISQUE", "FACTEUR_RISQUE_CHANGEMENT", "HYPOTHESES_EXPLICATIONS_RISQUES"}:
        category = "RISQUE"
        risk_type = "credit"
    elif themes & {"RISQUE_EMERGENT", "RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"}:
        category = "RISQUE"
        risk_type = "autre"
    elif "ESG_CLIMATIQUE" in themes:
        category = "RISQUE"
        risk_type = "autre"
    elif themes & {"GOUVERNANCE_RISQUES", "CONTROLE_CONFORMITE"}:
        category = "STRUCTURE"
        risk_type = "conformite"
    elif "STRUCTURE_RAPPORT" in themes:
        category = "STRUCTURE"
        risk_type = "autre"
    else:
        category = "STRUCTURE"
        risk_type = "autre"

    severity_map = {"MAJEUR": "ELEVEE", "MODERE": "MOYENNE", "MINEUR": "FAIBLE"}
    return {
        "category": category,
        "risk_type": risk_type,
        "relevance_score": severity_map.get(impact, "FAIBLE"),
        "risk_level": severity_map.get(impact, "FAIBLE"),
        "impact_description": "",
        "reference_reglementaire": "",
        "confidence": 0.85,
        "signals": {
            "regulatory_reference_added": "NOUVELLE_MENTION_REGLEMENTAIRE" in themes,
            "methodology_change": "MODIFICATION_METHODOLOGIE" in themes,
            "tone_changed": False,
            "forward_looking": False,
            "quantitative_changed": "MONTANT_REGLEMENTAIRE" in themes,
        },
    }


_COMPACT_HIGH_PRIORITY_THEMES = frozenset(
    {
        "RISQUE_EMERGENT",
        "RISQUE_MACRO_GEOPOLITIQUE",
        "MODIFICATION_METHODOLOGIE",
        "EXIGENCES_REGLEMENTAIRES",
        "NOUVELLE_MENTION_REGLEMENTAIRE",
        "MONTANT_REGLEMENTAIRE",
        "GOUVERNANCE_RISQUES",
    }
)


def _persisted_triage_from_compact(
    compact: dict[str, Any],
    *,
    change: dict[str, Any],
    bank_code: str = "",
) -> dict[str, Any]:
    """Ajoute localement les champs historiques sans les demander au LLM."""
    is_relevant = bool(compact.get("is_relevant", False))
    nouvelle_idee = bool(compact.get("nouvelle_idee", False))
    themes = list(compact.get("themes_amf") or [])
    bank_subject = analyst_bank_subject(bank_code)
    analyst_copy = _semantic_reason_payload(
        is_relevant=is_relevant,
        changement_constate=_ensure_bank_subject(
            str(compact.get("changement_constate") or ""),
            bank_subject,
        ),
        signification_metier=str(compact.get("signification_metier") or ""),
        comparaison_interbanques=str(
            compact.get("comparaison_interbanques") or ""
        ),
        limite_interpretation=str(compact.get("limite_interpretation") or ""),
        motif_non_pertinence=str(compact.get("motif_non_pertinence") or ""),
    )
    relevance_reason = analyst_copy["relevance_reason"]

    # Recalculate exclusion on change + GPT reason (catches CWB framed as methodology).
    bank_exclusion = _deterministic_bank_specific_exclusion(change) or (
        _deterministic_bank_specific_exclusion(
            {**change, "change_summary": relevance_reason}
        )
        if relevance_reason
        else None
    )

    # Post-LLM guardrail: never promote bank-specific noise to Majeur/Modéré.
    if bank_exclusion and is_relevant:
        logger.debug(
            "post_llm_guardrail override change_id=%s exclusion=%s",
            change.get("change_id"),
            bank_exclusion,
        )
        is_relevant = False
        nouvelle_idee = False
        themes = []
        factual, comparative, _subject = _analyst_exclusion_copy(
            change,
            bank_exclusion,
            bank_code=bank_code,
        )
        analyst_copy = _semantic_reason_payload(
            is_relevant=False,
            changement_constate=factual,
            motif_non_pertinence=comparative,
        )
        relevance_reason = analyst_copy["relevance_reason"]

    change_corpus = " ".join(
        str(change.get(field) or "")
        for field in (
            "source_text_t1",
            "source_text_t2",
            "semantic_text_t1",
            "semantic_text_t2",
            "change_summary",
        )
    )
    substantive_process_change = (
        nouvelle_idee
        and "CONTROLE_CONFORMITE" in themes
        and bool(_PROCESS_SIGNAL_RE.search(change_corpus))
    )
    high_priority = bool(set(themes) & _COMPACT_HIGH_PRIORITY_THEMES) or (
        substantive_process_change
    )

    if not is_relevant:
        impact_level = "MINEUR"
        action_requise = "aucune"
    elif nouvelle_idee and high_priority:
        impact_level = "MAJEUR"
        action_requise = "revue_prioritaire"
    elif nouvelle_idee:
        impact_level = "MODERE"
        action_requise = "investigation"
    else:
        impact_level = "MINEUR"
        action_requise = "information"

    triage: dict[str, Any] = {
        "compact_schema_version": "analyst_compact_v2",
        "bank_code": str(bank_code or "").strip().lower(),
        "bank_subject": bank_subject,
        "is_relevant": is_relevant,
        "themes_amf": themes,
        "nouvelle_idee": nouvelle_idee,
        **analyst_copy,
        "exclusion_reason": (
            None
            if is_relevant
            else (bank_exclusion or "non_pertinent_autre")
        ),
        "impact_level": impact_level,
        "action_requise": action_requise,
        "impact_it": "INDETERMINE",
        "impact_it_justification": "",
        "changement_posture": "INDETERMINE" if is_relevant else "AUCUN",
        "justification_posture": "",
        "statut_mise_en_oeuvre": "INDETERMINE",
        "confiance_posture": "INDETERMINE",
        "explanation": relevance_reason if is_relevant else "",
    }
    if bank_exclusion and not is_relevant and compact.get("is_relevant"):
        triage["source_guardrail"] = "post_llm_guardrail"
    triage["nouvelle_idee_justification"] = build_compact_triage_justification(
        change,
        triage,
    )
    legacy_fields = _derive_legacy_fields(triage)
    return {**triage, **legacy_fields, "source": TRIAGE_SOURCE_VERSION}
