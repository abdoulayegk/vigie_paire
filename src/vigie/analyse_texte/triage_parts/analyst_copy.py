"""Redaction des champs destines a l'analyste (justifications, extraits, libelles).

Extrait de ``triage.py`` sans modification.
"""

from __future__ import annotations

import re
from typing import Any

from vigie.comparaison.analyst_change_presentation import bank_subject as analyst_bank_subject

from .constants import _ANALYST_FIELD_END_RE


def _normalize_local_analyst_field(value: str, *, field_name: str) -> str:
    """Normalise et vérifie une unité sémantique produite localement."""
    normalized = " ".join(str(value or "").split())
    if not normalized:
        raise ValueError(f"{field_name} doit être non vide.")
    if not re.search(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]", normalized):
        raise ValueError(f"{field_name} doit contenir du contenu lexical.")
    if _ANALYST_FIELD_END_RE.search(normalized) is None:
        normalized = normalized.rstrip(" ,;:…") + "."
    return normalized


def _ensure_bank_subject(value: str, bank_subject: str) -> str:
    """Garantit un constat centré sur la banque, y compris pour un ancien texte."""
    normalized = " ".join(str(value or "").split())
    if not normalized:
        return normalized
    if normalized.casefold().startswith(bank_subject.casefold()):
        return normalized

    legacy_subjects = (
        "Le rapport courant",
        "Le rapport actuel",
        "La banque",
    )
    for legacy_subject in legacy_subjects:
        if normalized.casefold().startswith(legacy_subject.casefold()):
            return f"{bank_subject}{normalized[len(legacy_subject) :]}"

    lowered_starts = {
        "Le ": "le ",
        "La ": "la ",
        "Les ": "les ",
        "Un ": "un ",
        "Une ": "une ",
        "Des ": "des ",
        "Ce ": "ce ",
        "Cette ": "cette ",
        "Ces ": "ces ",
    }
    statement = normalized
    for prefix, replacement in lowered_starts.items():
        if statement.startswith(prefix):
            statement = f"{replacement}{statement[len(prefix) :]}"
            break
    return f"{bank_subject} indique que {statement}"


def _semantic_reason_payload(
    *,
    is_relevant: bool,
    changement_constate: str,
    signification_metier: str = "",
    comparaison_interbanques: str = "",
    limite_interpretation: str = "",
    motif_non_pertinence: str = "",
) -> dict[str, str]:
    """Construit les champs analystes et leur assemblage historique."""
    raw_fields = {
        "changement_constate": changement_constate,
        "signification_metier": signification_metier,
        "comparaison_interbanques": comparaison_interbanques,
        "limite_interpretation": limite_interpretation,
        "motif_non_pertinence": motif_non_pertinence,
    }
    applicable = (
        {
            "changement_constate",
            "signification_metier",
            "comparaison_interbanques",
            "limite_interpretation",
        }
        if is_relevant
        else {"changement_constate", "motif_non_pertinence"}
    )
    normalized_fields: dict[str, str] = {}
    for field_name, value in raw_fields.items():
        if field_name in applicable:
            normalized_fields[field_name] = _normalize_local_analyst_field(
                value,
                field_name=field_name,
            )
        else:
            normalized_fields[field_name] = ""
    reason_order = (
        (
            "changement_constate",
            "signification_metier",
            "comparaison_interbanques",
            "limite_interpretation",
        )
        if is_relevant
        else ("changement_constate", "motif_non_pertinence")
    )
    normalized_fields["relevance_reason"] = " ".join(normalized_fields[field_name] for field_name in reason_order)
    return normalized_fields


def _secondary_analyst_justification(
    *,
    subject_label: str,
    analyst_copy: dict[str, str],
    surveillance_note: str,
) -> str:
    """Compose la note historique à partir des mêmes unités structurées."""
    return (
        "NON — Nouvel élément à surveiller : Non.\n\n"
        f"Sujet détecté : {subject_label}.\n\n"
        f"Ce qui change : {analyst_copy['changement_constate']}\n\n"
        f"Pertinence métier : {analyst_copy['motif_non_pertinence']}\n\n"
        f"Point de surveillance : {surveillance_note}"
    )


def _local_relevance_reason(
    factual_change: str,
    comparative_interpretation: str,
) -> str:
    """Compatibilité : assemble deux unités sans recompter leurs phrases."""
    return _semantic_reason_payload(
        is_relevant=False,
        changement_constate=factual_change,
        motif_non_pertinence=comparative_interpretation,
    )["relevance_reason"]


def _excerpt_for_analyst(text: str, *, limit: int = 160) -> str:
    """Tronque un extrait source pour une phrase analysée lisible (une seule phrase)."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    # Conserve l'extrait dans une seule unité factuelle lisible.
    value = re.sub(r"[.!?]+", ",", value).strip(" ,;")
    if len(value) <= limit:
        return value
    space_idx = value.rfind(" ", 0, limit)
    if space_idx >= max(40, limit // 4):
        return value[:space_idx].rstrip(" ,;:") + "…"
    return value[:limit].rstrip(" ,;:") + "…"


def _analyst_exclusion_copy(
    change: dict[str, Any],
    exclusion_reason: str,
    *,
    bank_code: str = "",
) -> tuple[str, str, str]:
    """Textes analyste naturels pour un changement exclu (sans jargon pipeline)."""
    bank_subject = analyst_bank_subject(bank_code)
    diff_type = str(change.get("diff_type") or "").strip().lower()
    source_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    excerpt_t2 = _excerpt_for_analyst(source_t2)
    comparative = (
        "Ce changement n'apporte pas d'élément nouveau à comparer entre les banques pour la vigie prudentielle."
    )

    if exclusion_reason == "operation_interne_banque":
        subject = "Opération interne propre à la banque"
        if diff_type == "added" and excerpt_t2:
            factual = (
                f"{bank_subject} ajoute un passage sur "
                f"« {excerpt_t2} » lié à une "
                "opération propre à la banque (acquisition, rachat ou émission)."
            )
        elif diff_type == "added":
            factual = (
                f"{bank_subject} ajoute une information liée à une opération "
                "propre à la banque (acquisition, rachat ou émission)."
            )
        else:
            factual = (
                f"{bank_subject} modifie sa divulgation pour mentionner une "
                "opération propre à la banque (acquisition, rachat ou émission)."
            )
        return factual, comparative, subject

    if exclusion_reason == "variation_numerique_propre_banque":
        subject = "Variation chiffrée propre à la banque"
        if diff_type == "added":
            factual = (
                f"{bank_subject} ajoute des montants ou pourcentages "
                "propres à la banque, sans nouvelle exigence réglementaire."
            )
        else:
            factual = (
                f"{bank_subject} met uniquement à jour des chiffres, montants ou pourcentages propres à ses activités."
            )
        return factual, comparative, subject

    if exclusion_reason == "mise_a_jour_calendrier":
        subject = "Mise à jour de calendrier"
        factual = (
            f"{bank_subject} met uniquement à jour les dates ou échéances "
            "d’application, sans ajouter de nouvelle exigence."
        )
        return factual, comparative, subject

    # Cosmetic / generic exclusions
    subject = "Changement cosmétique"
    if exclusion_reason == "deplacement_texte":
        factual = f"{bank_subject} déplace la même divulgation sans ajouter de nouveau contenu."
        subject = "Déplacement de texte"
    elif exclusion_reason == "formatage_visuel":
        factual = f"{bank_subject} reformule le passage sans changement de fond (ponctuation ou formatage uniquement)."
    elif diff_type == "added":
        factual = f"{bank_subject} introduit une formulation différente sans changement de fond."
    else:
        factual = f"{bank_subject} reformule le passage sans changement de fond."
    return factual, comparative, subject
