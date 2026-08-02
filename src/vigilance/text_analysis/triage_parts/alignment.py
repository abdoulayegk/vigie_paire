"""Revue d'alignement semantique et verification de coherence du triage.

Extrait de ``triage.py`` sans modification.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from vigilance.analyst_change_presentation import bank_subject as analyst_bank_subject
from vigilance.text_analysis.normalization import _json_dumps
from vigilance.text_analysis.openai_client import _call_structured_completion_with_correction
from vigilance.text_comparison.change_segments import build_change_segments

from .analyst_copy import _secondary_analyst_justification, _semantic_reason_payload
from .constants import _FULL_EVIDENCE_VERIFICATION_MAX_TOKENS, _SEMANTIC_ALIGNMENT_DECISIONS, _SEMANTIC_REASON_FIELDS
from .evidence import _EvidencePacketCoherenceCheck
from .results import _default_triage

def _requires_alignment_review(change: dict[str, Any]) -> bool:
    """True only when the first GPT call explicitly cannot decide the relation."""
    decision = str(change.get("alignment_decision") or "").strip().lower()
    if decision in _SEMANTIC_ALIGNMENT_DECISIONS:
        return decision == "uncertain"
    # Cached artifacts from before semantic arbitration keep the former safe
    # fallback.  Fresh comparisons always carry ``alignment_decision``.
    return str(change.get("alignment_type") or "").strip().lower() == "ambiguous"




def _is_single_semantic_alignment_group(changes: list[dict[str, Any]]) -> bool:
    """Keeps the added/removed sides of one GPT decision in one triage call."""
    group_ids = {
        str(change.get("semantic_alignment_group_id") or "").strip()
        for change in changes
    }
    return len(changes) >= 2 and len(group_ids) == 1 and bool(next(iter(group_ids), ""))


def _alignment_review_result(
    change: dict[str, Any],
    *,
    bank_code: str = "",
) -> dict[str, Any]:
    """Preserves the evidence while preventing an unsupported automatic verdict."""
    bank_subject = analyst_bank_subject(bank_code)
    analyst_copy = _semantic_reason_payload(
        is_relevant=False,
        changement_constate=(
            f"{bank_subject} présente des passages qui pourraient décrire des "
            "divulgations différentes, sans preuve d’alignement suffisante pour "
            "conclure automatiquement."
        ),
        motif_non_pertinence=(
            "L’élément reste visible avec ses extraits sources et nécessite une "
            "revue avant toute qualification AMF."
        ),
    )
    triage = _default_triage(bank_code)
    triage.update(
        {
            "source": "alignment_review_required",
            "alignment_review_required": True,
            "alignment_review_reason": (
                str(change.get("alignment_rationale") or "").strip()
                or "L'alignement entre les deux passages reste ambigu après la comparaison initiale. "
                "Le changement est conservé pour revue, sans classification AMF automatique."
            ),
            **analyst_copy,
            "nouvelle_idee_justification": _secondary_analyst_justification(
                subject_label="Alignement à confirmer",
                analyst_copy=analyst_copy,
                surveillance_note=(
                    "Lire les extraits sources avant toute décision."
                ),
            ),
            # The analyst still sees the deterministic, verbatim difference;
            # no LLM-generated highlight is used for this unresolved pairing.
            "change_segments": build_change_segments(change),
        }
    )
    enriched = dict(change)
    enriched["genai_triage"] = triage
    return enriched


def _semantic_move_result(
    change: dict[str, Any],
    *,
    bank_code: str = "",
) -> dict[str, Any]:
    """Marks a GPT-confirmed text move as non-priority without human escalation."""
    bank_subject = analyst_bank_subject(bank_code)
    analyst_copy = _semantic_reason_payload(
        is_relevant=False,
        changement_constate=(
            f"{bank_subject} déplace une divulgation sans modifier "
            "substantiellement son sens, son niveau de détail ou son "
            "rattachement métier."
        ),
        motif_non_pertinence=(
            "Ce déplacement ne crée aucun nouvel élément à comparer entre les "
            "banques."
        ),
    )
    triage = _default_triage(bank_code)
    triage.update(
        {
            "source": "semantic_alignment_decision",
            "alignment_decision": "moved_text",
            "alignment_confidence": str(change.get("alignment_confidence") or "medium"),
            "alignment_rationale": str(change.get("alignment_rationale") or "").strip(),
            "exclusion_reason": "deplacement_texte",
            **analyst_copy,
            "nouvelle_idee_justification": _secondary_analyst_justification(
                subject_label="Texte déplacé",
                analyst_copy=analyst_copy,
                surveillance_note=(
                    "Aucun suivi prioritaire n’est requis pour ce déplacement."
                ),
            ),
            "change_segments": [],
        }
    )
    enriched = dict(change)
    enriched["genai_triage"] = triage
    return enriched


def _coherence_review_triage(
    change: dict[str, Any],
    reason: str,
    *,
    bank_code: str = "",
) -> dict[str, Any]:
    """Conserver un dossier visible quand le contrôle indépendant diverge."""
    bank_subject = analyst_bank_subject(bank_code)
    analyst_copy = _semantic_reason_payload(
        is_relevant=False,
        changement_constate=(
            f"{bank_subject} présente un changement dont la qualification métier "
            "ne concorde pas suffisamment avec la vérification indépendante des "
            "preuves complètes."
        ),
        motif_non_pertinence=(
            "Le dossier est conservé avec ses textes sources et ses pages afin "
            "qu’un analyste confirme le type de changement et sa pertinence "
            "avant toute conclusion de vigie."
        ),
    )
    triage = _default_triage(bank_code)
    triage.update(
        {
            "source": "triage_coherence_review_required",
            "coherence_review_required": True,
            "coherence_review_reason": str(reason or "").strip(),
            **analyst_copy,
            "nouvelle_idee_justification": _secondary_analyst_justification(
                subject_label="Cohérence à confirmer",
                analyst_copy=analyst_copy,
                surveillance_note=(
                    "Un analyste doit confirmer la qualification avant diffusion."
                ),
            ),
            "change_segments": build_change_segments(change),
        }
    )
    enriched = dict(change)
    enriched["genai_triage"] = triage
    return enriched


def _verify_triage_coherence(
    *,
    client: Any,
    model: str,
    change: dict[str, Any],
    triage: dict[str, Any],
    evidence_packets: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Contrôler chaque preuve exacte contre la décision métier proposée."""
    support_reasons: list[str] = []
    for packet in evidence_packets:
        response = _call_structured_completion_with_correction(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu vérifies une décision de vigie bancaire de façon indépendante. "
                        "Compare la décision proposée au paquet de texte exact fourni. "
                        "Tu traites un seul paquet : retourne un seul contrôle et ne "
                        "retourne ni liste ni packet_index. "
                        "Réponds supports si le paquet l'appuie, contradicts s'il la "
                        "contredit, insufficient s'il ne permet pas de conclure. N'invente "
                        "aucun fait absent du paquet."
                    ),
                },
                {
                    "role": "user",
                    "content": _json_dumps(
                        {
                            "diff_type": change.get("diff_type"),
                            "packet": packet,
                            "proposed_triage": {
                                "is_relevant": triage.get("is_relevant"),
                                "themes_amf": triage.get("themes_amf"),
                                "nouvelle_idee": triage.get("nouvelle_idee"),
                                **{
                                    field_name: triage.get(field_name, "")
                                    for field_name in _SEMANTIC_REASON_FIELDS
                                },
                                "relevance_reason": triage.get("relevance_reason"),
                            },
                        }
                    ),
                },
            ],
            response_format=_EvidencePacketCoherenceCheck,
            max_tokens=_FULL_EVIDENCE_VERIFICATION_MAX_TOKENS,
            max_retries=1,
            validation_retry_message=(
                "Réponds avec un seul objet contenant verdict (supports, contradicts "
                "ou insufficient) et reason. Ne retourne ni liste ni packet_index, "
                "et n’ajoute aucun thème."
            ),
            length_retry_message=(
                "La réponse précédente a dépassé la limite de sortie. Renvoie "
                "immédiatement un seul objet contenant verdict (supports, contradicts "
                "ou insufficient) et une reason factuelle concise (moins de 600 "
                "caractères), sans liste, sans packet_index, sans thème ni champ "
                "hors schéma."
            ),
        )
        if response.verdict == "contradicts":
            return False, response.reason
        if response.verdict == "supports":
            support_reasons.append(response.reason)

    if not support_reasons:
        return False, "Aucun paquet de preuve complet ne confirme la décision métier proposée."
    return True, " ".join(support_reasons)


def _change_index_from_validation_error(
    validation_error: ValidationError,
) -> int | None:
    """Récupère l'index métier depuis le payload ou la position du batch."""
    try:
        errors = validation_error.errors(include_input=True)
    except Exception:  # noqa: BLE001
        return None
    for error in errors:
        raw_input = error.get("input")
        if isinstance(raw_input, dict):
            try:
                change_index = int(raw_input.get("change_index"))
            except (TypeError, ValueError):
                change_index = 0
            if change_index >= 1:
                return change_index
    for error in errors:
        location = tuple(error.get("loc") or ())
        for offset, part in enumerate(location[:-1]):
            if part == "triages" and isinstance(location[offset + 1], int):
                return int(location[offset + 1]) + 1
    return None
