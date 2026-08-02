"""Paquets de preuve textuelle et relecture d'evidence pour le triage.

Extrait de ``triage.py`` sans modification.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vigilance.analyst_change_presentation import bank_subject as analyst_bank_subject
from vigilance.text_analysis.constants import _TRIAGE_SOURCE_SNIPPET_LIMIT
from vigilance.text_analysis.normalization import _json_dumps
from vigilance.text_analysis.openai_client import _call_structured_completion_with_correction
from vigilance.text_comparison.change_segments import build_change_segments

from .analyst_copy import _secondary_analyst_justification, _semantic_reason_payload
from .constants import _FULL_EVIDENCE_FACT_MAX_TOKENS, _FULL_EVIDENCE_PACKET_LIMIT
from .results import _default_triage

class _EvidencePacketObservation(BaseModel):
    """Constat factuel unique pour le paquet T1/T2 fourni dans l'appel."""

    model_config = ConfigDict(extra="forbid")

    factual_change: str = Field(..., min_length=12, max_length=700)


class _EvidencePacketCoherenceCheck(BaseModel):
    """Contrôle unique pour le paquet de preuve fourni dans l'appel."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["supports", "contradicts", "insufficient"]
    reason: str = Field(..., min_length=12, max_length=700)


def _split_full_evidence_text(text: str, *, limit: int = _FULL_EVIDENCE_PACKET_LIMIT) -> list[str]:
    """Découper un texte complet sans jamais retirer de caractères.

    La coupure privilégie les sauts de ligne puis les fins de phrase. Elle ne
    sert qu'à respecter le contexte d'un appel; la concaténation des paquets
    restitue intégralement le texte source.
    """
    value = str(text or "")
    if not value:
        return []
    if len(value) <= limit:
        return [value]

    packets: list[str] = []
    start = 0
    while start < len(value):
        end = min(start + limit, len(value))
        if end < len(value):
            boundary = max(
                value.rfind("\n", start + limit // 2, end),
                value.rfind(". ", start + limit // 2, end),
                value.rfind("; ", start + limit // 2, end),
            )
            if boundary > start:
                end = boundary + (1 if value[boundary] in ".;" else 0)
        packets.append(value[start:end])
        start = end
    return packets


def _build_full_evidence_packets(change: dict[str, Any]) -> list[dict[str, Any]]:
    """Construire des paquets T1/T2 exhaustifs pour un changement long."""
    source_t1 = str(change.get("source_text_t1") or change.get("semantic_text_t1") or "")
    source_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    packets_t1 = _split_full_evidence_text(source_t1)
    packets_t2 = _split_full_evidence_text(source_t2)
    packet_count = max(len(packets_t1), len(packets_t2), 1)
    return [
        {
            "packet_index": index + 1,
            "text_t1": packets_t1[index] if index < len(packets_t1) else "",
            "text_t2": packets_t2[index] if index < len(packets_t2) else "",
        }
        for index in range(packet_count)
    ]


def _requires_full_evidence_packets(change: dict[str, Any]) -> bool:
    """Indiquer si la preuve dépasse le contexte compact de triage."""
    return any(
        len(str(change.get(key) or "")) > _TRIAGE_SOURCE_SNIPPET_LIMIT
        for key in ("source_text_t1", "source_text_t2", "semantic_text_t1", "semantic_text_t2")
    )


def _collect_full_evidence_observations(
    *,
    client: Any,
    model: str,
    change: dict[str, Any],
    bank_code: str = "",
    section_key: str = "",
    change_index: int | None = None,
) -> list[dict[str, Any]]:
    """Lire toute preuve longue par appels factuels séparés et auditables."""
    bank_subject = analyst_bank_subject(bank_code)
    packets = _build_full_evidence_packets(change)
    observations: list[dict[str, Any]] = []
    for packet in packets:
        try:
            response = _call_structured_completion_with_correction(
                client,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu lis un seul paquet de preuve textuelle complète T1/T2. "
                            "Retourne exactement un constat factuel consolidé pour ce "
                            "paquet. Décris uniquement le fait observable entre les deux "
                            "textes, sans catégorie AMF, sans priorité, sans posture et "
                            "sans recommandation. Ne retourne ni liste ni packet_index : "
                            "le numéro du paquet est géré localement. La banque analysée "
                            f"est {bank_subject}; commence factual_change par "
                            f"{bank_subject} et un verbe d’action direct."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _json_dumps(
                            {
                                "diff_type": str(change.get("diff_type") or ""),
                                "packet": packet,
                            }
                        ),
                    },
                ],
                response_format=_EvidencePacketObservation,
                max_tokens=_FULL_EVIDENCE_FACT_MAX_TOKENS,
                max_retries=1,
                validation_retry_message=(
                    "Renvoie exactement un objet contenant uniquement factual_change, "
                    "sans liste, sans packet_index, sans qualification métier ni texte "
                    f"hors schéma. Commence factual_change par {bank_subject}."
                ),
                length_retry_message=(
                    "La réponse précédente a dépassé la limite de sortie. Renvoie "
                    "immédiatement un seul objet contenant factual_change, concis "
                    "(moins de 600 caractères), sans liste, sans packet_index, sans "
                    f"qualification métier ni champ hors schéma, commençant par "
                    f"{bank_subject}."
                ),
            )
        except Exception as exc:
            context_parts = [
                f"section={section_key or 'inconnue'}",
                f"change_index={change_index if change_index is not None else 'inconnu'}",
                f"packet_index={packet['packet_index']}",
            ]
            raise RuntimeError(
                "Lecture de preuve complète invalide "
                f"[{', '.join(context_parts)}] : {exc}"
            ) from exc
        observations.append(
            {
                "packet_index": packet["packet_index"],
                "factual_change": response.factual_change,
            }
        )
    return observations


def _evidence_read_review_triage(
    change: dict[str, Any],
    reason: str,
    *,
    bank_code: str = "",
) -> dict[str, Any]:
    """Conserver un changement dont la preuve complète n'a pas pu être lue."""
    bank_subject = analyst_bank_subject(bank_code)
    analyst_copy = _semantic_reason_payload(
        is_relevant=False,
        changement_constate=(
            f"{bank_subject} présente un changement dont la preuve complète "
            "n’a pas pu être validée automatiquement."
        ),
        motif_non_pertinence=(
            "Le dossier est conservé avec ses textes sources et ses pages afin "
            "qu’un analyste confirme le changement avant toute conclusion de vigie."
        ),
    )
    triage = _default_triage(bank_code)
    triage.update(
        {
            "source": "triage_evidence_review_required",
            "coherence_review_required": True,
            "coherence_review_reason": str(reason or "").strip(),
            **analyst_copy,
            "nouvelle_idee_justification": _secondary_analyst_justification(
                subject_label="Preuve complète à confirmer",
                analyst_copy=analyst_copy,
                surveillance_note=(
                    "Un analyste doit confirmer la lecture de la preuve avant diffusion."
                ),
            ),
            "change_segments": build_change_segments(change),
        }
    )
    enriched = dict(change)
    enriched["genai_triage"] = triage
    return enriched
