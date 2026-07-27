"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import logging
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vigilance.analyst_change_presentation import bank_subject as analyst_bank_subject
from vigilance.amf_taxonomy import (
    MaterialityAssessment,
    MaterialityLevel,
    THEMES_AMF_ANALYST_SUBJECTS,
    THEMES_AMF_DESCRIPTIONS,
    THEMES_AMF_PIPELINE_2,
    TRIAGE_SOURCE_VERSION,
    TriageAMFCompactLLMResultWithIndex,
    TriageAMFMaterialityLLMBatch,
    TriageValidationError,
    empty_triage_skeleton,
)
from vigilance.text_analysis.constants import (
    _NUMERIC_TOKEN_RE,
    _REGULATORY_REF_RE,
    _TRIAGE_BATCH_SIZE,
    _TRIAGE_SOURCE_SNIPPET_LIMIT,
)
from vigilance.text_analysis.normalization import _json_dumps
from vigilance.text_analysis.openai_client import (
    _call_structured_completion_with_correction,
    _embed_texts,
    _truncate_prompt_text,
)
from vigilance.text_analysis.precedent_memory import (
    PrecedentMemory,
    PrecedentQuery,
)
from vigilance.text_comparison.change_segments import build_change_segments
from vigilance.text_comparison.justification import build_compact_triage_justification

logger = logging.getLogger(__name__)

_MAX_TRIAGE_LLM_WORKERS = 6
_SEMANTIC_ALIGNMENT_DECISIONS = frozenset(
    {"same_disclosure", "distinct_disclosures", "moved_text", "uncertain"}
)
_COSMETIC_SEQUENCE_THRESHOLD = 0.97
_BANK_NOISE_SEQUENCE_THRESHOLD = 0.92
_TRIAGE_DEDUP_EMBEDDING_THRESHOLD = 0.92
_TRIAGE_EMBEDDING_TRUNCATE_CHARS = 1800
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_RELATED_CHANGE_CONTEXT_LIMIT = 6
_COMPACT_THEME_CANDIDATE_LIMIT = 6
_COMPACT_COMPLETION_BASE_TOKENS = 500
_COMPACT_COMPLETION_TOKENS_PER_CHANGE = 700
_COMPACT_COMPLETION_MAX_TOKENS = 2400
_FULL_EVIDENCE_PACKET_LIMIT = 2400
# Must stay above the token equivalent of max_length=700 on factual_change /
# reason so structured completions never hit finish_reason=length.
_FULL_EVIDENCE_FACT_MAX_TOKENS = 500
_FULL_EVIDENCE_VERIFICATION_MAX_TOKENS = 500
_SEMANTIC_REASON_FIELDS = (
    "changement_constate",
    "signification_metier",
    "motif_non_pertinence",
)
_ANALYST_FIELD_END_RE = re.compile(r"[.!?]+[\u00bb\u201d\"')\]]*$")
_MATERIALITY_RANK = {"MINEUR": 0, "MODERE": 1, "MAJEUR": 2}
_HARD_PREFILTER_EXCLUSIONS = frozenset(
    {
        "formatage_visuel",
        "deplacement_texte",
    }
)
_SENSITIVE_MATERIALITY_THEMES = frozenset(
    {
        "CAPITAL_REGLEMENTAIRE",
        "FONDS_PROPRES_REGLEMENTAIRES",
        "RATIOS_REGLEMENTAIRES",
        "LIQUIDITE",
        "EXIGENCES_REGLEMENTAIRES",
        "NOUVELLE_MENTION_REGLEMENTAIRE",
        "MONTANT_REGLEMENTAIRE",
        "DIVULGATION_RETRAIT",
        "MODIFICATION_TEXTE_RISQUE",
        "MODIFICATION_METHODOLOGIE",
        "FACTEUR_RISQUE_CHANGEMENT",
        "HYPOTHESES_EXPLICATIONS_RISQUES",
        "ESG_CLIMATIQUE",
        "RISQUE_EMERGENT",
        "RISQUE_DONNEES",
        "RISQUE_TIERS_CLOUD",
        "RISQUE_MACRO_GEOPOLITIQUE",
        "GOUVERNANCE_RISQUES",
        "CONTROLE_CONFORMITE",
        "SUJET_EMERGENT_HORS_GRILLE",
    }
)
_SENSITIVE_MATERIALITY_TEXT_RE = re.compile(
    r"\b(?:"
    r"capital|fonds?\s+propre|ad[ée]quation|suffisance|liquidit[ée]|"
    r"ratio|seuil|plancher|apr|actifs?\s+pond[ée]r[ée]s?|"
    r"gouvernance|comit[ée]|conseil|mandat|autor(?:it[ée]|ise|isation)|"
    r"responsabilit[ée]|supervision|reddition|ligne\s+de\s+d[ée]fense|"
    r"m[ée]thod(?:e|ologie)|mod[èe]le|processus|contr[oô]le|conformit[ée]|"
    r"exigence|r[èe]gle(?:mentaire)?|bsif|divulgation|transparence|"
    r"cyber|intelligence\s+artificielle|\bia\b|fraude|crypto|"
    r"sanction|donn[ée]e|tiers|nuage|climat|esg|risque"
    r")\b",
    flags=re.IGNORECASE,
)
_ISOLATED_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
    flags=re.IGNORECASE,
)
_VOLATILE_TOKEN_RE = re.compile(
    r"(?:"
    r"\b\d{1,2}\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)\s+\d{4}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:t|q)\s*[1-4]\s*[\-/–]?\s*\d{2,4}\b|"
    r"\bexercice\s+\d{4}\b|"
    r"\btrimestre\s+(?:de\s+)?\d{4}\b|"
    r"\b\d{4}\b|"
    r"\d[\d\s\u00a0.,]*\s*(?:%|m\$|g\$|mds?|millions?|milliards?)?\b"
    r")",
    flags=re.IGNORECASE,
)
_BANK_OPERATION_RE = re.compile(
    r"\b(?:"
    r"acquisition|acquérir|rachet|rachat|émission|émettre|dividende|"
    r"fusion|achat\s+d['’]actions|billets?\s+à\s+moyen\s+terme|"
    r"cwb|canadian\s+western\s+bank|transaction\s+d['’]entreprise|"
    r"offre\s+publique\s+d['’]achat|opa\b|spin[- ]?off"
    r")\b",
    flags=re.IGNORECASE,
)
_CALENDAR_UPDATE_RE = re.compile(
    r"\b(?:"
    r"jusqu['’]à\s+nouvel\s+ordre|report(?:é|er|ait)?|report|"
    r"calendrier|échéanc|exercice\s+\d{4}|à\s+compter|"
    r"progressiv|coefficient\s+de\s+plancher|plancher\s+de\s+fonds"
    r")\b",
    flags=re.IGNORECASE,
)
_METHODOLOGY_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"méthodolog|trimestriellement|périodiquement|mensuellement|"
    r"approche\s+standard|approche\s+interne|airb|modèle\s+interne|"
    r"sensibilités\s+standard|calcul(?:é|er)?\s+selon"
    r")\b",
    flags=re.IGNORECASE,
)
_PROCESS_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"processus|proc[ée]dure|flux\s+de\s+travail|workflow|"
    r"cha[iî]ne\s+de\s+traitement|mode\s+op[ée]ratoire"
    r")\b",
    flags=re.IGNORECASE,
)
_NEW_REGULATORY_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"b-15|ligne\s+directrice|tlac|bâle\s+iii|nouvelle\s+exigence|"
    r"entrée\s+en\s+vigueur|exigence\s+additionnelle|"
    r"cadre\s+réglementaire|avis\s+du\s+bsif"
    r")\b",
    flags=re.IGNORECASE,
)
_GOVERNANCE_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"gouvernance|comit[ée]s?|conseil\s+d['’]administration|mandat|"
    r"lignes?\s+de\s+d[ée]fense|responsabilit[ée]s?|supervision|"
    r"reddition\s+de\s+comptes|escalade|autorit[ée]\s+d[ée]cisionnelle|"
    r"droits?\s+d['’]approbation|culture\s+de\s+risque|"
    r"r[ée]mun[ée]ration|app[ée]tit\s+(?:pour\s+le|au)\s+risque"
    r")\b",
    flags=re.IGNORECASE,
)
_CALENDAR_SUBJECT_RE = re.compile(
    r"(?:"
    r"coefficient\s+de\s+plancher|plancher\s+des?\s+fonds\s+propres|"
    r"entrée\s+en\s+vigueur|report\s+des?\s+exigences|"
    r"calendrier\s+d['’]application|jusqu['’]à\s+nouvel\s+ordre"
    r")",
    flags=re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_THEME_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+", flags=re.IGNORECASE)
_THEME_STOPWORDS = frozenset(
    {
        "ajout",
        "changement",
        "dans",
        "des",
        "dune",
        "dun",
        "est",
        "les",
        "lié",
        "liée",
        "modification",
        "nouvelle",
        "pour",
        "rapport",
        "retrait",
        "risque",
        "une",
    }
)


class _EvidencePacketObservation(BaseModel):
    """Constat factuel unique pour le paquet T1/T2 fourni dans l'appel."""

    model_config = ConfigDict(extra="forbid")

    factual_change: str = Field(..., min_length=12, max_length=700)


class _EvidencePacketCoherenceCheck(BaseModel):
    """Contrôle unique pour le paquet de preuve fourni dans l'appel."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["supports", "contradicts", "insufficient"]
    reason: str = Field(..., min_length=12, max_length=700)


class _ConsolidatedDossierAssessment(MaterialityAssessment):
    """Jugement indépendant portant sur l'effet cumulé d'un dossier."""

    model_config = ConfigDict(extra="forbid")

    materiality_level: MaterialityLevel
    materiality_rationale: str = Field(
        ...,
        min_length=20,
        max_length=900,
    )


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
            return f"{bank_subject}{normalized[len(legacy_subject):]}"

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
            statement = f"{replacement}{statement[len(prefix):]}"
            break
    return f"{bank_subject} indique que {statement}"


def _semantic_reason_payload(
    *,
    is_relevant: bool,
    changement_constate: str,
    signification_metier: str = "",
    motif_non_pertinence: str = "",
) -> dict[str, str]:
    """Construit les champs analystes et leur assemblage historique."""
    raw_fields = {
        "changement_constate": changement_constate,
        "signification_metier": signification_metier,
        "motif_non_pertinence": motif_non_pertinence,
    }
    applicable = (
        {
            "changement_constate",
            "signification_metier",
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
        )
        if is_relevant
        else ("changement_constate", "motif_non_pertinence")
    )
    normalized_fields["relevance_reason"] = " ".join(
        normalized_fields[field_name] for field_name in reason_order
    )
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
    """Lire une preuve multipaquet par appels factuels séparés et auditables."""
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
            "compact_schema_version": "analyst_materiality_v4",
            "category": "NON_PERTINENT",
            "risk_type": "autre",
            "relevance_score": "FAIBLE",
            "risk_level": "FAIBLE",
            "impact_description": "",
            "reference_reglementaire": "",
            "confidence": 0.0,
            "materiality_level": None,
            "change_nature": [],
            "business_equivalence": "INDETERMINE",
            "materiality_confidence": "INDETERMINE",
            "evidence_sufficiency": "INSUFFISANTE",
            "decision_status": "A_CONFIRMER",
            "review_required": True,
            "supporting_evidence": [],
            "counterarguments": [],
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


def _requires_alignment_review(change: dict[str, Any]) -> bool:
    """True only when the first GPT call explicitly cannot decide the relation."""
    decision = str(change.get("alignment_decision") or "").strip().lower()
    if decision in _SEMANTIC_ALIGNMENT_DECISIONS:
        if decision == "uncertain":
            return True
        if decision == "moved_text":
            confidence = str(
                change.get("alignment_confidence") or ""
            ).strip().lower()
            return confidence != "high"
        return False
    # Cached artifacts from before semantic arbitration keep the former safe
    # fallback.  Fresh comparisons always carry ``alignment_decision``.
    return str(change.get("alignment_type") or "").strip().lower() == "ambiguous"


def _is_semantic_text_move(change: dict[str, Any]) -> bool:
    return (
        str(change.get("alignment_decision") or "").strip().lower()
        == "moved_text"
        and str(change.get("alignment_confidence") or "").strip().lower()
        == "high"
    )


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
            "Ce déplacement ne modifie ni le contenu métier ni la pratique "
            "divulguée."
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
            "materiality_level": "MINEUR",
            "impact_level": "MINEUR",
            "change_nature": ["DEPLACEMENT"],
            "business_equivalence": "CONFIRMEE",
            "materiality_confidence": "ELEVEE",
            "evidence_sufficiency": "SUFFISANTE",
            "decision_status": "CONFIRME",
            "review_required": False,
            "supporting_evidence": [
                "L'alignement sémantique confirme un déplacement sans modification substantielle."
            ],
            "counterarguments": [],
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
                        "Vérifie autant la pertinence et les thèmes que le niveau "
                        "de matérialité, la nature du changement et l'équivalence "
                        "métier proposées. "
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
                                "materiality_level": triage.get(
                                    "materiality_level"
                                ),
                                "change_nature": triage.get(
                                    "change_nature"
                                ),
                                "business_equivalence": triage.get(
                                    "business_equivalence"
                                ),
                                "materiality_confidence": triage.get(
                                    "materiality_confidence"
                                ),
                                "evidence_sufficiency": triage.get(
                                    "evidence_sufficiency"
                                ),
                                "decision_status": triage.get(
                                    "decision_status"
                                ),
                                "review_required": triage.get(
                                    "review_required"
                                ),
                                "supporting_evidence": triage.get(
                                    "supporting_evidence"
                                ),
                                "counterarguments": triage.get(
                                    "counterarguments"
                                ),
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


def _normalize_for_cosmetic(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _theme_tokens(value: str) -> set[str]:
    normalized = _normalize_for_cosmetic(value)
    return {
        token
        for token in _THEME_TOKEN_RE.findall(normalized)
        if len(token) >= 3 and token not in _THEME_STOPWORDS
    }


def _candidate_themes_for_change(
    change: dict[str, Any],
    *,
    section_key: str,
    limit: int = _COMPACT_THEME_CANDIDATE_LIMIT,
) -> list[dict[str, str]]:
    """Sélectionne localement une courte liste de thèmes AMF plausibles."""
    corpus = " ".join(
        str(value or "")
        for value in (
            change.get("change_summary"),
            change.get("source_text_t1"),
            change.get("source_text_t2"),
            change.get("semantic_text_t1"),
            change.get("semantic_text_t2"),
            change.get("subsection_heading"),
            section_key,
        )
    )
    corpus_tokens = _theme_tokens(corpus)
    scored: list[tuple[float, str]] = []
    for code, description in THEMES_AMF_DESCRIPTIONS.items():
        theme_text = f"{THEMES_AMF_ANALYST_SUBJECTS.get(code, '')} {description}"
        theme_tokens = _theme_tokens(theme_text)
        overlap = len(corpus_tokens & theme_tokens)
        coverage = overlap / max(len(theme_tokens), 1)
        score = float(overlap) + coverage
        if overlap:
            scored.append((score, code))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    diff_type = str(change.get("diff_type") or "").lower()
    forced = {
        "added": "DIVULGATION_AJOUT",
        "removed": "DIVULGATION_RETRAIT",
        "renamed": "STRUCTURE_RAPPORT",
    }.get(diff_type)
    if forced:
        selected.append(forced)

    for _, code in scored:
        if code not in selected:
            selected.append(code)
        if len(selected) >= max(limit - 1, 1):
            break

    if len(selected) < max(limit - 1, 1):
        section_fallbacks = {
            "gestion_capital": (
                "CAPITAL_REGLEMENTAIRE",
                "FONDS_PROPRES_REGLEMENTAIRES",
                "RATIOS_REGLEMENTAIRES",
                "EXIGENCES_REGLEMENTAIRES",
            ),
            "gestion_reglementation": (
                "NOUVELLE_MENTION_REGLEMENTAIRE",
                "EXIGENCES_REGLEMENTAIRES",
                "CONTROLE_CONFORMITE",
            ),
            "gestion_risques": (
                "MODIFICATION_TEXTE_RISQUE",
                "FACTEUR_RISQUE_CHANGEMENT",
                "GOUVERNANCE_RISQUES",
                "RISQUE_EMERGENT",
            ),
        }
        for code in section_fallbacks.get(section_key, ()):
            if code not in selected:
                selected.append(code)
            if len(selected) >= max(limit - 1, 1):
                break

    if "SUJET_EMERGENT_HORS_GRILLE" not in selected:
        selected.append("SUJET_EMERGENT_HORS_GRILLE")
    selected = selected[:limit]
    return [
        {
            "code": code,
            "label": THEMES_AMF_ANALYST_SUBJECTS[code],
            "description": THEMES_AMF_DESCRIPTIONS[code],
        }
        for code in selected
    ]


def _normalize_themes_amf(themes: list[str]) -> list[str]:
    """Accepte tout code de la taxonomie AMF ; remap les inconnus vers hors grille."""
    allowed = set(THEMES_AMF_DESCRIPTIONS)
    normalized: list[str] = []
    for theme in themes:
        code = str(theme or "").strip().upper()
        if not code:
            continue
        if code in allowed:
            if code not in normalized:
                normalized.append(code)
        elif "SUJET_EMERGENT_HORS_GRILLE" not in normalized:
            normalized.append("SUJET_EMERGENT_HORS_GRILLE")
            logger.debug(
                "theme_clamped unknown=%s -> SUJET_EMERGENT_HORS_GRILLE",
                code,
            )
    return normalized


def _sequence_ratio(left: str, right: str) -> float:
    left_norm = _normalize_for_cosmetic(left)
    right_norm = _normalize_for_cosmetic(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def _numeric_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _NUMERIC_TOKEN_RE.finditer(str(text or ""))}


def _regulatory_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _REGULATORY_REF_RE.finditer(str(text or ""))}


def _is_isolated_date_change(text_t1: str, text_t2: str) -> bool:
    """True when the only material difference looks like an isolated date update."""
    without_dates_t1 = _ISOLATED_DATE_RE.sub(" ", text_t1)
    without_dates_t2 = _ISOLATED_DATE_RE.sub(" ", text_t2)
    if _sequence_ratio(without_dates_t1, without_dates_t2) < 0.98:
        return False
    dates_t1 = {match.group(0).lower() for match in _ISOLATED_DATE_RE.finditer(text_t1)}
    dates_t2 = {match.group(0).lower() for match in _ISOLATED_DATE_RE.finditer(text_t2)}
    return bool(dates_t1 or dates_t2) and dates_t1 != dates_t2


def _mask_volatile_tokens(text: str) -> str:
    """Retire dates, trimestres et montants pour comparer le fond textuel."""
    masked = _VOLATILE_TOKEN_RE.sub(" ", str(text or ""))
    return _WHITESPACE_RE.sub(" ", masked).strip()


def _combined_change_text(change: dict[str, Any]) -> str:
    return " ".join(
        str(part or "")
        for part in (
            change.get("change_summary"),
            change.get("source_text_t1"),
            change.get("source_text_t2"),
            change.get("semantic_text_t1"),
            change.get("semantic_text_t2"),
        )
        if str(part or "").strip()
    )


def _has_methodology_signal(text_t1: str, text_t2: str) -> bool:
    markers_t1 = {m.group(0).lower() for m in _METHODOLOGY_SIGNAL_RE.finditer(text_t1)}
    markers_t2 = {m.group(0).lower() for m in _METHODOLOGY_SIGNAL_RE.finditer(text_t2)}
    return bool(markers_t1.symmetric_difference(markers_t2))


def _has_new_regulatory_substance(text_t1: str, text_t2: str) -> bool:
    """Conservé pertinent seulement si une mention réglementaire NOUVELLE apparaît en T2."""
    signals_t1 = {m.group(0).lower() for m in _NEW_REGULATORY_SIGNAL_RE.finditer(text_t1)}
    signals_t2 = {m.group(0).lower() for m in _NEW_REGULATORY_SIGNAL_RE.finditer(text_t2)}
    # Disappearance of a Bâle III / BSIF mention alone is reformulation, not substance.
    return bool(signals_t2 - signals_t1)


def _shares_calendar_subject(text_t1: str, text_t2: str) -> bool:
    subjects_t1 = {m.group(0).lower() for m in _CALENDAR_SUBJECT_RE.finditer(text_t1)}
    subjects_t2 = {m.group(0).lower() for m in _CALENDAR_SUBJECT_RE.finditer(text_t2)}
    return bool(subjects_t1 & subjects_t2)


def _has_calendar_reschedule_context(
    text_t1: str,
    text_t2: str,
    combined: str,
) -> bool:
    """True when the delta is a deferred-application update of a known requirement."""
    if not _CALENDAR_UPDATE_RE.search(combined):
        return False
    has_anchor = bool(
        _CALENDAR_SUBJECT_RE.search(combined)
        or re.search(
            r"\b(?:bsif|plancher\s+de\s+fonds|coefficient\s+de\s+plancher)\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    if not has_anchor:
        return False
    if _shares_calendar_subject(text_t1, text_t2):
        return True
    # Subject may appear only in T1 while T2 only updates the deferral wording.
    if _CALENDAR_SUBJECT_RE.search(text_t1) and _CALENDAR_UPDATE_RE.search(text_t2):
        return True
    return bool(
        re.search(
            r"\b(?:bsif|plancher|coefficient)\b",
            combined,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\b(?:report|retard|jusqu['’]à\s+nouvel\s+ordre)\b",
            combined,
            flags=re.IGNORECASE,
        )
    )


def _is_pure_new_regulatory_disclosure(change: dict[str, Any]) -> bool:
    """First mention of a shared regulatory requirement, without a bank deal."""
    diff_type = str(change.get("diff_type") or "").strip().lower()
    text_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    combined = _combined_change_text(change)
    if diff_type != "added" or not text_t2.strip():
        return False
    if _BANK_OPERATION_RE.search(combined):
        return False
    return bool(_NEW_REGULATORY_SIGNAL_RE.search(text_t2))


def _deterministic_bank_specific_exclusion(change: dict[str, Any]) -> str | None:
    """Repère dates, montants ou opérations sans modification prudentielle."""
    text_t1 = str(change.get("source_text_t1") or change.get("semantic_text_t1") or "")
    text_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    combined = _combined_change_text(change)
    diff_type = str(change.get("diff_type") or "").strip().lower()

    # Priority 1: bank-specific operations (acquisition, CWB, buyback, issuance).
    if _BANK_OPERATION_RE.search(combined):
        if _is_pure_new_regulatory_disclosure(change):
            pass
        else:
            return "operation_interne_banque"

    # Keep true methodology changes for analyst review — but only when no
    # bank operation already matched above.
    if text_t1.strip() and text_t2.strip():
        if _has_methodology_signal(text_t1, text_t2) and not _BANK_OPERATION_RE.search(
            combined
        ):
            return None
        if _has_new_regulatory_substance(text_t1, text_t2):
            masked_ratio = _sequence_ratio(
                _mask_volatile_tokens(text_t1),
                _mask_volatile_tokens(text_t2),
            )
            if (
                masked_ratio < _BANK_NOISE_SEQUENCE_THRESHOLD
                and not _BANK_OPERATION_RE.search(combined)
            ):
                return None

    if diff_type not in {"modified", "unchanged"}:
        return None
    if not text_t1.strip() or not text_t2.strip():
        return None

    masked_t1 = _mask_volatile_tokens(text_t1)
    masked_t2 = _mask_volatile_tokens(text_t2)
    if not masked_t1 or not masked_t2:
        return None

    masked_ratio = _sequence_ratio(masked_t1, masked_t2)
    numbers_differ = _numeric_tokens(text_t1) != _numeric_tokens(text_t2)
    dates_differ = _is_isolated_date_change(text_t1, text_t2) or (
        {m.group(0).lower() for m in _ISOLATED_DATE_RE.finditer(text_t1)}
        != {m.group(0).lower() for m in _ISOLATED_DATE_RE.finditer(text_t2)}
    )
    volatile_differ = numbers_differ or dates_differ

    # Calendar updates of a known requirement (e.g. BSIF floor deferral).
    if volatile_differ and _has_calendar_reschedule_context(text_t1, text_t2, combined):
        if not _has_methodology_signal(text_t1, text_t2):
            return "mise_a_jour_calendrier"

    if masked_ratio >= _BANK_NOISE_SEQUENCE_THRESHOLD and volatile_differ:
        if _CALENDAR_UPDATE_RE.search(combined) and dates_differ:
            return "mise_a_jour_calendrier"
        if dates_differ and not numbers_differ:
            return "mise_a_jour_calendrier"
        if numbers_differ:
            return "variation_numerique_propre_banque"
        return "variation_numerique_propre_banque"

    # Fallback: calendar reschedule when reformulation lowers similarity below 0.92.
    if (
        volatile_differ
        and dates_differ
        and _has_calendar_reschedule_context(text_t1, text_t2, combined)
        and not _has_methodology_signal(text_t1, text_t2)
    ):
        return "mise_a_jour_calendrier"
    return None


def _deterministic_cosmetic_exclusion(change: dict[str, Any]) -> str | None:
    """Return an exclusion reason when the change is manifestly cosmetic."""
    if _is_semantic_text_move(change):
        return "deplacement_texte"
    if str(change.get("alignment_type") or "").strip().lower() in {
        "global_reconciled_residual",
    } and str(change.get("alignment_decision") or "").strip().lower() in {
        "moved_text",
        "same_disclosure",
    }:
        # Residual after a confirmed move/resegmentation is already handled upstream.
        pass

    diff_type = str(change.get("diff_type") or "").strip().lower()
    if diff_type not in {"modified", "unchanged"}:
        return None

    text_t1 = str(change.get("source_text_t1") or "")
    text_t2 = str(change.get("source_text_t2") or "")
    if not text_t1.strip() or not text_t2.strip():
        return None

    if _numeric_tokens(text_t1) != _numeric_tokens(text_t2):
        return None
    if _regulatory_tokens(text_t1) != _regulatory_tokens(text_t2):
        return None

    compact_t1 = re.sub(r"[^\w]+", "", _normalize_for_cosmetic(text_t1), flags=re.UNICODE)
    compact_t2 = re.sub(r"[^\w]+", "", _normalize_for_cosmetic(text_t2), flags=re.UNICODE)
    if compact_t1 == compact_t2 and text_t1 != text_t2:
        return "formatage_visuel"

    # Une modification très courte peut changer le nom d'un comité, un mandat,
    # une responsabilité ou une ligne de défense. Ces cas doivent atteindre le
    # triage métier plutôt que d'être écartés selon leur seule similarité.
    if _GOVERNANCE_SIGNAL_RE.search(f"{text_t1} {text_t2}"):
        return None

    similarity = _sequence_ratio(text_t1, text_t2)
    if similarity >= _COSMETIC_SEQUENCE_THRESHOLD:
        return "reformulation_mineure"
    if _is_isolated_date_change(text_t1, text_t2):
        return "reformulation_mineure"
    return None


def _is_proven_pure_numeric_change(change: dict[str, Any]) -> bool:
    """True uniquement lorsque les chiffres sont la seule différence démontrée."""
    diff_type = str(change.get("diff_type") or "").strip().lower()
    if diff_type not in {"modified", "unchanged"}:
        return False
    text_t1 = str(change.get("source_text_t1") or change.get("semantic_text_t1") or "")
    text_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    if not text_t1.strip() or not text_t2.strip():
        return False
    if _numeric_tokens(text_t1) == _numeric_tokens(text_t2):
        return False

    combined = f"{text_t1} {text_t2}"
    if (
        _REGULATORY_REF_RE.search(combined)
        or _METHODOLOGY_SIGNAL_RE.search(combined)
        or _PROCESS_SIGNAL_RE.search(combined)
        or _GOVERNANCE_SIGNAL_RE.search(combined)
        or re.search(
            r"\b(?:seuil|plancher|minimum|maximum|cible|exigence|coefficient|"
            r"ratio\s+r[ée]glementaire|capital\s+requis)\b",
            combined,
            flags=re.IGNORECASE,
        )
    ):
        return False
    return _normalize_for_cosmetic(_mask_volatile_tokens(text_t1)) == (
        _normalize_for_cosmetic(_mask_volatile_tokens(text_t2))
    )


def _triage_advisory_signals(change: dict[str, Any]) -> list[str]:
    """Retourne des indices auditables sans imposer une décision métier."""
    signals: list[str] = []
    for signal in (
        _deterministic_cosmetic_exclusion(change),
        _deterministic_bank_specific_exclusion(change),
    ):
        if signal and signal not in signals:
            signals.append(signal)
    return signals


def _hard_prefilter_exclusion(change: dict[str, Any]) -> str | None:
    """Réserve l'exclusion automatique aux équivalences mécaniquement prouvées."""
    cosmetic = _deterministic_cosmetic_exclusion(change)
    if cosmetic in _HARD_PREFILTER_EXCLUSIONS:
        return cosmetic
    bank_signal = _deterministic_bank_specific_exclusion(change)
    if (
        bank_signal == "variation_numerique_propre_banque"
        and _is_proven_pure_numeric_change(change)
    ):
        return bank_signal
    return None


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
    non_relevance_reason = (
        "Ce changement ne modifie aucune pratique prudentielle, méthode, "
        "responsabilité ou exigence divulguée."
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
        return factual, non_relevance_reason, subject

    if exclusion_reason == "variation_numerique_propre_banque":
        subject = "Variation chiffrée propre à la banque"
        if diff_type == "added":
            factual = (
                f"{bank_subject} ajoute des montants ou pourcentages "
                "propres à la banque, sans nouvelle exigence réglementaire."
            )
        else:
            factual = (
                f"{bank_subject} met uniquement à jour des chiffres, montants "
                "ou pourcentages propres à ses activités."
            )
        return factual, non_relevance_reason, subject

    if exclusion_reason == "mise_a_jour_calendrier":
        subject = "Mise à jour de calendrier"
        factual = (
            f"{bank_subject} met uniquement à jour les dates ou échéances "
            "d’application, sans ajouter de nouvelle exigence."
        )
        return factual, non_relevance_reason, subject

    # Cosmetic / generic exclusions
    subject = "Changement cosmétique"
    if exclusion_reason == "deplacement_texte":
        factual = (
            f"{bank_subject} déplace la même divulgation sans ajouter de "
            "nouveau contenu."
        )
        subject = "Déplacement de texte"
    elif exclusion_reason == "formatage_visuel":
        factual = (
            f"{bank_subject} reformule le passage sans changement de fond "
            "(ponctuation ou formatage uniquement)."
        )
    elif diff_type == "added":
        factual = (
            f"{bank_subject} introduit une formulation différente "
            "sans changement de fond."
        )
    else:
        factual = f"{bank_subject} reformule le passage sans changement de fond."
    return factual, non_relevance_reason, subject


def _prefilter_triage_result(
    change: dict[str, Any],
    exclusion_reason: str,
    *,
    bank_code: str = "",
) -> dict[str, Any]:
    triage = _default_triage(bank_code)
    factual, non_relevance_reason, subject = _analyst_exclusion_copy(
        change,
        exclusion_reason,
        bank_code=bank_code,
    )
    analyst_copy = _semantic_reason_payload(
        is_relevant=False,
        changement_constate=factual,
        motif_non_pertinence=non_relevance_reason,
    )
    triage.update(
        {
            "source": "deterministic_prefilter",
            "exclusion_reason": exclusion_reason,
            "materiality_level": "MINEUR",
            "impact_level": "MINEUR",
            "change_nature": [
                {
                    "formatage_visuel": "FORMATAGE",
                    "deplacement_texte": "DEPLACEMENT",
                    "variation_numerique_propre_banque": "VARIATION_CHIFFREE",
                }.get(exclusion_reason, "REFORMULATION_EQUIVALENTE")
            ],
            "business_equivalence": "CONFIRMEE",
            "materiality_confidence": "ELEVEE",
            "evidence_sufficiency": "SUFFISANTE",
            "decision_status": "CONFIRME",
            "review_required": False,
            "supporting_evidence": [
                "La différence automatique est limitée à un élément mécaniquement vérifiable."
            ],
            "counterarguments": [],
            **analyst_copy,
            "nouvelle_idee_justification": (
                "NON — Nouvel élément à surveiller : Non.\n\n"
                f"Sujet détecté : {subject}.\n\n"
                f"Ce qui change : {factual}\n\n"
                f"Pertinence métier : {non_relevance_reason}\n\n"
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


def _triage_retrieval_text(change: dict[str, Any]) -> str:
    parts = [
        str(change.get("diff_type") or ""),
        str(change.get("subsection_heading") or ""),
        str(change.get("change_summary") or ""),
        str(change.get("source_text_t1") or "")[:600],
        str(change.get("source_text_t2") or "")[:600],
    ]
    text = " | ".join(part.strip() for part in parts if str(part or "").strip())
    if len(text) <= _TRIAGE_EMBEDDING_TRUNCATE_CHARS:
        return text
    return text[:_TRIAGE_EMBEDDING_TRUNCATE_CHARS]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _changes_compatible_for_dedup(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Never merge when nature, decision or evidence shape diverge."""
    if str(left.get("diff_type") or "") != str(right.get("diff_type") or ""):
        return False
    if str(left.get("alignment_decision") or "") != str(right.get("alignment_decision") or ""):
        return False
    left_has_t1 = bool(str(left.get("source_text_t1") or "").strip())
    right_has_t1 = bool(str(right.get("source_text_t1") or "").strip())
    left_has_t2 = bool(str(left.get("source_text_t2") or "").strip())
    right_has_t2 = bool(str(right.get("source_text_t2") or "").strip())
    if left_has_t1 != right_has_t1 or left_has_t2 != right_has_t2:
        return False
    return True


def _group_semantic_triage_duplicates(
    changes: list[dict[str, Any]],
    *,
    client: Any,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> list[list[int]]:
    """Group near-duplicate changes; returns lists of indices into ``changes``."""
    if len(changes) <= 1:
        return [[index] for index in range(len(changes))]

    try:
        embeddings = _embed_texts(
            client,
            [_triage_retrieval_text(change) for change in changes],
            model=embedding_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Déduplication triage embeddings indisponible: %s", exc)
        return [[index] for index in range(len(changes))]

    parents = list(range(len(changes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    for left_index in range(len(changes)):
        for right_index in range(left_index + 1, len(changes)):
            if not _changes_compatible_for_dedup(changes[left_index], changes[right_index]):
                continue
            score = _cosine_similarity(embeddings[left_index], embeddings[right_index])
            if score >= _TRIAGE_DEDUP_EMBEDDING_THRESHOLD:
                union(left_index, right_index)

    grouped: dict[int, list[int]] = {}
    for index in range(len(changes)):
        grouped.setdefault(find(index), []).append(index)
    return [sorted(members) for members in grouped.values()]


def _change_context_for_dossier(change: dict[str, Any]) -> dict[str, Any]:
    """Construit un contexte compact et factuel pour un changement relié."""
    return {
        "change_id": str(change.get("change_id") or ""),
        "diff_type": str(change.get("diff_type") or ""),
        "subsection_heading": str(change.get("subsection_heading") or ""),
        "change_summary": _truncate_prompt_text(
            str(change.get("change_summary") or ""),
            _TRIAGE_SOURCE_SNIPPET_LIMIT,
        ),
        "source_snippet_t1": _truncate_prompt_text(
            str(
                change.get("source_text_t1")
                or change.get("semantic_text_t1")
                or ""
            ),
            _TRIAGE_SOURCE_SNIPPET_LIMIT,
        ),
        "source_snippet_t2": _truncate_prompt_text(
            str(
                change.get("source_text_t2")
                or change.get("semantic_text_t2")
                or ""
            ),
            _TRIAGE_SOURCE_SNIPPET_LIMIT,
        ),
    }


def _annotate_triage_dossiers(
    changes: list[dict[str, Any]],
    *,
    section_key: str,
    groups: list[list[int]],
) -> list[dict[str, Any]]:
    """Ajoute le contexte relié sans copier le verdict d'un représentant."""
    group_by_index: dict[int, tuple[str, list[int]]] = {}
    for group_number, member_indexes in enumerate(groups, start=1):
        if len(member_indexes) <= 1:
            continue
        group_id = f"{section_key}_triage_dossier_{group_number:03d}"
        for member_index in member_indexes:
            group_by_index[member_index] = (group_id, member_indexes)

    annotated: list[dict[str, Any]] = []
    for index, change in enumerate(changes):
        enriched = dict(change)
        existing_dossier = change.get("triage_dossier")
        if isinstance(existing_dossier, dict) and existing_dossier:
            enriched["triage_dossier"] = dict(existing_dossier)
            annotated.append(enriched)
            continue
        related_indexes: list[int] = []
        group = group_by_index.get(index)
        if group is not None:
            _group_id, member_indexes = group
            related_indexes.extend(
                member_index
                for member_index in member_indexes
                if member_index != index
            )

        subsection = str(change.get("subsection_heading") or "").strip().casefold()
        if subsection:
            related_indexes.extend(
                candidate_index
                for candidate_index, candidate in enumerate(changes)
                if candidate_index != index
                and str(candidate.get("subsection_heading") or "").strip().casefold()
                == subsection
            )

        unique_related = list(dict.fromkeys(related_indexes))[
            :_RELATED_CHANGE_CONTEXT_LIMIT
        ]
        dossier: dict[str, Any] = {
            "related_changes": [
                _change_context_for_dossier(changes[related_index])
                for related_index in unique_related
            ],
            "classification_scope": (
                "Évaluer ce changement individuellement en tenant compte de "
                "l'effet cumulatif des changements reliés."
            ),
        }
        if group is not None:
            group_id, member_indexes = group
            dossier.update(
                {
                    "group_id": group_id,
                    "member_change_ids": [
                        str(changes[member_index].get("change_id") or "")
                        for member_index in member_indexes
                    ],
                    "verdict_propagation": False,
                }
            )
        enriched["triage_dossier"] = dossier
        annotated.append(enriched)
    return annotated


def _attach_consolidated_dossier_outcomes(
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose le niveau consolidé tout en conservant chaque décision individuelle."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for change in changes:
        dossier = change.get("triage_dossier") or {}
        group_id = str(dossier.get("group_id") or "")
        if group_id:
            grouped.setdefault(group_id, []).append(change)

    for group_id, members in grouped.items():
        levels = [
            str((member.get("genai_triage") or {}).get("impact_level") or "MINEUR").upper()
            for member in members
        ]
        consolidated_level = max(
            levels,
            key=lambda level: _MATERIALITY_RANK.get(level, -1),
        )
        relevant = any(
            bool((member.get("genai_triage") or {}).get("is_relevant"))
            for member in members
        )
        member_ids = [str(member.get("change_id") or "") for member in members]
        for member in members:
            triage = dict(member.get("genai_triage") or {})
            triage.update(
                {
                    "triage_group_id": group_id,
                    "triage_group_member_ids": member_ids,
                    "triage_group_verdict_propagated": False,
                    "consolidated_materiality_level": consolidated_level,
                    "consolidated_relevant": relevant,
                }
            )
            member["genai_triage"] = triage
            member["triage_dedup"] = {
                "group_id": group_id,
                "member_change_ids": member_ids,
                "propagated": False,
                "classification_mode": "all_members_with_shared_context",
            }
    return changes


def _evaluate_consolidated_dossier_materiality(
    *,
    client: Any,
    model: str,
    bank_code: str,
    section_key: str,
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Évalue séparément l'effet cumulé des dossiers reliés.

    Les verdicts individuels restent intacts. Le jugement de dossier peut
    toutefois produire un niveau consolidé supérieur lorsque plusieurs
    modifications cohérentes démontrent ensemble un changement de portée.
    """
    enriched = [
        {
            **change,
            "genai_triage": dict(change.get("genai_triage") or {}),
        }
        for change in changes
    ]
    _attach_consolidated_dossier_outcomes(enriched)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for change in enriched:
        dossier = change.get("triage_dossier") or {}
        group_id = str(dossier.get("group_id") or "")
        if group_id:
            grouped.setdefault(group_id, []).append(change)
    if not grouped:
        return enriched

    bank_subject = analyst_bank_subject(bank_code)
    for group_id, members in grouped.items():
        if len(members) <= 1:
            continue
        individual_levels = [
            str(
                (member.get("genai_triage") or {}).get("impact_level")
                or "MINEUR"
            ).upper()
            for member in members
        ]
        individual_max = max(
            individual_levels,
            key=lambda value: _MATERIALITY_RANK.get(value, -1),
        )
        dossier_input = {
            "group_id": group_id,
            "classification_scope": (
                "Évaluer uniquement l'effet cumulé démontré par les changements "
                "reliés, sans recopier leurs verdicts individuels."
            ),
            "members": [
                {
                    **_change_context_for_dossier(member),
                    "changement_constate": str(
                        (member.get("genai_triage") or {}).get(
                            "changement_constate"
                        )
                        or ""
                    ),
                    "themes_amf": list(
                        (member.get("genai_triage") or {}).get(
                            "themes_amf"
                        )
                        or []
                    ),
                }
                for member in members
            ],
        }
        try:
            assessment = _call_structured_completion_with_correction(
                client,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es l'évaluateur indépendant de matérialité "
                            "cumulative pour une vigie prudentielle. Les éléments "
                            "d'un dossier ont déjà été classés individuellement, "
                            "mais leurs niveaux ne te sont pas montrés. Détermine "
                            "si leur combinaison cohérente modifie une définition, "
                            "un périmètre, une responsabilité, une finalité, une "
                            "méthode, un contrôle, une obligation, le capital ou "
                            "un risque. Un simple nombre de changements ne suffit "
                            "jamais à élever le niveau. MAJEUR exige un effet "
                            "cumulatif substantiel démontré; MODERE couvre un effet "
                            "plausible ou une preuve partielle; MINEUR exige une "
                            "équivalence cumulative confirmée."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Renseigne tous les champs du schéma de matérialité "
                            "ainsi que materiality_rationale. Cite les faits "
                            "cumulatifs dans supporting_evidence, examine une "
                            "interprétation plus faible dans counterarguments et "
                            "exige une revue si la preuve n'est pas suffisante. "
                            f"Banque : {bank_subject}. Section : {section_key}.\n"
                            f"Dossier :\n{_json_dumps(dossier_input)}"
                        ),
                    },
                ],
                response_format=_ConsolidatedDossierAssessment,
                max_tokens=1400,
                max_retries=2,
                validation_retry_message=(
                    "Renvoie uniquement le schéma complet. Une décision "
                    "CONFIRME exige une preuve SUFFISANTE et une confiance "
                    "ELEVEE ou MOYENNE. MINEUR sans équivalence CONFIRMEE "
                    "exige review_required=true."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "consolidated dossier assessment unavailable group=%s: %s",
                group_id,
                exc,
            )
            for member in members:
                triage = member["genai_triage"]
                triage.update(
                    {
                        "consolidated_materiality_level": individual_max,
                        "consolidated_decision_status": "A_CONFIRMER",
                        "consolidated_review_required": True,
                        "consolidated_assessment_source": (
                            "individual_max_fallback"
                        ),
                        "consolidated_assessment_error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )
            continue

        assessed_level = str(assessment.materiality_level).upper()
        assessment_lower = (
            _MATERIALITY_RANK.get(assessed_level, -1)
            < _MATERIALITY_RANK.get(individual_max, -1)
        )
        consolidated_level = max(
            (individual_max, assessed_level),
            key=lambda value: _MATERIALITY_RANK.get(value, -1),
        )
        consolidated_review = (
            assessment.review_required or assessment_lower
        )
        consolidated_status = (
            "A_CONFIRMER"
            if consolidated_review
            else assessment.decision_status
        )
        assessment_audit = assessment.model_dump()
        assessment_audit["individual_max_materiality_level"] = (
            individual_max
        )
        assessment_audit["assessment_lower_than_individual_max"] = (
            assessment_lower
        )
        for member in members:
            triage = member["genai_triage"]
            triage.update(
                {
                    "consolidated_materiality_level": consolidated_level,
                    "consolidated_relevant": (
                        any(
                            bool(
                                (candidate.get("genai_triage") or {}).get(
                                    "is_relevant"
                                )
                            )
                            for candidate in members
                        )
                        or consolidated_level in {"MODERE", "MAJEUR"}
                    ),
                    "consolidated_decision_status": consolidated_status,
                    "consolidated_review_required": consolidated_review,
                    "consolidated_assessment_source": (
                        "independent_dossier_assessment"
                    ),
                    "consolidated_materiality_assessment": (
                        assessment_audit
                    ),
                }
            )
    return enriched


def _propagate_triage_to_group(
    *,
    representative: dict[str, Any],
    members: list[dict[str, Any]],
    group_id: str,
    bank_code: str = "",
) -> list[dict[str, Any]]:
    triage = dict(
        representative.get("genai_triage") or _default_triage(bank_code)
    )
    member_ids = [str(change.get("change_id") or "") for change in members]
    propagated: list[dict[str, Any]] = []
    for change in members:
        enriched = dict(change)
        member_triage = dict(triage)
        member_triage["triage_group_id"] = group_id
        member_triage["triage_group_member_ids"] = member_ids
        member_triage["triage_group_representative_id"] = str(
            representative.get("change_id") or ""
        )
        if str(change.get("change_id") or "") != str(representative.get("change_id") or ""):
            member_triage["source"] = f"{triage.get('source') or 'gpt'}_propagated"
        enriched["genai_triage"] = member_triage
        enriched["triage_dedup"] = {
            "group_id": group_id,
            "representative_change_id": str(representative.get("change_id") or ""),
            "member_change_ids": member_ids,
            "propagated": str(change.get("change_id") or "")
            != str(representative.get("change_id") or ""),
        }
        propagated.append(enriched)
    return propagated


_FEW_SHOT_TRIAGE_AMF = """\
Exemple 1 — ajout cyber pertinent
Input : {"bank_subject": "CIBC", "change_index": 1, "diff_type": "added", "change_summary": "Ajout d’exercices annuels de simulation de cyberattaque."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["RISQUE_EMERGENT", "CONTROLE_CONFORMITE"], "nouvelle_idee": true, "changement_constate": "CIBC ajoute des simulations annuelles de cyberattaque avec ses unités d’affaires.", "signification_metier": "Cette évolution rend explicite un mécanisme récurrent de préparation aux incidents cybernétiques.", "motif_non_pertinence": ""}

Exemple 2 — variation propre à la banque non pertinente
Input : {"bank_subject": "BMO", "change_index": 1, "diff_type": "modified", "change_summary": "Le portefeuille hypothécaire passe de 287 G$ à 294 G$."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "changement_constate": "BMO fait passer son portefeuille hypothécaire de 287 G$ à 294 G$, sans modifier la méthode de calcul ni le périmètre présenté.", "signification_metier": "", "motif_non_pertinence": "Cette variation reflète l’évolution normale des activités et ne modifie aucune pratique de gestion des risques."}

Exemple 3 — calendrier administratif non pertinent
Input : {"bank_subject": "RBC", "change_index": 1, "diff_type": "modified", "change_summary": "La date prévue de publication du rapport passe du 30 juin au 2 juillet, sans modification de l’information publiée."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "changement_constate": "RBC déplace du 30 juin au 2 juillet la date administrative de publication du rapport sans modifier l’information publiée.", "signification_metier": "", "motif_non_pertinence": "Ce déplacement administratif ne modifie ni une exigence prudentielle ni le contenu métier divulgué."}

Exemple 4 — acquisition interne non pertinente
Input : {"bank_subject": "BNC", "change_index": 1, "diff_type": "added", "change_summary": "Inclusion de CWB dans le calcul du risque opérationnel à la suite de l’acquisition."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "changement_constate": "BNC inclut CWB dans le calcul du risque opérationnel à la suite de son acquisition, sans décrire une nouvelle méthode de calcul.", "signification_metier": "", "motif_non_pertinence": "Cette opération propre à la banque ne modifie ni la méthode ni la pratique de gestion du risque opérationnel."}

Exemple 5 — rachat d’actions non pertinent
Input : {"bank_subject": "TD", "change_index": 1, "diff_type": "modified", "change_summary": "Mise à jour des montants de rachat d’actions ordinaires au semestre."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "changement_constate": "TD met à jour les montants de rachat d’actions ordinaires déjà présentés, sans modifier le cadre réglementaire associé.", "signification_metier": "", "motif_non_pertinence": "Cette transaction propre à la banque ne modifie pas le cadre prudentiel divulgué."}

Exemple 6 — transfert de responsabilité de gouvernance pertinent et substantiel
Input : {"bank_subject": "RBC", "change_index": 1, "diff_type": "modified", "change_summary": "L’approbation de l’appétit pour le risque passe du comité de direction au conseil d’administration."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["GOUVERNANCE_RISQUES"], "nouvelle_idee": true, "changement_constate": "RBC transfère au conseil d’administration l’approbation de l’appétit pour le risque auparavant confiée au comité de direction.", "signification_metier": "Ce transfert élève la décision au niveau de gouvernance ultime de la banque.", "motif_non_pertinence": ""}

Exemple 7 — comité renommé pertinent sans nouvelle idée substantielle
Input : {"bank_subject": "CIBC", "change_index": 1, "diff_type": "modified", "change_summary": "Le Comité de gestion des risques est renommé Comité des risques et de la conformité, sans modification de son mandat."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["GOUVERNANCE_RISQUES"], "nouvelle_idee": false, "changement_constate": "CIBC renomme le Comité de gestion des risques en Comité des risques et de la conformité tout en maintenant son mandat.", "signification_metier": "Cette désignation rend la conformité plus visible dans la structure déclarée de gouvernance.", "motif_non_pertinence": ""}

Exemple 8 — changement réel de méthodologie pertinent et substantiel
Input : {"bank_subject": "BMO", "change_index": 1, "diff_type": "modified", "change_summary": "La méthode standard de mesure du risque de crédit est remplacée par un modèle interne avancé."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["MODIFICATION_METHODOLOGIE"], "nouvelle_idee": true, "changement_constate": "BMO remplace la méthode standard de mesure du risque de crédit par un modèle interne avancé.", "signification_metier": "Cette nouvelle base méthodologique peut modifier la mesure et la sensibilité du risque déclaré.", "motif_non_pertinence": ""}

Exemple 9 — modification réelle de processus pertinente et substantielle
Input : {"bank_subject": "BNS", "change_index": 1, "diff_type": "modified", "change_summary": "Les alertes de conformité sont désormais validées par une deuxième équipe avant leur clôture."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["CONTROLE_CONFORMITE"], "nouvelle_idee": true, "changement_constate": "BNS ajoute une seconde validation au processus de clôture des alertes de conformité.", "signification_metier": "Cette étape supplémentaire formalise un contrôle indépendant avant la clôture des alertes.", "motif_non_pertinence": ""}
"""


_FEW_SHOT_MATERIALITY_AMF = """\
Exemple de matérialité A — terminologie prudentielle non démontrée équivalente
Input : {"bank_subject": "BMO", "change_index": 1, "diff_type": "modified", "change_summary_factuel_non_arbitre": "BMO remplace « suffisance du capital » par « adéquation des fonds propres »."}
Décision attendue : {"materiality_level": "MODERE", "change_nature": ["MODIFICATION_TERMINOLOGIE"], "business_equivalence": "NON_DEMONTREE", "materiality_confidence": "MOYENNE", "evidence_sufficiency": "PARTIELLE", "decision_status": "A_CONFIRMER", "review_required": true, "supporting_evidence": ["Le terme prudentiel central est remplacé et la preuve fournie ne démontre pas une stricte équivalence de référent ou de périmètre."], "counterarguments": ["Le remplacement pourrait refléter une harmonisation terminologique sans changement du cadre de capital."]}

Exemple de matérialité B — évolution consolidée de l'allocation du capital
Input : {"bank_subject": "BMO", "change_index": 1, "diff_type": "modified", "change_summary_factuel_non_arbitre": "BMO remplace les groupes d'exploitation par des unités d'exploitation et ajoute que le processus guide la répartition des ressources et optimise les rendements.", "related_change_dossier": {"related_changes": [{"change_summary": "BMO ajoute la surveillance et l'optimisation des rendements."}]}}
Décision attendue : {"materiality_level": "MAJEUR", "change_nature": ["MODIFICATION_PERIMETRE", "MODIFICATION_RESPONSABILITES"], "business_equivalence": "REFUTEE", "materiality_confidence": "ELEVEE", "evidence_sufficiency": "SUFFISANTE", "decision_status": "CONFIRME", "review_required": false, "supporting_evidence": ["Les changements reliés ajoutent explicitement des finalités de répartition des ressources et d'optimisation des rendements au processus de capital."], "counterarguments": ["Le remplacement de groupes par unités, pris isolément, pourrait être terminologique."]}

Exemple de matérialité C — renommage démontré équivalent
Input : {"bank_subject": "CIBC", "change_index": 1, "diff_type": "modified", "change_summary_factuel_non_arbitre": "CIBC modifie seulement l'acronyme d'un comité et maintient explicitement son mandat, son autorité et ses responsabilités."}
Décision attendue : {"materiality_level": "MINEUR", "change_nature": ["REFORMULATION_EQUIVALENTE"], "business_equivalence": "CONFIRMEE", "materiality_confidence": "ELEVEE", "evidence_sufficiency": "SUFFISANTE", "decision_status": "CONFIRME", "review_required": false, "supporting_evidence": ["Le texte confirme expressément que le mandat, l'autorité et les responsabilités demeurent inchangés."], "counterarguments": []}

Exemple de matérialité D — statut réglementaire devenu indéterminé
Input : {"bank_subject": "RBC", "change_index": 1, "diff_type": "modified", "change_summary_factuel_non_arbitre": "RBC remplace une date déterminée d'augmentation du plancher par un report jusqu'à nouvel ordre."}
Décision attendue : {"materiality_level": "MODERE", "change_nature": ["MODIFICATION_EXIGENCE_REGLEMENTAIRE", "MODIFICATION_STATUT_MISE_EN_OEUVRE"], "business_equivalence": "REFUTEE", "materiality_confidence": "ELEVEE", "evidence_sufficiency": "SUFFISANTE", "decision_status": "CONFIRME", "review_required": false, "supporting_evidence": ["L'horizon d'application passe d'une échéance déterminée à une durée indéterminée, ce qui modifie le statut de mise en œuvre déclaré."], "counterarguments": ["La nature technique du coefficient de plancher demeure inchangée."]}
"""


def _compact_materiality_snapshot(
    triage: TriageAMFCompactLLMResultWithIndex,
) -> dict[str, Any]:
    """Retourne uniquement les éléments nécessaires à l'audit du désaccord."""
    return {
        "is_relevant": triage.is_relevant,
        "themes_amf": list(triage.themes_amf),
        "nouvelle_idee": triage.nouvelle_idee,
        "materiality_level": triage.materiality_level,
        "change_nature": list(triage.change_nature),
        "business_equivalence": triage.business_equivalence,
        "materiality_confidence": triage.materiality_confidence,
        "evidence_sufficiency": triage.evidence_sufficiency,
        "decision_status": triage.decision_status,
        "review_required": triage.review_required,
        "supporting_evidence": list(triage.supporting_evidence),
        "counterarguments": list(triage.counterarguments),
    }


def _triage_input_has_sensitive_materiality_signal(
    triage_input: dict[str, Any],
) -> bool:
    """Détecte un domaine sensible même si le premier juge omet son thème."""
    dossier = triage_input.get("related_change_dossier") or {}
    candidate_codes = {
        str(
            candidate.get("code")
            if isinstance(candidate, dict)
            else candidate
        ).strip().upper()
        for candidate in (triage_input.get("candidate_themes") or [])
    }
    if candidate_codes & _SENSITIVE_MATERIALITY_THEMES:
        return True
    corpus = " ".join(
        (
            str(triage_input.get("source_snippet_t1") or ""),
            str(triage_input.get("source_snippet_t2") or ""),
            str(
                triage_input.get("change_summary_factuel_non_arbitre")
                or ""
            ),
            _json_dumps(dossier) if dossier else "",
            _json_dumps(
                triage_input.get("full_evidence_exact_packet") or {}
            ),
            _json_dumps(
                triage_input.get("full_evidence_observations") or []
            ),
        )
    )
    return bool(_SENSITIVE_MATERIALITY_TEXT_RE.search(corpus))


def _requires_blind_materiality_challenge(
    triage: TriageAMFCompactLLMResultWithIndex,
    *,
    triage_input: dict[str, Any] | None = None,
) -> bool:
    """Cible les MINEUR sensibles et toutes les décisions directes incertaines."""
    if triage.materiality_level is None:
        # Ancien payload/test sans décision directe : conserver la compatibilité.
        return False
    if (
        triage.review_required
        or triage.decision_status != "CONFIRME"
        or triage.materiality_confidence in {"FAIBLE", "INDETERMINE"}
        or triage.evidence_sufficiency in {"INSUFFISANTE", "INDETERMINE"}
    ):
        return True
    sensitive_minor = (
        triage.materiality_level == "MINEUR"
        and bool(set(triage.themes_amf) & _SENSITIVE_MATERIALITY_THEMES)
    )
    missed_sensitive_relevance = (
        not triage.is_relevant
        and triage_input is not None
        and _triage_input_has_sensitive_materiality_signal(triage_input)
    )
    return sensitive_minor or missed_sensitive_relevance


def _blind_materiality_challenge(
    *,
    client: Any,
    model: str,
    bank_subject: str,
    section_key: str,
    triage_input: dict[str, Any],
) -> TriageAMFCompactLLMResultWithIndex:
    """Produit une seconde lecture sans exposer la décision primaire."""
    challenge_input = {
        key: value
        for key, value in triage_input.items()
        if key
        in {
            "bank_subject",
            "change_index",
            "diff_type",
            "source_snippet_t1",
            "source_snippet_t2",
            "exact_change_segments",
            "change_summary_factuel_non_arbitre",
            "full_evidence_observations",
            "full_evidence_exact_packet",
            "advisory_signals",
            "related_change_dossier",
            "validated_analyst_precedents",
            "candidate_themes",
        }
    }
    challenge_input["change_index"] = 1
    batch = _call_structured_completion_with_correction(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es le contradicteur indépendant d'un triage prudentiel. "
                    "Tu ne vois pas la décision primaire. Évalue les preuves avant/après "
                    "et cherche activement une modification de définition, périmètre, "
                    "autorité, responsabilité, méthode, contrôle, obligation, statut "
                    "réglementaire, capital, risque ou transparence qui pourrait être "
                    "sous-classée. Conteste aussi un surclassement lorsque l'équivalence "
                    "est démontrée. Réponds uniquement avec le schéma compact complet, "
                    f"en français, et commence changement_constate par {bank_subject}. "
                    "MINEUR exige une équivalence métier positivement démontrée; sinon "
                    "choisis MODERE ou MAJEUR selon la preuve et exige une revue si "
                    "l'incertitude persiste."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rends exactement une entrée avec change_index=1. Évalue directement "
                    "materiality_level indépendamment de nouvelle_idee. Renseigne tous "
                    "les champs de pertinence, les trois champs analystes, change_nature, "
                    "business_equivalence, materiality_confidence, evidence_sufficiency, "
                    "decision_status, review_required, supporting_evidence et "
                    "counterarguments. Les précédents sont comparatifs et ne remplacent "
                    "jamais les preuves courantes.\n\n"
                    f"{_FEW_SHOT_MATERIALITY_AMF}\n\n"
                    f"Banque analysée : {bank_subject}\n"
                    f"Section : {section_key}\n"
                    f"Dossier aveugle :\n{_json_dumps(challenge_input)}"
                ),
            },
        ],
        response_format=TriageAMFMaterialityLLMBatch,
        max_tokens=1600,
        max_retries=2,
        validation_retry_message=(
            "Renvoie exactement une entrée complète avec change_index=1. "
            "MINEUR sans équivalence CONFIRMEE exige review_required=true; "
            "une décision CONFIRME exige une preuve SUFFISANTE."
        ),
    )
    if len(batch.triages) != 1 or batch.triages[0].change_index != 1:
        raise ValueError(
            "Le contradicteur doit retourner exactement change_index=1."
        )
    challenge = batch.triages[0]
    challenge.themes_amf = _normalize_themes_amf(list(challenge.themes_amf))
    return challenge


def _resolve_materiality_challenge(
    *,
    primary: TriageAMFCompactLLMResultWithIndex,
    challenger: TriageAMFCompactLLMResultWithIndex,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Réconcilie deux lectures sans masquer leur désaccord."""
    primary_level = str(primary.materiality_level or "MINEUR")
    challenger_level = str(challenger.materiality_level or "MINEUR")
    primary_rank = _MATERIALITY_RANK.get(primary_level, -1)
    challenger_rank = _MATERIALITY_RANK.get(challenger_level, -1)

    select_challenger = (
        challenger_rank > primary_rank
        or (challenger.is_relevant and not primary.is_relevant)
    )
    selected = challenger if select_challenger else primary
    resolved = selected.model_dump(exclude={"change_index"})
    disagreement = (
        primary.is_relevant != challenger.is_relevant
        or primary_level != challenger_level
        or set(primary.themes_amf) != set(challenger.themes_amf)
        or primary.business_equivalence != challenger.business_equivalence
    )
    if disagreement:
        resolved["decision_status"] = "A_CONFIRMER"
        resolved["review_required"] = True

    audit = {
        "blind": True,
        "primary": _compact_materiality_snapshot(primary),
        "challenger": _compact_materiality_snapshot(challenger),
        "disagreement": disagreement,
        "resolution": (
            "challenger_higher_materiality"
            if select_challenger
            else "primary_retained"
        ),
        "resolved_materiality_level": resolved.get("materiality_level"),
        "review_required": bool(resolved.get("review_required")),
    }
    return resolved, audit


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


def _legacy_impact_decision(
    *,
    is_relevant: bool,
    nouvelle_idee: bool,
    themes: list[str],
    change_corpus: str,
) -> tuple[str, str]:
    """Reproduit l'ancienne grille pour audit et comparaison en mode parallèle."""
    substantive_process_change = (
        nouvelle_idee
        and "CONTROLE_CONFORMITE" in themes
        and bool(_PROCESS_SIGNAL_RE.search(change_corpus))
    )
    high_priority = bool(set(themes) & _COMPACT_HIGH_PRIORITY_THEMES) or (
        substantive_process_change
    )
    if not is_relevant:
        return "MINEUR", "aucune"
    if nouvelle_idee and high_priority:
        return "MAJEUR", "revue_prioritaire"
    if nouvelle_idee:
        return "MODERE", "investigation"
    return "MINEUR", "information"


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
        motif_non_pertinence=str(compact.get("motif_non_pertinence") or ""),
    )
    relevance_reason = analyst_copy["relevance_reason"]

    # Broad deterministic matches are evidence for the semantic judge, never a
    # post-LLM veto. This prevents one acquisition/date token from erasing a
    # simultaneous change to risk, control, methodology or regulatory status.
    advisory_signals = _triage_advisory_signals(change)
    bank_exclusion = next(
        (
            signal
            for signal in advisory_signals
            if signal
            in {
                "operation_interne_banque",
                "variation_numerique_propre_banque",
                "mise_a_jour_calendrier",
            }
        ),
        None,
    )

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
    legacy_impact_level, legacy_action_requise = _legacy_impact_decision(
        is_relevant=is_relevant,
        nouvelle_idee=nouvelle_idee,
        themes=themes,
        change_corpus=change_corpus,
    )

    if not is_relevant:
        impact_level = "MINEUR"
        action_requise = "aucune"
    else:
        proposed_materiality = str(
            compact.get("materiality_level") or ""
        ).strip().upper()
        if proposed_materiality in _MATERIALITY_RANK:
            impact_level = proposed_materiality
            action_requise = {
                "MAJEUR": "revue_prioritaire",
                "MODERE": "investigation",
                "MINEUR": "information",
            }[impact_level]
        else:
            # Backward-compatible path for cached responses and legacy tests.
            impact_level = legacy_impact_level
            action_requise = legacy_action_requise

    direct_materiality = str(compact.get("materiality_level") or "").strip().upper()
    decision_basis = (
        "direct_materiality"
        if direct_materiality in _MATERIALITY_RANK
        else "legacy_fallback"
    )

    triage: dict[str, Any] = {
        "compact_schema_version": "analyst_materiality_v4",
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
        "materiality_level": (
            impact_level if decision_basis == "direct_materiality" else None
        ),
        "change_nature": list(compact.get("change_nature") or []),
        "business_equivalence": str(
            compact.get("business_equivalence") or "INDETERMINE"
        ),
        "materiality_confidence": str(
            compact.get("materiality_confidence") or "INDETERMINE"
        ),
        "evidence_sufficiency": str(
            compact.get("evidence_sufficiency") or "INDETERMINE"
        ),
        "decision_status": str(
            compact.get("decision_status") or "PROVISOIRE"
        ),
        "review_required": bool(compact.get("review_required", False)),
        "supporting_evidence": list(compact.get("supporting_evidence") or []),
        "counterarguments": list(compact.get("counterarguments") or []),
        "advisory_signals": advisory_signals,
        "legacy_impact_level": legacy_impact_level,
        "legacy_action_requise": legacy_action_requise,
        "materiality_decision_basis": decision_basis,
    }
    if not is_relevant and bank_exclusion:
        triage["exclusion_reason"] = bank_exclusion
    triage["nouvelle_idee_justification"] = build_compact_triage_justification(
        change,
        triage,
    )
    legacy_fields = _derive_legacy_fields(triage)
    return {**triage, **legacy_fields, "source": TRIAGE_SOURCE_VERSION}


def _triage_section_changes(
    *,
    client: Any,
    model: str,
    section_key: str,
    changes: list[dict[str, Any]],
    bank_code: str = "",
    precedent_memory: PrecedentMemory | None = None,
) -> list[dict[str, Any]]:
    """Qualifie metier les changements detectes et fusionne le triage.

    Le triage ne recalcule pas la diff textuelle: il prend les changements deja
    identifies, demande une qualification selective au modele, puis rattache le
    resultat a chaque changement pour la retention finale et le resume global.

    Aligne sur la taxonomie AMF appliquee au suivi prudentiel canadien. Le modèle produit
    le schéma AMF v2 (themes_amf multi-label, exclusion_reason, ...) ; les
    champs hérités (category, signals, ...) sont dérivés localement pour
    préserver la compatibilité aval.
    """
    if not changes:
        return []
    effective_bank_code = (
        str(bank_code or "").strip().lower()
        or str(changes[0].get("bank_code") or "").strip().lower()
    )
    bank_subject = analyst_bank_subject(effective_bank_code)

    # The first GPT call arbitrates the semantic relationship.  Only an
    # explicit ``uncertain`` result remains for a human; same and distinct
    # disclosures proceed to the AMF triage normally.
    if any(_requires_alignment_review(change) or _is_semantic_text_move(change) for change in changes):
        enriched: list[dict[str, Any]] = []
        for change in changes:
            if _requires_alignment_review(change):
                enriched.append(
                    _alignment_review_result(
                        change,
                        bank_code=effective_bank_code,
                    )
                )
            elif _is_semantic_text_move(change):
                enriched.append(
                    _semantic_move_result(
                        change,
                        bank_code=effective_bank_code,
                    )
                )
            else:
                enriched.extend(
                    _triage_section_changes(
                        client=client,
                        model=model,
                        section_key=section_key,
                        changes=[change],
                        bank_code=effective_bank_code,
                        precedent_memory=precedent_memory,
                    )
                )
        return enriched

    # Hard pre-filter only for mechanically demonstrated noise. Calendar,
    # acquisition, high-similarity and other broad signals still reach the
    # materiality judge and are supplied as advisory evidence.
    pending: list[dict[str, Any]] = []
    prefiltered: list[dict[str, Any]] = []
    for change in changes:
        exclusion = _hard_prefilter_exclusion(change)
        if exclusion:
            prefiltered.append(
                _prefilter_triage_result(
                    change,
                    exclusion,
                    bank_code=effective_bank_code,
                )
            )
        else:
            pending.append(change)
    if not pending:
        return prefiltered

    # Semantic near-duplicate grouping now builds a shared evidence dossier.
    # Every member remains independently classified; no representative verdict
    # is copied to the other changes.
    groups = _group_semantic_triage_duplicates(pending, client=client)
    pending = _annotate_triage_dossiers(
        pending,
        section_key=section_key,
        groups=groups,
    )

    if len(pending) > _TRIAGE_BATCH_SIZE and not _is_single_semantic_alignment_group(pending):
        chunks = [
            pending[start : start + _TRIAGE_BATCH_SIZE]
            for start in range(0, len(pending), _TRIAGE_BATCH_SIZE)
        ]
        max_workers = min(_MAX_TRIAGE_LLM_WORKERS, len(chunks))
        results_by_index: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    _triage_section_changes,
                    client=client,
                    model=model,
                    section_key=section_key,
                    changes=chunk,
                    bank_code=effective_bank_code,
                    precedent_memory=precedent_memory,
                ): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results_by_index[index] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Section triage failed for {section_key}/batch t{index:02d}: {exc}"
                    ) from exc

        enriched_batches: list[dict[str, Any]] = []
        for index in range(len(chunks)):
            enriched_batches.extend(results_by_index.get(index, []))
        return [
            *prefiltered,
            *_attach_consolidated_dossier_outcomes(enriched_batches),
        ]

    changes = pending
    triage_inputs = []
    exact_segments_by_index: dict[int, list[dict[str, str]]] = {}
    full_evidence_by_index: dict[int, list[dict[str, Any]]] = {}
    full_evidence_packets_by_index: dict[int, list[dict[str, Any]]] = {}
    direct_full_evidence_by_index: dict[int, dict[str, str]] = {}
    full_evidence_mode_by_index: dict[int, str] = {}
    full_evidence_failures_by_index: dict[int, str] = {}
    precedent_packets_by_index: dict[int, dict[str, Any]] = {}
    for idx, change in enumerate(changes, start=1):
        exact_segments = build_change_segments(change)
        exact_segments_by_index[idx] = exact_segments
        full_evidence = []
        if _requires_full_evidence_packets(change):
            packets = _build_full_evidence_packets(change)
            full_evidence_packets_by_index[idx] = packets
            if len(packets) == 1:
                # Le texte exact tient déjà dans le prompt principal : une
                # prélecture LLM le résumerait inutilement avant le triage.
                direct_full_evidence_by_index[idx] = {
                    "text_t1": str(packets[0].get("text_t1") or ""),
                    "text_t2": str(packets[0].get("text_t2") or ""),
                }
                full_evidence_mode_by_index[idx] = "direct_exact_packet"
            else:
                full_evidence_mode_by_index[idx] = "packet_observations"
                try:
                    full_evidence = _collect_full_evidence_observations(
                        client=client,
                        model=model,
                        change=change,
                        bank_code=effective_bank_code,
                        section_key=section_key,
                        change_index=idx,
                    )
                except Exception as exc:
                    failure_reason = str(exc)
                    full_evidence_failures_by_index[idx] = failure_reason
                    logger.error(
                        "full evidence read requires review section=%s "
                        "change_index=%d error=%s",
                        section_key,
                        idx,
                        failure_reason,
                    )
                else:
                    full_evidence_by_index[idx] = full_evidence
        exact_segments_for_prompt = [
            {
                "kind": str(segment.get("kind") or ""),
                "text_t1": _truncate_prompt_text(
                    str(segment.get("text_t1") or ""),
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
                "text_t2": _truncate_prompt_text(
                    str(segment.get("text_t2") or ""),
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
            }
            for segment in exact_segments
        ]
        candidate_themes = _candidate_themes_for_change(
            change,
            section_key=section_key,
        )
        precedent_packet: dict[str, Any] = {}
        if precedent_memory is not None and precedent_memory.precedents:
            query = PrecedentQuery(
                text_before=str(
                    change.get("source_text_t1")
                    or change.get("semantic_text_t1")
                    or ""
                ),
                text_after=str(
                    change.get("source_text_t2")
                    or change.get("semantic_text_t2")
                    or ""
                ),
                bank_code=effective_bank_code,
                section_key=section_key,
                change_nature=str(change.get("diff_type") or "").upper(),
                themes_amf=tuple(
                    str(candidate.get("code") or "")
                    for candidate in candidate_themes
                    if str(candidate.get("code") or "")
                ),
            )
            precedent_packet = precedent_memory.build_packet(query).to_dict()
            precedent_packets_by_index[idx] = precedent_packet

        triage_input = {
            "bank_subject": bank_subject,
            "change_index": idx,
            "diff_type": change["diff_type"],
            "source_snippet_t1": _truncate_prompt_text(
                change.get("source_text_t1")
                or change.get("semantic_text_t1")
                or "",
                _TRIAGE_SOURCE_SNIPPET_LIMIT,
            ),
            "source_snippet_t2": _truncate_prompt_text(
                change.get("source_text_t2")
                or change.get("semantic_text_t2")
                or "",
                _TRIAGE_SOURCE_SNIPPET_LIMIT,
            ),
            "exact_change_segments": exact_segments_for_prompt,
            "alignment_decision": str(change.get("alignment_decision") or ""),
            "alignment_confidence": str(change.get("alignment_confidence") or ""),
            "change_summary_factuel_non_arbitre": change.get("change_summary", ""),
            "full_evidence_observations": full_evidence,
            "advisory_signals": _triage_advisory_signals(change),
            "related_change_dossier": change.get("triage_dossier") or {},
            "validated_analyst_precedents": precedent_packet,
            "candidate_themes": candidate_themes,
        }
        direct_full_evidence = direct_full_evidence_by_index.get(idx)
        if direct_full_evidence is not None:
            triage_input["full_evidence_exact_packet"] = direct_full_evidence
        triage_inputs.append(triage_input)

    system_prompt = (
        "Tu qualifies des changements de divulgation d’une banque canadienne "
        "pour une vigie AMF. Réponds uniquement avec le schéma compact demandé. "
        f"La banque analysée est {bank_subject} et le champ "
        f"`changement_constate` doit commencer exactement par « {bank_subject} » "
        "suivi d’un verbe d’action direct, par exemple ajoute, retire, modifie, "
        "précise, transfère ou renomme. N’utilise jamais « le rapport courant », "
        "« le rapport précédent », « le passage », T1 ou T2 comme sujet du texte "
        "analyste. Établis directement la matérialité métier, mais sans analyse IT, "
        "posture, action recommandée ni répétition des textes sources. Rédige "
        "séparément, en "
        "français, des phrases complètes, professionnelles et faciles à comprendre "
        "dans `changement_constate`, `signification_metier` et "
        "`motif_non_pertinence`. Ne produis pas `relevance_reason`; il sera "
        "assemblé localement. La longueur du changement ne détermine jamais sa "
        "pertinence ou sa matérialité : une modification très courte peut être "
        "substantielle si elle change une définition, un périmètre, une autorité, "
        "une responsabilité, une méthode, un contrôle, une obligation, un statut "
        "réglementaire, le capital ou la transparence."
    )


    user_prompt = (
        f"Retourne exactement {len(changes)} entrée(s) dans `triages`, une par "
        "changement, avec les mêmes `change_index`, sans doublon ni entrée "
        "supplémentaire.\n\n"
        "Règles strictes :\n"
        "1. `is_relevant=true` seulement pour un changement substantiel utile "
        "à la vigie AMF; dans ce cas, choisis un ou deux codes. Préfère les "
        "`candidate_themes` de l’entrée ; tu peux aussi utiliser tout code de "
        "la taxonomie AMF complète listée ci-dessous ; si aucun ne convient, "
        "utilise `SUJET_EMERGENT_HORS_GRILLE`.\n"
        f"   Taxonomie AMF autorisée : {', '.join(THEMES_AMF_PIPELINE_2)}.\n"
        "2. `is_relevant=false` exige `themes_amf=[]` et `nouvelle_idee=false`. "
        "Les `advisory_signals` sont uniquement des indices mécaniques : ils ne "
        "constituent jamais un verdict. Une acquisition, une date, un chiffre ou "
        "une forte similarité ne rend pas tout le changement non pertinent si le "
        "même dossier modifie aussi un risque, un périmètre, une méthode, un "
        "contrôle, une obligation ou un statut de mise en œuvre. Un déplacement "
        "ou une reformulation n’est non pertinent que si l’équivalence complète "
        "est positivement démontrée.\n"
        f"3. `nouvelle_idee=true` seulement si {bank_subject} ajoute, retire "
        "ou modifie substantiellement une information qui n’était pas divulguée "
        "auparavant sous cette forme. Pour la gouvernance, considère comme substantiel "
        "tout changement démontré d’autorité décisionnelle, de mandat ou de rôle "
        "d’un comité, de ligne de défense, de responsabilité, de supervision, de reddition de "
        "comptes, de culture de risque, de rémunération liée au risque ou d’appétit "
        "pour le risque. Une phrase courte peut donc être une nouvelle idée; un "
        "simple renommage sans effet sur le mandat ne l’est pas.\n"
        "   Une modification réelle de méthodologie ou de processus est toujours "
        "substantielle et prioritaire : utilise `MODIFICATION_METHODOLOGIE` pour "
        "la méthode ou l’approche, et `CONTROLE_CONFORMITE` pour un processus de "
        "contrôle ou de conformité, avec `nouvelle_idee=true`. Une reformulation "
        "qui ne change ni le fonctionnement, ni les étapes, ni les acteurs, ni les "
        "contrôles demeure non substantielle. `nouvelle_idee` est un attribut "
        "descriptif indépendant : sa valeur ne commande jamais le niveau de "
        "matérialité. Un changement peut donc être MAJEUR avec "
        "`nouvelle_idee=false`.\n"
        "4. Chaque champ renseigné doit être non vide, lexical et terminé par "
        "« . », « ! » ou « ? ». Si `is_relevant=true`, renseigne "
        "`changement_constate` et `signification_metier`, puis laisse "
        "`motif_non_pertinence` vide. `changement_constate` décrit factuellement "
        f"l’action de {bank_subject}; `signification_metier` explique sa "
        "signification concrète. Si `is_relevant=false`, "
        "renseigne seulement `changement_constate` et `motif_non_pertinence`, puis "
        "laisse `signification_metier` vide. N’écris pas "
        "« Ce changement est pertinent pour la vigie AMF », « Ce changement "
        "n’est pas pertinent », « Pour la vigie », « Cette information est "
        "importante », « Il convient de noter que » ni « Dans le cadre de cette "
        "analyse ». "
        "Aucun titre, aucune liste, aucune rubrique et aucune consigne adressée "
        "à l’analyste. Interdit : fragment, chunk, T1, T2, termes anglais.\n"
        "5. Évalue directement `materiality_level`, indépendamment du thème et de "
        "`nouvelle_idee`. MINEUR exige une équivalence métier, un bruit ou un "
        "déplacement complet positivement démontré. MODERE s’applique lorsqu’un "
        "effet métier est plausible, lorsque l’équivalence n’est pas démontrée "
        "dans un domaine sensible ou lorsque la preuve reste partielle. MAJEUR "
        "exige une modification substantielle démontrée de définition, périmètre, "
        "gouvernance, responsabilité, méthode, contrôle, obligation réglementaire, "
        "capital, liquidité, risque ou transparence. Renseigne une à trois valeurs "
        "dans `change_nature`, puis `business_equivalence`, "
        "`materiality_confidence`, `evidence_sufficiency`, `decision_status`, "
        "`review_required`, `supporting_evidence` et `counterarguments`. Un MINEUR "
        "sans équivalence CONFIRMEE, une preuve insuffisante ou une confiance "
        "FAIBLE exige `decision_status=A_CONFIRMER` et `review_required=true`. "
        "Ne produis aucun champ d’action, de posture, d’impact IT, d’explication "
        "générale, de justification multi-rubriques ou `relevance_reason`.\n"
        "6. Si `full_evidence_exact_packet` est présent, il contient les textes "
        "T1/T2 complets et non tronqués pour ce changement. Fonde la qualification "
        "sur cette preuve exacte plutôt que sur les `source_snippet_*`. "
        "`full_evidence_observations` reste réservé aux preuves réellement "
        "découpées en plusieurs paquets.\n"
        "7. `related_change_dossier` fournit les changements reliés au même "
        "concept ou à la même sous-section. Classe chaque entrée individuellement, "
        "mais tiens compte de leur effet cumulatif. `alignment_decision="
        "same_disclosure` signifie seulement que les passages traitent du même "
        "sujet; ce n’est jamais une preuve d’équivalence métier. Le résumé factuel "
        "initial n’est pas un verdict et ne doit pas t’ancrer.\n"
        "8. `validated_analyst_precedents`, lorsqu’il est non vide, contient "
        "uniquement des décisions analystes validées et des cas contrastifs. "
        "Compare les ressemblances et les différences décisives, sans copier "
        "automatiquement leur niveau. Les preuves exactes du cas courant restent "
        "toujours prioritaires.\n\n"
        f"Adapte les exemples à la banque analysée : remplace toujours leur sujet "
        f"par {bank_subject} dans la réponse réelle.\n\n"
        "Les exemples numérotés 1 à 9 illustrent les champs de pertinence et de "
        "rédaction analyste; ils précèdent le schéma de matérialité directe. Dans "
        "la réponse réelle, ajoute toujours tous les champs de matérialité montrés "
        "dans les exemples A à D.\n\n"
        f"{_FEW_SHOT_TRIAGE_AMF}\n\n"
        f"{_FEW_SHOT_MATERIALITY_AMF}\n\n"
        f"Banque analysée : {bank_subject}\n"
        f"Section : {section_key}\n"
        f"Changements :\n{_json_dumps(triage_inputs)}"
    )
    compact_max_tokens = min(
        _COMPACT_COMPLETION_MAX_TOKENS,
        _COMPACT_COMPLETION_BASE_TOKENS
        + _COMPACT_COMPLETION_TOKENS_PER_CHANGE * len(changes),
    )

    try:
        batch = _call_structured_completion_with_correction(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=TriageAMFMaterialityLLMBatch,
            max_tokens=compact_max_tokens,
            max_retries=2,
            validation_retry_message=(
                "Renvoie le batch compact complet. Chaque change_index doit être "
                "présent exactement une fois. is_relevant=true exige un ou deux "
                "thèmes AMF (préfère candidate_themes, sinon tout code de la "
                "taxonomie AMF, sinon SUJET_EMERGENT_HORS_GRILLE); "
                "is_relevant=false exige themes_amf=[] et "
                "nouvelle_idee=false. Corrige uniquement les trois champs "
                "sémantiques : is_relevant=true exige changement_constate, "
                "signification_metier non vides, avec motif_non_pertinence vide; "
                "is_relevant=false exige changement_constate et "
                "motif_non_pertinence non vides, avec signification_metier vide. "
                f"Chaque changement_constate commence par {bank_subject} "
                "et chaque champ renseigné est lexical et ponctué. Renseigne aussi "
                "la décision directe materiality_level, une à trois valeurs "
                "change_nature, business_equivalence, materiality_confidence, "
                "evidence_sufficiency, decision_status, review_required, au moins "
                "une supporting_evidence lorsque la preuve n'est pas insuffisante, "
                "et counterarguments. Un MINEUR sans équivalence confirmée exige "
                "review_required=true. Ne produis pas relevance_reason."
            ),
            length_retry_message=(
                "Renvoie immédiatement le même batch compact complet, sans aucun "
                "commentaire hors schéma. Raccourcis séparément les champs "
                "changement_constate, signification_metier et "
                "motif_non_pertinence sans les fusionner. Respecte les champs "
                f"vides applicables et commence changement_constate par "
                f"{bank_subject}. Conserve tous les champs de matérialité directe "
                "et leurs preuves. Ne produis pas relevance_reason."
            ),
        )
    except ValidationError as exc:
        raise TriageValidationError(
            section_key=section_key,
            change_index=_change_index_from_validation_error(exc),
            raw_payload=None,
            validation_error=exc,
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Section triage failed for {section_key}: {exc}") from exc

    expected_indexes = list(range(1, len(changes) + 1))
    received_indexes = [triage.change_index for triage in batch.triages]
    if len(received_indexes) != len(expected_indexes) or sorted(received_indexes) != expected_indexes:
        validation_error = ValueError(
            "Le batch compact doit contenir exactement les change_index "
            f"{expected_indexes}; reçu {received_indexes}"
        )
        raise TriageValidationError(
            section_key=section_key,
            change_index=None,
            raw_payload=batch.model_dump(),
            validation_error=validation_error,
        )

    for triage_obj in batch.triages:
        candidate_codes = {
            candidate["code"]
            for candidate in triage_inputs[triage_obj.change_index - 1][
                "candidate_themes"
            ]
        }
        raw_themes = list(triage_obj.themes_amf or [])
        outside_candidates = [
            code
            for code in raw_themes
            if str(code or "").strip().upper() not in candidate_codes
            and str(code or "").strip().upper() in THEMES_AMF_DESCRIPTIONS
        ]
        if outside_candidates:
            logger.debug(
                "theme_accepted_outside_candidates section=%s change_index=%d themes=%s",
                section_key,
                triage_obj.change_index,
                outside_candidates,
            )
        # Soft-normalize: full AMF taxonomy accepted; unknown -> hors grille.
        triage_obj.themes_amf = _normalize_themes_amf(raw_themes)

    triage_map: dict[int, dict[str, Any]] = {}
    relevant_count = 0
    nouvelle_idee_count = 0
    for triage_obj in batch.triages:
        change = changes[triage_obj.change_index - 1]
        compact_dict = triage_obj.model_dump(exclude={"change_index"})
        challenge_audit: dict[str, Any] | None = None
        current_triage_input = triage_inputs[triage_obj.change_index - 1]
        if _requires_blind_materiality_challenge(
            triage_obj,
            triage_input=current_triage_input,
        ):
            try:
                challenger = _blind_materiality_challenge(
                    client=client,
                    model=model,
                    bank_subject=bank_subject,
                    section_key=section_key,
                    triage_input=current_triage_input,
                )
            except Exception as exc:  # noqa: BLE001
                compact_dict["decision_status"] = "A_CONFIRMER"
                compact_dict["review_required"] = True
                challenge_audit = {
                    "blind": True,
                    "primary": _compact_materiality_snapshot(triage_obj),
                    "challenger_error": f"{type(exc).__name__}: {exc}",
                    "disagreement": None,
                    "resolution": "analyst_review_required",
                    "resolved_materiality_level": compact_dict.get(
                        "materiality_level"
                    ),
                    "review_required": True,
                }
                logger.warning(
                    "blind materiality challenge unavailable section=%s "
                    "change_index=%d error=%s",
                    section_key,
                    triage_obj.change_index,
                    exc,
                )
            else:
                compact_dict, challenge_audit = _resolve_materiality_challenge(
                    primary=triage_obj,
                    challenger=challenger,
                )
        triage = _persisted_triage_from_compact(
            compact_dict,
            change=change,
            bank_code=effective_bank_code,
        )
        if challenge_audit is not None:
            triage["materiality_challenge"] = challenge_audit
        precedent_packet = precedent_packets_by_index.get(
            triage_obj.change_index
        )
        if precedent_packet:
            triage["analyst_precedent_packet"] = precedent_packet
        triage["change_segments"] = (
            exact_segments_by_index.get(triage_obj.change_index, [])
            if triage.get("is_relevant")
            else []
        )
        triage_map[triage_obj.change_index] = triage
        if triage.get("is_relevant"):
            relevant_count += 1
        if triage.get("nouvelle_idee"):
            nouvelle_idee_count += 1
        logger.info(
            "compact triage validated section=%s change_index=%d is_relevant=%s themes=%s nouvelle_idee=%s semantic_fields=%s",
            section_key,
            triage_obj.change_index,
            triage_obj.is_relevant,
            triage_obj.themes_amf,
            triage_obj.nouvelle_idee,
            [
                field_name
                for field_name in _SEMANTIC_REASON_FIELDS
                if getattr(triage_obj, field_name)
            ],
        )

    logger.info(
        "triage section summary section=%s total=%d relevant=%d nouvelles_idees=%d",
        section_key,
        len(batch.triages),
        relevant_count,
        nouvelle_idee_count,
    )

    enriched: list[dict[str, Any]] = []
    for idx, change in enumerate(changes, start=1):
        evidence_failure_reason = full_evidence_failures_by_index.get(idx)
        if evidence_failure_reason:
            enriched.append(
                _evidence_read_review_triage(
                    change,
                    evidence_failure_reason,
                    bank_code=effective_bank_code,
                )
            )
            continue
        triage = triage_map.get(idx, _default_triage(effective_bank_code))
        evidence_observations = full_evidence_by_index.get(idx, [])
        evidence_packets = full_evidence_packets_by_index.get(idx, [])
        if evidence_packets:
            coherent, coherence_reason = _verify_triage_coherence(
                client=client,
                model=model,
                change=change,
                triage=triage,
                evidence_packets=evidence_packets,
            )
            if not coherent:
                enriched.append(
                    _coherence_review_triage(
                        change,
                        coherence_reason,
                        bank_code=effective_bank_code,
                    )
                )
                continue
            triage["full_evidence_verified"] = True
            triage["full_evidence_mode"] = full_evidence_mode_by_index[idx]
            triage["full_evidence_observations"] = evidence_observations
        enriched_change = dict(change)
        enriched_change["genai_triage"] = triage
        enriched.append(enriched_change)
    return [
        *prefiltered,
        *_attach_consolidated_dossier_outcomes(enriched),
    ]
