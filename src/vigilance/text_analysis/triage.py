"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import logging
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from vigilance.amf_taxonomy import (
    COMPACT_RELEVANCE_REASON_SENTENCE_COUNT,
    THEMES_AMF_ANALYST_SUBJECTS,
    THEMES_AMF_DESCRIPTIONS,
    THEMES_AMF_PIPELINE_2,
    TRIAGE_SOURCE_VERSION,
    TriageAMFCompactLLMBatch,
    TriageValidationError,
    count_complete_sentences,
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
_COMPACT_THEME_CANDIDATE_LIMIT = 6
_COMPACT_COMPLETION_BASE_TOKENS = 350
_COMPACT_COMPLETION_TOKENS_PER_CHANGE = 320
_COMPACT_COMPLETION_MAX_TOKENS = 1200
_FULL_EVIDENCE_PACKET_LIMIT = 2400
_FULL_EVIDENCE_FACT_MAX_TOKENS = 300
_FULL_EVIDENCE_VERIFICATION_MAX_TOKENS = 220
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
    """Constat factuel sur un paquet complet de preuves T1/T2."""

    packet_index: int = Field(..., ge=1)
    factual_change: str = Field(..., min_length=12, max_length=700)


class _EvidencePacketBatch(BaseModel):
    """Réponse de l'étape factuelle, sans qualification AMF."""

    observations: list[_EvidencePacketObservation]


class _EvidencePacketCoherenceCheck(BaseModel):
    """Contrôle indépendant d'une décision sur une preuve exacte."""

    packet_index: int = Field(..., ge=1)
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
) -> list[dict[str, Any]]:
    """Lire toute preuve longue par appels factuels séparés et auditables."""
    packets = _build_full_evidence_packets(change)
    observations: list[dict[str, Any]] = []
    for packet in packets:
        response = _call_structured_completion_with_correction(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu lis une preuve textuelle complète T1/T2. Décris uniquement "
                        "le fait observable entre les deux textes, sans catégorie AMF, "
                        "sans priorité, sans posture et sans recommandation."
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
            response_format=_EvidencePacketBatch,
            max_tokens=_FULL_EVIDENCE_FACT_MAX_TOKENS,
            max_retries=1,
            validation_retry_message=(
                "Renvoie une seule observation factuelle pour le packet_index fourni, "
                "sans qualification métier ni texte hors schéma."
            ),
        )
        matching = [item for item in response.observations if item.packet_index == packet["packet_index"]]
        if len(matching) != 1:
            raise RuntimeError(
                "La lecture de preuve complète doit retourner exactement une observation "
                f"pour le paquet {packet['packet_index']}."
            )
        observations.append(
            {
                "packet_index": packet["packet_index"],
                "factual_change": matching[0].factual_change,
            }
        )
    return observations


def _local_relevance_reason(
    factual_change: str,
    comparative_interpretation: str,
) -> str:
    """Compose les deux phrases des qualifications locales sans remplissage."""
    sentences = [
        " ".join(str(sentence or "").split())
        for sentence in (factual_change, comparative_interpretation)
    ]
    if any(not sentence.endswith((".", "!", "?")) for sentence in sentences):
        raise ValueError("Chaque phrase locale doit être complète et ponctuée.")
    result = " ".join(sentences)
    if count_complete_sentences(result) != COMPACT_RELEVANCE_REASON_SENTENCE_COUNT:
        raise ValueError(
            "Une justification locale doit contenir exactement deux phrases complètes."
        )
    return result


def _default_triage() -> dict[str, Any]:
    """Retourne un triage par défaut conservateur (non pertinent).

    Schéma cible AMF v2 (``themes_amf``, ``exclusion_reason``) **plus** les
    champs hérités (``category``, ``signals``, ``confidence``, ...) maintenus
    avec valeurs par défaut pour préserver la compatibilité avec les
    consommateurs aval (review_export, review_models_v2, review_queue_normalizer)
    non encore migrés.
    """
    triage = empty_triage_skeleton()
    triage["source"] = TRIAGE_SOURCE_VERSION
    triage.update(
        {
            "category": "NON_PERTINENT",
            "risk_type": "autre",
            "relevance_score": "FAIBLE",
            "risk_level": "FAIBLE",
            "impact_description": "",
            "reference_reglementaire": "",
            "confidence": 0.0,
            "relevance_reason": _local_relevance_reason(
                "Le changement n’a pas reçu de qualification AMF exploitable.",
                "Il est conservé dans la file de revue sans être présenté comme "
                "une nouvelle idée, afin d’éviter une conclusion automatique "
                "non étayée par les éléments disponibles.",
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
        return decision == "uncertain"
    # Cached artifacts from before semantic arbitration keep the former safe
    # fallback.  Fresh comparisons always carry ``alignment_decision``.
    return str(change.get("alignment_type") or "").strip().lower() == "ambiguous"


def _is_semantic_text_move(change: dict[str, Any]) -> bool:
    return str(change.get("alignment_decision") or "").strip().lower() == "moved_text"


def _is_single_semantic_alignment_group(changes: list[dict[str, Any]]) -> bool:
    """Keeps the added/removed sides of one GPT decision in one triage call."""
    group_ids = {
        str(change.get("semantic_alignment_group_id") or "").strip()
        for change in changes
    }
    return len(changes) >= 2 and len(group_ids) == 1 and bool(next(iter(group_ids), ""))


def _alignment_review_result(change: dict[str, Any]) -> dict[str, Any]:
    """Preserves the evidence while preventing an unsupported automatic verdict."""
    triage = _default_triage()
    triage.update(
        {
            "source": "alignment_review_required",
            "alignment_review_required": True,
            "alignment_review_reason": (
                str(change.get("alignment_rationale") or "").strip()
                or "L'alignement entre les deux passages reste ambigu après la comparaison initiale. "
                "Le changement est conservé pour revue, sans classification AMF automatique."
            ),
            "relevance_reason": _local_relevance_reason(
                "Les passages pourraient décrire des divulgations différentes, "
                "mais l’alignement sémantique ne fournit pas une preuve suffisante "
                "pour conclure automatiquement.",
                "Le changement reste donc visible et doit être lu avec ses extraits "
                "sources avant toute décision.",
            ),
            # The analyst still sees the deterministic, verbatim difference;
            # no LLM-generated highlight is used for this unresolved pairing.
            "change_segments": build_change_segments(change),
        }
    )
    enriched = dict(change)
    enriched["genai_triage"] = triage
    return enriched


def _semantic_move_result(change: dict[str, Any]) -> dict[str, Any]:
    """Marks a GPT-confirmed text move as non-priority without human escalation."""
    triage = _default_triage()
    triage.update(
        {
            "source": "semantic_alignment_decision",
            "alignment_decision": "moved_text",
            "alignment_confidence": str(change.get("alignment_confidence") or "medium"),
            "alignment_rationale": str(change.get("alignment_rationale") or "").strip(),
            "exclusion_reason": "deplacement_texte",
            "relevance_reason": _local_relevance_reason(
                "La comparaison confirme que la divulgation a été déplacée sans "
                "modification substantielle de son sens, de son niveau de détail "
                "ou de son rattachement métier.",
                "Ce déplacement ne crée donc pas une nouvelle idée à surveiller.",
            ),
            "nouvelle_idee_justification": (
                "NON — Nouvel élément à surveiller : Non.\n\n"
                "Sujet détecté : Texte déplacé.\n\n"
                "Ce qui change : Le premier appel GPT a confirmé que la même divulgation "
                "a été déplacée sans changement sémantique substantiel.\n\n"
                "Pertinence métier : Ce déplacement ne modifie pas la substance de la "
                "divulgation ni la posture de risque.\n\n"
                "Point de surveillance : Aucun suivi prioritaire n'est requis pour ce déplacement."
            ),
            "change_segments": [],
        }
    )
    enriched = dict(change)
    enriched["genai_triage"] = triage
    return enriched


def _coherence_review_triage(change: dict[str, Any], reason: str) -> dict[str, Any]:
    """Conserver un dossier visible quand le contrôle indépendant diverge."""
    triage = _default_triage()
    triage.update(
        {
            "source": "triage_coherence_review_required",
            "coherence_review_required": True,
            "coherence_review_reason": str(reason or "").strip(),
            "relevance_reason": _local_relevance_reason(
                "La qualification métier proposée ne concorde pas suffisamment avec "
                "la vérification indépendante des preuves complètes.",
                "Le dossier est "
                "conservé avec ses textes sources et ses pages afin qu'un analyste "
                "confirme le type de changement, la pertinence et l'existence d'une "
                "nouvelle idée avant toute conclusion de vigie.",
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
                "Réponds avec le packet_index, supports, contradicts ou insufficient, "
                "et une raison factuelle courte sans ajouter de thème."
            ),
        )
        if response.packet_index != packet["packet_index"]:
            raise RuntimeError(
                "Le contrôle de cohérence doit répondre pour le paquet de preuve demandé."
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
    """Exclut dates/montants/opérations internes sans fond réglementaire inter-pairs."""
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


def _excerpt_for_analyst(text: str, *, limit: int = 160) -> str:
    """Tronque un extrait source pour une phrase analysée lisible (une seule phrase)."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    # Avoid internal sentence breaks that would break the 2-sentence relevance_reason.
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
) -> tuple[str, str, str]:
    """Textes analyste naturels pour un changement exclu (sans jargon pipeline)."""
    diff_type = str(change.get("diff_type") or "").strip().lower()
    source_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    excerpt_t2 = _excerpt_for_analyst(source_t2)
    comparative = (
        "Ce changement n'apporte pas d'élément nouveau à comparer entre les "
        "banques pour la vigie prudentielle."
    )

    if exclusion_reason == "operation_interne_banque":
        subject = "Opération interne propre à la banque"
        if diff_type == "added" and excerpt_t2:
            factual = (
                "Le rapport courant introduit un passage sur "
                f"« {excerpt_t2} », absent du rapport précédent et lié à une "
                "opération propre à la banque (acquisition, rachat ou émission)."
            )
        elif diff_type == "added":
            factual = (
                "Le rapport courant introduit une information liée à une "
                "opération propre à la banque (acquisition, rachat ou émission), "
                "absente du rapport précédent."
            )
        else:
            factual = (
                "Le passage est modifié et mentionne une opération propre à la "
                "banque (acquisition, rachat ou émission)."
            )
        return factual, comparative, subject

    if exclusion_reason == "variation_numerique_propre_banque":
        subject = "Variation chiffrée propre à la banque"
        if diff_type == "added":
            factual = (
                "Le rapport courant introduit des montants ou pourcentages "
                "propres à la banque, sans nouvelle exigence réglementaire."
            )
        else:
            factual = (
                "Seuls les chiffres, montants ou pourcentages diffèrent entre "
                "les deux versions."
            )
        return factual, comparative, subject

    if exclusion_reason == "mise_a_jour_calendrier":
        subject = "Mise à jour de calendrier"
        factual = (
            "Le passage est modifié : seules les dates ou échéances "
            "d'application changent, sans nouvelle exigence."
        )
        return factual, comparative, subject

    # Cosmetic / generic exclusions
    subject = "Changement cosmétique"
    if exclusion_reason == "deplacement_texte":
        factual = (
            "Le même texte apparaît ailleurs dans le rapport courant, "
            "sans nouveau contenu à comparer."
        )
        subject = "Déplacement de texte"
    elif exclusion_reason == "formatage_visuel":
        factual = (
            "Le passage est reformulé sans changement de fond "
            "(ponctuation ou formatage uniquement)."
        )
    elif diff_type == "added":
        factual = (
            "Le rapport courant introduit une formulation différente "
            "sans changement de fond."
        )
    else:
        factual = "Le passage est reformulé sans changement de fond."
    return factual, comparative, subject


def _prefilter_triage_result(change: dict[str, Any], exclusion_reason: str) -> dict[str, Any]:
    triage = _default_triage()
    factual, comparative, subject = _analyst_exclusion_copy(change, exclusion_reason)
    triage.update(
        {
            "source": "deterministic_prefilter",
            "exclusion_reason": exclusion_reason,
            "relevance_reason": _local_relevance_reason(factual, comparative),
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


def _cosmetic_triage_result(change: dict[str, Any], exclusion_reason: str) -> dict[str, Any]:
    """Compatibilité : délégué au préfiltre généraliste."""
    return _prefilter_triage_result(change, exclusion_reason)


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


def _propagate_triage_to_group(
    *,
    representative: dict[str, Any],
    members: list[dict[str, Any]],
    group_id: str,
) -> list[dict[str, Any]]:
    triage = dict(representative.get("genai_triage") or _default_triage())
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
Input : {"change_index": 1, "diff_type": "added", "change_summary": "Ajout d’exercices annuels de simulation de cyberattaque."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["RISQUE_EMERGENT", "CONTROLE_CONFORMITE"], "nouvelle_idee": true, "relevance_reason": "Le rapport courant indique que la banque réalise désormais des simulations annuelles de cyberattaque avec ses unités d’affaires, une pratique qui n’était pas mentionnée dans le rapport précédent. Cette évolution témoigne d’un renforcement des mécanismes de résilience et permet de comparer plus précisément le degré de préparation opérationnelle déclaré par les banques."}

Exemple 2 — variation propre à la banque non pertinente
Input : {"change_index": 1, "diff_type": "modified", "change_summary": "Le portefeuille hypothécaire passe de 287 G$ à 294 G$."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "relevance_reason": "Le portefeuille hypothécaire passe de 287 G$ à 294 G$, sans modification de la méthode de calcul ni du périmètre présenté. Cette variation reflète l’évolution normale des activités et n’apporte aucun nouvel élément sur les pratiques de gestion des risques à comparer entre les banques."}

Exemple 3 — calendrier d’application non pertinent
Input : {"change_index": 1, "diff_type": "modified", "change_summary": "Le BSIF reporte l’augmentation du coefficient de plancher jusqu’à nouvel ordre plutôt que jusqu’en 2027."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "relevance_reason": "Le rapport courant actualise seulement le calendrier d’application du coefficient de plancher annoncé par le BSIF, sans changer la nature de l’exigence. Cette mise à jour d’échéances n’apporte pas d’élément nouveau pour comparer les pratiques de gestion des fonds propres entre les banques."}

Exemple 4 — acquisition interne non pertinente
Input : {"change_index": 1, "diff_type": "added", "change_summary": "Inclusion de CWB dans le calcul du risque opérationnel à la suite de l’acquisition."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "relevance_reason": "Le rapport courant mentionne l’inclusion de CWB après une acquisition propre à la banque, sans décrire une nouvelle méthode de calcul partagée. Cette opération interne n’offre pas de base comparable sur les pratiques de gestion des risques entre institutions."}

Exemple 5 — rachat d’actions non pertinent
Input : {"change_index": 1, "diff_type": "modified", "change_summary": "Mise à jour des montants de rachat d’actions ordinaires au semestre."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "relevance_reason": "Le rapport courant met à jour les montants de rachat d’actions ordinaires déjà présentés, sans modifier le cadre réglementaire associé. Ce type de transaction propre à la banque n’éclaire pas la comparabilité des pratiques prudentielles entre pairs."}

Exemple 6 — transfert de responsabilité de gouvernance pertinent et substantiel
Input : {"change_index": 1, "diff_type": "modified", "change_summary": "L’approbation de l’appétit pour le risque passe du comité de direction au conseil d’administration."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["GOUVERNANCE_RISQUES"], "nouvelle_idee": true, "relevance_reason": "Le rapport courant transfère au conseil d’administration l’approbation de l’appétit pour le risque auparavant confiée au comité de direction. Ce transfert d’autorité modifie substantiellement la gouvernance et fournit un point important de comparaison des responsabilités entre les banques."}

Exemple 7 — comité renommé pertinent sans nouvelle idée substantielle
Input : {"change_index": 1, "diff_type": "modified", "change_summary": "Le Comité de gestion des risques est renommé Comité des risques et de la conformité, sans modification de son mandat."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["GOUVERNANCE_RISQUES"], "nouvelle_idee": false, "relevance_reason": "Le rapport courant renomme le Comité de gestion des risques en Comité des risques et de la conformité tout en maintenant son mandat. Ce changement de désignation reste pertinent pour suivre la structure de gouvernance entre les périodes, sans démontrer à lui seul une nouvelle responsabilité ou autorité."}
"""


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
) -> dict[str, Any]:
    """Ajoute localement les champs historiques sans les demander au LLM."""
    is_relevant = bool(compact.get("is_relevant", False))
    nouvelle_idee = bool(compact.get("nouvelle_idee", False))
    themes = list(compact.get("themes_amf") or [])
    relevance_reason = " ".join(
        str(compact.get("relevance_reason") or "").split()
    )

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
        )
        relevance_reason = _local_relevance_reason(factual, comparative)

    high_priority = bool(set(themes) & _COMPACT_HIGH_PRIORITY_THEMES)

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
        "compact_schema_version": "analyst_compact_v1",
        "is_relevant": is_relevant,
        "themes_amf": themes,
        "nouvelle_idee": nouvelle_idee,
        "relevance_reason": relevance_reason,
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


def _triage_section_changes(
    *,
    client: Any,
    model: str,
    section_key: str,
    changes: list[dict[str, Any]],
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

    # The first GPT call arbitrates the semantic relationship.  Only an
    # explicit ``uncertain`` result remains for a human; same and distinct
    # disclosures proceed to the AMF triage normally.
    if any(_requires_alignment_review(change) or _is_semantic_text_move(change) for change in changes):
        enriched: list[dict[str, Any]] = []
        for change in changes:
            if _requires_alignment_review(change):
                enriched.append(_alignment_review_result(change))
            elif _is_semantic_text_move(change):
                enriched.append(_semantic_move_result(change))
            else:
                enriched.extend(
                    _triage_section_changes(
                        client=client,
                        model=model,
                        section_key=section_key,
                        changes=[change],
                    )
                )
        return enriched

    # Deterministic cosmetic + bank-noise pre-filter before any AMF GPT call.
    pending: list[dict[str, Any]] = []
    prefiltered: list[dict[str, Any]] = []
    for change in changes:
        exclusion = (
            _deterministic_cosmetic_exclusion(change)
            or _deterministic_bank_specific_exclusion(change)
        )
        if exclusion:
            prefiltered.append(_prefilter_triage_result(change, exclusion))
        else:
            pending.append(change)
    if not pending:
        return prefiltered

    # Semantic near-duplicate grouping: one representative is triaged, then
    # the verdict is propagated with an auditable regrouping trace.
    groups = _group_semantic_triage_duplicates(pending, client=client)
    if any(len(group) > 1 for group in groups):
        grouped_results: list[dict[str, Any]] = []
        for group_index, member_indexes in enumerate(groups, start=1):
            members = [pending[index] for index in member_indexes]
            if len(members) == 1:
                grouped_results.extend(
                    _triage_section_changes(
                        client=client,
                        model=model,
                        section_key=section_key,
                        changes=members,
                    )
                )
                continue
            representative_results = _triage_section_changes(
                client=client,
                model=model,
                section_key=section_key,
                changes=[members[0]],
            )
            if not representative_results:
                continue
            group_id = f"{section_key}_triage_group_{group_index:03d}"
            grouped_results.extend(
                _propagate_triage_to_group(
                    representative=representative_results[0],
                    members=members,
                    group_id=group_id,
                )
            )
        return [*prefiltered, *grouped_results]

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
        return [*prefiltered, *enriched_batches]

    changes = pending
    triage_inputs = []
    exact_segments_by_index: dict[int, list[dict[str, str]]] = {}
    full_evidence_by_index: dict[int, list[dict[str, Any]]] = {}
    full_evidence_packets_by_index: dict[int, list[dict[str, Any]]] = {}
    for idx, change in enumerate(changes, start=1):
        exact_segments = build_change_segments(change)
        exact_segments_by_index[idx] = exact_segments
        full_evidence = []
        if _requires_full_evidence_packets(change):
            full_evidence_packets_by_index[idx] = _build_full_evidence_packets(change)
            full_evidence = _collect_full_evidence_observations(
                client=client,
                model=model,
                change=change,
            )
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
        triage_inputs.append(
            {
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
                "alignment_rationale": _truncate_prompt_text(
                    str(change.get("alignment_rationale") or ""),
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
                "change_summary": change.get("change_summary", ""),
                "full_evidence_observations": full_evidence,
                "candidate_themes": _candidate_themes_for_change(
                    change,
                    section_key=section_key,
                ),
            }
        )

    system_prompt = (
        "Tu qualifies les changements entre le rapport précédent et le rapport "
        "courant d’une banque canadienne pour une vigie AMF. Réponds uniquement "
        "avec le schéma compact demandé. Sois factuel, sans analyse IT, posture, "
        "niveau d’impact, action recommandée ni répétition des textes sources. "
        "Rédige chaque relevance_reason en français uniquement, en exactement "
        "deux phrases complètes, professionnelles et faciles à comprendre. "
        "N’utilise jamais fragment, chunk, T1, T2 ni termes anglais. La longueur "
        "du changement ne détermine jamais sa pertinence : une modification très "
        "courte peut être substantielle si elle touche la gouvernance."
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
        "Une variation chiffrée propre à la banque, une opération interne "
        "(acquisition, rachat, émission, dividende), une mise à jour de calendrier "
        "d’application, un déplacement identique, du formatage ou une reformulation "
        "sans nouveau fond sont non pertinents. Exception : le changement explicite "
        "du nom d’un comité ou d’une instance de gouvernance reste pertinent même "
        "si son mandat demeure identique; utilise alors `GOUVERNANCE_RISQUES` et "
        "`nouvelle_idee=false`.\n"
        "3. `nouvelle_idee=true` seulement si le rapport courant ajoute, retire "
        "ou modifie substantiellement une information absente sous cette forme "
        "dans le rapport précédent. Pour la gouvernance, considère comme substantiel "
        "tout changement démontré d’autorité décisionnelle, de mandat ou de rôle "
        "d’un comité, de ligne de défense, de responsabilité, de supervision, de reddition de "
        "comptes, de culture de risque, de rémunération liée au risque ou d’appétit "
        "pour le risque. Une phrase courte peut donc être une nouvelle idée; un "
        "simple renommage sans effet sur le mandat ne l’est pas.\n"
        "4. `relevance_reason` contient exactement deux phrases complètes, "
        "professionnelles et faciles à comprendre, rédigées pour une analyste. "
        "La première décrit factuellement le changement entre les rapports. "
        "La seconde interprète ce changement et précise ce qu’il apporte à "
        "l’analyse comparative entre les banques. N’écris pas « Ce changement "
        "est pertinent pour la vigie AMF » ni « Ce changement n’est pas pertinent ». "
        "Aucun titre, aucune liste, aucune rubrique et aucune consigne adressée "
        "à l’analyste. Interdit : fragment, chunk, T1, T2, termes anglais.\n"
        "5. Ne produis aucun champ d’impact, d’action, de posture, d’impact IT, "
        "d’explication générale ou de justification multi-rubriques.\n\n"
        f"{_FEW_SHOT_TRIAGE_AMF}\n\n"
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
            response_format=TriageAMFCompactLLMBatch,
            max_tokens=compact_max_tokens,
            max_retries=2,
            validation_retry_message=(
                "Renvoie le batch compact complet. Chaque change_index doit être "
                "présent exactement une fois. is_relevant=true exige un ou deux "
                "thèmes AMF (préfère candidate_themes, sinon tout code de la "
                "taxonomie AMF, sinon SUJET_EMERGENT_HORS_GRILLE); "
                "is_relevant=false exige themes_amf=[] et "
                "nouvelle_idee=false. Chaque relevance_reason doit contenir "
                "exactement deux phrases complètes : une description factuelle "
                "du changement, puis son interprétation et son apport à l’analyse "
                "comparative."
            ),
            length_retry_message=(
                "Renvoie immédiatement le même batch compact complet, sans aucun "
                "commentaire hors schéma. Garde exactement deux phrases complètes "
                "dans chaque relevance_reason : une description factuelle, puis "
                "une interprétation comparative."
            ),
        )
    except ValidationError as exc:
        raise TriageValidationError(
            section_key=section_key,
            change_index=None,
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
        triage = _persisted_triage_from_compact(
            compact_dict,
            change=change,
        )
        triage["change_segments"] = (
            exact_segments_by_index.get(triage_obj.change_index, [])
            if triage_obj.is_relevant
            else []
        )
        triage_map[triage_obj.change_index] = triage
        if triage_obj.is_relevant:
            relevant_count += 1
        if triage_obj.nouvelle_idee:
            nouvelle_idee_count += 1
        logger.info(
            "compact triage validated section=%s change_index=%d is_relevant=%s themes=%s nouvelle_idee=%s reason_sentences=%d",
            section_key,
            triage_obj.change_index,
            triage_obj.is_relevant,
            triage_obj.themes_amf,
            triage_obj.nouvelle_idee,
            count_complete_sentences(triage_obj.relevance_reason),
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
        triage = triage_map.get(idx, _default_triage())
        evidence_observations = full_evidence_by_index.get(idx, [])
        if evidence_observations:
            coherent, coherence_reason = _verify_triage_coherence(
                client=client,
                model=model,
                change=change,
                triage=triage,
                evidence_packets=full_evidence_packets_by_index[idx],
            )
            if not coherent:
                enriched.append(_coherence_review_triage(change, coherence_reason))
                continue
            triage["full_evidence_verified"] = True
            triage["full_evidence_observations"] = evidence_observations
        enriched_change = dict(change)
        enriched_change["genai_triage"] = triage
        enriched.append(enriched_change)
    return [*prefiltered, *enriched]
