"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import logging
import re
import unicodedata
from typing import Any

from pydantic import ValidationError

from vigilance.amf_taxonomy import (
    COMPACT_RELEVANCE_REASON_MAX_WORDS,
    COMPACT_RELEVANCE_REASON_MIN_WORDS,
    THEMES_AMF_ANALYST_SUBJECTS,
    THEMES_AMF_DESCRIPTIONS,
    TRIAGE_SOURCE_VERSION,
    TriageAMFCompactLLMBatch,
    TriageValidationError,
    count_words,
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
_TRIAGE_DEDUP_EMBEDDING_THRESHOLD = 0.92
_TRIAGE_EMBEDDING_TRUNCATE_CHARS = 1800
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_COMPACT_THEME_CANDIDATE_LIMIT = 6
_COMPACT_COMPLETION_BASE_TOKENS = 350
_COMPACT_COMPLETION_TOKENS_PER_CHANGE = 320
_COMPACT_COMPLETION_MAX_TOKENS = 1200
_ISOLATED_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
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


def _bounded_local_relevance_reason(value: str) -> str:
    """Normalise une raison locale dans la même plage que la sortie LLM."""
    words = " ".join(str(value or "").split()).split()
    filler = (
        "La décision reste visible afin que l’analyste puisse la confirmer ou "
        "la corriger directement à partir des textes sources comparés."
    ).split()
    while len(words) < COMPACT_RELEVANCE_REASON_MIN_WORDS:
        words.extend(filler)
    words = words[:COMPACT_RELEVANCE_REASON_MAX_WORDS]
    result = " ".join(words).strip()
    return result if result.endswith((".", "!", "?")) else f"{result}."


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
            "relevance_reason": _bounded_local_relevance_reason(
                "Le changement n’a pas reçu de qualification AMF exploitable. "
                "Il est conservé dans la file de revue sans être présenté comme "
                "une nouvelle idée, afin d’éviter une conclusion automatique "
                "non étayée par les éléments disponibles."
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
            "relevance_reason": _bounded_local_relevance_reason(
                "Les passages pourraient décrire des divulgations différentes, "
                "mais l’alignement sémantique ne fournit pas une preuve suffisante "
                "pour conclure automatiquement. Le changement reste donc visible "
                "et doit être lu avec ses extraits sources avant toute décision."
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
            "relevance_reason": _bounded_local_relevance_reason(
                "La comparaison confirme que la divulgation a été déplacée sans "
                "modification substantielle de son sens, de son niveau de détail "
                "ou de son rattachement métier. Ce déplacement ne crée donc pas "
                "une nouvelle idée à surveiller."
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

    similarity = _sequence_ratio(text_t1, text_t2)
    if similarity >= _COSMETIC_SEQUENCE_THRESHOLD:
        return "reformulation_mineure"
    if _is_isolated_date_change(text_t1, text_t2):
        return "reformulation_mineure"
    return None


def _cosmetic_triage_result(change: dict[str, Any], exclusion_reason: str) -> dict[str, Any]:
    triage = _default_triage()
    triage.update(
        {
            "source": "deterministic_prefilter",
            "exclusion_reason": exclusion_reason,
            "relevance_reason": _bounded_local_relevance_reason(
                "Le préfiltre déterministe identifie uniquement une différence "
                "de formulation, de présentation ou de date isolée. Aucun écart "
                "chiffré réglementaire, nouveau facteur de risque ou changement "
                "de méthode n’est détecté dans les passages comparés."
            ),
            "nouvelle_idee_justification": (
                "NON — Nouvel élément à surveiller : Non.\n\n"
                "Sujet détecté : Changement cosmétique.\n\n"
                "Ce qui change : Le pré-filtre déterministe a identifié un écart "
                "de formulation, de formatage ou de date isolée sans delta chiffré "
                "ni réglementaire.\n\n"
                "Pertinence métier : Ce changement ne modifie pas la substance de "
                "la divulgation ni la posture de risque.\n\n"
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
Input : {"change_index": 1, "diff_type": "added", "change_summary": "Ajout d’un contrôle contre les ransomwares."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["RISQUE_EMERGENT", "CONTROLE_CONFORMITE"], "nouvelle_idee": true, "relevance_reason": "Le rapport courant introduit explicitement un contrôle contre les ransomwares qui n’apparaissait pas dans la divulgation précédente. Cet ajout dépasse une reformulation, car il décrit une mesure concrète visant un risque cyber émergent. Pour la vigie, cette précision permet d’évaluer comment la banque renforce sa résilience opérationnelle, sa prévention des incidents et son encadrement des menaces numériques. Elle améliore aussi la comparabilité avec les autres institutions qui présentent leurs contrôles cyber. Le changement mérite donc une surveillance, puisque la présence d’un nouveau dispositif peut signaler une évolution de la gouvernance, des responsabilités ou des pratiques de gestion du risque technologique."}

Exemple 2 — variation propre à la banque non pertinente
Input : {"change_index": 1, "diff_type": "modified", "change_summary": "Le portefeuille hypothécaire passe de 287 G$ à 294 G$."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "relevance_reason": "Le changement porte uniquement sur la valeur courante du portefeuille hypothécaire de la banque, tandis que la nature de l’indicateur et la divulgation demeurent inchangées. Aucun nouveau seuil prudentiel, facteur de risque, contrôle, cadre réglementaire ou changement méthodologique n’est introduit. Cette mise à jour reflète l’évolution normale d’un montant propre à l’institution et ne crée pas une nouvelle idée comparable entre pairs. Elle reste visible pour permettre la validation humaine, mais elle ne justifie pas une surveillance prioritaire au titre de la taxonomie AMF. Les textes sources peuvent confirmer que seule la donnée quantitative a changé entre les deux rapports."}
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
        "exclusion_reason": None if is_relevant else "non_pertinent_autre",
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

    # Deterministic cosmetic pre-filter before any AMF GPT call.
    pending: list[dict[str, Any]] = []
    prefiltered: list[dict[str, Any]] = []
    for change in changes:
        exclusion = _deterministic_cosmetic_exclusion(change)
        if exclusion:
            prefiltered.append(_cosmetic_triage_result(change, exclusion))
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
    for idx, change in enumerate(changes, start=1):
        exact_segments = build_change_segments(change)
        exact_segments_by_index[idx] = exact_segments
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
        "niveau d’impact, action recommandée ni répétition des textes sources."
    )


    user_prompt = (
        f"Retourne exactement {len(changes)} entrée(s) dans `triages`, une par "
        "changement, avec les mêmes `change_index`, sans doublon ni entrée "
        "supplémentaire.\n\n"
        "Règles strictes :\n"
        "1. `is_relevant=true` seulement pour un changement substantiel utile "
        "à la vigie AMF; dans ce cas, choisis un ou deux codes uniquement parmi "
        "les `candidate_themes` de l’entrée.\n"
        "2. `is_relevant=false` exige `themes_amf=[]` et `nouvelle_idee=false`. "
        "Une variation chiffrée propre à la banque, un déplacement identique, "
        "du formatage ou une reformulation sans nouveau fond sont non pertinents.\n"
        "3. `nouvelle_idee=true` seulement si le rapport courant ajoute, retire "
        "ou modifie substantiellement une information absente sous cette forme "
        "dans le rapport précédent.\n"
        "4. `relevance_reason` explique concrètement pourquoi le changement est "
        "pertinent ou non pertinent. Il doit contenir strictement entre "
        f"{COMPACT_RELEVANCE_REASON_MIN_WORDS} et "
        f"{COMPACT_RELEVANCE_REASON_MAX_WORDS} mots, sans titre, liste, rubrique "
        "ni consigne adressée à l’analyste.\n"
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
                "candidate_themes; is_relevant=false exige themes_amf=[] et "
                "nouvelle_idee=false. relevance_reason doit contenir strictement "
                f"{COMPACT_RELEVANCE_REASON_MIN_WORDS} à "
                f"{COMPACT_RELEVANCE_REASON_MAX_WORDS} mots."
            ),
            length_retry_message=(
                "Renvoie immédiatement le même batch compact complet, sans aucun "
                "commentaire hors schéma. Garde chaque relevance_reason entre "
                f"{COMPACT_RELEVANCE_REASON_MIN_WORDS} et "
                f"{COMPACT_RELEVANCE_REASON_MAX_WORDS} mots."
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
        allowed_themes = {
            candidate["code"]
            for candidate in triage_inputs[triage_obj.change_index - 1][
                "candidate_themes"
            ]
        }
        unexpected_themes = set(triage_obj.themes_amf) - allowed_themes
        if unexpected_themes:
            validation_error = ValueError(
                "themes_amf contient des codes hors candidate_themes : "
                f"{sorted(unexpected_themes)}"
            )
            raise TriageValidationError(
                section_key=section_key,
                change_index=triage_obj.change_index,
                raw_payload=triage_obj.model_dump(),
                validation_error=validation_error,
            )

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
            "compact triage validated section=%s change_index=%d is_relevant=%s themes=%s nouvelle_idee=%s reason_words=%d",
            section_key,
            triage_obj.change_index,
            triage_obj.is_relevant,
            triage_obj.themes_amf,
            triage_obj.nouvelle_idee,
            count_words(triage_obj.relevance_reason),
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
        enriched_change = dict(change)
        enriched_change["genai_triage"] = triage
        enriched.append(enriched_change)
    return [*prefiltered, *enriched]
