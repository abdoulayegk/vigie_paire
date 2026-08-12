"""Renforcement sémantique de la localisation des sections depuis la TDM.

Les méthodes déterministes restent prioritaires. Ce module classe une courte
liste d'entrées de table des matières avec des embeddings, puis confie au LLM
la distinction entre grand chapitre, sous-section, mention et cas ambigu.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from vigie.extraction.section_taxonomy import canonicalize_section
from vigie.support.utils.genai import get_openai_api_key

from .models import LocatedSection, TocEntry, normalize_text

logger = logging.getLogger("vigie.extraction.section_locator")

SemanticConcept = Literal["capital_management", "risk_management", "regulatory_updates", "other"]
SemanticRole = Literal["main_section", "subsection", "mention", "combined_section", "ambiguous"]

_INTERNAL_SECTION_TYPES = {
    "capital_management": "gestion_capital",
    "risk_management": "gestion_risques",
    "regulatory_updates": "gestion_reglementation",
}
_MAJOR_ROLES = {"main_section", "combined_section"}

_CONCEPT_PROTOTYPES: dict[str, list[str]] = {
    "capital_management": [
        "Grand chapitre sur la gestion du capital, les fonds propres réglementaires et la solidité financière",
        "Gestion du capital et planification des fonds propres",
        "Gestion des fonds propres, ratios CET1, levier et capacité d'absorption des pertes",
        "Situation des fonds propres et adéquation du capital",
        "Capital management and regulatory capital adequacy",
    ],
    "risk_management": [
        "Grand chapitre sur les facteurs de risque et la gestion globale des risques",
        "Gestion du risque d'entreprise et gouvernance des risques",
        "Risques de crédit, de marché, de liquidité, opérationnels et émergents",
        "Facteurs de risque et dispositifs de maîtrise des risques",
        "Enterprise risk management and risk factors",
    ],
    "regulatory_updates": [
        "Section autonome sur les faits nouveaux et changements en matière de réglementation",
        "Contexte réglementaire, exigences prudentielles et perspectives réglementaires",
        "Évolution de la réglementation bancaire et nouvelles règles du régulateur",
        "Regulatory developments and regulatory outlook",
    ],
    "other": [
        "Chapitre comptable, résultats financiers, secteurs d'exploitation ou états financiers",
        "Glossaire, renseignements supplémentaires, transactions et contrôles",
        "Sous-section technique ou simple renvoi dans un autre chapitre",
    ],
}


class SemanticEntryDecision(BaseModel):
    """Décision structurée du LLM pour une entrée TDM."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    concept: SemanticConcept
    role: SemanticRole
    parent_candidate_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class SemanticDecisionBatch(BaseModel):
    """Lot de décisions structurées pour les entrées retenues."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[SemanticEntryDecision]
    warnings: list[str]


@dataclass(slots=True)
class SemanticTocResolution:
    """Sections sémantiques proposées et diagnostic associé."""

    sections: list[LocatedSection] = field(default_factory=list)
    status: str = "unavailable"
    decisions: list[SemanticEntryDecision] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


EmbeddingProvider = Callable[[list[str], str], list[list[float]]]
DecisionProvider = Callable[[list[dict[str, Any]], str], SemanticDecisionBatch]


def _dot_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculer une similarité cosinus robuste aux vecteurs non normalisés."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(value) ** 2 for value in left))
    norm_right = math.sqrt(sum(float(value) ** 2 for value in right))
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_left * norm_right)))


def _candidate_view(entries: list[TocEntry], index: int) -> str:
    """Construire la vue textuelle contextualisée envoyée aux embeddings."""
    entry = entries[index]
    previous_title = entries[index - 1].title if index > 0 else ""
    next_title = entries[index + 1].title if index + 1 < len(entries) else ""
    return (
        f"Titre TDM: {entry.title}\n"
        f"Page imprimée: {entry.page}\n"
        f"Niveau extrait: {entry.level}\n"
        f"Entrée précédente: {previous_title or '[aucune]'}\n"
        f"Entrée suivante: {next_title or '[aucune]'}"
    )


def _default_embedding_provider(api_key: str, timeout: float) -> EmbeddingProvider:
    """Créer le transport embeddings OpenAI par lots."""
    import openai  # noqa: PLC0415 - dépendance optionnelle chargée à l'exécution

    client = openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=1)

    def _embed(texts: list[str], model: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 96):
            response = client.embeddings.create(model=model, input=texts[start : start + 96])
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        return vectors

    return _embed


def _default_decision_provider(api_key: str, timeout: float) -> DecisionProvider:
    """Créer le transport LLM à sortie Pydantic structurée."""
    import openai  # noqa: PLC0415 - dépendance optionnelle chargée à l'exécution

    client = openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=1)

    def _decide(candidates: list[dict[str, Any]], model: str) -> SemanticDecisionBatch:
        prompt = """Tu classes des entrées ordonnées de la table des matières d'un rapport bancaire canadien.

OBJECTIF
- Identifier les grands chapitres de gestion du capital, gestion des risques et faits réglementaires.
- Distinguer strictement un grand chapitre d'une sous-section, d'une mention ou d'un renvoi.
- Préserver la hiérarchie : une nouvelle sous-section reste dans son grand chapitre jusqu'au prochain chapitre de même niveau ou supérieur.

RÈGLES
- « Facteurs de risque et gestion des risques » peut être le grand chapitre qui contient ensuite « Gestion des risques ».
- « Faits nouveaux en matière de réglementation » peut être une sous-section interne à capital, risques ou comptabilité; ne la classe pas automatiquement comme section autonome.
- Un titre de risque spécialisé (crédit, marché, IA, cyber, liquidité, modèles) est normalement une sous-section.
- parent_candidate_id doit référencer un candidat fourni ou être null.
- Utilise ambiguous lorsque le niveau ou le parent ne peut pas être déterminé avec confiance.
- Retourne exactement une décision par candidat fourni, sans inventer de candidat.

CANDIDATS
"""
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "Tu es un classificateur documentaire bancaire précis et conservateur."},
                {"role": "user", "content": prompt + json.dumps(candidates, ensure_ascii=False, indent=2)},
            ],
            response_format=SemanticDecisionBatch,
            temperature=0.0,
            max_completion_tokens=7000,
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError("semantic_section_llm_truncated")
        parsed = getattr(choice.message, "parsed", None)
        if parsed is None:
            raise RuntimeError("semantic_section_llm_empty")
        return parsed

    return _decide


def _embedding_scores(
    entries: list[TocEntry],
    *,
    embedding_model: str,
    embedding_provider: EmbeddingProvider,
) -> tuple[list[dict[str, float]], list[str]]:
    """Calculer le meilleur score par concept pour chaque entrée."""
    views = [_candidate_view(entries, index) for index in range(len(entries))]
    prototype_labels: list[str] = []
    prototype_texts: list[str] = []
    for concept, prototypes in _CONCEPT_PROTOTYPES.items():
        for prototype in prototypes:
            prototype_labels.append(concept)
            prototype_texts.append(prototype)

    vectors = embedding_provider(views + prototype_texts, embedding_model)
    if len(vectors) != len(views) + len(prototype_texts):
        raise RuntimeError("semantic_embedding_vector_count_mismatch")
    candidate_vectors = vectors[: len(views)]
    prototype_vectors = vectors[len(views) :]

    scores: list[dict[str, float]] = []
    for candidate_vector in candidate_vectors:
        by_concept = {concept: -1.0 for concept in _CONCEPT_PROTOTYPES}
        for concept, prototype_vector in zip(prototype_labels, prototype_vectors):
            by_concept[concept] = max(by_concept[concept], _dot_similarity(candidate_vector, prototype_vector))
        scores.append({key: round(value, 6) for key, value in by_concept.items()})
    return scores, views


def _select_candidate_indices(
    entries: list[TocEntry],
    scores: list[dict[str, float]],
    *,
    shortlist_per_concept: int,
    max_candidates: int,
) -> list[int]:
    """Sélectionner les meilleurs candidats et leur contexte immédiat."""
    selected: set[int] = set()
    for concept in ("capital_management", "risk_management", "regulatory_updates"):
        ranked = sorted(range(len(entries)), key=lambda index: scores[index].get(concept, -1.0), reverse=True)
        selected.update(ranked[:shortlist_per_concept])

    for index, entry in enumerate(entries):
        title_norm = normalize_text(entry.title)
        if entry.level == 0 or any(
            token in title_norm for token in ("capital", "fonds propres", "risque", "reglement")
        ):
            selected.add(index)

    with_context = set(selected)
    for index in selected:
        with_context.update(candidate for candidate in range(max(0, index - 2), min(len(entries), index + 3)))

    ranked_context = sorted(
        with_context,
        key=lambda index: (
            index not in selected,
            -max(scores[index].get(concept, -1.0) for concept in _CONCEPT_PROTOTYPES),
            index,
        ),
    )[:max_candidates]
    return sorted(ranked_context)


def _semantic_boundary(
    entries: list[TocEntry],
    start_entry: TocEntry,
    concept: str,
) -> tuple[int | None, str]:
    """Trouver le prochain chapitre sémantique sans couper sur une sous-section."""
    ordered = [
        entry
        for _, entry in sorted(
            enumerate(entries),
            key=lambda pair: (pair[1].page, pair[0]),
        )
    ]
    try:
        start_index = ordered.index(start_entry)
    except ValueError:
        return None, ""

    for entry in ordered[start_index + 1 :]:
        if entry.page < start_entry.page:
            continue
        if entry.semantic_role in {"subsection", "mention"}:
            continue
        if entry.semantic_role in _MAJOR_ROLES:
            if entry.semantic_concept == concept:
                continue
            return max(start_entry.page, entry.page - 1), "semantic_next_major_section"
        if not entry.semantic_role and entry.level == 0 and entry.page > start_entry.page:
            return entry.page - 1, "semantic_toc_level_fallback"
    return None, ""


def resolve_semantic_toc_sections(
    toc_entries: list[TocEntry],
    *,
    bank_code: str,
    config: dict[str, Any],
    embedding_provider: EmbeddingProvider | None = None,
    decision_provider: DecisionProvider | None = None,
) -> SemanticTocResolution:
    """Classifier la TDM et proposer uniquement des grands chapitres fiables."""
    semantic_config = config.get("section_semantic_localization", {}) if isinstance(config, dict) else {}
    if not semantic_config.get("enabled", False):
        return SemanticTocResolution(status="disabled", diagnostics={"status": "disabled"})
    if not toc_entries:
        return SemanticTocResolution(status="no_toc", diagnostics={"status": "no_toc"})

    api_key = get_openai_api_key()
    if embedding_provider is None or decision_provider is None:
        if not api_key:
            return SemanticTocResolution(
                status="unavailable",
                diagnostics={"status": "unavailable", "warnings": ["openai_api_key_missing"]},
            )
        timeout = float(semantic_config.get("timeout_sec", 120))
        embedding_provider = embedding_provider or _default_embedding_provider(api_key, timeout)
        decision_provider = decision_provider or _default_decision_provider(api_key, timeout)

    embedding_model = str(semantic_config.get("embedding_model", "text-embedding-3-large"))
    llm_model = str(semantic_config.get("llm_model", config.get("llm_models", {}).get("default_genai", "gpt-4o")))
    min_llm_candidate_confidence = float(semantic_config.get("min_llm_candidate_confidence", 0.7))
    min_llm_confidence = float(semantic_config.get("min_llm_confidence", 0.82))
    ambiguous_margin = float(semantic_config.get("ambiguous_margin", 0.04))
    allow_regulatory_discovery = bool(semantic_config.get("allow_regulatory_discovery", True))

    ordered_entries = [
        entry
        for _, entry in sorted(
            enumerate(toc_entries),
            key=lambda pair: (pair[1].page, pair[0]),
        )
    ]
    try:
        scores, views = _embedding_scores(
            ordered_entries,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
        )
        selected_indices = _select_candidate_indices(
            ordered_entries,
            scores,
            shortlist_per_concept=int(semantic_config.get("shortlist_per_concept", 6)),
            max_candidates=int(semantic_config.get("max_candidates", 48)),
        )
        payload: list[dict[str, Any]] = []
        id_to_entry: dict[str, TocEntry] = {}
        id_to_score: dict[str, dict[str, float]] = {}
        for index in selected_indices:
            candidate_id = f"toc_{index:03d}"
            entry = ordered_entries[index]
            ranked_scores = sorted(scores[index].items(), key=lambda item: item[1], reverse=True)
            payload.append(
                {
                    "candidate_id": candidate_id,
                    "title": entry.title,
                    "printed_page": entry.page,
                    "extracted_level": entry.level,
                    "context": views[index],
                    "embedding_scores": dict(ranked_scores),
                    "embedding_margin": round(ranked_scores[0][1] - ranked_scores[1][1], 6),
                }
            )
            id_to_entry[candidate_id] = entry
            id_to_score[candidate_id] = scores[index]

        batch = decision_provider(payload, llm_model)
    except Exception as exc:
        logger.warning("Localisation sémantique indisponible: %s", exc)
        return SemanticTocResolution(
            status="error",
            diagnostics={"status": "error", "warnings": [f"{type(exc).__name__}:{exc}"]},
        )

    allowed_ids = set(id_to_entry)
    decisions_by_id: dict[str, SemanticEntryDecision] = {}
    warnings = list(batch.warnings)
    for decision in batch.decisions:
        if decision.candidate_id not in allowed_ids or decision.candidate_id in decisions_by_id:
            warnings.append(f"invalid_or_duplicate_candidate:{decision.candidate_id}")
            continue
        if decision.parent_candidate_id and decision.parent_candidate_id not in allowed_ids:
            warnings.append(f"invalid_parent:{decision.candidate_id}:{decision.parent_candidate_id}")
            decision.parent_candidate_id = None
        decisions_by_id[decision.candidate_id] = decision

        entry = id_to_entry[decision.candidate_id]
        entry.semantic_concept = decision.concept
        entry.semantic_role = decision.role
        entry.semantic_confidence = decision.confidence
        parent = id_to_entry.get(decision.parent_candidate_id) if decision.parent_candidate_id else None
        entry.semantic_parent_title = parent.title if parent else None

    missing_decisions = allowed_ids - set(decisions_by_id)
    if missing_decisions:
        warnings.append(f"missing_decisions:{len(missing_decisions)}")
        diagnostics = {
            "status": "ambiguous",
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "bank_code": bank_code,
            "toc_entry_count": len(toc_entries),
            "candidate_count": len(allowed_ids),
            "decision_count": len(decisions_by_id),
            "concept_status": {},
            "warnings": warnings,
            "decisions": [decision.model_dump() for decision in decisions_by_id.values()],
        }
        return SemanticTocResolution(
            status="ambiguous",
            decisions=list(decisions_by_id.values()),
            diagnostics=diagnostics,
        )

    sections: list[LocatedSection] = []
    concept_status: dict[str, str] = {}
    ambiguous = False
    for concept in ("capital_management", "risk_management", "regulatory_updates"):
        if concept == "regulatory_updates" and not allow_regulatory_discovery:
            concept_status[concept] = "disabled"
            continue
        eligible = [
            decision
            for decision in decisions_by_id.values()
            if decision.concept == concept
            and decision.role in _MAJOR_ROLES
            and decision.confidence >= min_llm_candidate_confidence
        ]
        eligible.sort(
            key=lambda decision: (
                decision.confidence,
                id_to_score[decision.candidate_id].get(concept, -1.0),
            ),
            reverse=True,
        )
        if not eligible:
            concept_status[concept] = "not_found"
            continue

        best = eligible[0]
        if len(eligible) > 1:
            runner_up = eligible[1]
            best_page = id_to_entry[best.candidate_id].page
            runner_page = id_to_entry[runner_up.candidate_id].page
            if best_page != runner_page and best.confidence - runner_up.confidence <= ambiguous_margin:
                concept_status[concept] = "ambiguous"
                ambiguous = True
                continue

        entry = id_to_entry[best.candidate_id]
        end_page, end_method = _semantic_boundary(ordered_entries, entry, concept)
        sections.append(
            LocatedSection(
                section_type=_INTERNAL_SECTION_TYPES[concept],
                title_found=entry.title,
                start_page=entry.page,
                end_page=end_page,
                confidence=best.confidence,
                detection_method="toc_semantic",
                end_detection_method=end_method,
                semantic_role=best.role,
                semantic_confidence=best.confidence,
                semantic_status=("confirmed" if best.confidence >= min_llm_confidence else "vision_required"),
            )
        )
        concept_status[concept] = "confirmed" if best.confidence >= min_llm_confidence else "vision_required"

    status = "ambiguous" if ambiguous else ("resolved" if sections else "not_found")
    diagnostics = {
        "status": status,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "bank_code": bank_code,
        "toc_entry_count": len(toc_entries),
        "candidate_count": len(allowed_ids),
        "decision_count": len(decisions_by_id),
        "concept_status": concept_status,
        "warnings": warnings,
        "decisions": [decision.model_dump() for decision in decisions_by_id.values()],
    }
    return SemanticTocResolution(
        sections=sections,
        status=status,
        decisions=list(decisions_by_id.values()),
        diagnostics=diagnostics,
    )


def merge_semantic_sections(
    existing_sections: list[LocatedSection],
    semantic_sections: list[LocatedSection],
    *,
    replace_below_confidence: float = 0.7,
    conflict_page_gap: int = 3,
) -> tuple[list[LocatedSection], list[str]]:
    """Compléter ou remplacer prudemment les sections déterministes faibles."""
    merged = list(existing_sections)
    warnings: list[str] = []
    for candidate in semantic_sections:
        concept = canonicalize_section(candidate.section_type)
        matches = [section for section in merged if canonicalize_section(section.section_type) == concept]
        if not matches:
            merged.append(candidate)
            continue

        current = max(matches, key=lambda section: float(section.confidence or 0.0))
        page_gap = abs(int(current.start_page) - int(candidate.start_page))
        if current.confidence < replace_below_confidence and candidate.confidence > current.confidence:
            merged = [section for section in merged if canonicalize_section(section.section_type) != concept]
            merged.append(candidate)
        elif page_gap > conflict_page_gap:
            current.semantic_status = "ambiguous"
            warnings.append(
                f"semantic_page_conflict:{concept}:deterministic={current.start_page}:semantic={candidate.start_page}"
            )

    merged.sort(key=lambda section: section.start_page)
    return merged, warnings
