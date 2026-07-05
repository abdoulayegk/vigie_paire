"""Alignement local et GPT des sous-sections."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any


logger = logging.getLogger(__name__)

from .models import (
    NarrativeUnit,
    TextSubsectionAlignmentPlan,
    TextUnitAlignmentPlan,
    _AlignmentCandidate,
    _ComparisonTask,
    _SubsectionRecord,
)
from .openai_gateway import _call_json_completion, _call_structured_completion, _truncate_prompt_text
from .subsection_units import _build_subsection_records, _hierarchy_path_for_subsection
from .text_normalization import (
    _clamp_confidence,
    _jaccard,
    _json_dumps,
    _matching_tokens,
    _normalize_heading,
    _word_count,
)
from .text_topics import _is_specific_canonical_topic

def _alignment_candidate_for_texts(
    *,
    heading_t1: str,
    text_t1: str,
    topic_t1: str,
    heading_t2: str,
    text_t2: str,
    topic_t2: str,
) -> _AlignmentCandidate:
    """Score un possible alignement local entre deux textes."""
    heading_norm_t1 = _normalize_heading(heading_t1)
    heading_norm_t2 = _normalize_heading(heading_t2)
    title_similarity = SequenceMatcher(None, heading_norm_t1, heading_norm_t2).ratio()
    title_jaccard = _jaccard(_matching_tokens(heading_t1), _matching_tokens(heading_t2))
    content_similarity = _jaccard(_matching_tokens(text_t1), _matching_tokens(text_t2))
    canonical_match = (
        topic_t1 == topic_t2
        and _is_specific_canonical_topic(topic_t1)
        and topic_t1 != heading_norm_t1
        and topic_t2 != heading_norm_t2
    )

    weighted = max(
        content_similarity,
        (0.45 * title_similarity) + (0.25 * title_jaccard) + (0.30 * content_similarity),
    )
    if heading_norm_t1 and heading_norm_t1 == heading_norm_t2:
        return _AlignmentCandidate(1.0, "exact_heading", topic_t1, title_similarity, content_similarity, canonical_match)
    if canonical_match and (title_similarity >= 0.30 or content_similarity >= 0.10):
        return _AlignmentCandidate(
            _clamp_confidence(max(weighted, 0.78 + min(content_similarity, 0.25))),
            "canonical_topic",
            topic_t1,
            title_similarity,
            content_similarity,
            canonical_match,
        )
    if title_similarity >= 0.74 or title_jaccard >= 0.55:
        return _AlignmentCandidate(
            max(weighted, 0.74),
            "near_heading",
            topic_t1 if canonical_match else topic_t2,
            title_similarity,
            content_similarity,
            canonical_match,
        )
    if content_similarity >= 0.22:
        return _AlignmentCandidate(
            max(weighted, content_similarity),
            "semantic_match",
            topic_t1 if _is_specific_canonical_topic(topic_t1) else topic_t2,
            title_similarity,
            content_similarity,
            canonical_match,
        )
    return _AlignmentCandidate(
        weighted,
        "semantic_match",
        topic_t1 if _is_specific_canonical_topic(topic_t1) else topic_t2,
        title_similarity,
        content_similarity,
        canonical_match,
    )


def _alignment_candidate_for_records(
    previous: _SubsectionRecord,
    current: _SubsectionRecord,
) -> _AlignmentCandidate:
    """Score un alignement de sous-sections."""
    return _alignment_candidate_for_texts(
        heading_t1=previous.heading,
        text_t1=previous.body,
        topic_t1=previous.canonical_topic,
        heading_t2=current.heading,
        text_t2=current.body,
        topic_t2=current.canonical_topic,
    )


def _subsection_alignment_runtime_fields(
    *,
    previous_heading: str,
    current_heading: str,
    alignment_type: str,
) -> tuple[str, str | None]:
    """Normalise le type d'alignement et construit l'identifiant de restructuration."""
    if alignment_type == "renamed":
        return "renamed", None
    if _normalize_heading(previous_heading) == _normalize_heading(current_heading):
        return "exact_heading", None
    return "semantic_match", None


def _unit_prompt_item(unit: NarrativeUnit) -> dict[str, Any]:
    """Représentation compacte d'une unité narrative envoyée au LLM d'alignement."""
    return {
        "unit_index": unit.unit_index,
        "canonical_topic": unit.canonical_topic,
        "word_count": unit.word_count,
        "text": _truncate_prompt_text(unit.unit_text, 1_600),
    }


def _confidence_to_score(confidence: str, *, default: float = 0.70) -> float:
    """Convertit une confidence LLM textuelle en score numérique borné."""
    value = str(confidence or "").strip().lower()
    if value == "high":
        return 0.92
    if value == "medium":
        return 0.78
    if value == "low":
        return 0.40
    return default


def _client_supports_structured_outputs(client: Any) -> bool:
    """Indique si le client fourni expose ``beta.chat.completions.parse``."""
    return callable(getattr(getattr(getattr(getattr(client, "beta", None), "chat", None), "completions", None), "parse", None))


def _call_alignment_structured_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[Any],
) -> Any:
    """Appelle OpenAI en Pydantic structuré, avec fallback JSON pour les vieux mocks de tests."""
    if _client_supports_structured_outputs(client):
        return _call_structured_completion(
            client,
            model=model,
            messages=messages,
            response_format=response_format,
        )
    raw = _call_json_completion(client, model=model, messages=messages)
    return response_format.model_validate(raw)


def _gpt_align_units_in_subsection(
    *,
    client: Any,
    model: str,
    section_key: str,
    previous: _SubsectionRecord,
    current: _SubsectionRecord,
) -> dict[str, Any] | None:
    """Demande au LLM d'aligner les unités narratives dans une paire de sous-sections déjà bornée."""
    if not previous.units or not current.units:
        return None
    if len(previous.units) == 1 and len(current.units) == 1:
        return None

    try:
        plan = _call_alignment_structured_completion(
            client,
            model=model,
            response_format=TextUnitAlignmentPlan,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es expert en rapports bancaires réglementaires canadiens. "
                        "Tu alignes des unités narratives uniquement à l'intérieur d'une même "
                        "sous-section déjà appariée. Tu ne cherches jamais dans une autre "
                        "sous-section ni dans une autre grande section. Tu retournes uniquement "
                        "un objet JSON valide."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        'Format JSON de réponse: {"matches": [{"previous_unit_index": 1, '
                        '"current_unit_index": 2, "confidence": "high|medium|low", '
                        '"reason": "..."}], "group_matches": [{"previous_unit_indexes": [1], '
                        '"current_unit_indexes": [1, 2], "confidence": "high|medium|low", '
                        '"reason": "..."}], "removed_unit_indexes": [3], '
                        '"added_unit_indexes": [4], "ambiguous_previous_unit_indexes": [], '
                        '"ambiguous_current_unit_indexes": []}\n'
                        "Règles strictes:\n"
                        "- Les unités T1 et T2 viennent déjà de la même sous-section logique.\n"
                        "- Ne matche jamais par position ou par ordre.\n"
                        "- Matche seulement les unités qui portent le même sujet de fond.\n"
                        "- Une unité ne peut être utilisée qu'une seule fois.\n"
                        "- N'inclus que les matches de confidence high ou medium.\n"
                        "- Rédige tous les champs reason en français professionnel.\n"
                        "- Utilise matches pour les alignements 1 unité T1 -> 1 unité T2.\n"
                        "- Utilise group_matches quand une idée est découpée différemment, par exemple 1 unité T1 -> 2 unités T2 ou 2 unités T1 -> 1 unité T2.\n"
                        "- Les group_matches doivent rester locaux à cette sous-section et ne doivent jamais compenser une autre sous-section.\n"
                        "- Si une unité T1 n'a aucun équivalent T2 dans cette sous-section, mets son index dans removed_unit_indexes.\n"
                        "- Si une unité T2 n'a aucun équivalent T1 dans cette sous-section, mets son index dans added_unit_indexes.\n"
                        "- Si tu hésites, mets l'unité dans ambiguous_previous_unit_indexes ou ambiguous_current_unit_indexes.\n"
                        "- N'invente jamais d'index et ne cherche jamais hors de cette sous-section.\n\n"
                        f"Grande section: {section_key}\n"
                        f"Sous-section T1: {previous.heading}\n"
                        f"Sous-section T2: {current.heading}\n\n"
                        f"Unités T1:\n{_json_dumps([_unit_prompt_item(unit) for unit in previous.units])}\n\n"
                        f"Unités T2:\n{_json_dumps([_unit_prompt_item(unit) for unit in current.units])}"
                    ),
                },
            ],
        )
        raw = plan.model_dump()
    except Exception as exc:
        logger.warning(
            "Unit GPT alignment failed for %s/%s -> %s; full subsection fallback will be used: %s",
            section_key,
            previous.heading,
            current.heading,
            exc,
        )
        return None

    previous_indexes = {unit.unit_index for unit in previous.units}
    current_indexes = {unit.unit_index for unit in current.units}
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matches: list[dict[str, Any]] = []
    for item in raw.get("matches") or []:
        if not isinstance(item, dict):
            continue
        confidence = str(item.get("confidence") or "").strip().lower()
        if confidence not in {"high", "medium"}:
            continue
        try:
            previous_index = int(item.get("previous_unit_index"))
            current_index = int(item.get("current_unit_index"))
        except (TypeError, ValueError):
            continue
        if previous_index not in previous_indexes or current_index not in current_indexes:
            continue
        if previous_index in used_previous or current_index in used_current:
            continue
        used_previous.add(previous_index)
        used_current.add(current_index)
        matches.append(
            {
                "previous_unit_indexes": [previous_index],
                "current_unit_indexes": [current_index],
                "confidence": confidence,
                "reason": str(item.get("reason") or ""),
                "alignment_type": "llm_unit_match",
            }
        )

    for item in raw.get("group_matches") or []:
        if not isinstance(item, dict):
            continue
        confidence = str(item.get("confidence") or "").strip().lower()
        if confidence not in {"high", "medium"}:
            continue

        def _index_list(key: str, valid: set[int]) -> list[int]:
            indexes: list[int] = []
            for value in item.get(key) or []:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if index in valid and index not in indexes:
                    indexes.append(index)
            return indexes

        previous_group = _index_list("previous_unit_indexes", previous_indexes)
        current_group = _index_list("current_unit_indexes", current_indexes)
        if not previous_group or not current_group:
            continue
        if len(previous_group) == 1 and len(current_group) == 1:
            continue
        if any(index in used_previous for index in previous_group):
            continue
        if any(index in used_current for index in current_group):
            continue
        used_previous.update(previous_group)
        used_current.update(current_group)
        matches.append(
            {
                "previous_unit_indexes": previous_group,
                "current_unit_indexes": current_group,
                "confidence": confidence,
                "reason": str(item.get("reason") or ""),
                "alignment_type": "llm_unit_group_match",
            }
        )

    def _valid_indexes(key: str, valid: set[int], used: set[int]) -> list[int]:
        output: list[int] = []
        for value in raw.get(key) or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index in valid and index not in used and index not in output:
                output.append(index)
        return output

    removed = _valid_indexes("removed_unit_indexes", previous_indexes, used_previous)
    added = _valid_indexes("added_unit_indexes", current_indexes, used_current)
    ambiguous_previous = _valid_indexes("ambiguous_previous_unit_indexes", previous_indexes, used_previous)
    ambiguous_current = _valid_indexes("ambiguous_current_unit_indexes", current_indexes, used_current)

    accounted_previous = used_previous | set(removed) | set(ambiguous_previous)
    accounted_current = used_current | set(added) | set(ambiguous_current)
    if accounted_previous != previous_indexes or accounted_current != current_indexes:
        logger.info(
            "Unit GPT alignment incomplete for %s/%s -> %s; full subsection fallback will be used",
            section_key,
            previous.heading,
            current.heading,
        )
        return None
    if ambiguous_previous or ambiguous_current:
        logger.info(
            "Unit GPT alignment ambiguous for %s/%s -> %s; full subsection fallback will be used",
            section_key,
            previous.heading,
            current.heading,
        )
        return None
    if not matches:
        return None

    return {
        "matches": matches,
        "removed_unit_indexes": removed,
        "added_unit_indexes": added,
    }


def _comparison_tasks_for_unit_matches(
    *,
    client: Any,
    model: str,
    section_key: str,
    previous: _SubsectionRecord,
    current: _SubsectionRecord,
    alignment_type: str,
    confidence: str,
) -> list[_ComparisonTask] | None:
    """Construit des tâches unitaires depuis un plan d'alignement LLM strictement borné."""
    if not previous.units or not current.units:
        return None
    if len(previous.units) == 1 and len(current.units) == 1:
        return None

    plan = _gpt_align_units_in_subsection(
        client=client,
        model=model,
        section_key=section_key,
        previous=previous,
        current=current,
    )
    if plan is None:
        return None

    normalized_type, restructure_group_id = _subsection_alignment_runtime_fields(
        previous_heading=previous.heading,
        current_heading=current.heading,
        alignment_type=alignment_type,
    )
    heading_changed = _normalize_heading(previous.heading) != _normalize_heading(current.heading)
    emit_rename = heading_changed
    confidence_floor = 0.80 if confidence == "high" else 0.70
    tasks: list[_ComparisonTask] = []
    previous_by_index = {unit.unit_index: unit for unit in previous.units}
    current_by_index = {unit.unit_index: unit for unit in current.units}

    def _joined_unit_text(units: list[NarrativeUnit]) -> str:
        return "\n\n".join(unit.unit_text for unit in units if unit.unit_text.strip())

    def _match_indexes(match: dict[str, Any], plural_key: str, singular_key: str) -> list[int]:
        raw_indexes = match.get(plural_key)
        if raw_indexes is None and match.get(singular_key) is not None:
            raw_indexes = [match.get(singular_key)]
        indexes: list[int] = []
        for value in raw_indexes or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index not in indexes:
                indexes.append(index)
        return indexes

    def _match_sort_key(match: dict[str, Any]) -> int:
        indexes = _match_indexes(match, "previous_unit_indexes", "previous_unit_index")
        return min(indexes) if indexes else 0

    plan_matches = list(plan.get("matches") or [])
    for group_match in plan.get("group_matches") or []:
        if isinstance(group_match, dict):
            plan_matches.append({**group_match, "alignment_type": "llm_unit_group_match"})

    for match in sorted(plan_matches, key=_match_sort_key):
        previous_indexes = _match_indexes(match, "previous_unit_indexes", "previous_unit_index")
        current_indexes = _match_indexes(match, "current_unit_indexes", "current_unit_index")
        previous_units = [previous_by_index[int(index)] for index in previous_indexes]
        current_units = [current_by_index[int(index)] for index in current_indexes]
        canonical_topic = previous_units[0].canonical_topic if previous_units else previous.canonical_topic
        current_topic = current_units[0].canonical_topic if current_units else current.canonical_topic
        if canonical_topic != current_topic and current_topic:
            canonical_topic = current_topic
        previous_unit_indexes = [unit.unit_index for unit in previous_units]
        current_unit_indexes = [unit.unit_index for unit in current_units]
        alignment_label = str(match.get("alignment_type") or "llm_unit_match")
        if normalized_type == "renamed" and alignment_label == "llm_unit_match":
            alignment_label = "renamed_unit_match"
        tasks.append(
            _ComparisonTask(
                heading_t1=previous.heading,
                body_t1=_joined_unit_text(previous_units),
                heading_t2=current.heading,
                body_t2=_joined_unit_text(current_units),
                alignment_type=alignment_label,
                canonical_topic=canonical_topic,
                alignment_confidence=max(_confidence_to_score(str(match.get("confidence") or "")), confidence_floor),
                emit_rename=emit_rename,
                restructure_group_id=restructure_group_id,
                previous_unit_index=min(previous_unit_indexes) if previous_unit_indexes else None,
                current_unit_index=min(current_unit_indexes) if current_unit_indexes else None,
                previous_unit_indexes=previous_unit_indexes,
                current_unit_indexes=current_unit_indexes,
            )
        )
        emit_rename = False

    for previous_index in plan["removed_unit_indexes"]:
        previous_unit = previous_by_index[int(previous_index)]
        tasks.append(
            _ComparisonTask(
                heading_t1=previous.heading,
                body_t1=previous_unit.unit_text,
                heading_t2=current.heading,
                body_t2="",
                alignment_type="llm_unit_removed",
                canonical_topic=previous_unit.canonical_topic or previous.canonical_topic,
                alignment_confidence=confidence_floor,
                restructure_group_id=restructure_group_id,
                previous_unit_index=previous_unit.unit_index,
                synthetic_diff_type="removed",
            )
        )

    for current_index in plan["added_unit_indexes"]:
        current_unit = current_by_index[int(current_index)]
        tasks.append(
            _ComparisonTask(
                heading_t1=previous.heading,
                body_t1="",
                heading_t2=current.heading,
                body_t2=current_unit.unit_text,
                alignment_type="llm_unit_added",
                canonical_topic=current_unit.canonical_topic or current.canonical_topic,
                alignment_confidence=confidence_floor,
                restructure_group_id=restructure_group_id,
                current_unit_index=current_unit.unit_index,
                synthetic_diff_type="added",
            )
        )

    return sorted(tasks, key=lambda task: (task.previous_unit_index is None, task.previous_unit_index or task.current_unit_index or 0))


def _alignment_metadata(task: _ComparisonTask) -> dict[str, Any]:
    """Construit les métadonnées non cassantes d'alignement."""
    metadata: dict[str, Any] = {
        "alignment_type": task.alignment_type,
        "previous_subsection_heading": task.heading_t1 or "",
        "current_subsection_heading": task.heading_t2 or "",
        "canonical_topic": task.canonical_topic,
        "alignment_confidence": round(_clamp_confidence(task.alignment_confidence), 4),
    }
    if task.restructure_group_id:
        metadata["restructure_group_id"] = task.restructure_group_id
    if task.previous_unit_index is not None:
        metadata["previous_unit_index"] = task.previous_unit_index
    if task.current_unit_index is not None:
        metadata["current_unit_index"] = task.current_unit_index
    if task.previous_unit_indexes:
        metadata["previous_unit_indexes"] = list(task.previous_unit_indexes)
    if task.current_unit_indexes:
        metadata["current_unit_indexes"] = list(task.current_unit_indexes)
    return metadata


def _annotate_changes(changes: list[dict[str, Any]], task: _ComparisonTask) -> list[dict[str, Any]]:
    """Ajoute les métadonnées d'alignement aux changements produits."""
    metadata = _alignment_metadata(task)
    for change in changes:
        change.update(metadata)
        section_key = str(change.get("section_key") or "")
        previous_heading = str(change.get("previous_subsection_heading") or task.heading_t1 or "")
        current_heading = str(change.get("current_subsection_heading") or task.heading_t2 or "")
        if section_key and previous_heading:
            change["previous_hierarchy_path"] = _hierarchy_path_for_subsection(section_key, previous_heading)
        if section_key and current_heading:
            change["current_hierarchy_path"] = _hierarchy_path_for_subsection(section_key, current_heading)
    return changes


def _subsection_prompt_item(record: _SubsectionRecord) -> dict[str, Any]:
    """Représentation compacte envoyée à GPT pour l'alignement intra-section."""
    return {
        "heading": record.heading,
        "hierarchy_path": record.hierarchy_path,
        "canonical_topic": record.canonical_topic,
        "word_count": _word_count(record.body),
        "preview": _truncate_prompt_text(record.body, 1_000),
    }


def _gpt_align_subsections(
    *,
    client: Any,
    model: str,
    section_key: str,
    records_t1: dict[str, _SubsectionRecord],
    records_t2: dict[str, _SubsectionRecord],
) -> list[dict[str, Any]]:
    """Demande à GPT le plan d'alignement des sous-sections dans une même grande section."""
    if not records_t1 or not records_t2:
        return []
    try:
        plan = _call_alignment_structured_completion(
            client,
            model=model,
            response_format=TextSubsectionAlignmentPlan,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es expert en rapports bancaires réglementaires canadiens. "
                        "Tu construis un plan d'alignement entre sous-sections T1 et T2 "
                        "à l'intérieur d'une même grande section. Tu ne compares jamais "
                        "avec une autre grande section. Tu retournes uniquement un objet JSON valide."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        'Format JSON de réponse: {"matches": [{"heading_t1": "...", "heading_t2": "...", '
                        '"alignment_type": "exact_heading|renamed", '
                        '"confidence": "high|medium|low", "reason": "..."}]}\n'
                        "Règles strictes:\n"
                        "- Rester dans la grande section fournie uniquement.\n"
                        "- Aligner seulement un bloc T1 avec un seul bloc T2.\n"
                        "- Utiliser exact_heading si les titres sont le même bloc.\n"
                        "- Utiliser renamed seulement si le titre change mais que le bloc métier est clairement le même.\n"
                        "- Ne jamais produire split, merged, moved ou un alignement un-vers-plusieurs.\n"
                        "- N'inclure que confidence high ou medium.\n"
                        "- Rédige tous les champs reason en français professionnel.\n"
                        "- Si tu n'es pas sûr, ne pas inclure la paire.\n"
                        "- Retourner les headings EXACTEMENT comme fournis\n\n"
                        f"Grande section: {section_key}\n\n"
                        f"Sous-sections T1:\n{_json_dumps([_subsection_prompt_item(r) for r in records_t1.values()])}\n\n"
                        f"Sous-sections T2:\n{_json_dumps([_subsection_prompt_item(r) for r in records_t2.values()])}"
                    ),
                },
            ],
        )
        raw = plan.model_dump()
        headings_t1 = set(records_t1)
        headings_t2 = set(records_t2)
        used_t1: set[str] = set()
        used_t2: set[str] = set()
        matches: list[dict[str, Any]] = []
        for m in raw.get("matches") or []:
            if not isinstance(m, dict):
                continue
            conf = str(m.get("confidence") or "").lower()
            h1 = m.get("heading_t1") or ""
            h2 = m.get("heading_t2") or ""
            alignment_type = str(m.get("alignment_type") or "semantic_match").lower()
            if conf not in {"high", "medium"}:
                continue
            if h1 not in headings_t1 or h2 not in headings_t2:
                continue
            same_heading = _normalize_heading(h1) == _normalize_heading(h2)
            if same_heading:
                alignment_type = "exact_heading"
            elif alignment_type != "renamed" or conf != "high":
                continue
            if h1 in used_t1 or h2 in used_t2:
                continue
            matches.append(
                {
                    "heading_t1": h1,
                    "heading_t2": h2,
                    "alignment_type": alignment_type,
                    "confidence": conf,
                    "reason": str(m.get("reason") or ""),
                }
            )
            used_t1.add(h1)
            used_t2.add(h2)
        return matches
    except Exception as exc:
        logger.warning(
            "Subsection GPT alignment failed for %s — conservative fallback will be used: %s",
            section_key,
            exc,
        )
        return []


def _fallback_align_subsections(
    records_t1: dict[str, _SubsectionRecord],
    records_t2: dict[str, _SubsectionRecord],
) -> list[dict[str, Any]]:
    """Fallback local strict si GPT ne fournit pas de plan d'alignement.

    En vigie, un alignement entre sous-sections différentes doit venir d'un
    plan GPT explicite. Le fallback déterministe reste limité aux titres
    identiques pour éviter de comparer un bloc déplacé ailleurs dans la grande
    section sur la seule base d'un thème canonique.
    """
    matches: list[dict[str, Any]] = []
    used_t2: set[str] = set()

    for heading, previous in records_t1.items():
        current = records_t2.get(heading)
        if current is None or current.heading in used_t2:
            continue
        matches.append(
            {
                "heading_t1": heading,
                "heading_t2": current.heading,
                "alignment_type": "exact_heading",
                "confidence": "high",
                "reason": "Titre identique dans la même grande section.",
            }
        )
        used_t2.add(current.heading)
    return matches


def _is_allowed_subsection_match(match: dict[str, Any]) -> bool:
    """Applique la règle bloc-contre-bloc même si le plan GPT est injecté par test ou outil externe."""
    alignment_type = str(match.get("alignment_type") or "").strip().lower()
    if alignment_type in {"split", "merged", "moved"}:
        return False
    return alignment_type in {"exact_heading", "renamed", "near_heading", "canonical_topic", "semantic_match"}


def _comparison_task_for_subsection_match(
    *,
    previous: _SubsectionRecord,
    current: _SubsectionRecord,
    alignment_type: str,
    confidence: str,
    _reason: str,
) -> _ComparisonTask:
    """Construit une tâche de comparaison GPT au niveau sous-section complète."""
    candidate = _alignment_candidate_for_records(previous, current)
    normalized_type, restructure_group_id = _subsection_alignment_runtime_fields(
        previous_heading=previous.heading,
        current_heading=current.heading,
        alignment_type=alignment_type,
    )
    confidence_floor = 0.85 if confidence == "high" else 0.70
    return _ComparisonTask(
        heading_t1=previous.heading,
        body_t1=previous.body,
        heading_t2=current.heading,
        body_t2=current.body,
        alignment_type=normalized_type,
        canonical_topic=candidate.canonical_topic or previous.canonical_topic or current.canonical_topic,
        alignment_confidence=max(candidate.score, confidence_floor),
        emit_rename=_normalize_heading(previous.heading) != _normalize_heading(current.heading),
        restructure_group_id=restructure_group_id,
    )


def _build_comparison_tasks(
    *,
    client: Any,
    model: str,
    section_key: str,
    subs_t1: list[tuple[str, str]],
    subs_t2: list[tuple[str, str]],
) -> tuple[list[_ComparisonTask], list[_SubsectionRecord], list[_SubsectionRecord]]:
    """Aligne les sous-sections par GPT avant de produire des added/removed."""
    records_t1 = _build_subsection_records(section_key, subs_t1)
    records_t2 = _build_subsection_records(section_key, subs_t2)
    tasks: list[_ComparisonTask] = []
    consumed_t1: set[str] = set()
    consumed_t2: set[str] = set()

    alignment_matches = _gpt_align_subsections(
        client=client,
        model=model,
        section_key=section_key,
        records_t1=records_t1,
        records_t2=records_t2,
    )
    if not alignment_matches:
        alignment_matches = _fallback_align_subsections(records_t1, records_t2)

    for match in alignment_matches:
        h1 = str(match.get("heading_t1") or "")
        h2 = str(match.get("heading_t2") or "")
        alignment_type = str(match.get("alignment_type") or "semantic_match")
        if not _is_allowed_subsection_match(match):
            continue
        if _normalize_heading(h1) == _normalize_heading(h2):
            alignment_type = "exact_heading"
        elif alignment_type not in {"renamed"}:
            continue
        if h1 in consumed_t1 or h2 in consumed_t2:
            continue
        if h1 not in records_t1 or h2 not in records_t2:
            continue
        previous = records_t1[h1]
        current = records_t2[h2]
        unit_tasks = _comparison_tasks_for_unit_matches(
            client=client,
            model=model,
            section_key=section_key,
            previous=previous,
            current=current,
            alignment_type=alignment_type,
            confidence=str(match.get("confidence") or "medium"),
        )
        if unit_tasks is not None:
            tasks.extend(unit_tasks)
        else:
            tasks.append(
                _comparison_task_for_subsection_match(
                    previous=previous,
                    current=current,
                    alignment_type=alignment_type,
                    confidence=str(match.get("confidence") or "medium"),
                    _reason=str(match.get("reason") or ""),
                )
            )
        consumed_t1.add(h1)
        consumed_t2.add(h2)

    remaining_removed = [record for heading, record in records_t1.items() if heading not in consumed_t1]
    remaining_added = [record for heading, record in records_t2.items() if heading not in consumed_t2]
    return tasks, remaining_removed, remaining_added
