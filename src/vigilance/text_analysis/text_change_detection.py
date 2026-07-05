"""Detection des changements textuels entre deux versions markdown."""

from __future__ import annotations

import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

from .constants import _COMPARE_BATCH_SIZE, _COMPARE_BATCH_TEXT_CHAR_LIMIT
from .change_records import (
    _build_unpaired_changes,
    _synthetic_narrative_unit_change,
    _synthetic_subsection_change,
    _synthetic_subsection_rename_change,
)
from .models import (
    TextAnalysisQualityError,
    TextComparisonBatch,
    TextComparisonResultsBatch,
    _ComparisonTask,
)
from .openai_gateway import _call_json_completion, _call_structured_completion
from .subsection_alignment import (
    _alignment_metadata,
    _annotate_changes,
    _build_comparison_tasks,
)
from .subsection_units import _parse_subsections
from .text_normalization import _json_dumps, _normalize_heading, _sanitize_explanation, _sanitize_semantic_text
from .vigie_objectives import _vigie_objectives_prompt


def _client_supports_structured_outputs(client: Any) -> bool:
    """Indique si le client fourni expose ``beta.chat.completions.parse``."""
    return callable(getattr(getattr(getattr(getattr(client, "beta", None), "chat", None), "completions", None), "parse", None))


def _call_text_structured_completion(
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


def _task_heading_label_and_slug(task: _ComparisonTask) -> tuple[str, str]:
    """Retourne le libellé lisible et le slug stable d'une tâche de comparaison."""
    heading_label = task.heading_t1 or task.heading_t2 or "unknown"
    if task.heading_t1 and task.heading_t2 and _normalize_heading(task.heading_t1) != _normalize_heading(task.heading_t2):
        heading_label = f"{task.heading_t1} → {task.heading_t2}"
    heading_slug = re.sub(r"[^\w]+", "_", _normalize_heading(heading_label))[:40].strip("_") or "unknown"
    return heading_label, heading_slug


def _validate_comparison_changes(
    *,
    raw_changes: list[dict[str, Any]],
    section_key: str,
    heading_label: str,
    heading_slug: str,
    idx_offset: int,
) -> list[dict[str, Any]]:
    """Valide les changements GPT et les convertit au schéma interne."""
    validated: list[dict[str, Any]] = []
    for local_idx, item in enumerate(raw_changes, start=1):
        diff_type = str(item.get("diff_type") or "").strip().lower()
        if diff_type not in {"unchanged", "modified", "added", "removed"}:
            continue
        text_t1_item = str(item.get("text_t1") or "").strip()
        text_t2_item = str(item.get("text_t2") or "").strip()
        if diff_type in {"unchanged", "modified"} and not (text_t1_item and text_t2_item):
            continue
        if diff_type == "added" and not text_t2_item:
            continue
        if diff_type == "removed" and not text_t1_item:
            continue
        global_idx = idx_offset + local_idx
        status = str(item.get("status") or "").strip().upper()
        if status not in {"EXISTS", "MODIFIED", "ADDED", "REMOVED"}:
            status = {
                "unchanged": "EXISTS",
                "modified": "MODIFIED",
                "added": "ADDED",
                "removed": "REMOVED",
            }[diff_type]
        chunk_topic = _sanitize_explanation(str(item.get("topic") or ""))
        payload = {
            "change_id": f"{section_key}_{heading_slug}_change_{global_idx:03d}",
            "section_key": section_key,
            "subsection_heading": heading_label,
            "diff_type": diff_type,
            "comparison_status": status,
            "semantic_text_t1": _sanitize_semantic_text(text_t1_item),
            "semantic_text_t2": _sanitize_semantic_text(text_t2_item),
            "source_text_t1": text_t1_item,
            "source_text_t2": text_t2_item,
            "source_block_ids_t1": [],
            "source_block_ids_t2": [],
            "source_refs_t1": [],
            "source_refs_t2": [],
            "pages_t1": [],
            "pages_t2": [],
            "source_resolution_t1": "markdown",
            "source_resolution_t2": "markdown",
            "evidence_t1": {"pages": [], "snippet": text_t1_item[:400]},
            "evidence_t2": {"pages": [], "snippet": text_t2_item[:400]},
            "change_summary": _sanitize_explanation(str(item.get("change_summary") or "")),
        }
        if chunk_topic:
            payload["chunk_topic"] = chunk_topic
        validated.append(payload)
    return validated


def _compare_texts_single_call(
    *,
    client: Any,
    model: str,
    section_key: str,
    heading_label: str,
    heading_slug: str,
    text_t1: str,
    text_t2: str,
    idx_offset: int,
) -> list[dict[str, Any]]:
    """Appel GPT unique pour comparer deux corps de texte.

    Extrait la logique de comparaison GPT de ``_compare_section_texts`` pour
    permettre son appel répété par sous-section.
    """
    try:
        response = _call_text_structured_completion(
            client,
            model=model,
            response_format=TextComparisonBatch,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu compares deux versions d'une section de rapport bancaire. "
                        "Tu construis d'abord des chunks métier comparables, puis tu "
                        "cherches l'existence de chaque chunk T1 dans les chunks T2 de "
                        "la même section et sous-section comparées. Tu ne compares pas "
                        "par position ni par chunk_id."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Compare ces deux versions et retourne uniquement du JSON.\n"
                        'Format: {"changes":[{"diff_type":"unchanged|modified|added|removed",'
                        '"status":"EXISTS|MODIFIED|ADDED|REMOVED",'
                        '"topic":"libellé métier court du chunk",'
                        '"text_t1":"chunk T1 complet, vide si added",'
                        '"text_t2":"chunk T2 équivalent, vide si removed",'
                        '"change_summary":"explication concise du changement"}]}.\n'
                        "Langue de sortie: rédige topic et change_summary en français professionnel. "
                        "Les clés JSON et les valeurs d'énumération restent exactement au format demandé.\n"
                        "Méthode obligatoire:\n"
                        "1. Découpe T1 et T2 en chunks métier complets et comparables "
                        "(ex: Contexte réglementaire, Ratios minimaux, Exigences BISN). "
                        "Un chunk doit porter une idée complète, avec le minimum de texte utile.\n"
                        "2. Pour chaque chunk T1, cherche un équivalent de sens uniquement "
                        "dans les chunks T2 de cette même section/sous-section. Ne cherche "
                        "jamais dans une autre section ou sous-section.\n"
                        "3. Ne matche jamais par position, par ordre ou par chunk_id. Le "
                        "chunk_id sert seulement à identifier une ligne interne; le matching "
                        "se fait par sens.\n"
                        "4. Classifie chaque chunk T1:\n"
                        "- EXISTS / diff_type='unchanged': le même sens est présent dans T2, "
                        "même si la formulation change.\n"
                        "- MODIFIED / diff_type='modified': le même sujet existe dans T2, "
                        "mais le contenu substantif change (date, montant, seuil, portée, "
                        "exigence, méthode, responsabilité, nuance importante).\n"
                        "- REMOVED / diff_type='removed': aucun équivalent de sens n'existe "
                        "dans T2 pour ce chunk.\n"
                        "5. Après les chunks T1, classe les chunks T2 sans équivalent T1 "
                        "comme ADDED / diff_type='added'.\n"
                        "Règle anti-bruit: une reformulation sans changement de sens doit "
                        "être EXISTS/unchanged, pas MODIFIED.\n"
                        "Objectifs de vigie à garder en tête pour les résumés, sans inventer "
                        "de changement absent du texte:\n"
                        f"{_vigie_objectives_prompt()}\n\n"
                        f"Section: {section_key}\n\n"
                        f"=== T1 ===\n{text_t1}\n\n"
                        f"=== T2 ===\n{text_t2}\n"
                    ),
                },
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Section comparison failed for {section_key}/{heading_slug}: {exc}") from exc

    return _validate_comparison_changes(
        raw_changes=[change.model_dump() for change in response.changes],
        section_key=section_key,
        heading_label=heading_label,
        heading_slug=heading_slug,
        idx_offset=idx_offset,
    )


def _is_batchable_comparison_task(task: _ComparisonTask) -> bool:
    """Indique si une tâche peut être comparée dans un batch GPT d'unités narratives."""
    if task.synthetic_diff_type is not None or task.emit_rename:
        return False
    if not task.body_t1.strip() or not task.body_t2.strip():
        return False
    if task.previous_unit_index is None or task.current_unit_index is None:
        return False
    return max(len(task.body_t1), len(task.body_t2)) <= _COMPARE_BATCH_TEXT_CHAR_LIMIT


def _compare_texts_batch_call(
    *,
    client: Any,
    model: str,
    section_key: str,
    tasks: list[tuple[int, _ComparisonTask]],
) -> dict[int, list[dict[str, Any]]]:
    """Compare plusieurs paires de NarrativeUnit déjà alignées en un seul appel GPT."""
    if not tasks:
        return {}

    prompt_tasks = []
    valid_indexes: set[int] = set()
    for task_index, task in tasks:
        heading_label, _heading_slug = _task_heading_label_and_slug(task)
        valid_indexes.add(task_index)
        prompt_tasks.append(
            {
                "task_index": task_index,
                "heading_t1": task.heading_t1 or "",
                "heading_t2": task.heading_t2 or "",
                "heading_label": heading_label,
                "alignment_type": task.alignment_type,
                "canonical_topic": task.canonical_topic,
                "text_t1": task.body_t1,
                "text_t2": task.body_t2,
            }
        )

    try:
        response = _call_text_structured_completion(
            client,
            model=model,
            response_format=TextComparisonResultsBatch,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu compares plusieurs paires indépendantes d'unités narratives courtes "
                        "issues de rapports bancaires. Chaque task_index représente uniquement "
                        "une même section/sous-section à comparer. Tu construis des chunks "
                        "métier dans cette tâche, puis tu cherches l'existence sémantique des "
                        "chunks T1 dans les chunks T2 de la même tâche. Ne matche jamais par "
                        "position, par ordre ou par chunk_id."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Compare chaque tâche indépendamment et retourne uniquement du JSON.\n"
                        'Format: {"results":[{"task_index":1,"changes":[{"diff_type":"unchanged|modified|added|removed",'
                        '"status":"EXISTS|MODIFIED|ADDED|REMOVED",'
                        '"topic":"libellé métier court du chunk",'
                        '"text_t1":"texte T1, vide si added",'
                        '"text_t2":"texte T2, vide si removed",'
                        '"change_summary":"explication concise"}]}]}.\n'
                        "Langue de sortie: rédige topic et change_summary en français professionnel. "
                        "Les clés JSON et les valeurs d'énumération restent exactement au format demandé.\n"
                        "Méthode obligatoire:\n"
                        "- task_index doit reprendre exactement un task_index fourni.\n"
                        "- Ne compare jamais le texte d'une tâche avec une autre tâche, ni avec "
                        "une autre section/sous-section.\n"
                        "- Dans chaque tâche, découpe T1 et T2 en chunks métier complets et "
                        "comparables.\n"
                        "- Pour chaque chunk T1, cherche un équivalent de sens dans les chunks "
                        "T2 de cette même tâche seulement.\n"
                        "- EXISTS / diff_type='unchanged' = même sens présent dans T2, même si "
                        "la formulation change.\n"
                        "- MODIFIED / diff_type='modified' = même sujet présent mais contenu "
                        "substantif changé (date, montant, seuil, portée, exigence, méthode, "
                        "responsabilité, nuance importante).\n"
                        "- REMOVED / diff_type='removed' = aucun équivalent de sens dans T2.\n"
                        "- ADDED / diff_type='added' = chunk T2 sans équivalent T1.\n"
                        "Règle anti-bruit: une reformulation sans changement de sens doit être "
                        "EXISTS/unchanged, pas MODIFIED.\n\n"
                        "Objectifs de vigie à garder en tête pour les résumés, sans inventer "
                        "de changement absent du texte:\n"
                        f"{_vigie_objectives_prompt()}\n\n"
                        f"Section: {section_key}\n\n"
                        f"Tâches:\n{_json_dumps(prompt_tasks)}"
                    ),
                },
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Section comparison batch failed for {section_key}: {exc}") from exc

    results_by_index: dict[int, list[dict[str, Any]]] = {}
    for result in response.results:
        task_index = int(result.task_index)
        if task_index not in valid_indexes:
            continue
        results_by_index[task_index] = [item.model_dump() for item in result.changes]
    return results_by_index


def _compare_section_texts(
    *,
    client: Any,
    model: str,
    section_key: str,
    text_t1: str,
    text_t2: str,
) -> list[dict[str, Any]]:
    """Compare deux sections markdown T1/T2 sous-section par sous-section.

    Le texte est découpé selon les headings ### existants. Chaque paire de
    sous-sections fait l'objet d'un appel GPT séparé, évitant les dépassements
    de contexte sur les grandes sections comme ``Gestion des risques``.

    Les sous-sections sans contrepartie sont marquées ajoutées ou supprimées
    sans appel GPT. Si le texte ne contient aucune sous-section, un seul appel
    GPT est lancé sur le texte entier (comportement précédent).
    """
    # Safety: the .md canonique contient des marqueurs ``[p.N]`` pour la
    # reconstruction de l'index page→texte. Ils DOIVENT avoir été strippés
    # avant d'arriver ici par ``_extract_section_text_from_markdown``.
    if "[p." in text_t1 or "[p." in text_t2:
        raise TextAnalysisQualityError("Fuite de marqueurs de page vers le prompt GPT — strip manquant ?")
    if not text_t1.strip() and not text_t2.strip():
        return []

    subs_t1 = _parse_subsections(text_t1)
    subs_t2 = _parse_subsections(text_t2)

    has_real_subsections_t1 = any(heading != "__intro__" for heading, _body in subs_t1)
    has_real_subsections_t2 = any(heading != "__intro__" for heading, _body in subs_t2)

    # No markdown subsections on either side — fall back to single call (legacy behaviour)
    if not has_real_subsections_t1 and not has_real_subsections_t2:
        return _compare_texts_single_call(
            client=client,
            model=model,
            section_key=section_key,
            heading_label="",
            heading_slug="full",
            text_t1=text_t1,
            text_t2=text_t2,
            idx_offset=0,
        )

    comparison_tasks, remaining_removed, remaining_added = _build_comparison_tasks(
        client=client,
        model=model,
        section_key=section_key,
        subs_t1=subs_t1,
        subs_t2=subs_t2,
    )

    all_changes: list[dict[str, Any]] = []
    global_idx = 1
    logger.info(
        "text comparison section=%s tasks=%d remaining_removed=%d remaining_added=%d",
        section_key,
        len(comparison_tasks),
        len(remaining_removed),
        len(remaining_added),
    )

    task_position = 0
    while task_position < len(comparison_tasks):
        task = comparison_tasks[task_position]
        heading_label, heading_slug = _task_heading_label_and_slug(task)

        if task.emit_rename and task.heading_t1 and task.heading_t2:
            all_changes.append(
                _synthetic_subsection_rename_change(
                    section_key=section_key,
                    heading_t1=task.heading_t1,
                    heading_t2=task.heading_t2,
                    idx=global_idx,
                    alignment_type=task.alignment_type,
                    canonical_topic=task.canonical_topic,
                    alignment_confidence=task.alignment_confidence,
                    restructure_group_id=task.restructure_group_id,
                )
            )
            global_idx += 1

        if not task.body_t1.strip() and not task.body_t2.strip():
            task_position += 1
            continue

        if task.synthetic_diff_type in {"added", "removed"}:
            synthetic_heading = (
                (task.heading_t2 or task.heading_t1 or "unknown")
                if task.synthetic_diff_type == "added"
                else (task.heading_t1 or task.heading_t2 or "unknown")
            )
            has_unit_context = task.previous_unit_index is not None or task.current_unit_index is not None
            if has_unit_context:
                synthetic_change = _synthetic_narrative_unit_change(
                    section_key=section_key,
                    diff_type=task.synthetic_diff_type,
                    heading=synthetic_heading,
                    body_t1=task.body_t1 if task.synthetic_diff_type == "removed" else "",
                    body_t2=task.body_t2 if task.synthetic_diff_type == "added" else "",
                    idx=global_idx,
                    previous_unit_index=task.previous_unit_index,
                    current_unit_index=task.current_unit_index,
                    alignment_type=task.alignment_type,
                    canonical_topic=task.canonical_topic,
                    alignment_confidence=task.alignment_confidence,
                    previous_subsection_heading=task.heading_t1 or "",
                    current_subsection_heading=task.heading_t2 or "",
                    restructure_group_id=task.restructure_group_id,
                )
            else:
                synthetic_change = _synthetic_subsection_change(
                    section_key=section_key,
                    diff_type=task.synthetic_diff_type,
                    heading=synthetic_heading,
                    body_t1=task.body_t1 if task.synthetic_diff_type == "removed" else "",
                    body_t2=task.body_t2 if task.synthetic_diff_type == "added" else "",
                    idx=global_idx,
                    alignment_type=task.alignment_type,
                    canonical_topic=task.canonical_topic,
                    alignment_confidence=task.alignment_confidence,
                    previous_subsection_heading=task.heading_t1 or "",
                    current_subsection_heading=task.heading_t2 or "",
                    restructure_group_id=task.restructure_group_id,
                )
            synthetic_change.update(_alignment_metadata(task))
            all_changes.append(synthetic_change)
            global_idx += 1
            task_position += 1
            continue

        if _is_batchable_comparison_task(task):
            batch: list[tuple[int, _ComparisonTask]] = []
            cursor = task_position
            while cursor < len(comparison_tasks) and len(batch) < _COMPARE_BATCH_SIZE:
                candidate_task = comparison_tasks[cursor]
                if not _is_batchable_comparison_task(candidate_task):
                    break
                batch.append((len(batch) + 1, candidate_task))
                cursor += 1

            if len(batch) > 1:
                logger.info(
                    "text comparison batch section=%s tasks=%d-%d/%d size=%d",
                    section_key,
                    task_position + 1,
                    cursor,
                    len(comparison_tasks),
                    len(batch),
                )
                try:
                    batch_results = _compare_texts_batch_call(
                        client=client,
                        model=model,
                        section_key=section_key,
                        tasks=batch,
                    )
                    force_single_fallback = False
                except RuntimeError as exc:
                    logger.warning(
                        "Text comparison batch failed for %s tasks=%d-%d; falling back to single calls. Error: %s",
                        section_key,
                        task_position + 1,
                        cursor,
                        exc,
                    )
                    batch_results = {}
                    force_single_fallback = True

                for local_index, batch_task in batch:
                    batch_heading_label, batch_heading_slug = _task_heading_label_and_slug(batch_task)
                    raw_changes = batch_results.get(local_index)
                    if force_single_fallback or raw_changes is None:
                        subsection_changes = _compare_texts_single_call(
                            client=client,
                            model=model,
                            section_key=section_key,
                            heading_label=batch_heading_label,
                            heading_slug=batch_heading_slug,
                            text_t1=batch_task.body_t1,
                            text_t2=batch_task.body_t2,
                            idx_offset=global_idx - 1,
                        )
                    else:
                        subsection_changes = _validate_comparison_changes(
                            raw_changes=raw_changes,
                            section_key=section_key,
                            heading_label=batch_heading_label,
                            heading_slug=batch_heading_slug,
                            idx_offset=global_idx - 1,
                        )
                    _annotate_changes(subsection_changes, batch_task)
                    all_changes.extend(subsection_changes)
                    global_idx += len(subsection_changes)

                task_position = cursor
                continue

        subsection_changes = _compare_texts_single_call(
            client=client,
            model=model,
            section_key=section_key,
            heading_label=heading_label,
            heading_slug=heading_slug,
            text_t1=task.body_t1,
            text_t2=task.body_t2,
            idx_offset=global_idx - 1,
        )
        _annotate_changes(subsection_changes, task)
        all_changes.extend(subsection_changes)
        global_idx += len(subsection_changes)
        task_position += 1

    for record in remaining_removed:
        changes, global_idx = _build_unpaired_changes(
            section_key=section_key,
            diff_type="removed",
            record=record,
            idx=global_idx,
        )
        all_changes.extend(changes)

    for record in remaining_added:
        changes, global_idx = _build_unpaired_changes(
            section_key=section_key,
            diff_type="added",
            record=record,
            idx=global_idx,
        )
        all_changes.extend(changes)

    return all_changes
