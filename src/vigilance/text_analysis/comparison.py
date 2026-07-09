"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
from typing import Any

from vigilance.text_analysis.chunk_alignment import _align_chunks_tfidf, _format_alignments_for_prompt
from vigilance.text_analysis.chunk_alignment import ChunkAlignment
from vigilance.text_analysis.chunking import _chunk_subsection_text
from vigilance.text_analysis.constants import _SECTION_LABELS
from vigilance.text_analysis.models import TextAnalysisQualityError
from vigilance.text_analysis.normalization import _sanitize_explanation, _sanitize_semantic_text
from vigilance.text_analysis.openai_client import _call_json_completion
from vigilance.text_analysis.subsection_matching import (
    _gpt_match_orphan_headings,
    _normalize_heading,
    _pair_subsections,
    _parse_subsections,
    _synthetic_subsection_change,
    _synthetic_subsection_rename_change,
)


_MAX_COMPARISON_LLM_WORKERS = 6
_COMPARISON_BATCH_SIZES = {
    "matched_strong": 5,
    "matched_weak": 3,
    "ambiguous": 1,
    "possible_added": 1,
    "possible_removed": 1,
}


@dataclass(slots=True)
class ComparisonBatch:
    """Lot d'alignements envoyé dans un appel LLM de comparaison."""

    batch_id: str
    alignment_type: str
    alignments: list[ChunkAlignment]
    heading_label: str
    heading_slug: str
    idx_offset: int


def _prepare_subsection_alignments(
    *,
    section_key: str,
    subsection_heading_t1: str,
    subsection_heading_t2: str,
    body_t1: str,
    body_t2: str,
) -> list[ChunkAlignment]:
    """Prépare une paire de sous-sections en alignements locaux TF-IDF."""
    section_title = _SECTION_LABELS.get(section_key, section_key)
    chunks_t1 = _chunk_subsection_text(
        body_t1,
        subsection_heading=subsection_heading_t1,
        section_title=section_title,
    )
    chunks_t2 = _chunk_subsection_text(
        body_t2,
        subsection_heading=subsection_heading_t2,
        section_title=section_title,
    )
    if not chunks_t1:
        raise TextAnalysisQualityError(
            f"Sous-section appariée sans chunk T1: {section_key}/{subsection_heading_t1}"
        )
    if not chunks_t2:
        raise TextAnalysisQualityError(
            f"Sous-section appariée sans chunk T2: {section_key}/{subsection_heading_t2}"
        )
    return _align_chunks_tfidf(chunks_t1, chunks_t2)


def _batch_size_for_alignment_type(alignment_type: str) -> int:
    return _COMPARISON_BATCH_SIZES.get(alignment_type, 1)


def _build_comparison_batches(
    *,
    alignments: list[ChunkAlignment],
    heading_label: str,
    heading_slug: str,
) -> list[ComparisonBatch]:
    """Découpe les alignements en lots homogènes et ordonnés."""
    batches: list[ComparisonBatch] = []
    current_type = ""
    current: list[ChunkAlignment] = []

    def flush_current() -> None:
        nonlocal current, current_type
        if not current:
            return
        batch_index = len(batches)
        batches.append(
            ComparisonBatch(
                batch_id=f"b{batch_index:02d}",
                alignment_type=current_type,
                alignments=current,
                heading_label=heading_label,
                heading_slug=heading_slug,
                idx_offset=batch_index * 1000,
            )
        )
        current = []
        current_type = ""

    for alignment in alignments:
        alignment_type = alignment.alignment_type
        max_size = _batch_size_for_alignment_type(alignment_type)
        if current and (alignment_type != current_type or len(current) >= max_size):
            flush_current()
        current_type = alignment_type
        current.append(alignment)
    flush_current()
    return batches


def _reindex_changes(
    changes: list[dict[str, Any]],
    *,
    section_key: str,
    heading_slug: str,
    idx_offset: int,
) -> list[dict[str, Any]]:
    """Réindexe les changements fusionnés après appels LLM parallèles."""
    reindexed: list[dict[str, Any]] = []
    for local_idx, change in enumerate(changes, start=1):
        updated = dict(change)
        updated["change_id"] = f"{section_key}_{heading_slug}_change_{idx_offset + local_idx:03d}"
        reindexed.append(updated)
    return reindexed


def _compare_alignment_batch(
    *,
    client: Any,
    model: str,
    section_key: str,
    batch: ComparisonBatch,
) -> list[dict[str, Any]]:
    """Compare un lot d'alignements via un appel LLM."""
    text_t1, text_t2 = _format_alignments_for_prompt(batch.alignments)
    try:
        return _compare_texts_single_call(
            client=client,
            model=model,
            section_key=section_key,
            heading_label=batch.heading_label,
            heading_slug=batch.heading_slug,
            text_t1=text_t1,
            text_t2=text_t2,
            idx_offset=batch.idx_offset,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Batch comparison failed for {section_key}/{batch.heading_label}/{batch.batch_id}: {exc}"
        ) from exc


def _compare_alignment_batches(
    *,
    client: Any,
    model: str,
    section_key: str,
    batches: list[ComparisonBatch],
) -> list[dict[str, Any]]:
    """Compare les lots d'alignements en parallèle puis restitue l'ordre source."""
    if not batches:
        return []
    max_workers = min(_MAX_COMPARISON_LLM_WORKERS, len(batches))
    results_by_index: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _compare_alignment_batch,
                client=client,
                model=model,
                section_key=section_key,
                batch=batch,
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results_by_index[index] = future.result()

    merged: list[dict[str, Any]] = []
    for index in range(len(batches)):
        merged.extend(results_by_index.get(index, []))
    return merged


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
        raw = _call_json_completion(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu compares deux versions d'une section de rapport bancaire. "
                        "Identifie tous les changements observables paragraphe par paragraphe. "
                        "Ne masque pas les reformulations, les mises à jour de dates "
                        "ou les variations chiffrées : retourne-les comme changements, "
                        "le triage métier décidera ensuite de leur pertinence. "
                        "Lorsque le texte contient des blocs [c00], [c01], etc., "
                        "utilise ces bornes pour aligner les idées comparables, "
                        "mais ne recopie pas ces balises dans text_t1 ou text_t2. "
                        "Lorsque le texte contient des blocs [a00 | matched_strong], "
                        "[a00 | matched_weak], [a00 | ambiguous], [a00 | possible_added] "
                        "ou [a00 | possible_removed], ces alignements TF-IDF sont des "
                        "indices locaux dans la même sous-section, pas des verdicts. "
                        "Valide les cas faibles, ambigus, ajoutés ou supprimés possibles "
                        "avec les candidats fournis avant de décider added/removed/modified."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Compare ces deux versions et retourne uniquement du JSON.\n"
                        'Format: {"changes":[{"diff_type":"unchanged|modified|added|removed",'
                        '"text_t1":"texte du paragraphe en T1, vide si added",'
                        '"text_t2":"texte du paragraphe en T2, vide si removed",'
                        '"change_summary":"explication concise du changement"}]}.\n'
                        "unchanged = texte substantiellement identique, sans changement observable.\n"
                        "modified = texte correspondant changé, y compris reformulation, "
                        "mise à jour de date, variation chiffrée, changement de nuance "
                        "ou évolution substantielle.\n"
                        "added = idée nouvelle présente uniquement en T2.\n"
                        "removed = idée présente en T1, absente en T2.\n"
                        "Important : si le texte change mais que le sens semble identique, "
                        "retourne quand même diff_type='modified' avec un résumé indiquant "
                        "qu'il s'agit probablement d'une reformulation.\n"
                        f"Section: {section_key}\n\n"
                        f"=== T1 ===\n{text_t1}\n\n"
                        f"=== T2 ===\n{text_t2}\n"
                    ),
                },
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Section comparison failed for {section_key}/{heading_slug}: {exc}") from exc

    validated: list[dict[str, Any]] = []
    for local_idx, item in enumerate(raw.get("changes") or [], start=1):
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
        validated.append(
            {
                "change_id": f"{section_key}_{heading_slug}_change_{global_idx:03d}",
                "section_key": section_key,
                "subsection_heading": heading_label,
                "diff_type": diff_type,
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
        )
    return validated


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
    sans appel GPT. Les sections non vides sans ``###`` sont rejetées comme
    anomalie de qualité au lieu d'être comparées en bloc brut.
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

    has_subsections_t1 = any(heading != "__intro__" for heading, _body in subs_t1)
    has_subsections_t2 = any(heading != "__intro__" for heading, _body in subs_t2)

    if not has_subsections_t1 and not has_subsections_t2:
        raise TextAnalysisQualityError(
            f"Section non vide sans sous-sections ###: {section_key}"
        )

    pairs = _pair_subsections(subs_t1, subs_t2)

    # Phase 2 — Fuzzy rename resolution via GPT
    orphans_t1 = [h1 for h1, _b1, h2, _b2 in pairs if h2 is None and h1 is not None]
    orphans_t2 = [h2 for _h1, _b1, h2, _b2 in pairs if _h1 is None and h2 is not None]
    rename_matches = _gpt_match_orphan_headings(
        client=client,
        model=model,
        section_key=section_key,
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    # Build lookup: orphan_t1_heading → matched orphan_t2_heading (and vice versa)
    rename_t1_to_t2: dict[str, str] = {m["heading_t1"]: m["heading_t2"] for m in rename_matches}
    renamed_as_t2: set[str] = set(rename_t1_to_t2.values())

    # Augment pairs: replace (h1, body1, None, "") with (h1, body1, h2_matched, body2_matched)
    body_by_t2_heading: dict[str, str] = {h2: b2 for _, _, h2, b2 in pairs if h2 is not None}
    resolved_pairs: list[tuple[str | None, str, str | None, str]] = []
    for h1, body1, h2, body2 in pairs:
        if h2 is None and h1 is not None and h1 in rename_t1_to_t2:
            matched_h2 = rename_t1_to_t2[h1]
            matched_body2 = body_by_t2_heading.get(matched_h2, "")
            resolved_pairs.append((h1, body1, matched_h2, matched_body2))
        elif h1 is None and h2 is not None and h2 in renamed_as_t2:
            # Skip — already consumed by the T1 side above
            continue
        else:
            resolved_pairs.append((h1, body1, h2, body2))
    pairs = resolved_pairs

    all_changes: list[dict[str, Any]] = []
    global_idx = 1
    renamed_pairs: set[tuple[str, str]] = {(m["heading_t1"], m["heading_t2"]) for m in rename_matches}

    for h1, body1, h2, body2 in pairs:
        heading_label = h1 or h2 or "unknown"
        heading_slug = re.sub(r"[^\w]+", "_", _normalize_heading(heading_label))[:40].strip("_")

        # For renamed pairs, display as "T1 heading → T2 heading"
        is_renamed_pair = h1 is not None and h2 is not None and (h1, h2) in renamed_pairs
        if is_renamed_pair:
            all_changes.append(
                _synthetic_subsection_rename_change(
                    section_key=section_key,
                    heading_t1=h1,
                    heading_t2=h2,
                    idx=global_idx,
                )
            )
            global_idx += 1

        if h1 is not None and h2 is not None and (h1, h2) in renamed_pairs:
            heading_label = f"{h1} → {h2}"
            heading_slug = re.sub(r"[^\w]+", "_", _normalize_heading(h1))[:40].strip("_")

        if h2 is None:
            assert h1 is not None
            if not body1.strip():
                continue
            all_changes.append(
                _synthetic_subsection_change(
                    section_key=section_key,
                    diff_type="removed",
                    heading=h1,
                    body_t1=body1,
                    body_t2="",
                    idx=global_idx,
                )
            )
            global_idx += 1
            continue

        if h1 is None:
            assert h2 is not None
            if not body2.strip():
                continue
            all_changes.append(
                _synthetic_subsection_change(
                    section_key=section_key,
                    diff_type="added",
                    heading=h2,
                    body_t1="",
                    body_t2=body2,
                    idx=global_idx,
                )
            )
            global_idx += 1
            continue

        if not body1.strip() and not body2.strip():
            continue
        if not body1.strip():
            all_changes.append(
                _synthetic_subsection_change(
                    section_key=section_key,
                    diff_type="added",
                    heading=h2,
                    body_t1="",
                    body_t2=body2,
                    idx=global_idx,
                )
            )
            global_idx += 1
            continue
        if not body2.strip():
            all_changes.append(
                _synthetic_subsection_change(
                    section_key=section_key,
                    diff_type="removed",
                    heading=h1,
                    body_t1=body1,
                    body_t2="",
                    idx=global_idx,
                )
            )
            global_idx += 1
            continue

        alignments = _prepare_subsection_alignments(
            section_key=section_key,
            subsection_heading_t1=h1,
            subsection_heading_t2=h2,
            body_t1=body1,
            body_t2=body2,
        )
        batches = _build_comparison_batches(
            alignments=alignments,
            heading_label=heading_label,
            heading_slug=heading_slug,
        )
        subsection_changes = _compare_alignment_batches(
            client=client,
            model=model,
            section_key=section_key,
            batches=batches,
        )
        subsection_changes = _reindex_changes(
            subsection_changes,
            section_key=section_key,
            heading_slug=heading_slug,
            idx_offset=global_idx - 1,
        )
        all_changes.extend(subsection_changes)
        global_idx += len(subsection_changes)

    return all_changes
