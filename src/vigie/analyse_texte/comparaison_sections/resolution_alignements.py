"""Resolution et materialisation des decisions d alignement."""

from __future__ import annotations

import re
from typing import Any

from vigie.analyse_texte.chunk_alignment import ChunkAlignment
from vigie.analyse_texte.chunking import TextChunk
from vigie.analyse_texte.comparaison_sections.preparation_lots import (
    _atomic_unit_metadata,
)
from vigie.analyse_texte.normalization import _sanitize_semantic_text


def _normalize_for_alignment_contains(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _coerce_text_to_chunk(text: str, chunk: TextChunk | None) -> str | None:
    """Ramène le texte LLM au périmètre exact d'un chunk, sinon invalide."""
    value = str(text or "").strip()
    if chunk is None:
        return "" if not value else None
    if not value:
        return ""
    chunk_text = chunk.text.strip()
    if value in chunk_text:
        return value

    normalized_value = _normalize_for_alignment_contains(value)
    normalized_chunk = _normalize_for_alignment_contains(chunk_text)
    if not normalized_value:
        return ""
    if normalized_value == normalized_chunk:
        return chunk_text
    if normalized_chunk and normalized_chunk in normalized_value:
        return chunk_text
    if normalized_value in normalized_chunk:
        return chunk_text
    return None


_SEMANTIC_ALIGNMENT_DECISIONS = frozenset(
    {"same_disclosure", "distinct_disclosures", "moved_text", "uncertain"}
)


def _resolved_alignment_decision(change: dict[str, Any], alignment: ChunkAlignment) -> str:
    """Normalizes the first GPT call's semantic decision conservatively."""
    decision = str(change.get("alignment_decision") or "").strip().lower()
    if decision in _SEMANTIC_ALIGNMENT_DECISIONS:
        return decision
    # Cached / legacy responses do not have the new field.  Preserve the
    # previous conservative handling only for genuinely ambiguous matches.
    if alignment.alignment_type == "ambiguous":
        return "uncertain"
    return "same_disclosure"


def _resolved_alignment_confidence(change: dict[str, Any], decision: str) -> str:
    confidence = str(change.get("alignment_confidence") or "").strip().lower()
    if confidence in {"high", "medium", "low"}:
        return confidence
    return "low" if decision == "uncertain" else "medium"


def _attach_alignment_metadata(
    changes: list[dict[str, Any]],
    alignments: list[ChunkAlignment],
) -> list[dict[str, Any]]:
    """Valide les changements LLM et les borne à leur alignment/chunk source."""
    alignment_by_id = {alignment.alignment_id: alignment for alignment in alignments}
    scoped: list[dict[str, Any]] = []
    for change in changes:
        alignment_id = str(change.get("alignment_id") or "").strip()
        alignment = alignment_by_id.get(alignment_id)
        if alignment is None:
            continue

        text_t1 = _coerce_text_to_chunk(str(change.get("source_text_t1") or ""), alignment.chunk_t1)
        text_t2 = _coerce_text_to_chunk(str(change.get("source_text_t2") or ""), alignment.chunk_t2)
        if text_t1 is None or text_t2 is None:
            continue

        diff_type = str(change.get("diff_type") or "").lower()
        if diff_type in {"unchanged", "modified"} and not (text_t1 and text_t2):
            continue
        if diff_type == "added" and not text_t2:
            continue
        if diff_type == "removed" and not text_t1:
            continue

        scoped_change = dict(change)
        alignment_decision = _resolved_alignment_decision(scoped_change, alignment)
        scoped_change.update(
            {
                "source_scope": "chunk",
                "alignment_id": alignment.alignment_id,
                "alignment_type": alignment.alignment_type,
                "chunk_id_t1": alignment.chunk_t1.chunk_id if alignment.chunk_t1 else None,
                "chunk_id_t2": alignment.chunk_t2.chunk_id if alignment.chunk_t2 else None,
                "source_text_t1": text_t1,
                "source_text_t2": text_t2,
                "semantic_text_t1": _sanitize_semantic_text(text_t1),
                "semantic_text_t2": _sanitize_semantic_text(text_t2),
                "evidence_t1": {"pages": [], "snippet": text_t1[:400]},
                "evidence_t2": {"pages": [], "snippet": text_t2[:400]},
                "alignment_decision": alignment_decision,
                "alignment_confidence": _resolved_alignment_confidence(
                    scoped_change, alignment_decision
                ),
                "alignment_rationale": str(scoped_change.get("alignment_rationale") or "").strip(),
                "alignment_reason": alignment.reason,
                "tfidf_score": alignment.tfidf_score,
                "embedding_score": alignment.embedding_score,
                **_atomic_unit_metadata(alignment.chunk_t1, alignment.chunk_t2),
            }
        )
        scoped.append(scoped_change)
    return scoped


def _format_sub_items_breakdown(text: str, *, prefix: str = "Sous-éléments et clauses spécifiques retirés :") -> str:
    """Extrait les puces ou sous-phrases clés d'un bloc de texte supprimé pour lister les détails."""
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    items: list[str] = []

    for line in lines:
        cleaned = re.sub(r"^(?:[•\-*]|\d+[\.\)])\s*", "", line).strip()
        if len(cleaned) >= 15 and cleaned not in items:
            items.append(cleaned)
            if len(items) >= 5:
                break

    if len(items) <= 1:
        sentences = [s.strip() for s in re.split(r"[.!?]+", raw_text) if len(s.strip()) >= 15]
        if len(sentences) > 1:
            items = sentences[:4]

    if not items or len(items) <= 1:
        return ""

    bullet_list = "\n".join(f"  • {item}" for item in items)
    return f"\n\n{prefix}\n{bullet_list}"


def _materialize_semantic_alignment_decisions(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turns a GPT-confirmed distinct pairing into separate source changes.

    A lexical matcher can put two distinct events in one provisional pair.  If
    the comparison model explicitly confirms that they are distinct, retaining
    a single ``modified`` card would still imply a false correspondence.  Two
    one-sided records preserve the original evidence for the AMF triage.
    """
    materialized: list[dict[str, Any]] = []
    for change in changes:
        decision = str(change.get("alignment_decision") or "").strip().lower()
        text_t1 = str(change.get("source_text_t1") or "").strip()
        text_t2 = str(change.get("source_text_t2") or "").strip()
        if decision != "distinct_disclosures" or not text_t1 or not text_t2:
            materialized.append(change)
            continue

        rationale = str(change.get("alignment_rationale") or "").strip()
        base_summary = str(change.get("change_summary") or "").strip()

        sub_items_rem = _format_sub_items_breakdown(text_t1, prefix="Sous-éléments et clauses spécifiques retirés :")
        sub_items_add = _format_sub_items_breakdown(text_t2, prefix="Sous-éléments et clauses spécifiques ajoutés :")

        rationale_rem = (rationale + sub_items_rem).strip()
        rationale_add = (rationale + sub_items_add).strip()

        removed = dict(change)
        removed.update(
            {
                "diff_type": "removed",
                "alignment_id": f"{change.get('alignment_id')}:removed",
                "alignment_type": "semantic_distinct",
                "semantic_alignment_group_id": str(change.get("alignment_id") or ""),
                "source_text_t2": "",
                "semantic_text_t2": "",
                "evidence_t2": {"pages": [], "snippet": ""},
                "alignment_rationale": rationale_rem,
                "change_summary": (
                    f"Divulgation distincte retirée. {rationale_rem or base_summary}".strip()
                ),
            }
        )
        added = dict(change)
        added.update(
            {
                "diff_type": "added",
                "alignment_id": f"{change.get('alignment_id')}:added",
                "alignment_type": "semantic_distinct",
                "semantic_alignment_group_id": str(change.get("alignment_id") or ""),
                "source_text_t1": "",
                "semantic_text_t1": "",
                "evidence_t1": {"pages": [], "snippet": ""},
                "alignment_rationale": rationale_add,
                "change_summary": (
                    f"Divulgation distincte ajoutée. {rationale_add or base_summary}".strip()
                ),
            }
        )
        materialized.extend([removed, added])
    return materialized


def _deduplicate_alignment_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conserve une carte par alignment, même si le LLM liste plusieurs détails.

    Les détails restent dans le résumé concaténé ; le texte source demeure le
    chunk unique auquel ils se rapportent. Cela évite de répéter la même paire
    de paragraphes dans plusieurs cartes Dash.
    """
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    ordered_keys: list[tuple[str, str, str, str, str]] = []
    for change in changes:
        key = (
            str(change.get("section_key") or ""),
            str(change.get("subsection_heading") or ""),
            str(change.get("alignment_id") or ""),
            str(change.get("chunk_id_t1") or ""),
            str(change.get("chunk_id_t2") or ""),
        )
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(change)

    deduplicated: list[dict[str, Any]] = []
    for key in ordered_keys:
        group = grouped[key]
        if len(group) == 1 or not key[2]:
            deduplicated.extend(group)
            continue

        representative = next(
            (
                change
                for change in group
                if str(change.get("source_text_t1") or "").strip()
                and str(change.get("source_text_t2") or "").strip()
            ),
            group[0],
        )
        merged = dict(representative)
        summaries: list[str] = []
        for change in group:
            summary = str(change.get("change_summary") or "").strip()
            if summary and summary not in summaries:
                summaries.append(summary)
        merged["change_summary"] = " ; ".join(summaries)
        if str(merged.get("source_text_t1") or "").strip() and str(merged.get("source_text_t2") or "").strip():
            merged["diff_type"] = "modified"
        deduplicated.append(merged)
    return deduplicated
