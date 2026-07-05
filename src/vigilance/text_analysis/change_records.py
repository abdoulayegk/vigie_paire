"""Factories pour les changements textuels synthétiques."""

from __future__ import annotations

import re
from typing import Any

from .models import _SubsectionRecord
from .subsection_units import _hierarchy_path_for_subsection
from .text_normalization import _clamp_confidence, _normalize_heading, _sanitize_semantic_text
from .text_topics import _canonical_topic_for_text


def _synthetic_subsection_change(
    *,
    section_key: str,
    diff_type: str,
    heading: str,
    body_t1: str,
    body_t2: str,
    idx: int,
    alignment_type: str | None = None,
    canonical_topic: str | None = None,
    alignment_confidence: float = 0.0,
    previous_subsection_heading: str | None = None,
    current_subsection_heading: str | None = None,
    restructure_group_id: str | None = None,
) -> dict[str, Any]:
    """Crée un changement pour une sous-section entièrement ajoutée ou supprimée."""
    slug = re.sub(r"[^\w]+", "_", _normalize_heading(heading))[:40].strip("_")
    label = "ajoutée" if diff_type == "added" else "supprimée"
    payload = {
        "change_id": f"{section_key}_{slug}_change_{idx:03d}",
        "section_key": section_key,
        "subsection_heading": heading,
        "diff_type": diff_type,
        "semantic_text_t1": _sanitize_semantic_text(body_t1),
        "semantic_text_t2": _sanitize_semantic_text(body_t2),
        "source_text_t1": body_t1,
        "source_text_t2": body_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown",
        "source_resolution_t2": "markdown",
        "evidence_t1": {"pages": [], "snippet": body_t1[:400]},
        "evidence_t2": {"pages": [], "snippet": body_t2[:400]},
        "change_summary": f"Sous-section {label}: {heading}",
    }
    payload.update(
        {
            "alignment_type": alignment_type or ("true_added" if diff_type == "added" else "true_removed"),
            "previous_subsection_heading": previous_subsection_heading
            if previous_subsection_heading is not None
            else (heading if diff_type == "removed" else ""),
            "current_subsection_heading": current_subsection_heading
            if current_subsection_heading is not None
            else (heading if diff_type == "added" else ""),
            "canonical_topic": canonical_topic or _canonical_topic_for_text(heading, body_t1 or body_t2),
            "alignment_confidence": round(_clamp_confidence(alignment_confidence), 4),
        }
    )
    if restructure_group_id:
        payload["restructure_group_id"] = restructure_group_id
    previous_heading = str(payload.get("previous_subsection_heading") or "")
    current_heading = str(payload.get("current_subsection_heading") or "")
    if previous_heading:
        payload["previous_hierarchy_path"] = _hierarchy_path_for_subsection(section_key, previous_heading)
    if current_heading:
        payload["current_hierarchy_path"] = _hierarchy_path_for_subsection(section_key, current_heading)
    return payload


def _synthetic_narrative_unit_change(
    *,
    section_key: str,
    diff_type: str,
    heading: str,
    body_t1: str,
    body_t2: str,
    idx: int,
    previous_unit_index: int | None = None,
    current_unit_index: int | None = None,
    alignment_type: str | None = None,
    canonical_topic: str | None = None,
    alignment_confidence: float = 0.0,
    previous_subsection_heading: str | None = None,
    current_subsection_heading: str | None = None,
    restructure_group_id: str | None = None,
) -> dict[str, Any]:
    """Crée un changement pour une unité narrative ajoutée ou supprimée."""
    unit_index = previous_unit_index if diff_type == "removed" else current_unit_index
    slug_source = f"{heading}_unit_{unit_index or idx}"
    slug = re.sub(r"[^\w]+", "_", _normalize_heading(slug_source))[:40].strip("_")
    label = "ajoutée" if diff_type == "added" else "supprimée"
    payload = {
        "change_id": f"{section_key}_{slug}_change_{idx:03d}",
        "section_key": section_key,
        "subsection_heading": heading,
        "diff_type": diff_type,
        "semantic_text_t1": _sanitize_semantic_text(body_t1),
        "semantic_text_t2": _sanitize_semantic_text(body_t2),
        "source_text_t1": body_t1,
        "source_text_t2": body_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown",
        "source_resolution_t2": "markdown",
        "evidence_t1": {"pages": [], "snippet": body_t1[:400]},
        "evidence_t2": {"pages": [], "snippet": body_t2[:400]},
        "change_summary": f"Unité narrative {label}: {heading}",
        "alignment_type": alignment_type or ("unit_added" if diff_type == "added" else "unit_removed"),
        "previous_subsection_heading": previous_subsection_heading
        if previous_subsection_heading is not None
        else (heading if diff_type == "removed" else ""),
        "current_subsection_heading": current_subsection_heading
        if current_subsection_heading is not None
        else (heading if diff_type == "added" else ""),
        "canonical_topic": canonical_topic or _canonical_topic_for_text(heading, body_t1 or body_t2),
        "alignment_confidence": round(_clamp_confidence(alignment_confidence), 4),
    }
    if previous_unit_index is not None:
        payload["previous_unit_index"] = previous_unit_index
    if current_unit_index is not None:
        payload["current_unit_index"] = current_unit_index
    if restructure_group_id:
        payload["restructure_group_id"] = restructure_group_id
    previous_heading = str(payload.get("previous_subsection_heading") or "")
    current_heading = str(payload.get("current_subsection_heading") or "")
    if previous_heading:
        payload["previous_hierarchy_path"] = _hierarchy_path_for_subsection(section_key, previous_heading)
    if current_heading:
        payload["current_hierarchy_path"] = _hierarchy_path_for_subsection(section_key, current_heading)
    return payload


def _synthetic_subsection_rename_change(
    *,
    section_key: str,
    heading_t1: str,
    heading_t2: str,
    idx: int,
    alignment_type: str = "near_heading",
    canonical_topic: str | None = None,
    alignment_confidence: float = 1.0,
    restructure_group_id: str | None = None,
) -> dict[str, Any]:
    """Crée un changement explicite pour une sous-section renommée."""
    slug_source = f"{heading_t1}_{heading_t2}"
    slug = re.sub(r"[^\w]+", "_", _normalize_heading(slug_source))[:40].strip("_")
    summary = f"Sous-section renommée: {heading_t1} -> {heading_t2}"
    payload = {
        "change_id": f"{section_key}_{slug}_change_{idx:03d}",
        "section_key": section_key,
        "subsection_heading": f"{heading_t1} → {heading_t2}",
        "previous_subsection_heading": heading_t1,
        "current_subsection_heading": heading_t2,
        "diff_type": "renamed",
        "semantic_text_t1": _sanitize_semantic_text(heading_t1),
        "semantic_text_t2": _sanitize_semantic_text(heading_t2),
        "source_text_t1": heading_t1,
        "source_text_t2": heading_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown_heading",
        "source_resolution_t2": "markdown_heading",
        "evidence_t1": {"pages": [], "snippet": heading_t1},
        "evidence_t2": {"pages": [], "snippet": heading_t2},
        "change_summary": summary,
        "alignment_type": alignment_type,
        "canonical_topic": canonical_topic or _canonical_topic_for_text(heading_t1, heading_t2),
        "alignment_confidence": round(_clamp_confidence(alignment_confidence), 4),
    }
    if restructure_group_id:
        payload["restructure_group_id"] = restructure_group_id
    payload["previous_hierarchy_path"] = _hierarchy_path_for_subsection(section_key, heading_t1)
    payload["current_hierarchy_path"] = _hierarchy_path_for_subsection(section_key, heading_t2)
    return payload


def _build_unpaired_changes(
    *,
    section_key: str,
    diff_type: str,
    record: _SubsectionRecord,
    idx: int,
) -> tuple[list[dict[str, Any]], int]:
    """Produit un added/removed seulement après échec du plan d'alignement GPT."""
    body = record.body
    change = _synthetic_subsection_change(
        section_key=section_key,
        diff_type=diff_type,
        heading=record.heading,
        body_t1=body if diff_type == "removed" else "",
        body_t2=body if diff_type == "added" else "",
        idx=idx,
        canonical_topic=record.canonical_topic,
    )
    return [change], idx + 1
