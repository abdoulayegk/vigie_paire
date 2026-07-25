"""Modèle et réparations des segments issus du Markdown Docling."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from vigilance.text_analysis.boundary_repair import (
    BoundaryDisposition,
    RepairableBlock,
    classify_boundary,
)
from vigilance.text_analysis.canonical_cleanup import adjacent_duplicate_key


@dataclass(slots=True)
class DoclingSegment:
    """Segment parsé depuis le Markdown natif Docling."""

    kind: str
    text: str
    heading_level: int = 0
    follows_table: bool = False
    page: int | None = None
    bbox_norm: list[float] | None = None
    source_block_id: str | None = None
    source_block_type: str | None = None
    list_marker: str | None = None
    list_indent: int = 0


def _segment_audit_text(segment: DoclingSegment) -> str:
    """Restitue l'identité source d'un segment, marqueur ordonné compris."""
    marker = str(segment.list_marker or "").strip()
    text = str(segment.text or "").strip()
    if segment.kind == "list_item" and re.fullmatch(r"\d{1,3}[.)]", marker):
        return f"{marker} {text}".strip()
    return text


def _deduplicate_adjacent_docling_segments(
    segments: list[DoclingSegment],
    *,
    audit_events: list[dict[str, Any]] | None = None,
) -> list[DoclingSegment]:
    """Retire uniquement les doublons textuels strictement adjacents."""
    deduplicated: list[DoclingSegment] = []
    for segment in segments:
        if deduplicated:
            previous = deduplicated[-1]
            comparable_kinds = {"paragraph", "list_item"}
            if (
                segment.kind == previous.kind
                and segment.kind in comparable_kinds
                and adjacent_duplicate_key(segment.text)
                and adjacent_duplicate_key(segment.text) == adjacent_duplicate_key(previous.text)
            ):
                if audit_events is not None:
                    audit_events.append(
                        {
                            "action": "remove",
                            "reason": "adjacent_duplicate",
                            "kind": segment.kind,
                            "text": _segment_audit_text(segment),
                            "page": segment.page,
                            "source_block_id": segment.source_block_id,
                        }
                    )
                continue
        deduplicated.append(segment)
    return deduplicated


def _merge_docling_segments(previous: DoclingSegment, current: DoclingSegment) -> DoclingSegment:
    """Concatène deux fragments sans réécriture ni perte de caractères."""
    bbox = previous.bbox_norm
    if previous.page == current.page and previous.bbox_norm and current.bbox_norm:
        bbox = [
            min(previous.bbox_norm[0], current.bbox_norm[0]),
            min(previous.bbox_norm[1], current.bbox_norm[1]),
            max(previous.bbox_norm[2], current.bbox_norm[2]),
            max(previous.bbox_norm[3], current.bbox_norm[3]),
        ]
    source_ids = [
        value
        for value in (previous.source_block_id, current.source_block_id)
        if value
    ]
    return replace(
        previous,
        text=f"{previous.text.rstrip()} {current.text.lstrip()}",
        bbox_norm=bbox,
        source_block_id="+".join(source_ids) or None,
    )


def _repair_docling_segment_boundaries(
    segments: list[DoclingSegment],
    *,
    boundary_validator: Any | None = None,
    audit_events: list[dict[str, Any]] | None = None,
) -> list[DoclingSegment]:
    """Répare les continuations certaines et soumet les ambiguïtés à Vision."""
    repaired: list[DoclingSegment] = []
    for segment in segments:
        if not repaired:
            repaired.append(segment)
            continue
        previous = repaired[-1]
        page_gap = (
            abs(int(segment.page) - int(previous.page))
            if segment.page is not None and previous.page is not None
            else 0
        )
        decision = classify_boundary(
            RepairableBlock(kind=previous.kind, text=previous.text),
            RepairableBlock(
                kind=segment.kind,
                text=segment.text,
                hard_boundary_before=page_gap > 1,
            ),
        )
        should_merge = decision.disposition is BoundaryDisposition.MERGE
        vision_payload: dict[str, Any] | None = None
        if (
            decision.disposition is BoundaryDisposition.AMBIGUOUS
            and boundary_validator is not None
        ):
            try:
                validation = boundary_validator.validate(previous, segment)
                vision_payload = (
                    validation.model_dump()
                    if hasattr(validation, "model_dump")
                    else dict(validation)
                )
                should_merge = bool(vision_payload.get("apply_merge"))
            except Exception as exc:  # noqa: BLE001
                vision_payload = {
                    "apply_merge": False,
                    "status": "error_fail_closed",
                    "error": str(exc),
                }

        if should_merge:
            merged = _merge_docling_segments(previous, segment)
            repaired[-1] = merged
            if audit_events is not None:
                event: dict[str, Any] = {
                    "action": "merge",
                    "reason": (
                        "vision_confirmed_same_sentence"
                        if vision_payload is not None
                        else decision.reason
                    ),
                    "previous_text": previous.text,
                    "next_text": segment.text,
                    "result_text": merged.text,
                    "previous_page": previous.page,
                    "next_page": segment.page,
                }
                if vision_payload is not None:
                    event["vision"] = vision_payload
                audit_events.append(event)
            continue

        if decision.disposition is BoundaryDisposition.AMBIGUOUS and audit_events is not None:
            event = {
                "action": "keep_separate",
                "reason": decision.reason,
                "previous_text": previous.text,
                "next_text": segment.text,
                "previous_page": previous.page,
                "next_page": segment.page,
                "status": "ambiguous_fail_closed",
            }
            if vision_payload is not None:
                event["vision"] = vision_payload
            audit_events.append(event)
        repaired.append(segment)
    return repaired


def _repair_nonadjacent_dangling_boundaries(
    segments: list[DoclingSegment],
    *,
    boundary_validator: Any | None,
    audit_events: list[dict[str, Any]] | None = None,
    search_window: int = 12,
) -> list[DoclingSegment]:
    """Soumet à Vision une continuation éloignée par un ordre Docling douteux."""
    if boundary_validator is None:
        return segments

    repaired = list(segments)
    remove_indexes: set[int] = set()
    for previous_index, previous in enumerate(repaired):
        if previous_index in remove_indexes or previous.kind != "paragraph":
            continue
        immediate_next = (
            repaired[previous_index + 1]
            if previous_index + 1 < len(repaired)
            else None
        )
        if immediate_next is None or immediate_next.kind == "paragraph":
            continue

        candidates: list[tuple[int, DoclingSegment]] = []
        upper_bound = min(len(repaired), previous_index + search_window + 1)
        for current_index in range(previous_index + 2, upper_bound):
            current = repaired[current_index]
            if current_index in remove_indexes or current.kind != "paragraph":
                continue
            page_gap = (
                abs(int(current.page) - int(previous.page))
                if current.page is not None and previous.page is not None
                else 0
            )
            if page_gap > 1:
                continue
            decision = classify_boundary(
                RepairableBlock(kind="paragraph", text=previous.text),
                RepairableBlock(kind="paragraph", text=current.text),
            )
            if decision.disposition is BoundaryDisposition.MERGE:
                candidates.append((current_index, current))

        # Plusieurs reprises possibles rendent l'association elle-même ambiguë.
        if len(candidates) != 1:
            continue
        current_index, current = candidates[0]
        validation = boundary_validator.validate(previous, current)
        vision_payload = validation.model_dump()
        if not validation.apply_merge:
            if audit_events is not None:
                audit_events.append(
                    {
                        "action": "keep_separate",
                        "reason": "nonadjacent_boundary_vision_fail_closed",
                        "previous_text": previous.text,
                        "next_text": current.text,
                        "previous_page": previous.page,
                        "next_page": current.page,
                        "status": "ambiguous_fail_closed",
                        "vision": vision_payload,
                    }
                )
            continue

        merged = _merge_docling_segments(previous, current)
        repaired[previous_index] = merged
        remove_indexes.add(current_index)
        if audit_events is not None:
            audit_events.append(
                {
                    "action": "merge",
                    "reason": "vision_confirmed_nonadjacent_same_sentence",
                    "previous_text": previous.text,
                    "next_text": current.text,
                    "result_text": merged.text,
                    "previous_page": previous.page,
                    "next_page": current.page,
                    "vision": vision_payload,
                }
            )

    return [
        segment
        for index, segment in enumerate(repaired)
        if index not in remove_indexes
    ]
