"""Reconciliation des collisions de bbox produites par le locator pleine page."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from vigie.support.utils.indicator_cleaner import normalize_indicator_for_comparison


def _bbox_area(bbox: list[float]) -> float:
    """Calculer l'aire d'une boite englobante normalisee."""
    if not bbox or len(bbox) < 4:
        return 0.0
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return width * height


def _bbox_overlap_ratio(first: list[float], second: list[float]) -> float:
    """Calculer l'intersection rapportee a la plus petite des deux boites."""
    if len(first) < 4 or len(second) < 4:
        return 0.0
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    denominator = min(_bbox_area(first), _bbox_area(second))
    return intersection / denominator if denominator > 0 else 0.0


def _is_locator_merge_conflict(
    first_original: list[float],
    second_original: list[float],
    first_corrected: list[float],
    second_corrected: list[float],
) -> bool:
    """Detecter quand le locator fusionne deux blocs Docling distincts."""
    return bool(
        _bbox_overlap_ratio(first_corrected, second_corrected) >= 0.90
        and _bbox_overlap_ratio(first_original, second_original) < 0.20
    )


def _normalized_table_signals(values: list[str] | None) -> set[str]:
    """Construire des signaux semantiques tolerants aux variations OCR."""
    signals: set[str] = set()
    for value in list(values or []):
        raw = str(value or "").strip()
        if not raw:
            continue
        normalized = normalize_indicator_for_comparison(raw)
        if not normalized:
            normalized = re.sub(r"[^\w]+", " ", raw.casefold()).strip()
        if normalized:
            signals.add(normalized)
    return signals


def _signal_overlap(first: set[str], second: set[str]) -> float:
    """Mesurer le recouvrement par rapport au plus petit ensemble."""
    denominator = min(len(first), len(second))
    if denominator <= 0:
        return 0.0
    return len(first & second) / denominator


def _locator_tables_are_semantic_duplicates(
    first: Any,
    second: Any,
) -> bool:
    """Verifier si deux crops locator de meme region portent le meme tableau."""
    first_indicators = _normalized_table_signals(
        first.first_column_indicators_raw
        or first.first_column_indicators
    )
    second_indicators = _normalized_table_signals(
        second.first_column_indicators_raw
        or second.first_column_indicators
    )
    indicator_overlap = _signal_overlap(
        first_indicators,
        second_indicators,
    )
    if indicator_overlap < 0.80:
        return False

    first_headers = _normalized_table_signals(first.headers)
    second_headers = _normalized_table_signals(second.headers)
    header_overlap = _signal_overlap(first_headers, second_headers)
    first_title = _normalized_table_signals(
        [
            str(first.title or ""),
            str(first.debug_metrics.get("page_context_title") or ""),
        ]
    )
    second_title = _normalized_table_signals(
        [
            str(second.title or ""),
            str(second.debug_metrics.get("page_context_title") or ""),
        ]
    )
    title_overlap = _signal_overlap(first_title, second_title)
    return bool(header_overlap >= 0.60 or title_overlap >= 1.0)


def _locator_table_richness(table: Any) -> tuple[int, int, int, int]:
    """Classer deux extractions du meme crop en conservant la plus complete."""
    raw_indicators = [
        str(value or "").strip()
        for value in list(
            table.first_column_indicators_raw
            or table.first_column_indicators
            or []
        )
        if str(value or "").strip()
    ]
    headers = [
        str(value or "").strip()
        for value in list(table.headers or [])
        if str(value or "").strip()
    ]
    footnotes = [
        item
        for item in list(table.footnotes or [])
        if isinstance(item, dict)
        and any(str(value or "").strip() for value in item.values())
    ]
    return (
        len(raw_indicators),
        len(headers),
        len(footnotes),
        len(str(table.title or "").strip()),
    )


def _reconcile_on_demand_locator_merges(
    tables: list[Any],
) -> list[Any]:
    """Reduire les copies d'un meme tableau creees par le locator a la demande.

    Docling peut couper un tableau physique en deux blocs verticaux. Si le
    locator pleine page rattache ces blocs distincts a la meme bbox, chaque
    worker extrait ensuite le meme tableau complet. Un bloc voisin peut aussi
    rester sur sa bbox Docling tout en etant entierement contenu dans la bbox
    corrigee. La reconciliation exige au moins une sortie
    ``page_context_locator``, des bboxes Docling distinctes et un fort
    recouvrement semantique.
    """
    supported_sources = {"docling", "page_context_locator"}
    reconciled: list[Any] = []
    for table in tables:
        metrics = (
            table.debug_metrics
            if isinstance(table.debug_metrics, dict)
            else {}
        )
        table_source = str(metrics.get("bbox_source") or "")
        if table_source not in supported_sources:
            reconciled.append(table)
            continue
        original_bbox = metrics.get("bbox_original")
        final_bbox = table.bbox or metrics.get("bbox_final")
        if not (
            isinstance(original_bbox, (list, tuple))
            and len(original_bbox) == 4
            and isinstance(final_bbox, (list, tuple))
            and len(final_bbox) == 4
        ):
            reconciled.append(table)
            continue

        duplicate_index: int | None = None
        for position, existing in enumerate(reconciled):
            existing_metrics = (
                existing.debug_metrics
                if isinstance(existing.debug_metrics, dict)
                else {}
            )
            existing_source = str(existing_metrics.get("bbox_source") or "")
            existing_original = existing_metrics.get("bbox_original")
            existing_final = (
                existing.bbox
                or existing_metrics.get("bbox_final")
            )
            if (
                existing.page_number != table.page_number
                or existing_source not in supported_sources
                or "page_context_locator"
                not in {existing_source, table_source}
                or not isinstance(existing_original, (list, tuple))
                or len(existing_original) != 4
                or not isinstance(existing_final, (list, tuple))
                or len(existing_final) != 4
            ):
                continue
            if not _is_locator_merge_conflict(
                list(existing_original),
                list(original_bbox),
                list(existing_final),
                list(final_bbox),
            ):
                continue
            if not _locator_tables_are_semantic_duplicates(existing, table):
                continue
            duplicate_index = position
            break

        if duplicate_index is None:
            reconciled.append(table)
            continue

        existing = reconciled[duplicate_index]
        preferred = max(
            (existing, table),
            key=_locator_table_richness,
        )
        existing_metrics = dict(existing.debug_metrics or {})
        geometry_table = (
            existing
            if existing_metrics.get("bbox_source")
            == "page_context_locator"
            else table
        )
        geometry_metrics = dict(geometry_table.debug_metrics or {})
        preferred_metrics = dict(preferred.debug_metrics or {})
        for geometry_key in (
            "bbox_original",
            "bbox_final",
            "bbox_source",
            "bbox_confidence",
            "bbox_verified",
            "page_context_title",
            "page_context_continuation",
            "page_context_table_count",
        ):
            if geometry_key in geometry_metrics:
                preferred_metrics[geometry_key] = geometry_metrics[geometry_key]
        merged_ids = list(
            existing_metrics.get(
                "locator_merged_table_ids",
                [existing.table_id],
            )
        )
        if table.table_id not in merged_ids:
            merged_ids.append(table.table_id)
        original_bboxes = list(
            existing_metrics.get(
                "locator_original_bboxes",
                [list(existing_metrics.get("bbox_original") or [])],
            )
        )
        normalized_original = list(original_bbox)
        if normalized_original not in original_bboxes:
            original_bboxes.append(normalized_original)
        preferred_metrics.update(
            {
                "locator_merge_collapsed": True,
                "locator_merged_table_ids": merged_ids,
                "locator_original_bboxes": original_bboxes,
            }
        )
        reconciled[duplicate_index] = replace(
            preferred,
            table_id=existing.table_id,
            bbox=geometry_table.bbox,
            bbox_top=geometry_table.bbox_top,
            tables_on_page=geometry_table.tables_on_page,
            debug_metrics=preferred_metrics,
        )

    return reconciled
