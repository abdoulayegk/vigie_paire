"""Resolution des ancrages et metadonnees de preuves visuelles."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from vigie.extraction.section_taxonomy import canonicalize_section
from vigie.support.utils.matching_normalizer import (
    is_generic_title,
    normalize_for_matching,
    strip_temporal_expressions,
)
from vigie.support.utils.proof_rendering import normalize_proof_bbox


def _visual_sanity_meta(
    *,
    applied: bool,
    rejected_count: int,
    render_status: str,
    render_mode: str = "full",
) -> dict[str, Any]:
    """Construire le bloc de metadonnees de la verification visuelle."""
    return {
        "visual_sanity_applied": applied,
        "visual_sanity_rejected_count": int(rejected_count),
        "visual_sanity_scope": ["indicators", "footnotes", "tables"],
        "visual_sanity_render_mode": render_mode,
        "visual_sanity_render_status": render_status,
    }


def _normalize_table_anchor_section(value: Any) -> str:
    """Normaliser le nom de section pour l'ancrage visuel d'une table."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        normalized = canonicalize_section(raw)
    except Exception:
        normalized = raw
    return str(normalized or "").strip()


def _normalize_table_anchor_title(value: Any) -> str:
    """Normaliser le titre d'une table pour l'ancrage visuel."""
    raw = strip_temporal_expressions(str(value or ""), target="title", aggressive=True)
    return normalize_for_matching(raw, target="title")


def _snapshot_has_visual_render_anchor(snapshot: dict[str, Any]) -> bool:
    """Verifier qu'un snapshot peut etre rendu sur sa page PDF."""
    try:
        page = int(snapshot.get("page") or 0)
    except (TypeError, ValueError):
        return False
    return page > 0 and normalize_proof_bbox(snapshot.get("bbox")) is not None


def _normalized_anchor_values(
    snapshot: dict[str, Any],
    field: str,
    *,
    target: str,
) -> list[str]:
    """Normaliser une valeur scalaire ou une liste pour le score d'ancrage."""
    raw_value = snapshot.get(field)
    raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
    normalized: list[str] = []
    for value in raw_values:
        if target == "title" and is_generic_title(str(value or "")):
            continue
        text = (
            _normalize_table_anchor_title(value)
            if target == "title"
            else normalize_for_matching(str(value or ""), target=target)
        )
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _best_text_similarity(left: list[str], right: list[str]) -> float:
    """Retourner la meilleure similarite textuelle entre deux petits ensembles."""
    if not left or not right:
        return 0.0
    return max(SequenceMatcher(None, first, second).ratio() for first in left for second in right)


def _jaccard_anchor_values(left: list[str], right: list[str]) -> float:
    """Calculer le Jaccard de deux listes normalisees."""
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _resolve_visual_table_anchor(
    event_snapshot: dict[str, Any],
    opposite_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resoudre une ancre opposee avec plusieurs signaux et une marge.

    Le titre exact reste le signal le plus fort, mais un titre reformule ou
    absent peut etre compense par le resume, les indicateurs et les en-tetes.
    Aucun candidat n'est retenu si le score est faible ou ambigu.
    """
    event_section = _normalize_table_anchor_section(event_snapshot.get("section"))
    event_titles = _normalized_anchor_values(
        {
            "values": [
                event_snapshot.get("title"),
                event_snapshot.get("page_context_title"),
            ]
        },
        "values",
        target="title",
    )
    event_summaries = _normalized_anchor_values(
        event_snapshot,
        "table_summary",
        target="generic",
    )
    event_indicators = _normalized_anchor_values(
        event_snapshot,
        "indicators",
        target="indicator",
    )
    event_headers = _normalized_anchor_values(
        event_snapshot,
        "headers",
        target="header",
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in opposite_snapshots.values():
        if not _snapshot_has_visual_render_anchor(candidate):
            continue
        candidate_section = _normalize_table_anchor_section(candidate.get("section"))
        if event_section and candidate_section != event_section:
            continue

        candidate_titles = _normalized_anchor_values(
            {
                "values": [
                    candidate.get("title"),
                    candidate.get("page_context_title"),
                ]
            },
            "values",
            target="title",
        )
        candidate_summaries = _normalized_anchor_values(
            candidate,
            "table_summary",
            target="generic",
        )
        candidate_indicators = _normalized_anchor_values(
            candidate,
            "indicators",
            target="indicator",
        )
        candidate_headers = _normalized_anchor_values(
            candidate,
            "headers",
            target="header",
        )

        score = 0.0
        if set(event_titles) & set(candidate_titles):
            score += 6.0
        else:
            title_similarity = _best_text_similarity(
                event_titles,
                candidate_titles,
            )
            if title_similarity >= 0.72:
                score += 4.0 * title_similarity

        if set(event_summaries) & set(candidate_summaries):
            score += 4.0
        else:
            summary_similarity = _best_text_similarity(
                event_summaries,
                candidate_summaries,
            )
            if summary_similarity >= 0.76:
                score += 2.5 * summary_similarity

        indicator_overlap = _jaccard_anchor_values(
            event_indicators,
            candidate_indicators,
        )
        if indicator_overlap >= 0.20:
            score += 4.0 * indicator_overlap
        if event_indicators and candidate_indicators and event_indicators[0] == candidate_indicators[0]:
            score += 1.0

        header_overlap = _jaccard_anchor_values(
            event_headers,
            candidate_headers,
        )
        if header_overlap >= 0.25:
            score += 1.5 * header_overlap

        try:
            event_rows = int(event_snapshot.get("row_count") or 0)
            candidate_rows = int(candidate.get("row_count") or 0)
        except (TypeError, ValueError):
            event_rows = candidate_rows = 0
        if event_rows > 0 and candidate_rows > 0:
            size_ratio = min(event_rows, candidate_rows) / max(
                event_rows,
                candidate_rows,
            )
            if size_ratio >= 0.60:
                score += 0.5 * size_ratio

        if score > 0:
            scored.append((score, candidate))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 4.0 or best_score - second_score < 0.75:
        return None
    return best_candidate


def _infer_opposite_page_from_matched_pairs(
    event_snapshot: dict[str, Any],
    matched_pairs: list[dict[str, Any]],
    event_snapshots: dict[str, dict[str, Any]],
    opposite_snapshots: dict[str, dict[str, Any]],
    *,
    event_side: str,
) -> int | None:
    """Estimer la page opposee a partir des paires voisines deja appariees."""
    try:
        event_page = int(event_snapshot.get("page") or 0)
    except (TypeError, ValueError):
        return None
    if event_page < 1 or event_side not in {"previous", "current"}:
        return None

    event_id_key = "previous_table_id" if event_side == "previous" else "current_table_id"
    opposite_id_key = "current_table_id" if event_side == "previous" else "previous_table_id"
    page_map: dict[int, list[int]] = {}
    for pair in matched_pairs:
        event_id = str(pair.get(event_id_key, "") or "").strip()
        opposite_id = str(pair.get(opposite_id_key, "") or "").strip()
        event_anchor = event_snapshots.get(event_id, {})
        opposite_anchor = opposite_snapshots.get(opposite_id, {})
        try:
            event_anchor_page = int(event_anchor.get("page") or 0)
            opposite_anchor_page = int(opposite_anchor.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if event_anchor_page < 1 or opposite_anchor_page < 1:
            continue
        page_map.setdefault(event_anchor_page, []).append(opposite_anchor_page)

    anchors = sorted(
        (
            source_page,
            round(sum(target_pages) / len(target_pages)),
        )
        for source_page, target_pages in page_map.items()
        if target_pages
    )
    if not anchors:
        return None

    previous_anchor = next(
        (anchor for anchor in reversed(anchors) if anchor[0] <= event_page),
        None,
    )
    next_anchor = next(
        (anchor for anchor in anchors if anchor[0] >= event_page),
        None,
    )
    if previous_anchor and next_anchor:
        source_span = next_anchor[0] - previous_anchor[0]
        if source_span <= 0:
            return max(1, previous_anchor[1])
        progress = (event_page - previous_anchor[0]) / source_span
        inferred = previous_anchor[1] + progress * (next_anchor[1] - previous_anchor[1])
        return max(1, round(inferred))
    if previous_anchor:
        return max(1, event_page + previous_anchor[1] - previous_anchor[0])
    if next_anchor:
        return max(1, event_page + next_anchor[1] - next_anchor[0])
    return None
