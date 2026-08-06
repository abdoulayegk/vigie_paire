"""Contexte de mise en page pour les extensions dynamiques de recadrage.

Utilise les bounding boxes de tableaux detectes par Docling pour calculer les
extensions optimales haut/bas de chaque recadrage, en evitant les collisions
avec les tableaux voisins tout en capturant titres et notes de bas de page.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PageBboxEntry = tuple[int, list[float]]  # (table_index, [l, t, r, b])


def build_page_table_map(
    vision_items: list[tuple[int, int, list[float] | None, str, str | None]],
) -> dict[int, list[PageBboxEntry]]:
    """Regroupe les bboxes de tableaux par page, tries de haut en bas.

    Args:
        vision_items: Liste de (idx, page_num, bbox_norm, table_id, ref_text).

    Returns:
        ``{page_num: [(idx, [l, t, r, b]), ...]}`` trie par ``t`` (haut).
    """
    by_page: dict[int, list[PageBboxEntry]] = {}
    for idx, page_num, bbox, _table_id, _ref in vision_items:
        if bbox and len(bbox) >= 4:
            by_page.setdefault(page_num, []).append((idx, bbox))
    # Sort each page's tables by vertical position (top coordinate)
    for page_num in by_page:
        by_page[page_num].sort(key=lambda entry: entry[1][1])  # sort by t
    return by_page


def clamp_variant_crop_to_neighbors(
    table_idx: int,
    page_num: int,
    table_bbox: list[float],
    page_table_map: dict[int, list[PageBboxEntry]],
    *,
    bbox_override: list[float] | None = None,
    bottom_extension: float = 0.0,
    top_extension: float = 0.0,
    safety_margin: float = 0.005,
) -> tuple[list[float], float, float]:
    """Limiter un recadrage de secours aux tableaux voisins de la page.

    Les variantes peuvent agrandir la bbox ou ses extensions, mais ne doivent
    jamais traverser la marge de securite precedant le tableau suivant ou
    suivant le tableau precedent.
    """
    base = [float(value) for value in table_bbox[:4]]
    candidate = [float(value) for value in (bbox_override or table_bbox)[:4]]
    if len(base) < 4 or len(candidate) < 4:
        return base, max(0.0, bottom_extension), max(0.0, top_extension)

    left, top, right, bottom = candidate
    left = max(0.0, min(left, 1.0))
    top = max(0.0, min(top, 1.0))
    right = max(left, min(right, 1.0))
    bottom = max(top, min(bottom, 1.0))

    page_tables = page_table_map.get(page_num, [])
    current_pos = next(
        (position for position, (idx, _bbox) in enumerate(page_tables) if idx == table_idx),
        None,
    )
    upper_boundary = 0.0
    lower_boundary = 1.0
    if current_pos is not None:
        if current_pos > 0:
            upper_boundary = float(page_tables[current_pos - 1][1][3])
        if current_pos < len(page_tables) - 1:
            lower_boundary = float(page_tables[current_pos + 1][1][1])

    if current_pos is not None and current_pos > 0:
        top = max(top, upper_boundary + safety_margin)
    if current_pos is not None and current_pos < len(page_tables) - 1:
        bottom = min(bottom, lower_boundary - safety_margin)

    if right <= left or bottom <= top:
        left, top, right, bottom = base

    safe_top_extension = min(
        max(0.0, float(top_extension)),
        max(0.0, top - upper_boundary - safety_margin),
    )
    safe_bottom_extension = min(
        max(0.0, float(bottom_extension)),
        max(0.0, lower_boundary - bottom - safety_margin),
    )
    return (
        [left, top, right, bottom],
        safe_bottom_extension,
        safe_top_extension,
    )


def compute_dynamic_extensions(
    table_idx: int,
    page_num: int,
    table_bbox: list[float],
    page_table_map: dict[int, list[PageBboxEntry]],
    *,
    default_bottom: float = 0.12,
    default_top: float = 0.03,
    page_bottom_margin: float = 0.05,
    title_proximity_threshold: float = 0.08,
    min_gap_for_footnotes: float = 0.02,
) -> tuple[float, float]:
    """Calcule les extensions dynamiques haut et bas pour un recadrage de tableau.

    Args:
        table_idx: Index du tableau courant dans la liste vision_items.
        page_num: Numero de page du tableau courant.
        table_bbox: Bbox normalisee [l, t, r, b] du tableau courant.
        page_table_map: Sortie de :func:`build_page_table_map`.
        default_bottom: Extension bas par defaut (config).
        default_top: Extension haut par defaut (config).
        page_bottom_margin: Marge de securite depuis le bas de la page (normalisee).
        title_proximity_threshold: Ecart maximal au-dessus du tableau pour chercher un titre.
        min_gap_for_footnotes: Ecart minimal a laisser entre les tableaux.

    Returns:
        ``(top_extension, bottom_extension)`` en unites de page normalisees.
    """
    _l, t, _r, b = table_bbox
    page_tables = page_table_map.get(page_num, [])

    # Find position of current table in the sorted list
    current_pos: int | None = None
    for i, (idx, _bbox) in enumerate(page_tables):
        if idx == table_idx:
            current_pos = i
            break

    if current_pos is None:
        # Table not found in map (shouldn't happen), use defaults
        return default_top, default_bottom

    # ---------------------------------------------------------------------------
    # BOTTOM EXTENSION — capture footnotes
    # ---------------------------------------------------------------------------
    if current_pos < len(page_tables) - 1:
        # There is a table below: extend down to just before the next table's top
        next_table_top = page_tables[current_pos + 1][1][1]  # next bbox's t
        available_gap = max(0.0, next_table_top - b)

        if available_gap < 0.03:
            logger.warning(
                "Table %s on page %s is very close to the next table (gap %.1f%%). "
                "Footnotes or titles might be squeezed or misattributed.",
                table_idx,
                page_num,
                available_gap * 100,
            )

        # Leave a small safety margin so we don't bleed into the next table
        bottom_ext = max(0.0, available_gap - min_gap_for_footnotes)
    else:
        # Last table on the page: extend to bottom of page minus margin
        bottom_ext = max(0.0, (1.0 - page_bottom_margin) - b)

    # Clamp: never extend less than the configured default for single-table pages
    # but for multi-table pages, respect the inter-table gap even if smaller
    if len(page_tables) == 1:
        bottom_ext = max(bottom_ext, default_bottom)

    # Final safety: never extend past the page boundary
    max_possible_bottom = max(0.0, 1.0 - b)
    bottom_ext = min(bottom_ext, max_possible_bottom)

    # ---------------------------------------------------------------------------
    # TOP EXTENSION — capture title
    # ---------------------------------------------------------------------------
    if current_pos > 0:
        # There is a table above: extend up to just after the previous table's bottom
        prev_table_bottom = page_tables[current_pos - 1][1][3]  # prev bbox's b
        available_gap_above = max(0.0, t - prev_table_bottom)
        if available_gap_above <= title_proximity_threshold:
            # Gap is small enough that a title might be squeezed between tables
            top_ext = max(0.0, available_gap_above - min_gap_for_footnotes)
        else:
            # Large gap — there's probably text between the two tables;
            # take up to the proximity threshold
            top_ext = min(available_gap_above, title_proximity_threshold)
    else:
        # First table on the page: extend up to capture a title
        available_above = t  # distance from table top to page top
        if available_above <= title_proximity_threshold:
            top_ext = max(0.0, available_above - 0.01)  # small margin from page edge
        else:
            top_ext = title_proximity_threshold

    # Ensure minimum extension
    top_ext = max(top_ext, default_top)

    logger.debug(
        "dynamic_crop page=%s table_idx=%s tables_on_page=%s "
        "top_ext=%.3f (default=%.3f) bottom_ext=%.3f (default=%.3f)",
        page_num,
        table_idx,
        len(page_tables),
        top_ext,
        default_top,
        bottom_ext,
        default_bottom,
    )

    return top_ext, bottom_ext
