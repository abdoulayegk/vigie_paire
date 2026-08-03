"""Metadonnees de structure locale par page pour les tableaux (ordre vertical, role) pour l'appariement intra-page."""

from __future__ import annotations

from typing import Any


def _page_for_table(t: Any) -> int | None:
    """Extrait le numero de page d'un objet tableau."""
    page = getattr(t, "page_pdf", None)
    if page is not None:
        try:
            return int(page)
        except (TypeError, ValueError):
            pass
    page = getattr(t, "page_number", None)
    if page is not None:
        try:
            return int(page)
        except (TypeError, ValueError):
            pass
    return None


def _bbox_top_left(t: Any) -> tuple[float, float]:
    """Retourne (top, left) depuis la bbox du tableau pour le tri."""
    bbox = getattr(t, "bbox", None)
    if bbox is None:
        return (0.0, 0.0)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (float(bbox[1]), float(bbox[0]))
    if isinstance(bbox, dict):
        if "t" in bbox and "l" in bbox:
            return (float(bbox["t"]), float(bbox["l"]))
        if "y0" in bbox and "x0" in bbox:
            return (float(bbox["y0"]), float(bbox["x0"]))
        if "y" in bbox and "x" in bbox:
            return (float(bbox["y"]), float(bbox["x"]))
    return (0.0, 0.0)


def derive_page_local_structure(
    tables: list[Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Calcule la structure locale par page pour chaque tableau : index, compteur, bbox_top, role.

    Args:
        tables: Liste d'objets table-like avec ``table_id``, ``page``
            (``page_pdf`` ou ``page_number``) et ``bbox``.

    Returns:
        Pour chaque ``(table_id, page)``, un dictionnaire avec :
        - ``table_index_on_page`` : ordre vertical (base 1)
        - ``tables_on_page`` : nombre de tableaux sur la page
        - ``bbox_top`` : coordonnee y haute normalisee (depuis bbox)
        - ``page_local_role`` : ``"single"`` | ``"first"`` | ``"middle"`` | ``"last"``

        Tri intra-page : bbox_top croissant, puis bbox_left, puis ordre d'entree.
    """
    key_to_page: dict[tuple[str, int], int] = {}
    by_page: dict[int, list[tuple[int, str, float, float]]] = {}

    for idx, t in enumerate(tables):
        tid = getattr(t, "table_id", None)
        if not tid:
            continue
        page = _page_for_table(t)
        if page is None:
            continue
        bbox = getattr(t, "bbox", None)
        if (
            bbox is None
            or (isinstance(bbox, (list, tuple)) and len(bbox) < 4)
            or (isinstance(bbox, dict) and not bbox)
        ):
            continue
        top, left = _bbox_top_left(t)
        key_to_page[(str(tid), page)] = page
        by_page.setdefault(page, []).append((idx, str(tid), top, left))

    result: dict[tuple[str, int], dict[str, Any]] = {}

    for page, rows in by_page.items():
        rows_sorted = sorted(rows, key=lambda r: (r[2], r[3], r[0]))
        n = len(rows_sorted)
        for one_based, (_, tid, top, _) in enumerate(rows_sorted, start=1):
            if n == 1:
                role = "single"
            elif n == 2:
                role = "first" if one_based == 1 else "last"
            else:
                if one_based == 1:
                    role = "first"
                elif one_based == n:
                    role = "last"
                else:
                    role = "middle"
            result[(tid, page)] = {
                "table_index_on_page": one_based,
                "tables_on_page": n,
                "bbox_top": top,
                "page_local_role": role,
            }

    return result
