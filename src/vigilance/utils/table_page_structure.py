"""Page-local structure metadata for tables (vertical order, role) for same-page matching."""

from __future__ import annotations

from typing import Any


def _page_for_table(t: Any) -> int | None:
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
    """Return (top, left) from table bbox for sorting. Top = y_min, left = x_min."""
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


def _bbox_top_only(t: Any) -> float | None:
    bbox = getattr(t, "bbox", None)
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return float(bbox[1])
    if isinstance(bbox, dict):
        if "t" in bbox:
            return float(bbox["t"])
        if "y0" in bbox:
            return float(bbox["y0"])
        if "y" in bbox:
            return float(bbox["y"])
    return None


def derive_page_local_structure(
    tables: list[Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Compute page-local structure per table: index, count, bbox_top, role.

    Input: list of table-like objects with table_id, page (page_pdf or page_number), bbox.
    Output: for each (table_id, page), a dict with:
      - table_index_on_page: 1-based vertical order
      - tables_on_page: number of tables on that page
      - bbox_top: normalized top y (from bbox)
      - page_local_role: "single" | "first" | "middle" | "last"

    Sort within page: bbox_top ascending, then bbox_left, then input order.
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
        if bbox is None or (isinstance(bbox, (list, tuple)) and len(bbox) < 4) or (isinstance(bbox, dict) and not bbox):
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
