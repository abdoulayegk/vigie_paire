"""Helpers de formatage des pages pour les sorties analyste."""

from __future__ import annotations

import re
from typing import Any


def page_numbers(value: Any) -> list[int]:
    """Retourne une liste de pages entières, triées et dédupliquées."""
    if value is None:
        return []
    if isinstance(value, str):
        pages: list[int] = []
        for start_raw, end_raw in re.findall(r"(\d+)(?:\s*-\s*(\d+))?", value):
            start = int(start_raw)
            end = int(end_raw) if end_raw else start
            if start <= 0 or end <= 0:
                continue
            if end < start:
                start, end = end, start
            pages.extend(range(start, end + 1))
        return sorted(set(pages))
    raw_values = value if isinstance(value, list | tuple | set) else [value]
    pages: list[int] = []
    for raw in raw_values:
        if raw in (None, ""):
            continue
        try:
            page = int(raw)
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.append(page)
    return sorted(set(pages))


def format_page_interval(value: Any, *, prefix: str = "") -> str:
    """Formate des pages en intervalles lisibles, ex: ``69-71, 109``."""
    pages = page_numbers(value)
    if not pages:
        return ""

    ranges: list[tuple[int, int]] = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append((start, previous))
        start = previous = page
    ranges.append((start, previous))

    formatted = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)
    return f"{prefix}{formatted}" if prefix else formatted
