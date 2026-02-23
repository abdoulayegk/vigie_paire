"""Helpers to resolve semantic table titles from nearby text lines."""

from __future__ import annotations

import re

_TABLE_NUMBER_RE = re.compile(
    r"^\s*(?:tableau|table|tab\.?)\s*([0-9]+[a-z]?)\s*(?:[:\-–—.]?\s*(.*))?$",
    re.IGNORECASE,
)

_UNIT_RE = re.compile(
    r"\b(?:en|in)\s+(?:milliers?|millions?|milliards?|pourcentage|percent|%)\b",
    re.IGNORECASE,
)


def _clean_line(value: str | None) -> str:
    line = re.sub(r"\s+", " ", str(value or "")).strip()
    return line.strip(" -:;,")


def extract_table_number_and_inline_title(line: str | None) -> tuple[str | None, str | None]:
    """Extract table number and optional inline title from one line."""
    value = _clean_line(line)
    if not value:
        return None, None

    match = _TABLE_NUMBER_RE.match(value)
    if not match:
        return None, None

    number = (match.group(1) or "").strip() or None
    inline = _clean_line(match.group(2))
    return number, (inline or None)


def is_table_number_line(line: str | None) -> bool:
    """Return True when the line starts with a table number marker."""
    number, _ = extract_table_number_and_inline_title(line)
    return bool(number)


def is_unit_context_line(line: str | None) -> bool:
    """Return True for lines that mostly carry unit context."""
    value = _clean_line(line)
    if not value:
        return False
    return bool(_UNIT_RE.search(value))


def resolve_title_from_lines(
    lines: list[str],
    bank_code: str | None = None,
    first_row_cells: list[str] | None = None,
) -> dict[str, str]:
    """Resolve title metadata from a text window around a table."""
    _ = bank_code  # Reserved for future bank-specific rules.

    normalized_lines = [_clean_line(line) for line in lines if _clean_line(line)]
    unit_context = next((line for line in normalized_lines if is_unit_context_line(line)), "")

    for idx, line in enumerate(normalized_lines):
        number, inline = extract_table_number_and_inline_title(line)
        if not number:
            continue

        if inline:
            return {
                "title": inline,
                "title_raw": line,
                "table_number": number,
                "unit_context": unit_context,
                "resolution_method": "inline_numbered",
            }

        followup_title = ""
        for candidate in normalized_lines[idx + 1 :]:
            if is_unit_context_line(candidate) or is_table_number_line(candidate):
                continue
            followup_title = candidate
            break

        return {
            "title": followup_title,
            "title_raw": line,
            "table_number": number,
            "unit_context": unit_context,
            "resolution_method": "layout_anchor" if followup_title else "number_only",
        }

    fallback_title = next((line for line in normalized_lines if not is_unit_context_line(line)), "")
    if not fallback_title and first_row_cells:
        fallback_title = _clean_line(first_row_cells[0] if first_row_cells else "")

    return {
        "title": fallback_title,
        "title_raw": fallback_title,
        "table_number": "",
        "unit_context": unit_context,
        "resolution_method": "text_fallback",
    }
