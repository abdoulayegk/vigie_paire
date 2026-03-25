"""RBC-specific helpers for title reliability and first-column structure."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .indicator_cleaner import (
    is_header_footer_table_title,
    normalize_indicator_for_comparison,
    strip_dates_from_table_title,
    strip_note_refs_from_title,
    strip_units_from_table_title,
)
from .matching_normalizer import (
    _classify_excluded_line,
    is_date_only_line,
    normalize_for_matching,
)

_VALUE_TOKEN_RE = re.compile(
    r"(?:\d|[%$]|n\.s\.|n/s|n\.a\.|s\.o\.|—|–|-)",
    re.IGNORECASE,
)
_NARRATIVE_TITLE_RE = re.compile(
    r"^(?:le|ce|the)\s+(?:tableau|table)\b",
    re.IGNORECASE,
)
_GENERIC_TITLE_SET = frozenset(
    {
        "aux",
        "pour les trimestres clos",
        "pour le trimestre clos a cette date",
        "pour le trimestre clos",
        "pour le trimestre termine",
    }
)
_TOTAL_ROW_RE = re.compile(r"^\s*total\b", re.IGNORECASE)


def is_rbc_bank(bank_code: str | None) -> bool:
    return (str(bank_code or "").strip().lower()) == "rbc"


def classify_rbc_title_reliability(
    title: str | None,
    *,
    bank_code: str | None = None,
) -> str:
    """Classify RBC title reliability.

    Returns ``missing``, ``unreliable`` or ``reliable``.
    Non-RBC callers get a permissive answer so the helper is safe as a fallback.
    """
    value = str(title or "").strip()
    if not value:
        return "missing"

    if not is_rbc_bank(bank_code):
        return "reliable"

    if is_header_footer_table_title(value, bank_code):
        return "unreliable"

    no_notes = strip_note_refs_from_title(value).strip()
    if not no_notes:
        return "missing"
    if is_date_only_line(no_notes):
        return "unreliable"
    if _NARRATIVE_TITLE_RE.match(no_notes):
        return "unreliable"

    cleaned = strip_dates_from_table_title(no_notes)
    cleaned = strip_units_from_table_title(cleaned, bank_code=bank_code)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,./()")
    if not cleaned:
        return "unreliable"

    normalized = normalize_for_matching(cleaned, target="title")
    if not normalized or normalized in _GENERIC_TITLE_SET:
        return "unreliable"
    if len(re.findall(r"[a-z]", normalized)) < 4:
        return "unreliable"
    return "reliable"


def is_unreliable_rbc_title(
    title: str | None,
    *,
    bank_code: str | None = None,
) -> bool:
    return classify_rbc_title_reliability(title, bank_code=bank_code) != "reliable"


@dataclass(slots=True)
class RbcFirstColumnSignals:
    indicators_raw: list[str]
    indicators_clean: list[str]
    groups_raw: list[str]
    groups_clean: list[str]
    hierarchical_indicator_signature: list[str]


def _uniq(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _cell_looks_like_value(cell: Any) -> bool:
    text = str(cell or "").strip()
    if not text:
        return False
    return bool(_VALUE_TOKEN_RE.search(text))


def _clean_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _looks_nested(raw_label: str) -> bool:
    return bool(re.match(r"^\s{1,}", str(raw_label or "")))


def _classify_rbc_row(row: list[str]) -> tuple[str, str]:
    raw_label = str(row[0] if row else "" or "")
    label = _clean_label(raw_label)
    if not label:
        return ("noise", "")

    exclusion = _classify_excluded_line(label)
    if exclusion in {"unit", "date"}:
        return ("unit_context", label)
    if exclusion in {"footnote", "number", "total"}:
        return ("noise", label)
    if _TOTAL_ROW_RE.match(label):
        return ("noise", label)

    value_cells = [str(cell or "").strip() for cell in row[1:]]
    non_empty_values = [cell for cell in value_cells if cell]
    if not non_empty_values:
        return ("group_label", label)

    value_like_count = sum(
        1 for cell in non_empty_values if _cell_looks_like_value(cell)
    )
    if value_like_count == 0:
        return ("group_label", label)

    return ("indicator_row", label)


def _fallback_indicator_rows(raw_indicators: list[str]) -> list[str]:
    filtered: list[str] = []
    for raw in raw_indicators:
        label = _clean_label(raw)
        if not label:
            continue
        exclusion = _classify_excluded_line(label)
        if exclusion in {"unit", "date", "footnote", "number", "total"}:
            continue
        if _TOTAL_ROW_RE.match(label):
            continue
        filtered.append(label)
    return _uniq(filtered)


def build_rbc_first_column_signals(
    *,
    rows: list[list[str]] | None,
    raw_indicators: list[str] | None,
) -> RbcFirstColumnSignals:
    """Split RBC first-column content into true indicators and structural groups."""
    candidate_rows = [
        list(row) for row in (rows or []) if isinstance(row, list) and row
    ]
    indicator_rows: list[str] = []
    group_rows: list[str] = []
    hierarchical_rows: list[str] = []
    current_group: str | None = None

    for row in candidate_rows:
        raw_label = str(row[0] if row else "" or "")
        kind, label = _classify_rbc_row(row)
        if kind == "group_label":
            current_group = label
            group_rows.append(label)
            continue
        if kind != "indicator_row":
            continue

        indicator_rows.append(label)
        if current_group and _looks_nested(raw_label):
            hierarchical_rows.append(f"{current_group} > {label}")
        else:
            hierarchical_rows.append(label)

    fallback_rows = _fallback_indicator_rows(list(raw_indicators or []))
    use_row_signals = len(indicator_rows) >= max(2, min(3, len(fallback_rows) or 2))
    final_indicator_raw = _uniq(
        indicator_rows if use_row_signals or not fallback_rows else fallback_rows
    )

    if not hierarchical_rows or len(hierarchical_rows) != len(indicator_rows):
        hierarchical_rows = list(indicator_rows)

    if not use_row_signals and fallback_rows:
        hierarchical_rows = list(fallback_rows)

    indicators_clean = [
        normalized
        for normalized in (
            normalize_indicator_for_comparison(label) for label in final_indicator_raw
        )
        if normalized
    ]
    groups_clean = [
        normalized
        for normalized in (
            normalize_indicator_for_comparison(label) for label in _uniq(group_rows)
        )
        if normalized
    ]

    return RbcFirstColumnSignals(
        indicators_raw=final_indicator_raw,
        indicators_clean=_uniq(indicators_clean),
        groups_raw=_uniq(group_rows),
        groups_clean=_uniq(groups_clean),
        hierarchical_indicator_signature=_uniq(
            _clean_label(label) for label in hierarchical_rows
        ),
    )
