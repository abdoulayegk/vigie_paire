"""Footnote conversion helpers."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

# Canonical footnote type: list of {id, text} dicts.
FootnoteItem = dict[str, str]

_DICT_LIKE_RE = re.compile(r"^\s*\{.*\}\s*$", re.DOTALL)


def _parse_possible_mapping(value: Any) -> dict[str, Any] | None:
    """Parse a dict payload from dict or stringified-dict, else return None."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or not _DICT_LIKE_RE.match(raw):
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_id_text(mapping: dict[str, Any], index: int) -> tuple[str, str]:
    """Normalize marker/text with priority rules and anti-corruption fallback."""
    key = str(
        mapping.get("id")
        or mapping.get("ref")
        or mapping.get("marker")
        or index
    ).strip()

    text_value = mapping.get("text")
    if text_value is None or not str(text_value).strip():
        text_value = mapping.get("value")

    # Legacy corruption guard: text can itself contain stringified dict payload.
    parsed_text_mapping = _parse_possible_mapping(text_value)
    if parsed_text_mapping is not None:
        nested_key, nested_text = _normalize_id_text(parsed_text_mapping, index)
        if nested_text:
            return (nested_key or key or str(index), nested_text)

    text = str(text_value or "").strip()
    return (key or str(index), text)


def count_stringified_dict_suspects(items: list[Any] | None) -> int:
    """Count entries that look like stringified dict payloads (audit metric)."""
    suspects = 0
    for item in items or []:
        if isinstance(item, str) and _parse_possible_mapping(item) is not None:
            suspects += 1
            continue
        if isinstance(item, dict):
            text_value = item.get("text")
            if text_value is None:
                text_value = item.get("value")
            if isinstance(text_value, str) and _parse_possible_mapping(text_value) is not None:
                suspects += 1
    return suspects


def normalize_footnotes_to_canonical(
    items: list[Any] | None,
) -> list[FootnoteItem]:
    """Convert footnotes from various formats to canonical list[dict] with id and text.

    Handles:
    - None or empty -> []
    - list[str] (Docling legacy) -> [{"id": "1", "text": s}, ...] preserving order
    - list[dict] (Vision extraction) -> preserve id/marker and text; no repr strings
    - mixed types -> normalize each item safely
    """
    result: list[FootnoteItem] = []
    for i, item in enumerate(items or [], start=1):
        parsed_mapping = _parse_possible_mapping(item)
        if parsed_mapping is not None:
            kid, txt = _normalize_id_text(parsed_mapping, i)
            if txt:
                result.append({"id": kid or str(i), "text": txt})
        else:
            txt = str(item or "").strip()
            if txt:
                result.append({"id": str(i), "text": txt})
    return result


def footnotes_list_to_dict(items: list[Any]) -> dict[str, str]:
    """Convert a list-based footnote payload to a stable dict."""
    return {
        str(item.get("id") or "").strip(): str(item.get("text") or "").strip()
        for item in normalize_footnotes_to_canonical(items)
        if str(item.get("id") or "").strip() and str(item.get("text") or "").strip()
    }
