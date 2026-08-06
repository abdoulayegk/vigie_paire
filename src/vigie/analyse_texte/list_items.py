"""Reconnaissance et normalisation génériques des éléments de liste."""

from __future__ import annotations

import re
from dataclasses import dataclass


_UNICODE_BULLET_CHARS = "•◦▪‣⁃∙●○◉■□◆◇►▸▹‰\x81"
_UNICODE_BULLET_CLASS = re.escape(_UNICODE_BULLET_CHARS)
_LIST_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<marker>"
    r"\[\s*(?:x|X)?\s*\]"
    r"|[-*+](?=[ \t])"
    rf"|[{_UNICODE_BULLET_CLASS}]"
    r"|\d{1,3}[.)](?=[ \t])"
    r")"
    r"[ \t]*(?P<text>\S.*)$"
)
_REDUNDANT_UNORDERED_MARKER_RE = re.compile(
    r"^(?:"
    r"\[\s*(?:x|X)?\s*\]"
    r"|[-*+](?=[ \t])"
    rf"|[{_UNICODE_BULLET_CLASS}]"
    r")[ \t]*"
)
_ORDERED_MARKER_RE = re.compile(r"^\d{1,3}[.)]$")


@dataclass(frozen=True, slots=True)
class ParsedListItem:
    """Élément de liste reconnu dans une ligne de texte."""

    marker: str
    text: str
    indent: int


def _strip_redundant_unordered_markers(text: str) -> str:
    """Retire les glyphes de puce dupliqués par certains extracteurs PDF."""
    value = str(text or "").strip()
    while value:
        cleaned = _REDUNDANT_UNORDERED_MARKER_RE.sub("", value, count=1).strip()
        if cleaned == value:
            break
        value = cleaned
    return value


def parse_list_item_line(line: str) -> ParsedListItem | None:
    """Parse une ligne de liste explicite sans inférer une liste depuis sa prose."""
    match = _LIST_ITEM_RE.fullmatch(str(line or "").rstrip())
    if match is None:
        return None

    text = _strip_redundant_unordered_markers(match.group("text"))
    if not text:
        return None

    raw_marker = match.group("marker")
    marker = raw_marker if _ORDERED_MARKER_RE.fullmatch(raw_marker) else "-"
    expanded_indent = match.group("indent").expandtabs(4)
    return ParsedListItem(
        marker=marker,
        text=text,
        indent=min(len(expanded_indent), 12),
    )


def format_list_item_markdown(
    text: str,
    *,
    marker: str = "-",
    indent: int = 0,
) -> str:
    """Rend un élément de liste sous une forme Markdown stable."""
    normalized_text = _strip_redundant_unordered_markers(text)
    normalized_marker = marker if _ORDERED_MARKER_RE.fullmatch(str(marker or "")) else "-"
    normalized_indent = max(0, min(int(indent or 0), 12))
    return f"{' ' * normalized_indent}{normalized_marker} {normalized_text}".rstrip()
