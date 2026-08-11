"""Mise en evidence des segments modifies dans le texte compare.

Extrait de ``page_text_analysis.py`` sans modification.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from dash import html

# Styles inline pour les highlights — couleurs métier banque
# (ambre=retiré, vert=ajouté).
_HIGHLIGHT_REMOVED_STYLE = {
    "backgroundColor": "#fef3c7",
    "color": "#92400e",
    "padding": "0 2px",
    "borderRadius": "2px",
    "fontWeight": "500",
}
_HIGHLIGHT_ADDED_STYLE = {
    "backgroundColor": "#dcfce7",
    "color": "#14532d",
    "padding": "0 2px",
    "borderRadius": "2px",
    "fontWeight": "500",
}


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fusionne des intervalles de caractères chevauchants."""
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_highlight_intervals(text: str, highlights: list[str]) -> list[tuple[int, int]]:
    """Retourne les positions des fragments GPT retrouvables dans le texte."""
    intervals: list[tuple[int, int]] = []
    lower_text = text.lower()
    for highlight in highlights:
        if not highlight or not highlight.strip():
            continue
        needle = highlight.lower()
        start = 0
        while True:
            idx = lower_text.find(needle, start)
            if idx < 0:
                break
            intervals.append((idx, idx + len(highlight)))
            start = idx + len(highlight)
    return _merge_intervals(intervals)


def _highlight_text_by_intervals(
    text: str,
    intervals: list[tuple[int, int]],
    style: dict[str, str],
) -> list:
    """Découpe ``text`` en spans selon des intervalles déjà calculés."""
    if not text:
        return []
    merged = _merge_intervals(intervals)
    if not merged:
        return [html.Span(text)]

    spans: list = []
    cursor = 0
    for start, end in merged:
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        if cursor < start:
            spans.append(html.Span(text[cursor:start]))
        if start < end:
            spans.append(html.Span(text[start:end], style=style))
        cursor = end
    if cursor < len(text):
        spans.append(html.Span(text[cursor:]))
    return spans


def _highlight_text(text: str, highlights: list[str], style: dict[str, str]) -> list:
    """Découpe ``text`` en spans dont les portions matching ``highlights`` portent ``style``.

    Recherche par ``str.find()`` insensible à la casse mais avec le texte
    verbatim de GPT. Si un highlight n'est pas trouvable dans le texte source
    (hallucination GPT), il est silencieusement ignoré.

    Args:
        text: Texte source complet (T1 ou T2).
        highlights: Liste de fragments à surligner.
        style: Dict de style CSS appliqué aux spans surlignés.

    Returns:
        Liste de ``html.Span`` (alternance segments normaux / surlignés).
    """
    if not text:
        return []
    if not highlights:
        return [html.Span(text)]

    return _highlight_text_by_intervals(text, _find_highlight_intervals(text, highlights), style)


def _change_segments_are_usable(change_segments: list[dict]) -> bool:
    """Ecarte les segments trop fragmentés qui produisent du faux surlignage."""
    lengths: list[int] = []
    for seg in change_segments:
        if not isinstance(seg, dict):
            continue
        parts = [
            str(seg.get("text_t1") or "").strip(),
            str(seg.get("text_t2") or "").strip(),
        ]
        substantive = [part for part in parts if re.search(r"\w", part, flags=re.UNICODE)]
        if not substantive:
            continue
        longest = max(len(part) for part in substantive)
        if longest < 3:
            return False
        lengths.append(longest)

    if not lengths:
        return False
    if len(lengths) >= 8:
        tiny_count = sum(1 for length in lengths if length < 12)
        if tiny_count / len(lengths) >= 0.35:
            return False
    return True


def _token_intervals(text: str) -> list[tuple[str, int, int]]:
    """Tokenise un texte en mots avec positions pour calculer un diff lisible."""
    return [(match.group(0).lower(), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _diff_highlight_intervals(text_t1: str, text_t2: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Calcule les intervalles modifiés directement depuis le diff T1/T2."""
    tokens_t1 = _token_intervals(text_t1)
    tokens_t2 = _token_intervals(text_t2)
    if not tokens_t1 and not tokens_t2:
        return [], []
    if tokens_t1 and not tokens_t2:
        return [(0, len(text_t1))], []
    if tokens_t2 and not tokens_t1:
        return [], [(0, len(text_t2))]

    words_t1 = [token for token, _, _ in tokens_t1]
    words_t2 = [token for token, _, _ in tokens_t2]
    intervals_t1: list[tuple[int, int]] = []
    intervals_t2: list[tuple[int, int]] = []
    matcher = SequenceMatcher(None, words_t1, words_t2, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"delete", "replace"} and i1 < i2:
            intervals_t1.append((tokens_t1[i1][1], tokens_t1[i2 - 1][2]))
        if tag in {"insert", "replace"} and j1 < j2:
            intervals_t2.append((tokens_t2[j1][1], tokens_t2[j2 - 1][2]))
    return _merge_intervals(intervals_t1), _merge_intervals(intervals_t2)
