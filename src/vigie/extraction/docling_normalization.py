"""Outils de normalisation pour l'extraction Docling/pdfs."""

from __future__ import annotations

import re

# Lignes de notes: (1), [2], a), b), Note 1, Note 2., etc.
_FOOTNOTE_ROW_RE = re.compile(
    r"^\s*(?:[\(\[]\d+[\)\]]|[a-z]\)|note\s*\d+\s*[.:\-–—]?)\s*",
    re.IGNORECASE,
)


def _is_footnote_row(row: list[str]) -> bool:
    """Detecter les lignes de notes de bas de tableau dans le grid Docling.

    Retourne ``True`` pour les lignes dont la premiere cellule ressemble a
    une definition de note : ``(1)``, ``[2]``, ``1)``, etc.

    Args:
        row: Cellules d'une ligne de tableau.

    Returns:
        ``True`` si la ligne est une note de bas de tableau.
    """
    if not row:
        return True

    first = str(row[0]).strip() if row[0] else ""
    if not first:
        return False
    return bool(_FOOTNOTE_ROW_RE.match(first)) and len(first) < 100


def _extract_table_context_split(
    page_text: str,
    table_title: str | None,
    chars_before: int = 300,
    chars_after: int = 400,
) -> tuple[str, str]:
    """Extraire le contexte avant et apres un tableau (pour table_type_classifier).

    Returns:
        Tuple (context_before, context_after)
    """
    if not page_text:
        return ("", "")

    # Si pas de titre, prendre le debut de la page comme reference
    if not table_title or not table_title.strip():
        before = page_text[:chars_before].strip()
        after = page_text[chars_before : chars_before + chars_after].strip()
        return (before, after)

    title_pos = page_text.lower().find(table_title.lower()[:30])
    if title_pos == -1:
        return ("", "")

    start_before = max(0, title_pos - chars_before)
    context_before = page_text[start_before:title_pos].strip()

    end_title = title_pos + len(table_title)
    end_after = min(len(page_text), end_title + chars_after)
    context_after = page_text[end_title:end_after].strip()

    return (context_before, context_after)
