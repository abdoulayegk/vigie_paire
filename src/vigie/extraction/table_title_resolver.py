"""Fonctions utilitaires pour resoudre les titres semantiques de tableaux a partir des lignes de texte avoisinantes."""

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
    """Nettoyer une ligne de texte (espaces, tirets, ponctuation peripherique)."""
    line = re.sub(r"\s+", " ", str(value or "")).strip()
    return line.strip(" -:;,")


def extract_table_number_and_inline_title(
    line: str | None,
) -> tuple[str | None, str | None]:
    """Extraire le numero de tableau et le titre inline optionnel d'une ligne.

    Args:
        line: Ligne de texte brute.

    Returns:
        Tuple ``(numero_tableau, titre_inline)`` ; chaque element peut etre ``None``.
    """
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
    """Retourner ``True`` si la ligne commence par un marqueur de numero de tableau."""
    number, _ = extract_table_number_and_inline_title(line)
    return bool(number)


def is_unit_context_line(line: str | None) -> bool:
    """Retourner ``True`` pour les lignes portant principalement un contexte d'unite."""
    value = _clean_line(line)
    if not value:
        return False
    return bool(_UNIT_RE.search(value))


def resolve_title_from_lines(
    lines: list[str],
    bank_code: str | None = None,
    first_row_cells: list[str] | None = None,
) -> dict[str, str]:
    """Resoudre les metadonnees du titre a partir d'une fenetre de texte autour d'un tableau.

    Args:
        lines: Lignes de texte proches du tableau.
        bank_code: Code banque (reserve pour regles specifiques futures).
        first_row_cells: Cellules de la premiere ligne du tableau (fallback).

    Returns:
        Dictionnaire avec ``title``, ``title_raw``, ``table_number``,
        ``unit_context`` et ``resolution_method``.
    """
    _ = bank_code  # Reserved for future bank-specific rules.

    normalized_lines = [_clean_line(line) for line in lines if _clean_line(line)]
    unit_context = next(
        (line for line in normalized_lines if is_unit_context_line(line)), ""
    )

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

    fallback_title = next(
        (line for line in normalized_lines if not is_unit_context_line(line)), ""
    )
    if not fallback_title and first_row_cells:
        fallback_title = _clean_line(first_row_cells[0] if first_row_cells else "")

    return {
        "title": fallback_title,
        "title_raw": fallback_title,
        "table_number": "",
        "unit_context": unit_context,
        "resolution_method": "text_fallback",
    }
