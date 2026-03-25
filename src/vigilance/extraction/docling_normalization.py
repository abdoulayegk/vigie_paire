"""Outils de normalisation pour l'extraction Docling/pdfs."""

from __future__ import annotations

import re

# Lignes de notes: (1), [2], a), b), Note 1, Note 2., etc.
_FOOTNOTE_ROW_RE = re.compile(
    r"^\s*(?:[\(\[]\d+[\)\]]|[a-z]\)|note\s*\d+\s*[.:\-–—]?)\s*",
    re.IGNORECASE,
)


def _find_table_title_in_text(page_text: str, table_idx: int) -> str | None:
    """
    Chercher un titre de tableau dans le texte de la page.

    Args:
        page_text: Texte de la page
        table_idx: Index du tableau sur la page

    Returns:
        Titre trouve ou None
    """
    # Patterns pour les titres de tableaux
    patterns = [
        re.compile(r"(TABLEAU\s+\d+[a-zA-Z]?\s*[:\-–—]?\s*[^\n]+)", re.IGNORECASE),
        re.compile(r"(TABLE\s+\d+[a-zA-Z]?\s*[:\-–—]?\s*[^\n]+)", re.IGNORECASE),
        re.compile(r"^(T\d+[A-Za-z]?\s+[^\n]+)", re.MULTILINE),
    ]

    for pattern in patterns:
        matches = pattern.findall(page_text)
        if matches and table_idx < len(matches):
            return matches[table_idx].strip()

    return None


def _reconstruct_table_rows(table_data: list) -> list:
    """
    Reconstruire les lignes d'un tableau en fusionnant les cellules fragmentees.

    Probleme: pdfplumber peut fragmenter une ligne sur plusieurs lignes
    Solution: Fusionner les lignes qui semblent etre des continuations
    """
    if not table_data:
        return []

    reconstructed = []
    current_row = None

    for row in table_data:
        if not row:
            continue

        # Nettoyer les cellules
        cleaned = [str(cell).strip() if cell else "" for cell in row]

        # Verifier si c'est une nouvelle ligne ou une continuation
        first_cell = cleaned[0] if cleaned else ""

        # Si la premiere cellule est vide et qu'on a une ligne en cours,
        # c'est probablement une continuation
        if not first_cell and current_row is not None:
            # Fusionner avec la ligne precedente
            for i, cell in enumerate(cleaned):
                if cell and i < len(current_row):
                    if current_row[i]:
                        current_row[i] += " " + cell
                    else:
                        current_row[i] = cell
        else:
            # Nouvelle ligne
            if current_row is not None:
                reconstructed.append(current_row)
            current_row = cleaned

    # Ajouter la derniere ligne
    if current_row is not None:
        reconstructed.append(current_row)

    # Filtrer les lignes vides
    return [row for row in reconstructed if any(cell for cell in row)]


def _merge_fragmented_cells(rows: list[list[str]]) -> list[list[str]]:
    """
    Fusionner les cellules fragmentees par Docling (texte coupe entre lignes).

    Si une ligne a la premiere cellule vide ou tres courte et que la suivante
    semble etre une continuation (pas de majuscule en debut, pas de chiffre seul),
    fusionner avec la ligne precedente.
    """
    if not rows:
        return []

    reconstructed = []
    current_row: list[str] | None = None

    for row in rows:
        if not row:
            continue

        cleaned = [str(c).strip() if c else "" for c in row]
        first_cell = cleaned[0] if cleaned else ""

        # Premiere cellule vide ou tres courte: possible continuation
        is_continuation = (
            current_row is not None
            and (not first_cell or len(first_cell) <= 2)
            and first_cell != "0"
            and not (len(first_cell) == 1 and first_cell.isdigit())
        )

        # Verifier si la ligne actuelle ressemble a une continuation
        if is_continuation and first_cell:
            next_looks_like_new = (
                first_cell[0].isupper() if first_cell else False
            ) or (first_cell.isdigit() and len(first_cell) <= 4)
            if next_looks_like_new:
                is_continuation = False

        if is_continuation and current_row is not None:
            for i, cell in enumerate(cleaned):
                if cell and i < len(current_row):
                    current_row[i] = (current_row[i] or "") + " " + cell
                elif cell and i >= len(current_row):
                    current_row.append(cell)
        else:
            if current_row is not None:
                reconstructed.append(current_row)
            current_row = cleaned

    if current_row is not None:
        reconstructed.append(current_row)

    return [r for r in reconstructed if any(c for c in r)]


def _is_footnote_row(row: list[str]) -> bool:
    """
    Detecter les lignes de notes de bas de tableau dans le grid Docling.

    Retourne True pour les lignes dont la premiere cellule ressemble a
    une definition de note: (1), [2], 1), etc.
    """
    if not row:
        return True

    first = str(row[0]).strip() if row[0] else ""
    if not first:
        return False
    return bool(_FOOTNOTE_ROW_RE.match(first)) and len(first) < 100


def _extract_table_context(page_text: str, table_title: str | None) -> str:
    """
    Extraire le contexte textuel autour d'un tableau.
    """
    if not table_title or not page_text:
        return ""

    # Trouver la position du titre dans le texte
    title_pos = page_text.lower().find(table_title.lower()[:20])
    if title_pos == -1:
        return ""

    # Extraire le contexte (200 caracteres avant et apres)
    start = max(0, title_pos - 200)
    end = min(len(page_text), title_pos + len(table_title) + 500)

    return page_text[start:end]


def _extract_table_context_split(
    page_text: str,
    table_title: str | None,
    chars_before: int = 300,
    chars_after: int = 400,
) -> tuple[str, str]:
    """
    Extraire le contexte avant et apres un tableau (pour table_type_classifier).

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
