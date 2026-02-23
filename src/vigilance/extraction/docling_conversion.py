"""Conversion des structures Docling vers un format standardisé."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _extract_table_footnotes(table) -> list[str]:
    """Extraire les notes de bas de page associees a un tableau Docling."""
    footnotes = []
    try:
        if hasattr(table, "footnotes"):
            for fn in table.footnotes:
                footnotes.append(str(fn))
    except Exception:
        pass
    return footnotes


def _docling_table_to_dict(table, page_num: int) -> dict:
    """
    Convertir un tableau Docling en dictionnaire standardisé.
    """
    try:
        # Docling v2.x structure
        if hasattr(table, "export_to_dataframe"):
            df = table.export_to_dataframe()
            headers = list(df.columns)
            rows = df.values.tolist()
        elif hasattr(table, "data") and hasattr(table.data, "table_cells"):
            # Extraction manuelle des cellules
            cells = table.data.table_cells
            headers, rows = _parse_docling_cells(cells)
        else:
            # Fallback: essayer d'accéder aux attributs communs
            headers = []
            rows = []

            if hasattr(table, "text"):
                # Parser le texte brut du tableau
                lines = table.text.strip().split("\n")
                if lines:
                    headers = lines[0].split("\t") if "\t" in lines[0] else lines[0].split()
                    rows = [
                        line.split("\t") if "\t" in line else line.split() for line in lines[1:]
                    ]

        # Nettoyer les données
        headers = [str(h).strip() if h else "" for h in headers]
        rows = [[str(cell).strip() if cell else "" for cell in row] for row in rows]

        # Filtrer les lignes vides
        rows = [row for row in rows if any(cell for cell in row)]

        # Extraire les indicateurs (première colonne)
        indicators = [row[0].strip() for row in rows if row and len(row) > 0 and row[0].strip()]

        footnotes = _extract_table_footnotes(table)

        return {
            "table_id": f"docling_p{page_num}_{id(table)}",
            "page_number": page_num,
            "title": None,  # Docling ne fournit pas toujours le titre
            "headers": headers,
            "rows": rows,
            "first_column_indicators": indicators,
            "footnotes": footnotes,
            "extraction_method": "docling",
        }

    except Exception as e:
        logger.warning(f"Erreur conversion tableau Docling: {e}")
        return {}


def _parse_docling_cells(cells) -> tuple[list, list]:
    """
    Parser les cellules Docling en headers et rows.
    """
    if not cells:
        return [], []

    # Organiser par ligne
    rows_dict = {}
    max_col = 0

    for cell in cells:
        row_idx = cell.row_index if hasattr(cell, "row_index") else 0
        col_idx = cell.col_index if hasattr(cell, "col_index") else 0
        text = cell.text if hasattr(cell, "text") else str(cell)

        if row_idx not in rows_dict:
            rows_dict[row_idx] = {}
        rows_dict[row_idx][col_idx] = text
        max_col = max(max_col, col_idx)

    # Convertir en liste
    result_rows = []
    for row_idx in sorted(rows_dict.keys()):
        row = []
        for col_idx in range(max_col + 1):
            row.append(rows_dict[row_idx].get(col_idx, ""))
        result_rows.append(row)

    # Première ligne = headers
    if result_rows:
        headers = result_rows[0]
        rows = result_rows[1:]
        return headers, rows

    return [], []
