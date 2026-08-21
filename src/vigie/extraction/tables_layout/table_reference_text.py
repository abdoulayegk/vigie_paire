"""Texte de reference pour Vision a partir des blocs table PyMuPDF Layout."""

from __future__ import annotations

from typing import Any

from vigie.extraction.docling_bbox_helpers import _build_indicator_reference_text


def build_reference_text_from_table_block(
    table_payload: dict[str, Any] | None,
    *,
    max_chars: int,
) -> str | None:
    """Construire un ``reference_text`` a partir du bloc table Layout.

    Preferer le markdown du tableau, sinon les lignes ``extract``, puis un
    aplatissement simple des cellules. Le filtrage final reutilise le helper
    Docling afin de garder le meme role cote Vision.

    Args:
        table_payload: Sous-objet ``table`` du bloc JSON Layout.
        max_chars: Plafond de caracteres.

    Returns:
        Texte filtre, ou ``None`` si trop court / inutilisable.
    """
    if not isinstance(table_payload, dict) or max_chars <= 0:
        return None

    raw = ""
    markdown = table_payload.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        raw = markdown.strip()
    else:
        extract = table_payload.get("extract")
        if isinstance(extract, list) and extract:
            lines: list[str] = []
            for row in extract:
                if isinstance(row, (list, tuple)):
                    cells = [str(cell or "").strip() for cell in row]
                    lines.append(" | ".join(cell for cell in cells if cell))
                else:
                    text = str(row or "").strip()
                    if text:
                        lines.append(text)
            raw = "\n".join(line for line in lines if line)
        else:
            cells = table_payload.get("cells")
            if isinstance(cells, list) and cells:
                pieces: list[str] = []
                for cell in cells:
                    if isinstance(cell, dict):
                        text = str(cell.get("text") or cell.get("content") or "").strip()
                    else:
                        text = str(cell or "").strip()
                    if text:
                        pieces.append(text)
                raw = "\n".join(pieces)

    if len(raw.strip()) <= 20:
        return None
    return _build_indicator_reference_text(raw, max_chars=max_chars)
