"""Modele commun des ancres de tableaux pour les moteurs de localisation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableAnchor:
    """Ancre structurelle d'un tableau detecte dans un PDF.

    Attributes:
        table_id: Identifiant stable du tableau dans le run.
        page_number: Numero de page 1-indexe.
        bbox: Boite normalisee ``[l, t, r, b]`` dans ``[0, 1]``, ou ``None``.
        reference_text: Texte de reference optionnel pour Vision.
        source: Moteur d'origine (``tables_layout`` ou ``docling``).
    """

    table_id: str
    page_number: int
    bbox: list[float] | None
    reference_text: str | None = None
    source: str = ""


@dataclass
class TableLocationResult:
    """Resultat d'une passe de localisation structurelle."""

    anchors: list[TableAnchor] = field(default_factory=list)
    text_content: str = ""
    total_pages: int = 0
    inventory_pages: list[int] = field(default_factory=list)
