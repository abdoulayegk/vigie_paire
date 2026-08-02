"""Structures de donnees et normalisation de texte du localisateur de sections.

Extrait de ``section_locator.py`` sans modification : ``normalize_text`` et les
quatre dataclasses decrivant un element visuel, une entree de table des matieres,
une section localisee et le mapping complet d'un document.
"""

import unicodedata
from dataclasses import dataclass, field


def normalize_text(text: str) -> str:
    """Normaliser le texte en supprimant les accents et en mettant en minuscules.

    Permet de matcher "réglementation" avec "reglementation", etc.

    Args:
        text: Texte a normaliser

    Returns:
        Texte sans accents, en minuscules
    """
    if not text:
        return ""
    # NFD decompose les caracteres accentues (e + accent)
    # encode/decode supprime les caracteres non-ASCII (les accents)
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("utf-8")
    return ascii_text.lower()


@dataclass
class VisualTextElement:
    """Element de texte avec ses caracteristiques visuelles."""

    text: str
    page: int
    x0: float  # Position horizontale gauche
    y0: float  # Position verticale haute
    x1: float  # Position horizontale droite
    y1: float  # Position verticale basse
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False
    is_uppercase: bool = False
    line_number: int = 0  # Position relative sur la page
    page_width: float = 0.0
    page_height: float = 0.0

    @property
    def height(self) -> float:
        """Hauteur de l'element en points."""
        return abs(self.y1 - self.y0)

    @property
    def width(self) -> float:
        """Largeur de l'element en points."""
        return abs(self.x1 - self.x0)

    @property
    def is_likely_header(self) -> bool:
        """Determiner si l'element a les caracteristiques d'un titre."""
        # Criteres: grande taille, gras, ou majuscules
        return self.font_size >= 12.0 or self.is_bold or (self.is_uppercase and len(self.text) > 10)

    @property
    def bbox_norm(self) -> list[float] | None:
        """Retourner la bbox normalisee [x0, y0, x1, y1] si la taille de page est connue."""
        if self.page_width <= 0 or self.page_height <= 0:
            return None
        return [
            max(0.0, min(1.0, self.x0 / self.page_width)),
            max(0.0, min(1.0, self.y0 / self.page_height)),
            max(0.0, min(1.0, self.x1 / self.page_width)),
            max(0.0, min(1.0, self.y1 / self.page_height)),
        ]


@dataclass
class TocEntry:
    """Entree de la Table des matieres."""

    title: str
    page: int
    level: int = 0  # 0 = section principale, 1+ = sous-section
    raw_line: str = ""

    def __repr__(self):
        """Representation textuelle courte de l'entree TDM."""
        return f"TocEntry('{self.title[:30]}...', page={self.page}, level={self.level})"


@dataclass
class LocatedSection:
    """Represente une section localisee dans le document."""

    section_type: str  # "gestion_capital" ou "gestion_risques"
    title_found: str
    start_page: int
    end_page: int | None = None
    confidence: float = 0.0
    detection_method: str = ""  # "toc", "scan", "manual_override", "following_section"
    end_detection_method: str = ""  # Comment la fin a ete determinee
    detected_span: int | None = None
    final_span: int | None = None
    constraint_applied: bool = False
    constraint_reason: str = ""
    anchor_page: int | None = None
    anchor_text: str | None = None
    anchor_bbox_norm: list[float] | None = None
    anchor_found: bool = False
    end_anchor_page: int | None = None
    end_anchor_text: str | None = None
    end_anchor_bbox_norm: list[float] | None = None


SHARED_PAGE_TOP_THRESHOLD = 0.12


@dataclass
class SectionMapping:
    """Mapping complet des sections pour un document."""

    bank_code: str
    bank_name: str
    quarter: str
    year: int
    file_path: str
    sections: list[LocatedSection] = field(default_factory=list)
    total_pages: int = 0
    toc_entries: list[TocEntry] = field(default_factory=list)  # TDM complete
    toc_score: float = 0.0
    toc_reliable: bool = False
    toc_used: bool = False
    override_applied: bool = False
    boundary_validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convertir le mapping de sections en dictionnaire serialisable.

        Returns:
            Dictionnaire contenant toutes les informations du mapping.
        """
        sections_dict = {}
        for section in self.sections:
            sections_dict[section.section_type] = {
                "pages": f"{section.start_page}-{section.end_page}" if section.end_page else str(section.start_page),
                "start_page": section.start_page,
                "end_page": section.end_page,
                "title_found": section.title_found,
                "confidence": section.confidence,
                "detection_method": section.detection_method,
                "end_detection_method": section.end_detection_method,
                "detected_span": section.detected_span,
                "final_span": section.final_span,
                "constraint_applied": section.constraint_applied,
                "constraint_reason": section.constraint_reason,
                "anchor_page": section.anchor_page,
                "anchor_text": section.anchor_text,
                "anchor_bbox_norm": section.anchor_bbox_norm,
                "anchor_found": section.anchor_found,
                "end_anchor_page": section.end_anchor_page,
                "end_anchor_text": section.end_anchor_text,
                "end_anchor_bbox_norm": section.end_anchor_bbox_norm,
            }

        return {
            "bank_code": self.bank_code,
            "bank_name": self.bank_name,
            "quarter": self.quarter,
            "year": self.year,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "sections": sections_dict,
            "toc_entry_count": len(self.toc_entries),
            "toc_score": self.toc_score,
            "toc_reliable": self.toc_reliable,
            "toc_used": self.toc_used,
            "override_applied": self.override_applied,
            "boundary_validation": self.boundary_validation,
        }
