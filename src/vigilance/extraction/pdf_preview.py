"""
Module de preview PDF pour la verification des sections detectees.

Ce module fournit des fonctions pour:
- Rendre des pages PDF en images
- Extraire le texte d'une plage de pages
- Generer des previews pour verification dans Streamlit
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Import conditionnel de PyMuPDF
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF non disponible - preview PDF desactive")

try:
    from vigilance.utils.pdf_open import open_pdf_safely
except ImportError:
    open_pdf_safely = None  # type: ignore[assignment, misc]

# Import conditionnel de PIL
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class PagePreview:
    """Preview d'une page PDF."""

    page_number: int
    image_bytes: bytes
    width: int
    height: int
    text_content: str = ""


@dataclass
class SectionPreview:
    """Preview d'une section complete."""

    section_type: str
    start_page: int
    end_page: int
    pages: list[PagePreview]
    total_text: str = ""
    title_found: str = ""
    confidence: float = 0.0


def render_pdf_page(
    pdf_path: str | Path, page_number: int, scale: float = 1.5, format: str = "png"
) -> bytes | None:
    """
    Rendre une page PDF en image.

    Args:
        pdf_path: Chemin vers le PDF
        page_number: Numero de page (1-indexed)
        scale: Echelle de rendu (0.5 = 50% resolution)
        format: Format de sortie (png, jpeg)

    Returns:
        Bytes de l'image ou None si erreur
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF requis pour le rendu PDF")
        return None

    try:
        doc = open_pdf_safely(pdf_path) if open_pdf_safely else fitz.open(str(pdf_path))

        # Conversion 1-indexed vers 0-indexed
        page_idx = page_number - 1

        if page_idx < 0 or page_idx >= len(doc):
            logger.error(f"Page {page_number} hors limites (1-{len(doc)})")
            doc.close()
            return None

        page = doc[page_idx]

        # Matrice de transformation pour le scale
        matrix = fitz.Matrix(scale, scale)

        # Rendre la page en pixmap
        pix = page.get_pixmap(matrix=matrix)

        # Convertir en bytes
        if format.lower() == "png":
            image_bytes = pix.tobytes("png")
        elif format.lower() in ["jpg", "jpeg"]:
            image_bytes = pix.tobytes("jpeg")
        else:
            image_bytes = pix.tobytes("png")

        doc.close()
        return image_bytes

    except Exception as e:
        logger.error(f"Erreur rendu page {page_number}: {e}")
        return None


def render_pdf_pages(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
    scale: float = 1.5,
    max_pages: int = 5,
) -> list[PagePreview]:
    """
    Rendre plusieurs pages PDF en images.

    Args:
        pdf_path: Chemin vers le PDF
        start_page: Page de debut (1-indexed)
        end_page: Page de fin (1-indexed)
        scale: Echelle de rendu
        max_pages: Nombre max de pages a rendre

    Returns:
        Liste de PagePreview
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF requis pour le rendu PDF")
        return []

    previews = []

    try:
        doc = open_pdf_safely(pdf_path) if open_pdf_safely else fitz.open(str(pdf_path))
        total_pages = len(doc)

        # Limiter les pages
        actual_end = min(end_page, total_pages)
        actual_start = max(1, start_page)
        pages_to_render = min(actual_end - actual_start + 1, max_pages)

        matrix = fitz.Matrix(scale, scale)

        for i in range(pages_to_render):
            page_num = actual_start + i
            page_idx = page_num - 1

            if page_idx >= len(doc):
                break

            page = doc[page_idx]

            # Rendre l'image
            pix = page.get_pixmap(matrix=matrix)
            image_bytes = pix.tobytes("png")

            # Extraire le texte
            text = page.get_text()

            preview = PagePreview(
                page_number=page_num,
                image_bytes=image_bytes,
                width=pix.width,
                height=pix.height,
                text_content=text,
            )
            previews.append(preview)

        doc.close()

    except Exception as e:
        logger.error(f"Erreur rendu pages {start_page}-{end_page}: {e}")

    return previews


def extract_text_from_pages(pdf_path: str | Path, start_page: int, end_page: int) -> str:
    """
    Extraire le texte d'une plage de pages.

    Args:
        pdf_path: Chemin vers le PDF
        start_page: Page de debut (1-indexed)
        end_page: Page de fin (1-indexed)

    Returns:
        Texte extrait
    """
    if not PYMUPDF_AVAILABLE:
        # Fallback vers pdfplumber
        try:
            import pdfplumber

            text_parts = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num in range(start_page - 1, min(end_page, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    text = page.extract_text() or ""
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Erreur extraction texte: {e}")
            return ""

    try:
        doc = open_pdf_safely(pdf_path) if open_pdf_safely else fitz.open(str(pdf_path))
        text_parts = []

        for page_num in range(start_page - 1, min(end_page, len(doc))):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

        doc.close()
        return "\n\n".join(text_parts)

    except Exception as e:
        logger.error(f"Erreur extraction texte: {e}")
        return ""


def get_pdf_info(pdf_path: str | Path) -> dict:
    """
    Obtenir les informations d'un PDF.

    Args:
        pdf_path: Chemin vers le PDF

    Returns:
        Dict avec total_pages, metadata, etc.
    """
    if not PYMUPDF_AVAILABLE:
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                return {
                    "total_pages": len(pdf.pages),
                    "metadata": pdf.metadata or {},
                    "available": True,
                }
        except Exception as e:
            return {"error": str(e), "available": False}

    try:
        doc = open_pdf_safely(pdf_path) if open_pdf_safely else fitz.open(str(pdf_path))
        info = {"total_pages": len(doc), "metadata": doc.metadata or {}, "available": True}
        doc.close()
        return info
    except Exception as e:
        return {"error": str(e), "available": False}


def create_section_preview(
    pdf_path: str | Path,
    section_type: str,
    start_page: int,
    end_page: int,
    title_found: str = "",
    confidence: float = 0.0,
    max_preview_pages: int = 3,
    scale: float = 0.5,
) -> SectionPreview:
    """
    Creer un preview complet d'une section.

    Args:
        pdf_path: Chemin vers le PDF
        section_type: Type de section (gestion_capital, gestion_risques)
        start_page: Page de debut
        end_page: Page de fin
        title_found: Titre trouve
        confidence: Score de confiance
        max_preview_pages: Nombre max de pages en preview
        scale: Echelle de rendu

    Returns:
        SectionPreview avec images et texte
    """
    # Rendre les premieres pages
    pages = render_pdf_pages(
        pdf_path, start_page, end_page, scale=scale, max_pages=max_preview_pages
    )

    # Extraire le texte complet
    total_text = extract_text_from_pages(pdf_path, start_page, end_page)

    return SectionPreview(
        section_type=section_type,
        start_page=start_page,
        end_page=end_page,
        pages=pages,
        total_text=total_text,
        title_found=title_found,
        confidence=confidence,
    )


def create_thumbnail(pdf_path: str | Path, page_number: int, width: int = 200) -> bytes | None:
    """
    Creer une vignette d'une page.

    Args:
        pdf_path: Chemin vers le PDF
        page_number: Numero de page (1-indexed)
        width: Largeur cible en pixels

    Returns:
        Bytes de l'image thumbnail
    """
    if not PYMUPDF_AVAILABLE:
        return None

    try:
        doc = open_pdf_safely(pdf_path) if open_pdf_safely else fitz.open(str(pdf_path))
        page_idx = page_number - 1

        if page_idx < 0 or page_idx >= len(doc):
            doc.close()
            return None

        page = doc[page_idx]

        # Calculer le scale pour la largeur cible
        page_width = page.rect.width
        scale = width / page_width

        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)

        image_bytes = pix.tobytes("png")
        doc.close()

        return image_bytes

    except Exception as e:
        logger.error(f"Erreur creation thumbnail: {e}")
        return None


def compare_pdf_sections(
    pdf_path_t1: str | Path,
    pdf_path_t2: str | Path,
    section_t1: tuple[int, int],  # (start, end)
    section_t2: tuple[int, int],  # (start, end)
    section_type: str = "",
) -> dict:
    """
    Comparer les sections de deux PDFs pour verification.

    Args:
        pdf_path_t1: PDF T1
        pdf_path_t2: PDF T2
        section_t1: (start_page, end_page) pour T1
        section_t2: (start_page, end_page) pour T2
        section_type: Type de section

    Returns:
        Dict avec previews des deux cotes
    """
    preview_t1 = create_section_preview(
        pdf_path_t1, section_type, section_t1[0], section_t1[1], max_preview_pages=2
    )

    preview_t2 = create_section_preview(
        pdf_path_t2, section_type, section_t2[0], section_t2[1], max_preview_pages=2
    )

    return {
        "section_type": section_type,
        "t1": {
            "pages": f"{section_t1[0]}-{section_t1[1]}",
            "preview": preview_t1,
            "page_count": section_t1[1] - section_t1[0] + 1,
        },
        "t2": {
            "pages": f"{section_t2[0]}-{section_t2[1]}",
            "preview": preview_t2,
            "page_count": section_t2[1] - section_t2[0] + 1,
        },
    }


# Fonctions utilitaires pour Streamlit
def is_preview_available() -> bool:
    """Verifier si le preview PDF est disponible."""
    return PYMUPDF_AVAILABLE


def get_preview_status() -> dict:
    """Obtenir le statut des dependances de preview."""
    return {"pymupdf": PYMUPDF_AVAILABLE, "pil": PIL_AVAILABLE, "ready": PYMUPDF_AVAILABLE}
