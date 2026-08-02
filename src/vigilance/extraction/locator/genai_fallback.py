"""Repli GenAI lorsque les strategies deterministes ne trouvent pas les sections cibles.

Extrait de ``section_locator.py`` sans modification des corps de methodes.
Mixin consomme par ``SectionLocator``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import LocatedSection

# Nom de logger conserve a l'identique apres le decoupage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigilance.extraction.section_locator")


class GenAIFallbackMixin:
    """Repli GenAI lorsque les strategies deterministes ne trouvent pas les sections cibles."""

    def _needs_genai_fallback(self, sections: list[LocatedSection]) -> bool:
        """Determiner si le fallback GenAI est necessaire.

        Criteres:
        - Moins de 2 sections trouvees
        - Confiance moyenne inferieure a 0.7

        Args:
            sections: Sections deja detectees

        Returns:
            True si GenAI fallback necessaire
        """
        # Cas 1: Moins de 2 sections trouvees
        if len(sections) < 2:
            logger.info("GenAI fallback: moins de 2 sections trouvees")
            return True

        # Cas 2: Confiance moyenne trop faible
        avg_confidence = sum(s.confidence for s in sections) / len(sections)
        if avg_confidence < 0.7:
            logger.info(f"GenAI fallback: confiance moyenne faible ({avg_confidence:.2f})")
            return True

        return False

    def _detect_with_genai(self, pdf_path: Path) -> list[LocatedSection]:
        """Utiliser GenAI pour detecter les sections.

        Args:
            pdf_path: Chemin vers le PDF

        Returns:
            Liste de LocatedSection detectees par GenAI
        """
        try:
            from .genai_toc_detector import GenAITOCDetector
        except ImportError:
            logger.warning("genai_toc_detector non disponible")
            return []

        try:
            detector = GenAITOCDetector()
            genai_results = detector.find_and_extract_sections(pdf_path)

            # Convertir en LocatedSection
            sections = []
            for result in genai_results:
                section = LocatedSection(
                    section_type=result.section_type,
                    title_found=result.title_found,
                    start_page=result.start_page,
                    end_page=None,  # Sera determine plus tard
                    confidence=result.confidence,
                    detection_method="genai_fallback",
                    end_detection_method="",
                )
                sections.append(section)

            return sections

        except Exception as e:
            logger.error(f"Erreur GenAI fallback: {e}")
            return []
