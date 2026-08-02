"""Detection des sections dans le texte et association aux tableaux.

Extrait de ``docling_processor.py`` sans modification des corps de
methodes. Mixin consomme par ``DoclingProcessor``.
"""

from __future__ import annotations

from .models import ExtractedTable
from .runtime import COMPILED_SECTION_PATTERNS


class SectionAssociationMixin:
    """Detection des sections dans le texte et association aux tableaux."""

    def _detect_sections_in_text(self, text_content: str) -> list[dict]:
        """Détecte les titres de sections dans le texte du document.

        Args:
            text_content: Contenu textuel du document

        Returns:
            Liste de sections détectées avec leur position, nom et phase
        """
        sections = []
        lines = text_content.split("\n")

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()

            # Ignorer les lignes trop longues (pas un titre) ou trop courtes
            if len(line_stripped) < 5 or len(line_stripped) > 80:
                continue

            for pattern, section_name, phase in COMPILED_SECTION_PATTERNS:
                if pattern.match(line_stripped):
                    sections.append(
                        {
                            "line_num": line_num,
                            "name": section_name,
                            "phase": phase,
                            "original_text": line_stripped,
                        }
                    )
                    break

        return sections

    def _associate_tables_with_sections(self, tables: list[ExtractedTable], text_content: str) -> list[ExtractedTable]:
        """Associe chaque tableau à sa section parente basée sur la position dans le document.

        Args:
            tables: Liste des tableaux extraits
            text_content: Contenu textuel du document

        Returns:
            Liste des tableaux avec section associée
        """
        sections = self._detect_sections_in_text(text_content)

        if not sections:
            return tables

        # Pour chaque tableau, trouver la section précédente la plus proche
        for table in tables:
            # Chercher dans le texte autour du titre du tableau

            # Trouver la section la plus proche avant ce tableau (basé sur page)
            best_section = None
            for section in sections:
                # On ne peut pas directement mapper ligne -> page sans page_breaks
                # Donc on utilise une heuristique : sections apparaissent avant leurs tableaux
                # Pour l'instant, on assigne la dernière section vue
                if section["line_num"] < (table.page_number * 50):  # Estimation ~50 lignes/page
                    best_section = section

            if best_section:
                table.section = best_section["name"]
                table.section_phase = best_section["phase"]

        return tables


# -----------------------------------------------------------------------------
# Fonctions utilitaires d'extraction (API publique)
# -----------------------------------------------------------------------------
