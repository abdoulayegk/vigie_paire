"""Détection par la mise en page : éléments visuels, en-têtes et ancres.

Mixin consommé par ``SectionLocator``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path

import pdfplumber

from .models import (
    SHARED_PAGE_TOP_THRESHOLD,
    LocatedSection,
    TocEntry,
    VisualTextElement,
    normalize_text,
)
from .patterns import SECTION_TITLE_ALIASES

# Nom de logger conservé à l'identique après le découpage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigie.extraction.section_locator")


class VisualLayoutMixin:
    """Détection par la mise en page : éléments visuels, en-têtes et ancres."""

    def _extract_visual_elements(self, pdf_path: Path) -> dict[int, list[VisualTextElement]]:
        """Extraire les elements de texte avec leurs caracteristiques visuelles.

        Utilise pdfplumber pour obtenir:
        - Taille de police
        - Nom de la police (pour detecter le gras)
        - Position sur la page

        Args:
            pdf_path: Chemin vers le PDF

        Returns:
            Dict {page_number: liste de VisualTextElement}
        """
        visual_elements: dict[int, list[VisualTextElement]] = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_elements = []

                    # Extraire les caracteres individuels avec leurs proprietes
                    chars = page.chars or []

                    if not chars:
                        continue

                    # Regrouper les caracteres par ligne (meme position Y approximative)
                    lines: dict[int, list] = {}
                    tolerance = 3  # Tolerance pour regrouper sur la meme ligne

                    for char in chars:
                        y_pos = round(char.get("top", 0) / tolerance)
                        if y_pos not in lines:
                            lines[y_pos] = []
                        lines[y_pos].append(char)

                    # Traiter chaque ligne - construire UNE entree par ligne
                    for line_idx, (y_key, line_chars) in enumerate(sorted(lines.items())):
                        # Trier par position X
                        line_chars.sort(key=lambda c: c.get("x0", 0))

                        # Collecter toutes les informations de la ligne
                        line_text = "".join(c.get("text", "") for c in line_chars)

                        if len(line_text.strip()) < 5:
                            continue

                        # Taille de police: prendre le MAX (pas la moyenne)
                        sizes = [c.get("size", 0) for c in line_chars if c.get("size", 0) > 0]
                        max_font_size = max(sizes) if sizes else 0

                        # Police: verifier si au moins un caractere est en gras
                        fonts = set(c.get("fontname", "") for c in line_chars)
                        is_bold = any(self._is_bold_font(f) for f in fonts)

                        # Position
                        x0 = min(c.get("x0", 0) for c in line_chars)
                        y0 = min(c.get("top", 0) for c in line_chars)
                        x1 = max(c.get("x1", 0) for c in line_chars)
                        y1 = max(c.get("bottom", 0) for c in line_chars)

                        # Creer l'element
                        elem = VisualTextElement(
                            text=line_text.strip(),
                            page=page_num,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            font_size=max_font_size,
                            font_name=next(iter(fonts), ""),
                            is_bold=is_bold,
                            is_uppercase=line_text.strip().isupper(),
                            line_number=line_idx,
                            page_width=float(getattr(page, "width", 0) or 0),
                            page_height=float(getattr(page, "height", 0) or 0),
                        )
                        page_elements.append(elem)

                    visual_elements[page_num] = page_elements

        except Exception as e:
            logger.warning(f"Erreur extraction visuelle: {e}")
            return {}

        return visual_elements

    def _is_bold_font(self, font_name: str) -> bool:
        """Determiner si le nom de police indique une graisse en gras."""
        if not font_name:
            return False
        font_lower = font_name.lower()
        return any(marker in font_lower for marker in ["bold", "heavy", "black", "demi", "semi", "medium"])

    def _merge_adjacent_elements(self, elements: list[VisualTextElement]) -> list[VisualTextElement]:
        """Fusionner les elements adjacents sur la meme ligne.

        Args:
            elements: Liste d'elements a fusionner

        Returns:
            Liste d'elements fusionnes par ligne
        """
        if not elements:
            return []

        # Trier par ligne puis par position X
        elements.sort(key=lambda e: (e.line_number, e.x0))

        merged = []
        current = None

        for elem in elements:
            if current is None:
                current = elem
                continue

            # Meme ligne et proche (ecart < 50 pixels)?
            if elem.line_number == current.line_number and elem.x0 - current.x1 < 50:
                # Fusionner
                current = VisualTextElement(
                    text=current.text + " " + elem.text,
                    page=current.page,
                    x0=current.x0,
                    y0=min(current.y0, elem.y0),
                    x1=elem.x1,
                    y1=max(current.y1, elem.y1),
                    font_size=(current.font_size + elem.font_size) / 2,
                    font_name=current.font_name,
                    is_bold=current.is_bold or elem.is_bold,
                    is_uppercase=current.is_uppercase and elem.is_uppercase,
                    line_number=current.line_number,
                    page_width=current.page_width or elem.page_width,
                    page_height=current.page_height or elem.page_height,
                )
            else:
                merged.append(current)
                current = elem

        if current:
            merged.append(current)

        return merged

    def _detect_section_headers_visual(
        self,
        visual_elements: dict[int, list[VisualTextElement]],
        text_by_page: dict[int, str],
    ) -> list[LocatedSection]:
        """Detecter les titres de sections en utilisant les caracteristiques visuelles.

        Cette methode cherche les elements qui:
        - Ont une grande taille de police (> moyenne de la page)
        - Sont en gras
        - Sont en majuscules
        - Sont positionnes en haut de page
        - Correspondent aux patterns de sections cibles

        Args:
            visual_elements: Elements visuels par page
            text_by_page: Texte par page (pour validation contextuelle)

        Returns:
            Liste des sections detectees visuellement
        """
        # Collecter TOUS les candidats d'abord, puis choisir le meilleur par type
        candidates: dict[str, list[tuple[LocatedSection, float]]] = {
            "gestion_capital": [],
            "gestion_risques": [],
        }

        # Calculer la taille de police moyenne du document
        all_sizes = []
        for page_elements in visual_elements.values():
            for elem in page_elements:
                if elem.font_size > 0:
                    all_sizes.append(elem.font_size)

        if not all_sizes:
            return []

        avg_font_size = sum(all_sizes) / len(all_sizes)
        header_threshold = avg_font_size * 1.2  # 20% plus grand que la moyenne

        logger.debug(f"Detection visuelle: taille moyenne={avg_font_size:.1f}, seuil titres={header_threshold:.1f}")

        # Scanner les pages (ignorer les premieres pages = TDM, intro)
        for page_num in sorted(visual_elements.keys()):
            if page_num < 5:
                continue

            page_elements = visual_elements[page_num]

            for elem in page_elements:
                # Verifier si c'est potentiellement un titre
                is_header_candidate = (
                    elem.font_size >= header_threshold or elem.is_bold or (elem.is_uppercase and len(elem.text) > 15)
                )

                if not is_header_candidate:
                    continue

                # Position: titre en haut de page (premier tiers)
                # Ou en debut de ligne (x0 proche de la marge gauche)
                is_top_of_page = elem.line_number < 10
                is_left_aligned = elem.x0 < 150  # Marge gauche typique

                if not (is_top_of_page or is_left_aligned):
                    continue

                # Verifier si le texte correspond a un pattern de section

                for section_type, config in self.compiled_patterns.items():
                    # Verifier les patterns
                    for pattern in config["regex"]:
                        if pattern.search(elem.text):
                            # Calculer un score de confiance base sur les caracteristiques visuelles
                            visual_score = self._calculate_visual_confidence(
                                elem, avg_font_size, is_top_of_page, is_left_aligned
                            )

                            # Creer une section temporaire pour validation contextuelle
                            temp_section = LocatedSection(
                                section_type=section_type,
                                title_found=elem.text,
                                start_page=page_num,
                                end_page=min(page_num + 10, max(text_by_page.keys())),
                                detection_method="visual_temp",
                            )

                            # Validation contextuelle
                            is_valid, content_score = self._validate_section_content(temp_section, text_by_page)

                            final_confidence = visual_score * 0.6 + content_score * 0.4

                            if final_confidence > 0.3:  # Seuil plus bas pour collecter
                                section = LocatedSection(
                                    section_type=section_type,
                                    title_found=elem.text,
                                    start_page=page_num,
                                    confidence=final_confidence,
                                    detection_method="visual",
                                )
                                # Ajouter aux candidats avec le score visuel brut
                                # (taille de police comme critere de departage)
                                candidates[section_type].append((section, elem.font_size))
                                logger.debug(
                                    f"Candidat visuel: {section_type} page {page_num} "
                                    f"(taille={elem.font_size:.1f}, gras={elem.is_bold}, "
                                    f"conf={final_confidence:.2f})"
                                )
                            break

        # Selectionner le meilleur candidat pour chaque type de section
        # Critere: priorite a la taille de police (titres plus grands = plus fiables)
        sections = []
        for section_type, section_candidates in candidates.items():
            if not section_candidates:
                continue

            # Trier par taille de police (desc) puis par confiance (desc)
            section_candidates.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
            best_section, best_size = section_candidates[0]

            # Verifier que le meilleur a une confiance acceptable
            if best_section.confidence > 0.4:
                sections.append(best_section)
                logger.info(
                    f"Section detectee visuellement: {section_type} page {best_section.start_page} "
                    f"(taille={best_size:.1f}, conf={best_section.confidence:.2f}) "
                    f"[{len(section_candidates)} candidats]"
                )

        return sections

    def _calculate_visual_confidence(
        self,
        elem: VisualTextElement,
        avg_font_size: float,
        is_top_of_page: bool,
        is_left_aligned: bool,
    ) -> float:
        """Calculer un score de confiance base sur les caracteristiques visuelles.

        Args:
            elem: Element visuel
            avg_font_size: Taille moyenne de police du document
            is_top_of_page: Element en haut de page
            is_left_aligned: Element aligne a gauche

        Returns:
            Score entre 0 et 1
        """
        score = 0.0

        # Taille de police (max 0.35)
        if elem.font_size > avg_font_size * 1.5:
            score += 0.35  # Beaucoup plus grand
        elif elem.font_size > avg_font_size * 1.2:
            score += 0.25  # Plus grand
        elif elem.font_size > avg_font_size:
            score += 0.15

        # Gras (max 0.25)
        if elem.is_bold:
            score += 0.25

        # Majuscules (max 0.15)
        if elem.is_uppercase:
            score += 0.15

        # Position (max 0.25)
        if is_top_of_page:
            score += 0.15
        if is_left_aligned:
            score += 0.10

        return min(1.0, score)

    def _find_boundary_header_on_page(
        self,
        page: int,
        patterns: list[re.Pattern],
        title_candidates: list[str],
        visual_elements: dict[int, list[VisualTextElement]],
    ) -> VisualTextElement | None:
        """Localise le titre de la section suivante sur une page partagée."""
        matches: list[VisualTextElement] = []
        for elem in visual_elements.get(page, []):
            if not self._matches_boundary_title(elem.text, patterns, title_candidates):
                continue
            bbox = elem.bbox_norm
            if not bbox or float(bbox[1]) <= SHARED_PAGE_TOP_THRESHOLD:
                continue
            matches.append(elem)
        if not matches:
            return None
        headed = [elem for elem in matches if elem.is_likely_header]
        pool = headed or matches
        return max(pool, key=lambda elem: float(elem.y0))

    def _refine_shared_page_boundaries(
        self,
        sections: list[LocatedSection],
        toc_entries: list[TocEntry],
        visual_elements: dict[int, list[VisualTextElement]],
    ) -> list[LocatedSection]:
        """Étend end_page à la page partagée quand la section suivante ne commence pas en haut."""
        if not sections:
            return sections

        refined: list[LocatedSection] = []
        for section in sections:
            if section.end_page is None:
                refined.append(section)
                continue

            boundary_page = int(section.end_page) + 1
            patterns = self.following_patterns.get(section.section_type, [])
            title_candidates = list(self._next_toc_boundary_title_candidates(section, toc_entries))
            end_anchor = str(section.end_anchor_text or "").strip()
            if end_anchor and end_anchor not in title_candidates:
                title_candidates.insert(0, end_anchor)
            boundary = self._find_boundary_header_on_page(
                boundary_page,
                patterns,
                title_candidates,
                visual_elements,
            )
            bbox_norm = boundary.bbox_norm if boundary is not None else None
            if boundary is not None and bbox_norm:
                logger.info(
                    "Page partagée détectée pour %s: extension p.%s -> p.%s, frontière '%s' y=%.3f",
                    section.section_type,
                    section.end_page,
                    boundary_page,
                    boundary.text[:60],
                    float(bbox_norm[1]),
                )
                refined.append(
                    replace(
                        section,
                        end_page=boundary_page,
                        end_anchor_page=boundary_page,
                        end_anchor_text=boundary.text,
                        end_anchor_bbox_norm=list(bbox_norm),
                        end_detection_method=f"{section.end_detection_method}+shared_page"
                        if section.end_detection_method
                        else "shared_page",
                    )
                )
                continue

            if end_anchor and int(section.end_anchor_page or 0) == boundary_page:
                logger.info(
                    "Page partagée TOC pour %s: extension p.%s -> p.%s sans bbox visuelle",
                    section.section_type,
                    section.end_page,
                    boundary_page,
                )
                refined.append(
                    replace(
                        section,
                        end_page=boundary_page,
                        end_anchor_page=boundary_page,
                        end_anchor_text=end_anchor,
                        end_detection_method=f"{section.end_detection_method}+shared_page_toc"
                        if section.end_detection_method
                        else "shared_page_toc",
                    )
                )
                continue

            refined.append(section)
        return refined

    def _resolve_section_anchor(
        self,
        section: LocatedSection,
        visual_elements: dict[int, list[VisualTextElement]],
    ) -> LocatedSection:
        """Resoudre une ancre intra-page a partir du bloc titre reel de la section."""
        if not section.start_page:
            return replace(section, anchor_found=False)

        page_elements = visual_elements.get(section.start_page, [])
        if not page_elements:
            return replace(section, anchor_found=False)

        candidates = self._get_section_anchor_candidates(section)
        if not candidates:
            return replace(section, anchor_found=False)

        def _candidate_sort_key(elem: VisualTextElement) -> tuple[int, float, float, int]:
            """Clé de tri privilégiant les en-têtes, position verticale, grande police."""
            return (
                0 if elem.is_likely_header else 1,
                float(elem.y0),
                -float(elem.font_size),
                int(elem.line_number),
            )

        for candidate_text in candidates:
            candidate_variants = self._title_match_variants(candidate_text)
            matches = [elem for elem in page_elements if self._title_match_variants(elem.text) & candidate_variants]
            if not matches:
                continue

            best = sorted(matches, key=_candidate_sort_key)[0]
            bbox_norm = best.bbox_norm
            if not bbox_norm:
                continue

            return replace(
                section,
                anchor_page=section.start_page,
                anchor_text=best.text,
                anchor_bbox_norm=bbox_norm,
                anchor_found=True,
            )

        return replace(section, anchor_found=False)

    def _resolve_section_anchors(
        self,
        sections: list[LocatedSection],
        visual_elements: dict[int, list[VisualTextElement]],
    ) -> list[LocatedSection]:
        """Resoudre les ancres de toutes les sections localisees."""
        resolved: list[LocatedSection] = []
        for section in sections:
            anchored = self._resolve_section_anchor(section, visual_elements)
            if anchored.anchor_found:
                logger.info(
                    "Ancre section resolue: %s page %s -> '%s'",
                    anchored.section_type,
                    anchored.anchor_page,
                    anchored.anchor_text,
                )
            elif section.detection_method.startswith("manual_override"):
                logger.debug(
                    "Ancre section non resolue pour override manuel: %s page %s",
                    section.section_type,
                    section.start_page,
                )
            else:
                logger.warning(
                    "Ancre section introuvable: %s page %s title_found='%s'",
                    section.section_type,
                    section.start_page,
                    section.title_found,
                )
            resolved.append(anchored)
        return resolved

    def _get_section_anchor_candidates(self, section: LocatedSection) -> list[str]:
        """Retourner les libelles exacts a tester pour l'ancre de debut de section."""
        candidates: list[str] = []
        title_found = str(section.title_found or "").strip()
        if title_found:
            candidates.append(title_found)

        for section_key in self._section_alias_keys(section.section_type):
            for alias in SECTION_TITLE_ALIASES.get(section_key, []):
                alias = str(alias or "").strip()
                if alias and normalize_text(alias) not in {normalize_text(existing) for existing in candidates}:
                    candidates.append(alias)

            for alias in self._get_config_section_names(section_key):
                alias = str(alias or "").strip()
                if alias and normalize_text(alias) not in {normalize_text(existing) for existing in candidates}:
                    candidates.append(alias)

        return candidates

    def _matches_boundary_title(
        self,
        text: str,
        patterns: list[re.Pattern],
        title_candidates: list[str],
    ) -> bool:
        """Indique si un bloc titre correspond à une frontière de section suivante."""
        stripped = str(text or "").strip()
        if not stripped:
            return False
        unstuttered = self._unstutter_pdf_text(stripped)
        value = normalize_text(stripped)
        for title in title_candidates:
            title_norm = normalize_text(title)
            if not title_norm:
                continue
            if title_norm in value or value in title_norm or self._text_similarity(value, title_norm) > 0.75:
                return True
        for pattern in patterns:
            if pattern.search(stripped) or pattern.search(unstuttered):
                return True
        return False
