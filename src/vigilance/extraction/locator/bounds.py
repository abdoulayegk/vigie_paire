"""Determination et correction des bornes de section (page de fin, contraintes de longueur).

Extrait de ``section_locator.py`` sans modification des corps de methodes.
Mixin consomme par ``SectionLocator``.
"""

from __future__ import annotations

import logging
import re

from .models import LocatedSection, TocEntry, normalize_text
from .patterns import SECTION_PATTERNS

# Nom de logger conserve a l'identique apres le decoupage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigilance.extraction.section_locator")


class BoundsMixin:
    """Determination et correction des bornes de section (page de fin, contraintes de longueur)."""

    def _determine_end_pages(
        self,
        sections: list[LocatedSection],
        text_by_page: dict[int, str],
        toc_entries: list[TocEntry],
        total_pages: int,
    ) -> list[LocatedSection]:
        """Determiner les pages de fin avec la logique hybride a 3 niveaux.

        Priorite:
        1. Override manuel (deja applique)
        2. TDM - page debut section suivante
        3. Scan - detection pattern section suivante
        4. Fallback - estimation contextuelle

        Args:
            sections: Sections avec start_page
            text_by_page: Texte par page
            toc_entries: Entrees TDM
            total_pages: Nombre total de pages

        Returns:
            Sections avec end_page determine
        """
        if not sections:
            return sections

        # Trier par page de debut
        sections = sorted(sections, key=lambda s: s.start_page)

        for i, section in enumerate(sections):
            # Les rapports annuels T4 ont une structure de TDM souvent plate:
            # un sous-thème (p. ex. propriété intellectuelle) peut y ressembler
            # à un nouveau chapitre. Une fin pré-calculée depuis cette TDM ne
            # doit donc jamais court-circuiter la recherche du vrai successeur.
            if self._is_annual_t4_target_section(section) and not section.detection_method.startswith(
                "manual_override"
            ):
                end_page, end_method = self._detect_annual_t4_section_end(
                    section.section_type, section.start_page, text_by_page, total_pages
                )
                section.end_page = end_page
                section.end_detection_method = end_method
                self._apply_section_length_constraints(section, total_pages, source="annual_t4_boundary")
                continue

            # Si end_page deja defini (override, TDM, etc.), appliquer quand meme
            # les contraintes de longueur avant de continuer.
            if section.end_page is not None:
                self._apply_section_length_constraints(section, total_pages, source="predefined")
                continue

            constraints = self._get_section_length_constraints(section.section_type)
            default_length = constraints["default_length"]

            end_page = None
            end_method = ""

            # Niveau 2: Scanner pour la section suivante explicite. Les titres
            # "followed_by" bornent mieux les sections vigie que la prochaine
            # section cible quand des blocs intermediaires existent.
            if not end_page:
                end_page, end_method = self._detect_section_end(
                    section.section_type, section.start_page, text_by_page, total_pages
                )

            # Niveau 3: utiliser la prochaine section cible detectee quand elle
            # existe et qu'aucun titre de fin plus precis n'a ete trouve.
            if not end_page and i + 1 < len(sections):
                end_page = sections[i + 1].start_page - 1
                end_method = "next_target_section"

            # Niveau 4: Utiliser la TDM si aucune borne locale n'a ete trouvee.
            if toc_entries and not end_page:
                end_page, end_method = self._find_end_from_toc(section.section_type, section.start_page, toc_entries)

            # Niveau 5: Fallback - estimation contextuelle
            if not end_page:
                # Estimation contextuelle bornee par contraintes de la section
                end_page = min(section.start_page + default_length - 1, total_pages)
                end_method = "estimation"

            section.end_page = end_page
            section.end_detection_method = end_method

            # Affiner les limites avec les sous-sections (Amélioration 3)
            if section.section_type in {"gestion_capital", "gestion_risques"}:
                section = self._refine_bounds_with_subsections(section, text_by_page)

            self._apply_section_length_constraints(section, total_pages, source="determine_end")

        return sections

    def _detect_section_end(
        self,
        section_type: str,
        start_page: int,
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> tuple[int | None, str]:
        """Detecter la fin d'une section en scannant le PDF.

        Cherche les patterns des sections qui suivent typiquement cette section.

        Args:
            section_type: Type de section
            start_page: Page de debut
            text_by_page: Texte par page
            total_pages: Nombre total de pages

        Returns:
            Tuple (end_page, method) ou (None, "")
        """
        following_patterns = self.following_patterns.get(section_type, [])

        if not following_patterns:
            return None, ""

        # Scanner les pages apres le debut de la section
        constraints = self._get_section_length_constraints(section_type)
        min_length = constraints["min_length"]
        max_length = constraints["max_length"]

        search_start = start_page + min_length
        search_end = min(start_page + max_length, total_pages)

        for page_num in range(search_start, search_end + 1):
            page_text = text_by_page.get(page_num, "")
            lines = page_text.split("\n")

            for line in lines:
                line_stripped = line.strip()
                line_unstuttered = self._unstutter_pdf_text(line_stripped)

                # Verifier si c'est un titre potentiel
                if not (
                    self._is_likely_section_title(line_stripped, page_text)
                    or self._is_likely_section_title(line_unstuttered, page_text)
                ):
                    continue
                if self._is_weak_section_scan_line(line_stripped, section_type) or self._is_weak_section_scan_line(
                    line_unstuttered, section_type
                ):
                    continue

                # Verifier contre les patterns des sections suivantes
                for pattern in following_patterns:
                    if pattern.search(line_stripped) or pattern.search(line_unstuttered):
                        # Verifier que ce n'est pas une sous-section
                        if self._is_risk_subsection(line_stripped):
                            continue

                        logger.debug(f"Fin de {section_type} detectee page {page_num}: {line_stripped[:40]}...")
                        return page_num - 1, "following_section_scan"

        return None, ""

    def _find_next_section_by_pattern(
        self, section_type: str, start_page: int, toc_entries: list[TocEntry]
    ) -> tuple[int, str] | None:
        """Trouver la prochaine section via les patterns 'followed_by' et les sections cibles.

        Cette methode cherche dans les entrees TDM :
        1. Les sections qui correspondent aux patterns 'followed_by'
        2. Les autres sections cibles (capital_management si on cherche risk_management, etc.)

        Args:
            section_type: Type de section actuelle (gestion_capital ou gestion_risques)
            start_page: Page de debut de la section actuelle
            toc_entries: Entrees TDM

        Returns:
            Tuple (page, title) ou None si aucune section suivante trouvee
        """
        if not self.bank_code or not self.bank_config:
            return None

        # Obtenir les sections suivantes configurees
        config_name = "capital_management" if section_type == "gestion_capital" else "risk_management"
        bank_data = self.bank_config.get("banks", {}).get(self.bank_code, {})
        sections = bank_data.get("sections", {})
        section_config = sections.get(config_name, {})
        followed_by = section_config.get("followed_by", [])

        # Obtenir aussi les noms de l'autre section cible
        other_config_name = "risk_management" if section_type == "gestion_capital" else "capital_management"
        other_section_config = sections.get(other_config_name, {})
        other_section_names = other_section_config.get("names", [])

        # Combiner les patterns a chercher
        patterns_to_search = list(followed_by)  # D'abord les patterns 'followed_by'
        patterns_to_search.extend(other_section_names)  # Puis les autres sections cibles

        if not patterns_to_search:
            return None

        # Obtenir la longueur minimale de la section courante
        min_length = self._get_section_length_constraints(section_type)["min_length"]

        # Chercher dans les entrees TDM apres start_page
        min_end_page = start_page + min_length
        entries_after = [e for e in toc_entries if e.page >= min_end_page]

        if not entries_after:
            return None

        # Trier par page croissante
        entries_after.sort(key=lambda e: e.page)

        # Chercher la premiere entree qui correspond a un pattern
        for entry in entries_after:
            entry_title_normalized = normalize_text(entry.title)

            for pattern_name in patterns_to_search:
                pattern_normalized = normalize_text(pattern_name)

                # Match partiel ou exact
                if (
                    pattern_normalized in entry_title_normalized
                    or entry_title_normalized in pattern_normalized
                    or self._text_similarity(entry_title_normalized, pattern_normalized) > 0.7
                ):
                    # Determiner la source du match pour le log
                    if pattern_name in followed_by:
                        source = "followed_by"
                    else:
                        source = f"other_target_section ({other_config_name})"

                    logger.debug(
                        f"Section suivante trouvee par pattern: '{entry.title}' page {entry.page} "
                        f"(pattern: '{pattern_name}', section: {section_type}, source: {source})"
                    )
                    return (entry.page, entry.title)

        return None

    def _estimate_end_pages(self, sections: list[LocatedSection], total_pages: int) -> list[LocatedSection]:
        """Estimer les pages de fin pour les sections (methode legacy).

        Args:
            sections: Sections avec start_page
            total_pages: Nombre total de pages

        Returns:
            Sections avec end_page estime
        """
        if not sections:
            return sections

        # Trier par page de debut
        sections = sorted(sections, key=lambda s: s.start_page)

        for i, section in enumerate(sections):
            if section.end_page is None:
                if i + 1 < len(sections):
                    # La section se termine avant la section suivante
                    section.end_page = sections[i + 1].start_page - 1
                    section.end_detection_method = "next_section"
                else:
                    # Derniere section: estimer la fin
                    constraints = self._get_section_length_constraints(section.section_type)
                    estimated_length = constraints["default_length"]
                    section.end_page = min(section.start_page + estimated_length - 1, total_pages)
                    section.end_detection_method = "estimation"

                self._apply_section_length_constraints(section, total_pages, source="legacy_estimate")

        return sections

    def _correct_section_bounds(
        self,
        section: LocatedSection,
        toc_detections: list[TocEntry],
        scan_detections: list[LocatedSection],
    ) -> LocatedSection | None:
        """Corriger les limites d'une section selon le consensus des methodes.

        Utilise la mediane ponderee des pages detectees par toutes les methodes.

        Args:
            section: Section a corriger
            toc_detections: Entrees TDM correspondant a cette section
            scan_detections: Sections detectees par scan

        Returns:
            Section corrigee ou None si aucune correction n'est possible
        """
        # Collecter toutes les pages de debut
        start_pages = []
        if section.start_page:
            start_pages.append((section.start_page, 1.0))

        for toc_entry in toc_detections:
            if toc_entry.page:
                start_pages.append((toc_entry.page, 0.8))

        for scan_section in scan_detections:
            if scan_section.start_page:
                start_pages.append((scan_section.start_page, 0.6))

        # Collecter toutes les pages de fin
        end_pages = []
        if section.end_page:
            end_pages.append((section.end_page, 1.0))

        for scan_section in scan_detections:
            if scan_section.end_page:
                end_pages.append((scan_section.end_page, 0.6))

        # Calculer la mediane ponderee pour le debut
        corrected_start = section.start_page
        if start_pages:
            sorted_starts = sorted(start_pages, key=lambda x: x[0])
            total_weight = sum(w for _, w in sorted_starts)

            if total_weight > 0:
                # Calculer la mediane ponderee
                cumulative_weight = 0
                median_weight = total_weight / 2

                for page, weight in sorted_starts:
                    cumulative_weight += weight
                    if cumulative_weight >= median_weight:
                        corrected_start = page
                        break

        # Calculer la mediane ponderee pour la fin
        corrected_end = section.end_page
        if end_pages:
            sorted_ends = sorted(end_pages, key=lambda x: x[0])
            total_weight = sum(w for _, w in sorted_ends)

            if total_weight > 0:
                cumulative_weight = 0
                median_weight = total_weight / 2

                for page, weight in sorted_ends:
                    cumulative_weight += weight
                    if cumulative_weight >= median_weight:
                        corrected_end = page
                        break

        # Verifier si une correction est necessaire
        start_changed = corrected_start != section.start_page
        end_changed = corrected_end != section.end_page

        if start_changed or end_changed:
            # Creer une copie corrigee
            corrected = LocatedSection(
                section_type=section.section_type,
                title_found=section.title_found,
                start_page=corrected_start,
                end_page=corrected_end,
                confidence=section.confidence,
                detection_method=f"{section.detection_method}_corrected",
                end_detection_method=f"{section.end_detection_method}_corrected",
            )

            logger.info(
                f"Limites corrigees pour {section.section_type}: "
                f"{section.start_page}-{section.end_page} -> {corrected_start}-{corrected_end}"
            )

            return corrected

        return None

    def _apply_section_length_constraints(
        self, section: LocatedSection, total_pages: int, source: str = ""
    ) -> LocatedSection:
        """Appliquer les contraintes min/max/default de longueur a une section.

        Args:
            section: Section a contraindre
            total_pages: Nombre total de pages du document
            source: Etiquette indiquant l'origine de l'appel (pour le log)

        Returns:
            La section modifiee en place avec les contraintes appliquees.
        """
        constraints = self._get_section_length_constraints(section.section_type)
        min_length = constraints["min_length"]
        max_length = constraints["max_length"]
        default_length = constraints["default_length"]

        if not section.start_page:
            return section

        reason_parts: list[str] = []
        applied = section.constraint_applied

        # Fin absente -> fallback deterministic
        if section.end_page is None:
            section.end_page = min(total_pages, section.start_page + default_length - 1)
            reason_parts.append(f"end_missing->default_{default_length}")
            applied = True

        if section.end_page is not None:
            detected_span = max(1, section.end_page - section.start_page + 1)
            section.detected_span = detected_span

            # Respecter le minimum
            if detected_span < min_length:
                section.end_page = min(total_pages, section.start_page + min_length - 1)
                reason_parts.append(f"min_enforced_{detected_span}->{min_length}")
                applied = True

            # Respecter le maximum
            current_span = max(1, section.end_page - section.start_page + 1)
            if current_span > max_length:
                section.end_page = min(total_pages, section.start_page + max_length - 1)
                reason_parts.append(f"max_enforced_{current_span}->{max_length}")
                applied = True

            section.final_span = max(1, section.end_page - section.start_page + 1)

        if applied and reason_parts:
            suffix = f" [{source}]" if source else ""
            section.constraint_reason = "; ".join(reason_parts) + suffix
            section.constraint_applied = True
            logger.info(
                f"Contrainte section appliquee ({section.section_type}): {section.constraint_reason} "
                f"(pages {section.start_page}-{section.end_page})"
            )
        elif section.end_page is not None:
            # Renseigner aussi dans le cas sans ajustement
            section.detected_span = section.detected_span or (section.end_page - section.start_page + 1)
            section.final_span = section.final_span or section.detected_span

        return section

    def _get_section_length_constraints(self, section_type: str) -> dict[str, int]:
        """Recuperer les contraintes de longueur pour un type de section.

        Priorite:
        1. Default code (gestion_reglementation = section courte 1-3 pages)
        2. section_boundary_detection.section_length_overrides
        3. banks.<bank>.sections.<section>.length_constraints
        """
        boundary_config = self.bank_config.get("section_boundary_detection", {})
        constraints = {
            "min_length": int(boundary_config.get("min_section_length", 3)),
            "max_length": int(boundary_config.get("max_section_length", 50)),
            "default_length": int(boundary_config.get("default_section_length", 20)),
        }

        # Default metier pour la section reglementaire (regulatory_updates)
        if section_type == "gestion_reglementation":
            constraints.update({"min_length": 1, "max_length": 3, "default_length": 3})

        # Overrides globaux optionnels
        overrides = boundary_config.get("section_length_overrides", {})
        override = overrides.get(section_type, {}) if isinstance(overrides, dict) else {}
        if override:
            constraints["min_length"] = int(
                override.get("min_length", override.get("min_pages", constraints["min_length"]))
            )
            constraints["max_length"] = int(
                override.get("max_length", override.get("max_pages", constraints["max_length"]))
            )
            constraints["default_length"] = int(
                override.get(
                    "default_length",
                    override.get("default_span", constraints["default_length"]),
                )
            )

        # Overrides par banque optionnels
        section_name_map = {
            "capital_management": "capital_management",
            "risk_management": "risk_management",
            "regulatory_updates": "regulatory_updates",
            "gestion_capital": "capital_management",
            "gestion_risques": "risk_management",
            "gestion_reglementation": "regulatory_updates",
        }
        if self.bank_code:
            bank_sections = self.bank_config.get("banks", {}).get(self.bank_code, {}).get("sections", {})
            section_name = section_name_map.get(section_type)
            section_cfg = bank_sections.get(section_name, {}) if section_name else {}
            bank_override = section_cfg.get("length_constraints", {})
            if isinstance(bank_override, dict) and bank_override:
                constraints["min_length"] = int(
                    bank_override.get(
                        "min_length",
                        bank_override.get("min_pages", constraints["min_length"]),
                    )
                )
                constraints["max_length"] = int(
                    bank_override.get(
                        "max_length",
                        bank_override.get("max_pages", constraints["max_length"]),
                    )
                )
                constraints["default_length"] = int(
                    bank_override.get(
                        "default_length",
                        bank_override.get("default_span", constraints["default_length"]),
                    )
                )

        # Normalisation defensive
        constraints["min_length"] = max(1, constraints["min_length"])
        constraints["max_length"] = max(constraints["min_length"], constraints["max_length"])
        constraints["default_length"] = min(
            max(constraints["default_length"], constraints["min_length"]),
            constraints["max_length"],
        )

        # Les sections annuelles T4 peuvent être plus longues que dans les
        # rapports trimestriels. Cette exception reste limitée au T4.
        if self._is_t4_quarter() and section_type in {"gestion_capital", "gestion_risques"}:
            constraints["max_length"] = max(constraints["max_length"], 120)
            constraints["default_length"] = max(constraints["default_length"], 60)
        return constraints

    def _is_section_bounds_suspicious(self, section: LocatedSection, total_pages: int) -> bool:
        """Verifier si les bornes d'une section semblent anormales.

        Args:
            section: Section a verifier
            total_pages: Nombre total de pages du document

        Returns:
            True si les bornes sont suspectes (trop courtes, trop longues, etc.).
        """
        if not section.start_page or not section.end_page:
            return True

        constraints = self._get_section_length_constraints(section.section_type)
        min_length = constraints["min_length"]
        max_length = constraints["max_length"]
        length = section.end_page - section.start_page + 1
        if length < min_length:
            return True

        if length > max_length:
            return True

        if total_pages and length > total_pages * 0.8:
            return True

        return False

    def _refine_bounds_with_subsections(self, section: LocatedSection, text_by_page: dict[int, str]) -> LocatedSection:
        """Affiner les limites d'une section en detectant les sous-sections.

        Pour "Gestion des risques":
        - Detecte "Risque de credit", "Risque de marche", etc.
        - La section commence au premier sous-titre
        - La section se termine avant la prochaine section principale

        Args:
            section: Section a affiner
            text_by_page: Texte par page

        Returns:
            Section avec limites affinees si des sous-sections sont trouvees
        """
        # NOTE: L'affinement des limites de section a ete desactive car il causait
        # des sections trop courtes (1-2 pages au lieu de 20+).
        #
        # L'ancienne logique:
        # 1. Cherchait des sous-sections dans les 5 premieres pages et deplacait le debut
        # 2. Cherchait des sous-sections dans les 10 dernieres pages et coupait la fin
        #
        # Problemes:
        # - La page de debut detectee par scan/TOC est generalement correcte
        # - Deplacer le debut faisait perdre le titre de section et l'introduction
        # - Couper la fin a la premiere sous-section trouvee eliminait le reste de la section
        #
        # Solution: Les limites de section sont maintenant determinees uniquement par:
        # - Le scan de titres (detection_method: scan)
        # - La table des matieres (detection_method: toc)
        # - Les overrides manuels (detection_method: manual_override)
        # - La detection de la section suivante (end_detection_method: following_section_*)

        return section

    def _get_subsection_patterns(self, section_type: str) -> list[re.Pattern]:
        """Obtenir les patterns regex pour les sous-sections d'un type de section.

        Args:
            section_type: Type de section (gestion_capital ou gestion_risques)

        Returns:
            Liste de patterns regex compiles pour les sous-sections
        """
        # Patterns de sous-sections selon le type
        subsection_patterns_dict = {
            "gestion_risques": [
                r"risque\s+de\s+cr[eé]dit",
                r"risque\s+de\s+march[eé]",
                r"risque\s+de\s+liquidit[eé]",
                r"risque\s+op[eé]rationnel",
                r"risque\s+de\s+taux\s+d['']inter[eé]t",
                r"risque\s+de\s+change",
                r"credit\s+risk",
                r"market\s+risk",
                r"liquidity\s+risk",
                r"operational\s+risk",
            ],
            "gestion_capital": [
                r"ratio\s+CET1",
                r"ratio\s+de\s+levier",
                r"ratio\s+de\s+liquidit[eé]",
                r"fonds\s+propres\s+r[eé]glementaires",
                r"capital\s+r[eé]glementaire",
                r"Tier\s+1",
                r"Tier\s+2",
                r"TLAC",
                r"LCR",
                r"NSFR",
            ],
        }

        patterns = subsection_patterns_dict.get(section_type, [])

        # Utiliser aussi les patterns depuis SECTION_PATTERNS si disponibles
        if section_type in SECTION_PATTERNS:
            config_subsections = SECTION_PATTERNS[section_type].get("subsections", [])
            patterns.extend(config_subsections)

        # Compiler les patterns
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

        return compiled
